use cgmath::{InnerSpace, Quaternion, Rotation, Vector2, Vector3};
use half::f16;
use image::{ImageBuffer, Rgba};
use serde::Serialize;

use crate::{
    camera::{ Camera, OrthographicProjection, Projection},
    cmap::ColorMap,
    renderer::{FrameBuffer, ImageUpscaler, RenderSettings, VolumeRenderer},
    volume::{Volume, VolumeGPU},
    WGPUContext,
};

async fn render_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut VolumeRenderer,
    upscaler: &ImageUpscaler,
    volume: &VolumeGPU,
    camera: Camera,
    render_settings: &RenderSettings,
    bg: wgpu::Color,
    resolution: Vector2<u32>,
) -> anyhow::Result<(ImageBuffer<Rgba<u8>, Vec<u8>>, RenderStatistics)> {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render texture"),
        size: wgpu::Extent3d {
            width: resolution.x,
            height: resolution.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: renderer.format(),
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let render_resolution = [
        (resolution.x as f32 / render_settings.render_scale) as u32,
        (resolution.y as f32 / render_settings.render_scale) as u32,
    ];
    let frame_buffer = FrameBuffer::new(device, render_resolution[0],render_resolution[1], renderer.format());

    let frame_data = renderer.prepare(device, &volume, &camera, &render_settings, render_resolution ,0);

    let queryset = device.create_query_set(
        &wgpu::QuerySetDescriptor {
            label: Some("query set"),
            ty: wgpu::QueryType::Timestamp,
            count: 4,
        },
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render encoder"),
    });
    encoder.write_timestamp(&queryset, 0);
    {
        let mut render_pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &frame_buffer.color().create_view(&Default::default()),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &frame_buffer.grad_x().create_view(&Default::default()),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &frame_buffer.grad_y().create_view(&Default::default()),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &frame_buffer.grad_xy().create_view(&Default::default()),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                ..Default::default()
            })
            .forget_lifetime();

        renderer.render(queue, &mut render_pass, &frame_data);
    }
    encoder.write_timestamp(&queryset, 1);

    encoder.write_timestamp(&queryset, 2);

    {
        let mut render_pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        upscaler.render(&mut render_pass, &frame_data.settings_bg, &frame_buffer);
    }
    encoder.write_timestamp(&queryset, 3);

    let timestamp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp buffer"),
        size: 4 * std::mem::size_of::<u64>() as u64,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.resolve_query_set(&queryset, 0..4, &timestamp_buffer, 0);

    let done = queue.submit(std::iter::once(encoder.finish()));
    let buff = download_buffer(device, &timestamp_buffer, Some(done)).await;
    let timestamps: Vec<u64> = buff.chunks_exact(8).map(|v| {
        u64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]])
    }).collect();

    let render_time =(timestamps[1] - timestamps[0]) as f32 * queue.get_timestamp_period()*1e-6;
    let upscaling_time = (timestamps[3] - timestamps[2]) as f32 * queue.get_timestamp_period()*1e-6;

    let img = download_texture(&target, device, queue).await;
    return Ok((img, RenderStatistics {
        render_time,
        upscaling_time,
    }));
}

#[derive(Debug, Clone,Serialize)]
pub struct RenderStatistics {
    pub render_time: f32,
    pub upscaling_time: f32,
}

pub async fn render_volume(
    volumes: Vec<Volume>,
    cmap: ColorMap,
    resolution: Vector2<u32>,
    frames: &[f32],
    bg: wgpu::Color,
    vmin: Option<f32>,
    vmax: Option<f32>,
    distance_scale: f32,
    spatial_interpolation: wgpu::FilterMode,
    temporal_interpolation: wgpu::FilterMode,
    axis_scale: Option<Vector3<f32>>,
    direction: Option<Vector3<f32>>,
) -> anyhow::Result<Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let wgpu_context = WGPUContext::new(&instance, None).await;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    let aabb = volumes[0].aabb.clone();
    let volume_gpu: Vec<VolumeGPU> = volumes
        .into_iter()
        .map(|v| VolumeGPU::new(device, queue, v))
        .collect();

    let render_format = wgpu::TextureFormat::Rgba16Float;

    let mut renderer = VolumeRenderer::new(&device, &queue, render_format);

    let upscaler = ImageUpscaler::new(device, render_format);

    let ratio = resolution.x as f32 / resolution.y as f32;
    let radius = aabb.radius();
    let center = aabb.center();
    let offset = direction.unwrap_or(Vector3::new(1., 1., -2.));
    let camera = Camera::new(
        center + offset,
        Quaternion::look_at(-offset, Vector3::unit_y()),
        Projection::Orthographic(OrthographicProjection::new(
            Vector2::new(ratio, 1.) * 2. * radius,
            1e-4,
            100.,
        )),
    );

    let mut images: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> = Vec::with_capacity(frames.len());
    for time in frames {
        let (img,_) = render_view(
            device,
            queue,
            &mut renderer,
            &upscaler,
            &volume_gpu[0],
            camera,
            &RenderSettings {
                time: *time,
                vmin,
                vmax,
                distance_scale,
                spatial_filter: spatial_interpolation,
                temporal_filter: temporal_interpolation,
                axis_scale: axis_scale.unwrap_or(Vector3::new(1., 1., 1.)).into(),
                cmap: cmap.clone(),
                ..Default::default()
            },
            bg,
            resolution,
        )
        .await?;
        images.push(img);
    }
    Ok(images)
}


pub async fn render_volume_directions(
    volumes: Vec<Volume>,
    resolution: Vector2<u32>,
    bg: wgpu::Color,
    directions: Vec<Vector3<f32>>,
    render_settings: &RenderSettings
) -> anyhow::Result<(Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>,RenderStatistics)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let wgpu_context = WGPUContext::new(&instance, None).await;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    let aabb = volumes[0].aabb.clone();
    let volume_gpu: Vec<VolumeGPU> = volumes
        .into_iter()
        .map(|v| VolumeGPU::new(device, queue, v))
        .collect();

    let render_format = wgpu::TextureFormat::Rgba16Float;

    let mut renderer = VolumeRenderer::new(&device, &queue, render_format);

    let upscaler = ImageUpscaler::new(device, render_format);

    let ratio = resolution.x as f32 / resolution.y as f32;
    let radius = aabb.radius();
    let center = aabb.center();

    let mut stats_avg = RenderStatistics {
        render_time: 0.,
        upscaling_time: 0.,
    };

    let mut images: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> = Vec::with_capacity(directions.len());
    for direction in &directions {
        let mut up = Vector3::unit_y();
        if direction.dot(up).abs() > 0.95 {
            up = Vector3::unit_z();
        }

        let camera = Camera::new(
            center + direction,
            Quaternion::look_at(-*direction, up),
            Projection::Orthographic(OrthographicProjection::new(
                Vector2::new(ratio, 1.) * 2. * radius,
                1e-4,
                100.,
            )),
        );
        let (img,stats) = render_view(
            device,
            queue,
            &mut renderer,
            &upscaler,
            &volume_gpu[0],
            camera,
            &render_settings,
            bg,
            resolution,
        )
        .await?;

        
        stats_avg.render_time += stats.render_time / (&directions).len() as f32;
        stats_avg.upscaling_time += stats.upscaling_time / (&directions).len() as f32;
        images.push(img);
    }
    Ok((images, stats_avg))
}

pub async fn download_texture(
    texture: &wgpu::Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let texture_format = texture.format();

    let texel_size: u32 = texture_format.block_copy_size(None).unwrap();
    let fb_size = texture.size();
    let align: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1;
    let bytes_per_row = (texel_size * fb_size.width) + align & !align;

    let output_buffer_size = (bytes_per_row * fb_size.height) as wgpu::BufferAddress;

    let output_buffer_desc = wgpu::BufferDescriptor {
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: Some("texture download buffer"),
        mapped_at_creation: false,
    };
    let staging_buffer = device.create_buffer(&output_buffer_desc);

    let mut encoder: wgpu::CommandEncoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download frame buffer encoder"),
        });

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(fb_size.height),
            },
        },
        fb_size,
    );
    let sub_idx = queue.submit(std::iter::once(encoder.finish()));

    let mut image = {
        let data: wgpu::BufferView<'_> =
            download_buffer(device, &staging_buffer, Some(sub_idx)).await;

        let data_u8 = data
            .chunks_exact(2)
            .map(|v| (f16::from_le_bytes([v[0], v[1]]).to_f32() * 255.).clamp(0., 255.) as u8)
            .collect::<Vec<_>>();

        ImageBuffer::<Rgba<_>, _>::from_raw(
            bytes_per_row / texel_size,
            fb_size.height,
            data_u8.to_vec(),
        )
        .unwrap()
    };
    staging_buffer.unmap();

    return image::imageops::crop(&mut image, 0, 0, fb_size.width, fb_size.height).to_image();
}

async fn download_buffer<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
    wait_idx: Option<wgpu::SubmissionIndex>,
) -> wgpu::BufferView<'a> {
    let slice = buffer.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(match wait_idx {
        Some(idx) => wgpu::Maintain::WaitForSubmissionIndex(idx),
        None => wgpu::Maintain::Wait,
    });
    rx.receive().await.unwrap().unwrap();

    let view = slice.get_mapped_range();
    return view;
}
