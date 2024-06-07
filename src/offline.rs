use cgmath::Vector2;
use image::{ImageBuffer, Rgba};

use crate::{
    camera::{Camera, OrthographicProjection, Projection},
    cmap::{ColorMapGPU, GenericColorMap, COLORMAP_RESOLUTION},
    renderer::{RenderSettings, VolumeRenderer},
    volume::{Volume, VolumeGPU},
    WGPUContext,
};

async fn render_view<P: Projection>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut VolumeRenderer,
    volume: &VolumeGPU,
    cmap: &ColorMapGPU,
    camera: Camera<P>,
    render_settings: &RenderSettings,
    bg: wgpu::Color,
    resolution: Vector2<u32>,
) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render encoder"),
    });
    let frame_data = renderer.prepare(device, volume, &camera, &render_settings, cmap);
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(bg),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        renderer.render(&mut render_pass, &frame_data);
    }
    queue.submit(std::iter::once(encoder.finish()));
    let img = download_texture(&target, device, queue).await;
    return Ok(img);
}

pub async fn render_volume(
    volumes: Vec<Volume>,
    cmap: GenericColorMap,
    resolution: Vector2<u32>,
    frames: &[f32],
    bg: wgpu::Color,
    vmin: Option<f32>,
    vmax: Option<f32>,
    distance_scale: f32,
    spatial_interpolation: wgpu::FilterMode,
    temporal_interpolation: wgpu::FilterMode,
) -> anyhow::Result<Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let wgpu_context = WGPUContext::new(&instance, None).await;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    let aabb = volumes[0].aabb.clone();
    let volume_gpu: Vec<VolumeGPU> = volumes
        .into_iter()
        .map(|v| VolumeGPU::new(device, queue, v))
        .collect();
    let cmap_gpu = ColorMapGPU::new(&cmap, device, queue, COLORMAP_RESOLUTION);

    let render_format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let mut renderer = VolumeRenderer::new(&device, render_format);

    let ratio = resolution.x as f32 / resolution.y as f32;
    let radius = aabb.radius();
    let camera = Camera::new_aabb_iso(
        aabb,
        OrthographicProjection::new(Vector2::new(ratio, 1.) * radius * 2., 0.01, 1000.),
    );

    let mut images: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> = Vec::with_capacity(frames.len());
    for time in frames {
        let img = render_view(
            device,
            queue,
            &mut renderer,
            &volume_gpu[0],
            &cmap_gpu,
            camera,
            &RenderSettings {
                time: *time,
                vmin,
                vmax,
                distance_scale,
                spatial_filter: spatial_interpolation,
                temporal_filter: temporal_interpolation,
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
        wgpu::ImageCopyBufferBase {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
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

        ImageBuffer::<Rgba<u8>, _>::from_raw(
            bytes_per_row / texel_size,
            fb_size.height,
            data.to_vec(),
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
