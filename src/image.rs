use crate::{
    camera::{GenericCamera, PerspectiveProjection},
    cmap::ColorMapGPU,
    renderer::{RenderSettings, VolumeRenderer},
    volume::VolumeGPU,
};
use cgmath::{Deg, EuclideanSpace, InnerSpace, Point3, Quaternion, Rotation, Vector2, Vector3};
use clap::Parser;
use half::f16;
use image::{ImageBuffer, Rgba};
#[allow(unused_imports)]
use std::{fs::File, path::PathBuf, time::Duration};
#[allow(unused_imports)]
#[derive(Debug, Parser)]
#[command(author, version)]
#[command(about = "Dataset offline renderer. Renders to PNG files", long_about = None)]
struct Opt {
    /// input file
    input: PathBuf,

    colormap: PathBuf,

    /// image output directory
    img_out: PathBuf,

    #[arg(long, default_value_t = false)]
    channel_first: bool,
}

async fn render_to_image(
    volume: &VolumeGPU,
    cmap: &ColorMapGPU,
    renderer: &mut VolumeRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resolution: Vector2<u32>,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
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
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render encoder"),
    });
    let r = volume.volume.aabb.radius();
    let corner = Vector3::new(1., -1., 1.);
    let view_dir = Quaternion::look_at(-corner, Vector3::unit_y());
    let camera = GenericCamera::new(
        Point3::from_vec(corner.normalize()) * r * 3.,
        view_dir,
        PerspectiveProjection::new(resolution, Deg(45.), 0.01, 1000.),
    );
    let frame_data = renderer.prepare(
        &device,
        &volume,
        &camera,
        &RenderSettings {
            clipping_aabb: None,
            time: 0.,
            step_size: 1e-4,
            spatial_filter: wgpu::FilterMode::Linear,
            temporal_filter: wgpu::FilterMode::Linear,
            distance_scale: 1.,
            vmin: None,
            vmax: None,
        },
        cmap,
    );
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
    return img;
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
            data.to_vec()
                .chunks(2)
                .map(|c| (f16::from_le_bytes([c[0], c[1]]).to_f32().clamp(0., 1.) * 255.) as u8)
                .collect::<Vec<u8>>(),
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
