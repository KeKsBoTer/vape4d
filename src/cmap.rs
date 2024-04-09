use std::io::{Read, Seek};

use wgpu::{util::DeviceExt, Extent3d};

#[derive(Debug)]
pub struct ColorMap {
    pub texture: wgpu::Texture,
    pub bindgroup: wgpu::BindGroup,
}


impl ColorMap {
    pub fn from_npz<'a, R>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        reader: &'a mut R,
    ) -> anyhow::Result<Self>
    where
        R: Read + Seek,
    {
        let npz_file = npyz::NpyFile::new(reader)?;
        let n = npz_file.shape()[0];
        let data: Vec<u8> = npz_file.into_vec::<f32>()?.into_iter().map(|x|(x*255.) as u8).collect();
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("cmap texture"),
                size: Extent3d {
                    width: n as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data.as_slice(),
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cmap sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cmap bind group"),
            layout: &Self::bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        Ok(Self {
            texture,
            bindgroup,
        })
    }

    pub(crate) fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cmap bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }
}
