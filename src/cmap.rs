use std::io::{Read, Seek};

use anyhow::Ok;
use cgmath::{Vector2, Vector4};
use wgpu::{util::DeviceExt, Extent3d};

#[derive(Debug)]
pub struct ColorMap {
    pub values: Vec<Vector4<u8>>,
    pub texture: Option<wgpu::Texture>,
    pub bindgroup: Option<wgpu::BindGroup>,
}

impl ColorMap {
    pub fn new(values: Vec<Vector4<u8>>) -> Self {
        Self {
            texture: None,
            bindgroup: None,
            values,
        }
    }

    pub fn upload2gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.texture = Some(device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("cmap texture"),
                size: Extent3d {
                    width: self.values.len() as u32,
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
            bytemuck::cast_slice(&self.values),
        ));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cmap sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        self.bindgroup = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cmap bind group"),
                layout: &Self::bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self
                                .texture
                                .as_ref()
                                .unwrap()
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            }),
        );
    }

    pub fn from_npy<'a, R>(reader: R) -> anyhow::Result<Self>
    where
        R: Read + Seek,
    {
        let npz_file = npyz::NpyFile::new(reader)?;
        let values: Vec<_> = npz_file
            .into_vec::<f32>()?
            .chunks_exact(4)
            .map(|v| {
                Vector4::new(
                    (v[0] * 255.) as u8,
                    (v[1] * 255.) as u8,
                    (v[2] * 255.) as u8,
                    (v[3] * 255.) as u8,
                )
            })
            .collect();
        Ok(Self::new(values))
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

pub fn rasterize_tf(points: &[Vector2<f32>], n: u32) -> Vec<u8> {
    assert!(points.len() >= 2, "spline must have at least 2 points");
    let mut values = vec![0; n as usize];
    let mut last_i = 0;
    let mut current_i = 1;
    for i in 0..n {
        let x = i as f32 / (n - 1) as f32;
        let last = points[last_i];
        let current = points[current_i];
        if (last.x - current.x).abs() < 0.5 / n as f32 {
            values[i as usize] = (last.y * 255.) as u8;
        } else {
            let y = last.y + (current.y - last.y) * (x - last.x) / (current.x - last.x);
            values[i as usize] = (y * 255.) as u8;
        }
        if x > points[current_i].x {
            last_i = current_i;
            current_i += 1;
        }
    }
    values
}
