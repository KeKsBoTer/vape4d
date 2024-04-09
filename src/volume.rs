use bytemuck::Zeroable;
use cgmath::{BaseNum,  EuclideanSpace, MetricSpace, Point3, Vector3};
use npyz::npz::{self};
use num_traits::Float;
use std::{
    io::{Read, Seek},
    iter,
};
use wgpu::util::{DeviceExt, TextureDataOrder};

pub struct Volume {
    pub timesteps: u32,
    pub resolution: Vector3<u32>,
    bind_groups: Vec<wgpu::BindGroup>,
    pub(crate) aabb: Aabb<f32>,
    pub min_value: f32,
    pub max_value: f32,
}

impl Volume {
    pub fn load_npz<'a, R>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        reader: &'a mut R,
    ) -> anyhow::Result<Self>
    where
        R: Read + Seek,
    {
        let mut npz_file = npz::NpzArchive::new(reader)?;
        let array = npz_file.by_name("trj")?.unwrap();
        let timesteps = array.shape()[0] as u32;
        let resolution = [
            array.shape()[2] as u32,
            array.shape()[3] as u32,
            array.shape()[4] as u32,
        ];
        let mut data: Vec<f32> = array.into_vec()?;
        let mut max_value = data
            .iter()
            .max_by(|a,b| a.total_cmp(b)).unwrap().clone();
        let min_value = data
        .iter()
        .min_by(|a,b| a.total_cmp(b)).unwrap().clone();

        if min_value == max_value {
            max_value = min_value + 1.0;
        }

        data = data.iter().map(|x| (x - min_value) / (max_value - min_value)).collect(); 

        let volumes: Vec<wgpu::Texture> = (0..timesteps)
            .map(|i| {
                device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some(format!("volume texture {}", i).as_str()),
                        size: wgpu::Extent3d {
                            width: resolution[2],
                            height: resolution[1],
                            depth_or_array_layers: resolution[0],
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::R32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    TextureDataOrder::LayerMajor,
                    bytemuck::cast_slice(
                        &data[(i * resolution[0] * resolution[1] * resolution[2]) as usize
                            ..((i + 1) * resolution[0] * resolution[1] * resolution[2]) as usize],
                    ),
                )
            })
            .collect();

        let bind_groups = volumes
            .iter()
            .zip(
                volumes
                    .iter()
                    .skip(1)
                    .chain(iter::once(volumes.last().unwrap())),
            )
            .map(|(a, b)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("volume bind group"),
                    layout: &Self::bind_group_layout(device),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &a.create_view(&wgpu::TextureViewDescriptor::default()),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &b.create_view(&wgpu::TextureViewDescriptor::default()),
                            ),
                        },
                    ],
                })
            })
            .collect();
        Ok(Self {
            timesteps,
            resolution: resolution.into(),
            bind_groups,
            aabb: Aabb::unit(),
            max_value,
            min_value,
        })
    }

    pub(crate) fn bind_group(&self, i: usize) -> Option<&wgpu::BindGroup> {
        self.bind_groups.get(i)
    }

    pub(crate) fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }
}

#[repr(C)]
#[derive(Zeroable, Clone, Copy, Debug)]
pub struct Aabb<F: Float + BaseNum> {
    pub min: Point3<F>,
    pub max: Point3<F>,
}

impl<F: Float + BaseNum> Aabb<F> {


    pub fn grow(&mut self, pos: &Point3<F>) {
        self.min.x = self.min.x.min(pos.x);
        self.min.y = self.min.y.min(pos.y);
        self.min.z = self.min.z.min(pos.z);

        self.max.x = self.max.x.max(pos.x);
        self.max.y = self.max.y.max(pos.y);
        self.max.z = self.max.z.max(pos.z);
    }

    pub fn unit() -> Self {
        Self {
            min: Point3::new(-F::zero(), -F::zero(), -F::zero()),
            max: Point3::new(F::one(), F::one(), F::one()),
        }
    }

    pub fn center(&self) -> Point3<F> {
        self.min.midpoint(self.max)
    }

    /// radius of a sphere that contains the aabb
    pub fn radius(&self) -> F {
        self.min.distance(self.max) / (F::one() + F::one())
    }

    pub fn size(&self) -> Vector3<F> {
        self.max - self.min
    }

    pub fn grow_union(&mut self, other: &Aabb<F>) {
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);

        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }
}
