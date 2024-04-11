use bytemuck::Zeroable;
use cgmath::{BaseNum, EuclideanSpace, MetricSpace, Point3, Vector3};
use half::f16;
use npyz::npz::{self};
use num_traits::Float;
use std::io::{Read, Seek};
use wgpu::util::{DeviceExt, TextureDataOrder};

pub struct Volume {
    pub timesteps: u32,
    pub resolution: Vector3<u32>,
    pub textures: Vec<wgpu::Texture>,
    pub aabb: Aabb<f32>,
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
        let data: Vec<f32> = array.into_vec()?;
        let mut max_value = data.iter().max_by(|a, b| a.total_cmp(b)).unwrap().clone();
        let min_value = data.iter().min_by(|a, b| a.total_cmp(b)).unwrap().clone();

        if min_value == max_value {
            max_value = min_value + 1.0;
        }

        let data16: Vec<_> = data
            .iter()
            .map(|x| f16::from_f32((x - min_value) / (max_value - min_value)))
            .collect();

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
                        format: wgpu::TextureFormat::R16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    TextureDataOrder::LayerMajor,
                    bytemuck::cast_slice(
                        &data16[(i * resolution[0] * resolution[1] * resolution[2]) as usize
                            ..((i + 1) * resolution[0] * resolution[1] * resolution[2]) as usize],
                    ),
                )
            })
            .collect();

        Ok(Self {
            timesteps,
            resolution: resolution.into(),
            aabb: Aabb::unit(),
            max_value,
            min_value,
            textures: volumes,
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
}
