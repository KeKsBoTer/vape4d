use bytemuck::Zeroable;
use cgmath::{BaseNum, EuclideanSpace, MetricSpace, Point3, Vector3, Zero};
use half::f16;
use npyz::{npz, NpyFile};
use num_traits::Float;
use std::io::{Read, Seek};
use wgpu::util::{DeviceExt, TextureDataOrder};

pub struct Volume {
    pub timesteps: u32,
    pub resolution: Vector3<u32>,
    pub aabb: Aabb<f32>,
    pub min_value: f32,
    pub max_value: f32,
    data: Vec<f16>,
}

impl Volume {
    pub fn load_npy<'a, R>(reader: &'a mut R,time_first:bool) -> anyhow::Result<Vec<Self>>
    where
        R: Read + Seek,
    {
        let array = NpyFile::new(reader)?;
        Self::read(array,time_first)
    }

    pub fn read<'a, R>(array: NpyFile<R>,time_first:bool) -> anyhow::Result<Vec<Self>>
    where
        R: Read,
    {
        let time_dim = if time_first {0} else {1};
        let channel_dim = if time_first {1} else {0};
        let timesteps = array.shape()[time_dim] as usize;
        let channels = array.shape()[channel_dim] as usize;
        let strides = array.strides().to_vec();
        let resolution = [
            array.shape()[2] as u32,
            array.shape()[3] as u32,
            array.shape()[4] as u32,
        ];
        let numel = resolution.iter().product::<u32>() as usize;

        let mut max_value = f32::MIN;
        let mut min_value = f32::MAX;
        let mut volumes:Vec<Vec<f16>> = vec![vec![f16::zero();numel*timesteps ];channels];
        for (i,v) in array.data::<f32>()?.into_iter().enumerate() {
            let v = v.unwrap();
            if v > max_value {
                max_value = v;
            }
            if v < min_value {
                min_value = v;
            }   
            if time_first{
                let t= i / strides[0] as usize; 
                let c = (i-strides[0] as usize*t) / strides[1] as usize;
                volumes[c][t*numel + (i-strides[0] as usize*t-strides[1] as usize*c)] = f16::from_f32(v);
            }else{
                let c= i / strides[0] as usize; 
                let t = (i-strides[0] as usize*c) / strides[1] as usize;
                volumes[c][t*numel + (i-strides[0] as usize*c-strides[1] as usize*t)] = f16::from_f32(v);
            }
        }

        if min_value == max_value {
            max_value = min_value + 1.0;
        }

        Ok((0..channels as usize).map(|c|{
            Self {
                timesteps: timesteps as u32,
                resolution: resolution.into(),
                aabb: Aabb::unit(),
                max_value,
                min_value,
                data:volumes[c].clone(),
            }
        }).collect())
    }

    pub fn load_npz<'a, R>(reader: R,time_first:bool) -> anyhow::Result<Vec<Self>>
    where
        R: Read + Seek,
    {
        let mut reader = npz::NpzArchive::new(reader)?;
        let arr_name = reader
            .array_names()
            .next()
            .ok_or(anyhow::format_err!("no array present"))?
            .to_string();
        let array = reader.by_name(arr_name.as_str())?.unwrap();
        Self::read(array,time_first)
    }
}

pub(crate) struct VolumeGPU {
    pub(crate) textures: Vec<wgpu::Texture>,
    pub(crate) volume: Volume,
}

impl VolumeGPU{
    pub fn new( device: &wgpu::Device, queue: &wgpu::Queue,volume:Volume) -> Self {
        let textures = 
                (0..volume.timesteps)
                    .map(|i| {
                        device.create_texture_with_data(
                            queue,
                            &wgpu::TextureDescriptor {
                                label: Some(format!("volume texture {}", i).as_str()),
                                size: wgpu::Extent3d {
                                    width: volume.resolution[2],
                                    height: volume.resolution[1],
                                    depth_or_array_layers: volume.resolution[0],
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D3,
                                format: wgpu::TextureFormat::R16Float,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING
                                    | wgpu::TextureUsages::COPY_DST,
                                view_formats: &[],
                            },
                            TextureDataOrder::LayerMajor,
                            bytemuck::cast_slice(
                                &volume.data[(i
                                    * volume.resolution[0]
                                    * volume.resolution[1]
                                    * volume.resolution[2])
                                    as usize
                                    ..((i + 1)
                                        * volume.resolution[0]
                                        * volume.resolution[1]
                                        * volume.resolution[2])
                                        as usize],
                            ),
                        )
                    })
                    .collect();
        Self {
            textures,
            volume,
        }
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