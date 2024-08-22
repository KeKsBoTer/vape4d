use bytemuck::Zeroable;
use cgmath::{BaseNum, EuclideanSpace, MetricSpace, Point3, Vector3, Zero};
use half::f16;
#[cfg(target_arch = "wasm32")]
use instant::Instant;
use nifti::{InMemNiftiObject, IntoNdArray, NiftiObject, NiftiVolume};
use npyz::{npz, Deserialize, NpyFile};
use num_traits::Float;
#[cfg(feature = "python")]
use numpy::ndarray::ArrayViewD;
use std::io::{Read, Seek};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
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
    #[cfg(feature = "python")]
    pub fn from_array(data: ArrayViewD<f16>) -> Self {
        let shape = data.shape().to_vec();
        let resolution = [shape[1] as u32, shape[2] as u32, shape[3] as u32];
        let vec_data = data.to_slice().unwrap().to_vec();

        let res_min = resolution.iter().min().unwrap();
        let aabb = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(
                resolution[2] as f32 / *res_min as f32,
                resolution[1] as f32 / *res_min as f32,
                resolution[0] as f32 / *res_min as f32,
            ),
        };
        let vmin = *vec_data.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let vmax = *vec_data.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        Self {
            timesteps: shape[0] as u32,
            resolution: resolution.into(),
            aabb,
            min_value: vmin.to_f32(),
            max_value: vmax.to_f32(),
            data: vec_data,
        }
    }

    pub fn load_npy<'a, R>(reader: R) -> anyhow::Result<Vec<Self>>
    where
        R: Read + Seek,
    {
        let array = NpyFile::new(reader)?;
        Self::read(array)
    }

    pub fn read_dyn<'a, R, P>(array: NpyFile<R>) -> anyhow::Result<Vec<Self>>
    where
        R: Read,
        P: Into<f64> + Deserialize,
    {
        let start = Instant::now();
        let time_dim = 0;
        let channel_dim = 1;
        let timesteps = array.shape()[time_dim] as usize;
        let channels = array.shape()[channel_dim] as usize;
        log::debug!("size: {:?}", array.shape());
        if array.shape().len() != 5 {
            anyhow::bail!("unsupported shape: {:?}", array.shape());
        }
        let resolution = [
            array.shape()[2] as u32,
            array.shape()[3] as u32,
            array.shape()[4] as u32,
        ];
        let numel = resolution.iter().product::<u32>() as usize;

        let strides = array.strides().to_vec();
        let mut volumes: Vec<Vec<f16>> = vec![vec![f16::zero(); numel * timesteps]; channels];

        let mut min_value = f32::MAX;
        let mut max_value = f32::MIN;
        for (i, v) in array.data::<P>()?.enumerate() {
            let v64: f64 = v.unwrap().into();
            let v32: f32 = v64 as f32;
            let v = f16::from_f32(v32);
            if v32 > max_value {
                max_value = v32;
            }
            if v32 < min_value {
                min_value = v32;
            }

            let t = i / strides[0] as usize;
            let c = (i - strides[0] as usize * t) / strides[1] as usize;
            let idx = t * numel + (i - strides[0] as usize * t - strides[1] as usize * c);

            volumes[c][idx] = v;
        }

        if min_value == max_value {
            max_value = min_value + 1.0;
        }
        let res_min = resolution.iter().min().unwrap();

        let aabb = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(
                resolution[2] as f32 / *res_min as f32,
                resolution[1] as f32 / *res_min as f32,
                resolution[0] as f32 / *res_min as f32,
            ),
        };

        let results = (0..channels as usize)
            .map(|c| Self {
                timesteps: timesteps as u32,
                resolution: resolution.into(),
                aabb,
                max_value,
                min_value,
                data: volumes[c].clone(),
            })
            .collect();
        log::info!("read volume in {:?}", start.elapsed());
        Ok(results)
    }

    pub fn read<'a, R>(array: NpyFile<R>) -> anyhow::Result<Vec<Self>>
    where
        R: Read,
    {
        match array.dtype() {
            npyz::DType::Plain(d) => match d.type_char() {
                npyz::TypeChar::Float => match d.num_bytes().unwrap() {
                    2 => Self::read_dyn::<_, f16>(array),
                    4 => Self::read_dyn::<_, f32>(array),
                    8 => Self::read_dyn::<_, f64>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                npyz::TypeChar::Uint => match d.num_bytes().unwrap() {
                    1 => Self::read_dyn::<_, u8>(array),
                    2 => Self::read_dyn::<_, u16>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                npyz::TypeChar::Int => match d.num_bytes().unwrap() {
                    1 => Self::read_dyn::<_, i8>(array),
                    2 => Self::read_dyn::<_, i16>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                _ => anyhow::bail!("unsupported type {:}", d),
            },
            d => anyhow::bail!("unsupported type {:}", d.descr()),
        }
    }
    pub fn load<'a, R>(mut reader: R) -> anyhow::Result<Vec<Self>>
    where
        R: Read + Seek,
    {
        let start = Instant::now();
        let mut buffer = [0; 4];
        reader.read_exact(&mut buffer)?;
        let is_npz = buffer == *b"\x50\x4B\x03\x04";

        reader.seek(std::io::SeekFrom::Start(344))?;
        reader.read_exact(&mut buffer)?;
        let is_nifti = buffer == *b"\x6E\x2B\x31\x00";
        reader.seek(std::io::SeekFrom::Start(0))?;

        let volume = if is_nifti {
            Self::load_nifti(reader)
        }else 
        if is_npz {
            Self::load_npz(reader)
        } else {
            Self::load_npy(reader)
        };
        log::info!("loaded volume in {:?}", start.elapsed());
        return volume;
    }

    pub fn load_npz<'a, R>(reader: R) -> anyhow::Result<Vec<Self>>
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
        Self::read(array)
    }

    pub fn load_nifti<'a, R>(reader: R) -> anyhow::Result<Vec<Self>>
    where
        R: Read + Seek,
    {
        let obj = InMemNiftiObject::from_reader(reader)?;
        // use obj
        let volume = obj.into_volume();
        let shape = volume.dim().to_vec();
        let dim = volume.dimensionality();
        if dim != 3 {
            anyhow::bail!("unsupported dimensionality: {}", dim);
        }
        let data = volume.into_ndarray::<f32>()?;
        let min_value = data.iter().fold(f32::MAX, |a, &b| a.min(b)) as f32;
        let mut max_value = data.iter().fold(f32::MIN, |a, &b| a.max(b)) as f32;
        if min_value == max_value {
            max_value = min_value + 1.0;
        }
        let res_min = shape.iter().min().unwrap();

        let aabb = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(
                shape[0] as f32 / *res_min as f32,
                shape[1] as f32 / *res_min as f32,
                shape[2] as f32 / *res_min as f32,
            ),
        };
        log::info!("volume shape is: {:?}, interpreted as [WxHxD]", shape);

        return Ok(vec![Self {
            timesteps: 1,
            resolution: Vector3::new(shape[2] as u32, shape[1] as u32, shape[0] as u32),
            aabb,
            min_value,
            max_value,
            data: data.as_standard_layout().as_slice().unwrap().iter().map(|&v| f16::from_f32(v)).collect(),
        }]);
    }
}

pub struct VolumeGPU {
    pub(crate) textures: Vec<wgpu::Texture>,
    pub(crate) volume: Volume,
}

impl VolumeGPU {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, volume: Volume) -> Self {
        let textures = (0..volume.timesteps)
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
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    TextureDataOrder::LayerMajor,
                    bytemuck::cast_slice(
                        &volume.data[(i
                            * volume.resolution[0]
                            * volume.resolution[1]
                            * volume.resolution[2]) as usize
                            ..((i + 1)
                                * volume.resolution[0]
                                * volume.resolution[1]
                                * volume.resolution[2]) as usize],
                    ),
                )
            })
            .collect();
        Self { textures, volume }
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
            min: Point3::new(F::zero(), F::zero(), F::zero()),
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
