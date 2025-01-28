use bytemuck::Zeroable;
use cgmath::{BaseNum, EuclideanSpace, MetricSpace, One, Point3, Vector3};
use half::f16;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
use ndarray::{Array5, ArrayViewD, Axis, Ix5, StrideShape};
use npyz::{npz, Deserialize, NpyFile};
use num_traits::Float;
use std::io::{Read, Seek};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use wgpu::util::{DeviceExt, TextureDataOrder};

const HISTOGRAM_BINS: usize = 128;
#[derive(Clone)]
pub struct Volume {
    pub aabb: Aabb<f32>,
    pub min_value: f32,
    pub max_value: f32,
    /// array of shape [channels, timesteps, z, y, x]
    data: Array5<f16>,

    /// one histogram per channel
    histograms: Vec<[f32; HISTOGRAM_BINS]>,
}

impl Volume {
    pub fn from_array(arr: ArrayViewD<f16>) -> Result<Self, anyhow::Error> {

        let shape = arr.shape().to_vec();

        let arr = match shape.len() {
            3 => arr.insert_axis(Axis(0)).insert_axis(Axis(0)),
            4 => arr.insert_axis(Axis(0)),
            _ => arr,
        };
        let data: Array5<f16> = arr.into_dimensionality::<Ix5>()?.into_owned();
        let shape = data.shape().to_vec();
        let resolution = [shape[2] as u32, shape[3] as u32, shape[4] as u32];

        let res_min = resolution.iter().min().unwrap();
        let aabb = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(
                resolution[2] as f32 / *res_min as f32,
                resolution[1] as f32 / *res_min as f32,
                resolution[0] as f32 / *res_min as f32,
            ),
        };

        let (min_value, mut max_value) = data
            .iter()
            .fold((f16::MAX, f16::MIN), |(acc_min, acc_max), b| {
                (acc_min.min(*b), acc_max.max(*b))
            });

        if min_value >= max_value {
            max_value = min_value + f16::one();
        }


        let channels = data.shape()[0];
        let mut histograms = vec![[0f32; HISTOGRAM_BINS]; channels];
        let numel: usize = data.shape().iter().skip(1).product();
        for (c, channel_data) in data.axis_iter(Axis(0)).enumerate() {
            for &value in channel_data.iter() {
                let bin = ((value.to_f32() - min_value.to_f32()) / (max_value - min_value).to_f32()
                    * (HISTOGRAM_BINS - 1) as f32)
                    .floor() as usize;
                histograms[c][bin] += 1. / numel as f32;
            }
        }


        Ok(Self {
            aabb,
            min_value: min_value.to_f32(),
            max_value: max_value.to_f32(),
            data: data,
            histograms
        })
    }

    pub fn load_npy<'a, R>(reader: R, time_first: bool) -> anyhow::Result<Self>
    where
        R: Read + Seek,
    {
        let array = NpyFile::new(reader)?;
        Self::read(array, time_first)
    }

    pub fn channels(&self) -> usize {
        self.data.shape()[0]
    }

    pub fn timesteps(&self) -> usize {
        self.data.shape()[1]
    }

    /// returns width, height, depth
    pub fn size(&self) -> Vector3<u32> {
        Vector3 {
            x: self.data.shape()[4] as u32, // width
            y: self.data.shape()[3] as u32, // height
            z: self.data.shape()[2] as u32, // depth
        }
    }

    pub fn read_dyn<'a, R, P>(array: NpyFile<R>, time_first: bool) -> anyhow::Result<Self>
    where
        R: Read,
        P: Into<f64> + Copy + Deserialize,
    {
        let start = Instant::now();

        let shape: Vec<u64> = array.shape().to_vec();
        log::debug!("size: {:?}", &shape);

        if shape.len() != 5 {
            anyhow::bail!("unsupported shape: {:?}", shape);
        }

        // TODO take order into account
        let shape_stride: StrideShape<_> = (
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
            shape[4] as usize,
        )
            .into();

        let data = array.into_vec::<P>()?;
        let mut arr = Array5::from_shape_vec(shape_stride, data)?.map(|v| {
            let v64: f64 = (*v).into();
            f16::from_f64(v64)
        });


        if time_first {
            arr.swap_axes(0, 1);
        }
        log::info!("read volume in {:?}", start.elapsed());
        return Self::from_array(arr.view().into_dyn());
    }

    pub fn read<'a, R>(array: NpyFile<R>, time_first: bool) -> anyhow::Result<Self>
    where
        R: Read,
    {
        match array.dtype() {
            npyz::DType::Plain(d) => match d.type_char() {
                npyz::TypeChar::Float => match d.num_bytes().unwrap() {
                    2 => Self::read_dyn::<_, f16>(array, time_first),
                    4 => Self::read_dyn::<_, f32>(array, time_first),
                    8 => Self::read_dyn::<_, f64>(array, time_first),
                    _ => anyhow::bail!("unsupported float type {:}", d),
                },
                npyz::TypeChar::Uint => match d.num_bytes().unwrap() {
                    1 => Self::read_dyn::<_, u8>(array, time_first),
                    2 => Self::read_dyn::<_, u16>(array, time_first),
                    _ => anyhow::bail!("unsupported unsigned type {:}", d),
                },
                npyz::TypeChar::Int => match d.num_bytes().unwrap() {
                    1 => Self::read_dyn::<_, i8>(array, time_first),
                    2 => Self::read_dyn::<_, i16>(array, time_first),
                    _ => anyhow::bail!("unsupported signed type {:?}", d),
                },
                _ => anyhow::bail!("unsupported type {:}", d),
            },
            d => anyhow::bail!("unsupported type {:}", d.descr()),
        }
    }
    pub fn load_numpy<'a, R>(mut reader: R, time_first: bool) -> anyhow::Result<Self>
    where
        R: Read + Seek,
    {
        let mut buffer = [0; 4];
        reader.read_exact(&mut buffer)?;
        reader.seek(std::io::SeekFrom::Current(-4))?;
        let is_npz = buffer == *b"\x50\x4B\x03\x04";

        if is_npz {
            Self::load_npz(reader, time_first)
        } else {
            Self::load_npy(reader, time_first)
        }
    }

    pub fn load_npz<'a, R>(reader: R, time_first: bool) -> anyhow::Result<Self>
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
        Self::read(array, time_first)
    }

    pub fn get_histogram(&self, c: usize) -> &[f32; HISTOGRAM_BINS] {
        &self.histograms[c]
    }
}

pub struct VolumeGPU {
    textures: Vec<Vec<wgpu::Texture>>,
    pub(crate) volume: Volume,
}

impl VolumeGPU {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, volume: Volume) -> Self {
        let resolution = volume.size();

        let textures = volume
            .data
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(c, data)| {
                data.axis_iter(Axis(0))
                    .enumerate()
                    .map(|(t, data)| {
                        device.create_texture_with_data(
                            queue,
                            &wgpu::TextureDescriptor {
                                label: Some(
                                    format!("volume texture channel {}, timestep {}", c, t)
                                        .as_str(),
                                ),
                                size: wgpu::Extent3d {
                                    width: resolution.x,
                                    height: resolution.y,
                                    depth_or_array_layers: resolution.z,
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
                            bytemuck::cast_slice(data.as_slice().unwrap()),
                        )
                    })
                    .collect()
            })
            .collect();

        Self { textures, volume }
    }

    pub fn get_texture(&self, channel: usize, timestep: usize) -> &wgpu::Texture {
        &self.textures[channel][timestep]
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

    pub fn size(&self) -> Vector3<F> {
        self.max - self.min
    }
}
