use bytemuck::Zeroable;
use cgmath::{
    BaseNum, EuclideanSpace, Matrix, Matrix4, MetricSpace, One, Point3, SquareMatrix, Vector3,
    Vector4,
};
use egui_probe::EguiProbe;
use half::f16;

use ndarray::{
    Array, Array5, ArrayBase, Axis, Ix5, IxDyn, OwnedRepr,
};
use nifti::{InMemNiftiObject, IntoNdArray, NiftiObject, NiftiVolume};

use npyz::{npz, Deserialize, NpyFile};
use num_traits::Float;
use std::io::{Read, Seek};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use wgpu::util::{DeviceExt, TextureDataOrder};

#[derive(Clone)]
pub struct Volume {
    pub aabb: Aabb<f32>,
    pub min_value: f32,
    pub max_value: f32,
    /// array of shape [channels, timesteps, z, y, x]
    data: Array5<f16>,
}

impl Volume {
    pub fn from_array(arr: ArrayBase<OwnedRepr<f16>, IxDyn>,axis_scale:Vector3<f32>) -> Result<Self, anyhow::Error> {
        let shape = arr.shape().to_vec();

        let arr = match shape.len() {
            3 => arr.insert_axis(Axis(0)).insert_axis(Axis(0)),
            4 => arr.insert_axis(Axis(0)),
            _ => arr,
        };
        let data: Array5<f16> = arr
            .into_dimensionality::<Ix5>()?
            .as_standard_layout()
            .into_owned();
        let shape = data.shape().to_vec();
        let resolution = [shape[2] as u32, shape[3] as u32, shape[4] as u32];

        let res_min = resolution.iter().min().unwrap();
        let aabb = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(
                resolution[2] as f32 / *res_min as f32*axis_scale.x,
                resolution[1] as f32 / *res_min as f32*axis_scale.y,
                resolution[0] as f32 / *res_min as f32*axis_scale.z,
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

        Ok(Self {
            aabb,
            min_value: min_value.to_f32(),
            max_value: max_value.to_f32(),
            data: data,
        })
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

    pub fn load<'a, R>(mut reader: R) -> anyhow::Result<Self>
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

        let (_,scale,data) = if is_nifti {
            Self::load_nifti(reader)
        } else if is_npz {
            Self::load_npz(reader)
        } else {
            Self::load_npy(reader)
        }?;
        let volume = Self::from_array(data,scale);
        log::info!("loading volume took: {:?}", start.elapsed());
        return volume;
    }

    fn load_npz<'a, R>(reader: R) -> anyhow::Result<(f64,Vector3<f32>,ArrayBase<OwnedRepr<f16>, IxDyn>)>
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
        Self::read_npy(array)
    }

    fn load_npy<'a, R>(reader: R) -> anyhow::Result<(f64,Vector3<f32>,ArrayBase<OwnedRepr<f16>, IxDyn>)>
    where
        R: Read + Seek,
    {
        let array = NpyFile::new(reader)?;
        Self::read_npy(array)
    }

    fn read_npy_dyn<'a, R, P>(
        npy_file: NpyFile<R>,
    ) -> anyhow::Result<(f64,Vector3<f32>,ArrayBase<OwnedRepr<f16>, IxDyn>)>
    where
        R: Read,
        P: Into<f64> + Deserialize + Clone,
    {
        let shape: Vec<usize> = npy_file.shape().iter().map(|v| *v as usize).collect();
        let data = npy_file.into_vec::<P>()?;
        let array_s = Array::from_shape_vec(IxDyn(&shape), data)?;
        let scale = Vector3::new(1.,1.,1.);
        return Ok((1.,scale,array_s.map(|v| f16::from_f64((*v).clone().into()))));
    }

    fn read_npy<'a, R>(array: NpyFile<R>) -> anyhow::Result<(f64,Vector3<f32>,ArrayBase<OwnedRepr<f16>, IxDyn>)>
    where
        R: Read,
    {
        match array.dtype() {
            npyz::DType::Plain(d) => match d.type_char() {
                npyz::TypeChar::Float => match d.num_bytes().unwrap() {
                    2 => Self::read_npy_dyn::<_, f16>(array),
                    4 => Self::read_npy_dyn::<_, f32>(array),
                    8 => Self::read_npy_dyn::<_, f64>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                npyz::TypeChar::Uint => match d.num_bytes().unwrap() {
                    1 => Self::read_npy_dyn::<_, u8>(array),
                    2 => Self::read_npy_dyn::<_, u16>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                npyz::TypeChar::Int => match d.num_bytes().unwrap() {
                    1 => Self::read_npy_dyn::<_, i8>(array),
                    2 => Self::read_npy_dyn::<_, i16>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                _ => anyhow::bail!("unsupported type {:}", d),
            },
            d => anyhow::bail!("unsupported type {:}", d.descr()),
        }
    }

    fn load_nifti<'a, R>(
        reader: R,
    ) ->anyhow::Result<(f64,Vector3<f32>,ArrayBase<OwnedRepr<f16>, IxDyn>)>
    where
        R: Read + Seek,
    {
        let obj = InMemNiftiObject::from_reader(reader)?;
        let header = obj.header();
        let transform = Matrix4::from_cols(
            header.srow_x.into(),
            header.srow_y.into(),
            header.srow_z.into(),
            Vector4::unit_w(),
        )
        .transpose();

        // Check if the transform matrix only has values on the diagonal
        if !transform.is_diagonal() {
            log::warn!("Transform matrix is not diagonal, only diagonal values will be used");
        }

        let scale = Vector3::new(header.pixdim[1], header.pixdim[2], header.pixdim[3]);
        let volume = obj.into_volume();

        let max_v = match volume.data_type(){
            nifti::NiftiType::Uint8 => u8::MAX as f64,
            nifti::NiftiType::Int16 => i16::MAX as f64,
            nifti::NiftiType::Int32 => i32::MAX as f64,
            nifti::NiftiType::Float32 => 1.,
            nifti::NiftiType::Float64 => 1.,
            nifti::NiftiType::Int8 => i8::MAX as f64,
            nifti::NiftiType::Uint16 => i32::MAX as f64,
            nifti::NiftiType::Uint32 => u32::MAX as f64,
            nifti::NiftiType::Int64 => i64::MAX as f64,
            nifti::NiftiType::Uint64 => u64::MAX as f64,
            nifti::NiftiType::Float128 => 1.,
            d=>anyhow::bail!("unsupported type {:?}", d),
        };
        let data = volume.into_ndarray::<f32>()?;
        return Ok((max_v,scale, data.map(|v| {
            let v_n = *v as f64;// / max_v;
            f16::from_f64(v_n)
        })));
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

impl EguiProbe for Aabb<f32> {
    fn probe(&mut self, ui: &mut egui::Ui, _style: &egui_probe::Style) -> egui::Response {
        ui.collapsing("AABB", |ui| {
            ui.collapsing("min", |ui| {
                ui.add(egui::DragValue::new(&mut self.min.x));
                ui.add(egui::DragValue::new(&mut self.min.y));
                ui.add(egui::DragValue::new(&mut self.min.z));
            });
            ui.collapsing("max", |ui| {
                ui.add(egui::DragValue::new(&mut self.min.x));
                ui.add(egui::DragValue::new(&mut self.min.y));
                ui.add(egui::DragValue::new(&mut self.min.z));
            });
        })
        .header_response
    }
}
