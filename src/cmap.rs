use std::{
    hash::{Hash, Hasher},
    io::{Read, Seek, SeekFrom, Write},
};

#[cfg(feature = "colormaps")]
use std::{collections::HashMap, io::Cursor};

use anyhow::Ok;
use cgmath::Vector4;
#[cfg(feature = "colormaps")]
use include_dir::Dir;
use npyz::WriterBuilder;
#[cfg(feature = "python")]
use numpy::ndarray::{ArrayViewD, Axis};
use wgpu::{util::DeviceExt, Extent3d};

#[cfg(feature = "colormaps")]
use once_cell::sync::Lazy;

#[cfg(feature = "colormaps")]
use include_dir::include_dir;

#[cfg(feature = "colormaps")]
static COLORMAPS_MATPLOTLIB: include_dir::Dir = include_dir!("colormaps/matplotlib");
#[cfg(feature = "colormaps")]
static COLORMAPS_SEABORN: include_dir::Dir = include_dir!("colormaps/seaborn");
#[cfg(feature = "colormaps")]
static COLORMAPS_CMASHER: include_dir::Dir = include_dir!("colormaps/cmasher");

#[cfg(feature = "colormaps")]
fn load_cmaps(dir: &Dir) -> HashMap<String, GenericColorMap> {
    let cmaps: HashMap<String, GenericColorMap> = dir
        .files()
        .filter_map(|f| {
            let file_name = f.path();
            let reader = Cursor::new(f.contents());
            let name = file_name.file_stem().unwrap().to_str().unwrap().to_string();
            let cmap = GenericColorMap::read(reader).unwrap();
            return Some((name, cmap));
        })
        .collect();
    cmaps
}

// list of predefined colormaps
#[cfg(feature = "colormaps")]
pub static COLORMAPS: Lazy<HashMap<String, HashMap<String, GenericColorMap>>> = Lazy::new(|| {
    let mut cmaps = HashMap::new();
    cmaps.insert("matplotlib".to_string(), load_cmaps(&COLORMAPS_MATPLOTLIB));
    cmaps.insert("seaborn".to_string(), load_cmaps(&COLORMAPS_SEABORN));
    cmaps.insert("cmasher".to_string(), load_cmaps(&COLORMAPS_CMASHER));

    cmaps
});

#[derive(Debug, Clone)]
pub struct ListedColorMap(Vec<Vector4<u8>>);

impl Hash for ListedColorMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl ListedColorMap {
    pub fn new(values: Vec<Vector4<u8>>) -> Self {
        Self(values)
    }

    #[cfg(feature = "python")]
    pub fn from_array(data: ArrayViewD<f32>) -> Self {
        Self(
            data.axis_iter(Axis(0))
                .map(|v| {
                    Vector4::new(
                        (v[0] * 255.) as u8,
                        (v[1] * 255.) as u8,
                        (v[2] * 255.) as u8,
                        (v[3] * 255.) as u8,
                    )
                })
                .collect(),
        )
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
    pub fn update_alpha(&mut self, alphas: &[u8]) {
        if alphas.len() != self.0.len() {
            panic!("Alpha channel must have the same length as the color map")
        }
        self.0 = self
            .0
            .iter()
            .zip(alphas)
            .map(|(v, a)| Vector4::new(v[0], v[1], v[2], *a))
            .collect();
    }

    pub fn save_npy<F: Write>(&self, f: F) -> anyhow::Result<()> {
        let mut out_file = npyz::WriteOptions::new()
            .order(npyz::Order::C)
            .shape(&[self.0.len() as u64, 4])
            .default_dtype()
            .writer(f)
            .begin_nd()?;
        out_file.extend(self.0.iter().flat_map(|c| {
            vec![
                c.x as f32 / 255.,
                c.y as f32 / 255.,
                c.z as f32 / 255.,
                c.w as f32 / 255.,
            ]
        }))?;
        out_file.finish()?;
        Ok(())
    }
}

impl<'a> ColorMap for &'a ListedColorMap {
    type Item = ListedColorMap;
    fn sample(&self, x: f32) -> Vector4<u8> {
        let n = self.0.len() as f32;
        let i = (x * n).min(n - 1.0).max(0.0) as usize;
        self.0[i] // TODO linear interpolation
    }

    fn reverse(&self) -> ListedColorMap {
        let mut cmap = self.0.clone();
        cmap.reverse();
        ListedColorMap::new(cmap)
    }
}

pub struct ColorMapGPU {
    texture: wgpu::Texture,
    bindgroup: wgpu::BindGroup,
}

pub trait ColorMap {
    type Item;
    fn sample(&self, x: f32) -> Vector4<u8>;

    fn rasterize(&self, n: usize) -> Vec<Vector4<u8>> {
        (0..n)
            .map(|i| self.sample(i as f32 / (n - 1) as f32))
            .collect()
    }

    fn reverse(&self) -> Self::Item;
}

impl ColorMapGPU {
    pub fn new(cmap: impl ColorMap, device: &wgpu::Device, queue: &wgpu::Queue, n: u32) -> Self {
        let (texture, bindgroup) = Self::create(device, queue, cmap, n);
        Self { texture, bindgroup }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn bindgroup(&self) -> &wgpu::BindGroup {
        &self.bindgroup
    }

    fn create(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cmap: impl ColorMap,
        n: u32,
    ) -> (wgpu::Texture, wgpu::BindGroup) {
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("cmap texture"),
                size: Extent3d {
                    width: n,
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
            bytemuck::cast_slice(&cmap.rasterize(n as usize)),
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
        (texture, bindgroup)
    }

    fn size(&self) -> u32 {
        return self.texture.size().width;
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

    pub fn update(&self, queue: &wgpu::Queue, cmap: impl ColorMap) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&cmap.rasterize(self.size() as usize)),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: self.size() as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LinearSegmentedColorMap {
    /// x, y0,y1
    #[serde(alias = "red")]
    pub r: Vec<(f32, f32, f32)>,
    #[serde(alias = "green")]
    pub g: Vec<(f32, f32, f32)>,
    #[serde(alias = "blue")]
    pub b: Vec<(f32, f32, f32)>,
    #[serde(alias = "alpha")]
    pub a: Option<Vec<(f32, f32, f32)>>,
}

impl LinearSegmentedColorMap {
    pub fn new(
        r: Vec<(f32, f32, f32)>,
        g: Vec<(f32, f32, f32)>,
        b: Vec<(f32, f32, f32)>,
        a: Option<Vec<(f32, f32, f32)>>,
    ) -> anyhow::Result<Self> {
        if !Self::check_values(&r) {
            return Err(anyhow::anyhow!(
                "x values for red are not in (0,1) or ascending"
            ));
        };

        if !Self::check_values(&g) {
            return Err(anyhow::anyhow!(
                "x values for green are not in (0,1) or ascending"
            ));
        };

        if !Self::check_values(&b) {
            return Err(anyhow::anyhow!(
                "x values for blue are not in (0,1) or ascending"
            ));
        };

        if let Some(a) = &a {
            if !Self::check_values(&a) {
                return Err(anyhow::anyhow!(
                    "x values for alpha are not in (0,1) or ascending"
                ));
            };
        }
        Ok(Self { r, g, b, a })
    }

    pub fn from_json<R: Read>(reader: R) -> anyhow::Result<Self> {
        Ok(serde_json::from_reader(reader)?)
    }

    fn check_values(v: &Vec<(f32, f32, f32)>) -> bool {
        let mut last_x = 0.0;
        for (x, _, _) in v.iter() {
            if x < &last_x || x > &1.0 || x < &0.0 {
                return false;
            }
            last_x = *x;
        }
        return true;
    }
    pub fn from_color_map(cmap: impl ColorMap, n: u32) -> Self {
        let mut r = vec![];
        let mut g = vec![];
        let mut b = vec![];
        let mut a = vec![];
        for (i, v) in cmap.rasterize(n as usize).iter().enumerate() {
            let x = i as f32 / (n - 1) as f32;
            r.push((x, v.x as f32 / 255., v.x as f32 / 255.));
            g.push((x, v.y as f32 / 255., v.y as f32 / 255.));
            b.push((x, v.z as f32 / 255., v.z as f32 / 255.));
            a.push((x, v.w as f32 / 255., v.w as f32 / 255.));
        }

        // merge neighboring points with the same alpha value
        merge_neighbours(&mut r);
        merge_neighbours(&mut g);
        merge_neighbours(&mut b);
        merge_neighbours(&mut a);

        Self {
            r,
            g,
            b,
            a: Some(a),
        }
    }
}

impl ColorMap for &LinearSegmentedColorMap {
    type Item = LinearSegmentedColorMap;
    fn sample(&self, x: f32) -> Vector4<u8> {
        let a = self
            .a
            .as_ref()
            .map(|a| sample_channel(x, &a))
            .unwrap_or(1.0);
        Vector4::new(
            (sample_channel(x, &self.r) * 255.) as u8,
            (sample_channel(x, &self.g) * 255.) as u8,
            (sample_channel(x, &self.b) * 255.) as u8,
            (a * 255.) as u8,
        )
    }
    fn reverse(&self) -> Self::Item {
        let mut r: Vec<_> = self
            .r
            .iter()
            .map(|(x, y1, y2)| (1.0 - x, *y1, *y2))
            .collect();
        let mut g: Vec<_> = self
            .g
            .iter()
            .map(|(x, y1, y2)| (1.0 - x, *y1, *y2))
            .collect();
        let mut b: Vec<_> = self
            .b
            .iter()
            .map(|(x, y1, y2)| (1.0 - x, *y1, *y2))
            .collect();
        let mut a: Option<Vec<(f32, f32, f32)>> = self
            .a
            .clone()
            .map(|a| a.iter().map(|(x, y1, y2)| (1.0 - x, *y1, *y2)).collect());
        r.reverse();
        g.reverse();
        b.reverse();
        if let Some(a) = &mut a {
            a.reverse();
        }
        LinearSegmentedColorMap { r, g, b, a }
    }
}
impl Hash for LinearSegmentedColorMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for c in [&self.r, &self.g, &self.b].iter() {
            c.iter().for_each(|(a, b, c)| {
                state.write_u32(a.to_bits());
                state.write_u32(b.to_bits());
                state.write_u32(c.to_bits())
            });
        }
        if let Some(a) = &self.a {
            a.iter().for_each(|(a, b, c)| {
                state.write_u32(a.to_bits());
                state.write_u32(b.to_bits());
                state.write_u32(c.to_bits())
            });
        }
    }
}

pub fn rasterize_tf(points: &[(f32, f32, f32)], n: u32) -> Vec<u8> {
    assert!(points.len() >= 2, "spline must have at least 2 points");
    let mut values = vec![0; n as usize];
    let mut last_i = 0;
    let mut current_i = 1;
    for i in 0..n {
        let x = i as f32 / (n - 1) as f32;
        let last = points[last_i];
        let current = points[current_i];
        if (last.0 - current.0).abs() < 0.5 / n as f32 {
            values[i as usize] = (last.2 * 255.) as u8;
        } else {
            let y = last.2 + (current.1 - last.2) * (x - last.0) / (current.0 - last.0);
            values[i as usize] = (y * 255.) as u8;
        }
        if x > points[current_i].0 {
            last_i = current_i;
            current_i += 1;
        }
    }
    values
}

fn sample_channel(x: f32, values: &[(f32, f32, f32)]) -> f32 {
    for i in 0..values.len() - 1 {
        let (x0, _, y0) = values[i];
        let (x1, y1, _) = values[i + 1];
        if x0 <= x && x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    return 0.0;
}

fn merge_neighbours(values: &mut Vec<(f32, f32, f32)>) {
    let mut i = 1;
    while i < values.len() - 1 {
        let (_, y0, y1) = values[i];
        if y0 == y1 {
            let y_prev = values[i - 1].2;
            let y_next = values[i + 1].1;
            if y_prev == y_next {
                values.remove(i);
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}

pub const COLORMAP_RESOLUTION: u32 = 256;

#[derive(Debug, Clone, Hash)]
pub enum GenericColorMap {
    Listed(ListedColorMap),
    LinearSegmented(LinearSegmentedColorMap),
}

impl GenericColorMap {
    pub fn read<R: Read + Seek>(mut reader: R) -> anyhow::Result<Self> {
        let mut start = [0; 6];
        reader.read_exact(&mut start)?;
        reader.seek(SeekFrom::Start(0))?;
        if start.eq(b"\x93NUMPY") {
            // numpy file
            Ok(GenericColorMap::Listed(ListedColorMap::from_npy(reader)?))
        } else {
            // json file
            Ok(GenericColorMap::LinearSegmented(
                LinearSegmentedColorMap::from_json(reader)?,
            ))
        }
    }

    pub fn into_linear_segmented(&self, n: u32) -> LinearSegmentedColorMap {
        match self {
            crate::cmap::GenericColorMap::Listed(c) => {
                LinearSegmentedColorMap::from_color_map(c, n)
            }
            crate::cmap::GenericColorMap::LinearSegmented(c) => c.clone(),
        }
    }

    /// if all alpha values are 1.0, the alpha channel is considered boring
    #[allow(unused)]
    pub(crate) fn has_boring_alpha_channel(&self) -> bool {
        match self {
            GenericColorMap::Listed(c) => c.0.iter().all(|v| v.w == 255),
            GenericColorMap::LinearSegmented(c) => {
                c.a.as_ref()
                    .map(|a| a.iter().all(|(_, _, a)| *a == 1.0))
                    .unwrap_or(true)
            }
        }
    }
}

impl<'a> ColorMap for &'a GenericColorMap {
    fn sample(&self, x: f32) -> Vector4<u8> {
        match self {
            GenericColorMap::Listed(c) => c.sample(x),
            GenericColorMap::LinearSegmented(c) => c.sample(x),
        }
    }

    type Item = GenericColorMap;

    fn reverse(&self) -> Self::Item {
        match self {
            GenericColorMap::Listed(c) => GenericColorMap::Listed(c.reverse()),
            GenericColorMap::LinearSegmented(c) => GenericColorMap::LinearSegmented(c.reverse()),
        }
    }
}
