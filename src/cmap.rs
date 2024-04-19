use std::{collections::HashMap, hash::{Hash, Hasher}, io::{Cursor, Read, Seek}};

use anyhow::Ok;
use cgmath::{Vector2, Vector4};
use once_cell::sync::Lazy;
use wgpu::{util::DeviceExt, Extent3d};


use include_dir::{Dir,include_dir};

// list of predefined colormaps
static COLORMAP_DIR: Dir = include_dir!("colormaps");
pub static COLORMAPS: Lazy<HashMap<String,ColorMap>> = Lazy::new(||{
    let cmaps: HashMap<String,ColorMap> = COLORMAP_DIR.files()
    .filter_map(|f| {
        let reader = Cursor::new(f.contents());
        let name = f.path().file_stem().unwrap().to_str().unwrap().to_string();
        let cmap = ColorMap::from_npy(reader).unwrap();
        Some((name.clone(),cmap))
    })
    .collect();
    cmaps
});

#[derive(Debug, Clone)]
pub struct ColorMap(Vec<Vector4<u8>>);

impl Hash for ColorMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl ColorMap {
    pub fn new(values: Vec<Vector4<u8>>) -> Self {
        Self (values)
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
    
    pub(crate) fn values(&self) -> &Vec<Vector4<u8>> {
        &self.0
    }

    pub fn update_alpha(&mut self,alphas:&[u8]){
        if alphas.len() != self.0.len(){
            panic!("Alpha channel must have the same length as the color map")
        }
        self.0 = self.0
        .iter()
        .zip(alphas)
        .map(|(v, a)| Vector4::new(v[0], v[1], v[2], *a))
        .collect();
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

pub struct ColorMapGPU{
    texture: wgpu::Texture,
    bindgroup: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    color_map: ColorMap,
}

impl ColorMapGPU{
    pub fn new(cmap:ColorMap, device: &wgpu::Device, queue: &wgpu::Queue) -> Self{
        let (texture, bindgroup, sampler) = Self::create(device,queue,&cmap);
        Self{
            texture,
            bindgroup,
            sampler,
            color_map: cmap,
        }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn bindgroup(&self) -> &wgpu::BindGroup {
        &self.bindgroup
    }

    pub fn color_map(&self) -> &ColorMap {
        &self.color_map
    }

    fn create(device: &wgpu::Device, queue: &wgpu::Queue,cmap:&ColorMap) -> ( wgpu::Texture, wgpu::BindGroup, wgpu::Sampler) {
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("cmap texture"),
                size: Extent3d {
                    width: cmap.0.len() as u32,
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
            bytemuck::cast_slice(&cmap.0),
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
        (texture, bindgroup, sampler)
    }


    pub fn set_color_map(&mut self, cmap: ColorMap,device: &wgpu::Device, queue: &wgpu::Queue){
        if self.color_map.values().len() == cmap.values().len(){
            self.update(queue);
        }else{
            let (texture, bindgroup, sampler) = Self::create(device,queue,&cmap);
            self.texture = texture;
            self.bindgroup = bindgroup;
            self.sampler = sampler;
            
        }
        self.color_map = cmap;
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

    fn update(&self, queue: &wgpu::Queue){
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.color_map.0),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: self.color_map.0.len() as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
}