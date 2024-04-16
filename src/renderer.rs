use crate::{
    camera::{GenericCamera, Projection, VIEWPORT_Y_FLIP},
    cmap::ColorMap,
    volume::{Aabb, Volume},
};

use cgmath::{EuclideanSpace, Matrix4, SquareMatrix, Vector4, Zero};
use wgpu::util::DeviceExt;

pub struct VolumeRenderer {
    pipeline: wgpu::RenderPipeline,
    sampler_nearest: wgpu::Sampler,
    sampler_linear: wgpu::Sampler,
}

impl VolumeRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[
                &Self::bind_group_layout(device),
                &ColorMap::bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/raymarch.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        VolumeRenderer {
            pipeline,
            sampler_nearest,
            sampler_linear,
        }
    }

    pub fn prepare<'a,P: Projection>(
        &mut self,
        device: &wgpu::Device,
        volume: &Volume,
        camera: &GenericCamera<P>,
        render_settings: &RenderSettings,
        cmap: &'a ColorMap,
    ) -> PerFrameData<'a> {
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("camera buffer"),
            contents: bytemuck::bytes_of(&CameraUniform::from(camera)),
            usage: wgpu::BufferUsages::UNIFORM,
        });


        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("settnigs buffer"),
            contents: bytemuck::bytes_of(&RenderSettingsUniform::from_settings(&render_settings, volume)),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let step = ((volume.timesteps - 1) as f32 * render_settings.time) as usize;
        let bind_group = 
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("volume renderer bind group"),
                layout: &Self::bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &volume.textures[step]
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &volume.textures[(step + 1) % volume.timesteps as usize]
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            if render_settings.spatial_filter == wgpu::FilterMode::Nearest {
                                &self.sampler_nearest
                            } else {
                                &self.sampler_linear
                            },
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource:wgpu::BindingResource::Buffer(camera_buffer.as_entire_buffer_binding()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource:wgpu::BindingResource::Buffer(settings_buffer.as_entire_buffer_binding()),
                    },
                ],
            });
        PerFrameData {
            bind_group,
            cmap_bind_group: &cmap.bindgroup,
        }
    }

    pub fn render<'rpass>(
        &'rpass self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
        frame_data: &'rpass PerFrameData,
    ) {
        render_pass.set_bind_group(0, &frame_data.bind_group, &[]);
        render_pass.set_bind_group(1, frame_data.cmap_bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline);

        render_pass.draw(0..4, 0..1);
    }

    fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume renderer bind group layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}

pub struct PerFrameData<'a> {
    bind_group: wgpu::BindGroup,
    cmap_bind_group: &'a wgpu::BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// the cameras view matrix
    pub(crate) view_matrix: Matrix4<f32>,
    /// inverse view matrix
    pub(crate) view_inv_matrix: Matrix4<f32>,

    // the cameras projection matrix
    pub(crate) proj_matrix: Matrix4<f32>,

    // inverse projection matrix
    pub(crate) proj_inv_matrix: Matrix4<f32>,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_matrix: Matrix4::identity(),
            view_inv_matrix: Matrix4::identity(),
            proj_matrix: Matrix4::identity(),
            proj_inv_matrix: Matrix4::identity(),
        }
    }
}

impl CameraUniform {

    pub(crate) fn set_view_mat(&mut self, view_matrix: Matrix4<f32>) {
        self.view_matrix = view_matrix;
        self.view_inv_matrix = view_matrix.invert().unwrap();
    }

    pub(crate) fn set_proj_mat(&mut self, proj_matrix: Matrix4<f32>) {
        self.proj_matrix = VIEWPORT_Y_FLIP * proj_matrix;
        self.proj_inv_matrix = proj_matrix.invert().unwrap();
    }

    pub fn set_camera(&mut self, camera: &GenericCamera<impl Projection>) {
        self.set_proj_mat(camera.proj_matrix());
        self.set_view_mat(camera.view_matrix());
    }
}

impl<P:Projection> From<&GenericCamera<P>> for CameraUniform {
    fn from(camera: &GenericCamera<P>) -> Self {
        let mut uniform = CameraUniform::default();
        uniform.set_camera(camera);
        uniform
    }
}

#[derive(Debug, Clone)]
pub struct RenderSettings {
    pub clipping_aabb: Aabb<f32>,
    pub time: f32,
    pub step_size: f32,
    pub spatial_filter: wgpu::FilterMode,
    pub temporal_filter: wgpu::FilterMode,
    pub distance_scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderSettingsUniform {
    volume_aabb_min: Vector4<f32>,
    volume_aabb_max: Vector4<f32>,
    clipping_min: Vector4<f32>,
    clipping_max: Vector4<f32>,
    time: f32,
    time_steps: u32,
    step_size: f32,
    temporal_filter: u32,
    distance_scale: f32,
    _pad: [u32; 3],
}

impl RenderSettingsUniform {
    pub fn from_settings(settings: &RenderSettings, volume: &Volume) -> Self {
        let volume_aabb = volume.aabb;

        Self {
            volume_aabb_min: volume_aabb.min.to_vec().extend(0.),
            volume_aabb_max: volume_aabb.max.to_vec().extend(0.),
            time: settings.time,
            time_steps: volume.timesteps as u32,
            clipping_min: settings.clipping_aabb.min.to_vec().extend(0.),
            clipping_max: settings.clipping_aabb.max.to_vec().extend(0.),
            step_size: settings.step_size,
            temporal_filter: settings.temporal_filter as u32,
            distance_scale: settings.distance_scale,
            _pad: [0; 3],
        }
    }
}

impl Default for RenderSettingsUniform {
    fn default() -> Self {
        Self {
            volume_aabb_min: Vector4::new(-1., -1., -1., 0.),
            volume_aabb_max: Vector4::new(1., 1., 1., 0.),
            clipping_min: Vector4::zero(),
            clipping_max: Vector4::new(1., 1., 1., 0.),
            time: 0.,
            time_steps: 1,
            step_size: 0.01,
            temporal_filter: wgpu::FilterMode::Nearest as u32,
            distance_scale: 1.,
            _pad: [0; 3],
        }
    }
}
