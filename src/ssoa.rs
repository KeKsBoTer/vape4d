use std::num::NonZero;

use wgpu::util::DeviceExt;

use crate::{
    camera::OrthographicCamera,
    renderer::CameraUniform,
};

pub struct SSAO {
    ssao_pipeline: wgpu::RenderPipeline,
    blur_vert_pipeline: wgpu::RenderPipeline,
    blur_hor_pipeline: wgpu::RenderPipeline,
    bg_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl SSAO {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat, // outgoing target texture format
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/ssao.wgsl"));

        let bg_layout = Self::bind_group_layout(device);

        let ssao_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao Pipeline layout"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });

        let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssoa render pipeline"),
            layout: Some(&ssao_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "ssao_frag",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline layout"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });

        let blur_vert_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur render pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "blur_vert_frag",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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

        let blur_hor_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur render pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "blur_hor_frag",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::SrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssoa texture sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        return Self {
            ssao_pipeline,
            blur_vert_pipeline,
            blur_hor_pipeline,
            bg_layout,
            sampler,
        };
    }

    fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao bindgroup layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // ssao render output
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // ssao render output
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // ssao render output
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZero::new(std::mem::size_of::<CameraUniform>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn render<'a>(
        &self,
        encoder: &'a mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        textures: &SSAOTextures,
        camera: &OrthographicCamera,
        target_texture: &wgpu::TextureView,
    ) {
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::bytes_of(&CameraUniform::from(camera)),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let ssao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao bind_group"),
            layout: &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&textures.normal_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });
        // ssao effect
        {
            let mut ssao_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &textures.ssao,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            ssao_pass.set_bind_group(0, &ssao_bg, &[]);
            ssao_pass.set_pipeline(&self.ssao_pipeline);

            ssao_pass.draw(0..4, 0..1);
        }

        // bluring the ssao
        // vertical pass
        {
            let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao bind_group"),
                layout: &self.bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&textures.ssao),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: camera_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut blur_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &textures.blur,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            blur_pass.set_bind_group(0, &blur_bg, &[]);
            blur_pass.set_pipeline(&self.blur_vert_pipeline);

            blur_pass.draw(0..4, 0..1);
        }
        // horizontal_pass
        {
            let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao bind_group"),
                layout: &self.bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&textures.blur),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: camera_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut blur_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_texture,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            blur_pass.set_bind_group(0, &blur_bg, &[]);
            blur_pass.set_pipeline(&self.blur_hor_pipeline);

            blur_pass.draw(0..4, 0..1);
        }
    }
}

pub struct SSAOTextures {
    pub normal_depth: wgpu::TextureView,
    pub ssao: wgpu::TextureView,
    pub blur: wgpu::TextureView,
}

impl SSAOTextures {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let normal_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normal and depth texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let ssoa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blur texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        return Self {
            normal_depth: normal_depth_texture.create_view(&Default::default()),
            ssao: ssoa_texture.create_view(&Default::default()),
            blur: blur_texture.create_view(&Default::default()),
        };
    }
}
