use std::{
    collections::HashMap, fs, io::{BufReader, Read, Seek}, sync::Arc
};

use camera::{GenericCamera, PerspectiveCamera};
use controller::CameraController;
#[cfg(target_arch = "wasm32")]
use instant::{Duration, Instant};
use renderer::{RenderSettings, VolumeRenderer};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use wgpu::Backends;

use cgmath::{
    vec3, Deg, EuclideanSpace, InnerSpace, Point3, Quaternion, Rotation, Vector2,
    Vector3,
};
use egui::{Color32, TextureId};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::PhysicalKey,
    window::{Window, WindowBuilder},
};

use crate::{
    camera::PerspectiveProjection,
    cmap::ColorMap,
    volume::{Aabb, Volume},
};

mod camera;
mod cmap;
mod controller;
mod debug;
mod lines;
mod renderer;
mod ui;
mod ui_renderer;
mod uniform;
mod volume;

pub struct RenderConfig {
    pub no_vsync: bool,
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new_instance() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        return WGPUContext::new(&instance, None).await;
    }

    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();

        let required_features = wgpu::Features::FLOAT32_FILTERABLE;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features,
                    required_limits: wgpu::Limits {
                        max_bind_groups: 5,
                        ..wgpu::Limits::downlevel_webgl2_defaults()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        Self {
            device,
            queue,
            adapter,
        }
    }
}

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scale_factor: f32,

    controller: CameraController,
    camera: GenericCamera<PerspectiveProjection>,
    ui_renderer: ui_renderer::EguiWGPU,
    ui_visible: bool,

    background_color: egui::Color32,

    line_renderer: lines::LineRenderer,
    debug_lines: debug::DebugLines,

    volume: Volume,
    renderer: VolumeRenderer,

    render_settings: RenderSettings,
    cmaps: HashMap<String,(cmap::ColorMap,TextureId)>,
    selected_cmap: String,

    playing:bool,
    animation_duration:Duration,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R: Read + Seek>(
        window: Window,
        pc_file: R,
        render_config: &RenderConfig,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size();
        if size == PhysicalSize::new(0, 0) {
            size = PhysicalSize::new(800, 600);
        }

        let window = Arc::new(window);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        log::info!("device: {:?}", wgpu_context.adapter.get_info().name);

        let device = &wgpu_context.device;
        let queue = &wgpu_context.queue;

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();

        let render_format = surface_format;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.remove_srgb_suffix()],
        };
        surface.configure(&device, &config);

        let mut ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);

        let mut buff_reader = BufReader::new(pc_file);
        let volume = Volume::load_npz(device, queue, &mut buff_reader)?;
        let renderer = VolumeRenderer::new(device, surface_format);

        let line_renderer = lines::LineRenderer::new(device, render_format);
        let debug_lines = debug::DebugLines::new(&volume);

        let files = find_files_with_extension_in_folder("colormaps", ".npy");
        let cmaps: HashMap<String,(ColorMap,TextureId)> = files
            .iter()
            .map(|f| {
                let mut reader =
                    BufReader::new(fs::File::open(format!("colormaps/{}", f)).unwrap());
                let name = f.split(".").next().unwrap().to_string();
                let cmap = ColorMap::from_npz(device, queue, &mut reader).unwrap();
                let egui_texture:TextureId = ui_renderer.renderer.register_native_texture(
                    &device,
                    &cmap.texture.create_view(&Default::default()),
                    wgpu::FilterMode::Linear,
                );
                (name.clone(),(cmap, egui_texture))
            })
            .collect();

        

        let render_settings = RenderSettings {
            clipping_aabb: Aabb::unit(),
            use_dda: false,
            time: 0.,
            step_size: 2. / 1000.,
            spatial_filter: wgpu::FilterMode::Linear,
            temporal_filter: wgpu::FilterMode::Linear,
            distance_scale: 1.,
        };

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = volume.aabb.center();
        let r = volume.aabb.radius();
        let corner = vec3(1., -1., 1.);
        let view_dir = Quaternion::look_at(-corner, Vector3::unit_y());
        let camera = PerspectiveCamera::new(
            Point3::from_vec(corner.normalize()) * r * 3.,
            view_dir,
            PerspectiveProjection::new(
                Vector2::new(size.width, size.height),
                Deg(45.),
                0.01,
                1000.,
            ),
        );

        // let camera = GenericCamera::new(
        //     volume.aabb.center()-vec3(0., 0., 3.),
        //     Quaternion::one(),
        //     OrthographicProjection::new(
        //         Vector2::new(size.width, size.height)/300,
        //         1000.,
        //         -1000.,
        //     ),
        // );
        let selected_cmap = "magma".to_string();

        let animation_duration =Duration::from_secs_f32(volume.timesteps as f32*0.05);

        Ok(Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            controller,
            ui_renderer,
            ui_visible: true,
            background_color: Color32::BLACK,
            camera,

            volume,
            renderer,
            line_renderer,
            debug_lines,
            render_settings,
            cmaps,
            selected_cmap,
            animation_duration,
            playing: true,
        })
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.camera
                .projection
                .resize(new_size.width, new_size.height);
            self.surface
                .configure(&self.wgpu_context.device, &self.config);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    fn update(&mut self, dt: Duration) {
        self.controller.update_camera(&mut self.camera, dt);
        
        if self.playing {
            self.render_settings.time += dt.as_secs_f32() / self.animation_duration.as_secs_f32();
            self.render_settings.time = self.render_settings.time.fract();
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let window_size = self.window.inner_size();
        if window_size.width != self.config.width || window_size.height != self.config.height {
            self.resize(window_size, None);
        }

        let output = self.surface.get_current_texture()?;
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format),
            ..Default::default()
        });

        // do prepare stuff

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        self.renderer.prepare(
            &mut encoder,
            &self.wgpu_context.device,
            &self.wgpu_context.queue,
            &self.volume,
            self.camera.clone(),
            &self.render_settings,
        );

        self.line_renderer.prepare(
            &mut encoder,
            &self.wgpu_context.queue,
            &self.wgpu_context.device,
            &self.debug_lines.visible_lines(),
        );
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_rgb,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.background_color.r() as f64 / 255.,
                            g: self.background_color.g() as f64 / 255.,
                            b: self.background_color.b() as f64 / 255.,
                            a: 1.,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.renderer
                .render(&mut render_pass, &self.volume, &self.cmaps[&self.selected_cmap].0);
            self.line_renderer
                .render(&mut render_pass, &self.renderer.camera);
        }

        self.wgpu_context
            .queue
            .submit(std::iter::once(encoder.finish()));

        if self.ui_visible {
            // ui rendering
            self.ui_renderer.begin_frame(&self.window);
            ui::ui(self);

            let shapes = self.ui_renderer.end_frame(&self.window);

            self.ui_renderer.paint(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &view_rgb,
                shapes,
            );
        }

        output.present();
        Ok(())
    }
}

fn find_files_with_extension_in_folder(folder_path: &str, extension: &str) -> Vec<String> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(folder_path) {
        for entry in entries {
            if let Ok(entry) = entry {
                if let Some(file_name) = entry.file_name().to_str() {
                    if file_name.ends_with(extension) {
                        files.push(file_name.to_string());
                    }
                }
            }
        }
    }
    files
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(file: R, config: RenderConfig) {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let window_size = PhysicalSize::new(800, 600);

    let window = WindowBuilder::new()
        .with_title("web-splats")
        .with_inner_size(window_size)
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                doc.get_element_by_id("loading-display")
                    .unwrap()
                    .set_text_content(Some("Unpacking"));
                doc.body()
            })
            .and_then(|body| {
                let canvas = window.canvas().unwrap();
                canvas.set_id("window-canvas");
                canvas.set_width(body.client_width() as u32);
                canvas.set_height(body.client_height() as u32);
                let elm = web_sys::Element::from(canvas);
                elm.set_attribute("style", "width: 100%; height: 100%;")
                    .unwrap();
                body.append_child(&elm).ok()
            })
            .expect("couldn't append canvas to document body");
    }

    let mut state = WindowContext::new(window, file, &config).await.unwrap();

    let mut last = Instant::now();

    event_loop.run(move |event,target| 
        
        match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() && !state.ui_renderer.on_event(&state.window,event) => match event {
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size, None);
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                ..
            } => {
                state.scale_factor = *scale_factor as f32;
            }
            WindowEvent::CloseRequested => {log::info!("close!");target.exit()},
            WindowEvent::ModifiersChanged(m)=>{
                state.controller.alt_pressed = m.state().alt_key();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key{
                state
                    .controller
                    .process_keyboard(key, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                    state.controller.process_scroll(*dy )
                }
                winit::event::MouseScrollDelta::PixelDelta(p) => {
                    state.controller.process_scroll(p.y as f32 / 100.)
                }
            },
            WindowEvent::MouseInput { state:button_state, button, .. }=>{
                match button {
                    winit::event::MouseButton::Left =>                         state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                    winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                    _=>{}
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now-last;
                last = now;
                state.update(dt);
    
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size(), None),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) =>target.exit(),
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => println!("error: {:?}", e),
                }
            }
            _ => {}
        },
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion{ delta, },
            .. // We're not using device_id currently
        } => {
            state.controller.process_mouse(delta.0 as f32, delta.1 as f32)
        }
        
        Event::AboutToWait => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window.request_redraw();
        }
        _ => {},
    }).unwrap();
}
