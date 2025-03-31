use camera::{Camera, OrthographicProjection};
use cmap::ColorMap;
use controller::CameraController;
use egui::FullOutput;
use renderer::{RenderSettings, VolumeRenderer};
use std::{path::PathBuf, sync::Arc};
use volume::VolumeGPU;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::*;

use cgmath::{Vector2, Vector3};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

use crate::volume::Volume;

pub mod camera;
pub mod cmap;
mod controller;
pub mod offline;
#[cfg(feature = "python")]
pub mod py;
pub mod renderer;
mod ui;
mod ui_renderer;
#[cfg(not(target_arch = "wasm32"))]
mod viewer;
#[cfg(not(target_arch = "wasm32"))]
pub use viewer::viewer;

pub mod volume;

#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub no_vsync: bool,
    pub background_color: wgpu::Color,
    pub show_colormap_editor: bool,
    pub vmin: Option<f32>,
    pub vmax: Option<f32>,
    pub distance_scale: f32,
    pub axis_scale: Vector3<f32>,

    pub show_cmap_select: bool,
    pub duration: Option<Duration>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            no_vsync: false,
            background_color: wgpu::Color::BLACK,
            show_colormap_editor: true,
            vmin: None,
            vmax: None,
            distance_scale: 1.0,
            axis_scale: Vector3::new(1.0, 1.0, 1.0),
            show_cmap_select: true,
            duration: None,
        }
    }
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();

        let required_features = wgpu::Features::default();

        log::debug!("adapter: {:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features,
                    required_limits: adapter.limits(),
                    label: None,
                    memory_hints: wgpu::MemoryHints::Performance,
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
    camera: Camera<OrthographicProjection>,
    ui_renderer: ui_renderer::EguiWGPU,
    ui_visible: bool,

    background_color: wgpu::Color,

    volume: VolumeGPU,
    renderer: VolumeRenderer,

    render_settings: RenderSettings,

    playing: bool,
    animation_duration: Duration,
    num_columns: u32,
    selected_channel: Option<usize>,

    colormap_editor_visible: bool,

    cmap_select_visible: bool,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new(
        window: Window,
        volume: Volume,
        cmap: ColorMap,
        render_config: RenderConfig,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            size = LogicalSize::new(800, 600).to_physical(window.scale_factor());
        }
        let window = Arc::new(window);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all().symmetric_difference(wgpu::Backends::BROWSER_WEBGPU),
            ..Default::default()
        });

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        log::info!("device: {:?}", wgpu_context.adapter.get_info().name);

        let device = &wgpu_context.device;
        let queue = &wgpu_context.queue;

        let max_size = device.limits().max_texture_dimension_2d;
        window.set_max_inner_size(Some(PhysicalSize::new(max_size, max_size)));

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();
        let surface_format = surface_format;

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
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);

        let renderer = VolumeRenderer::new(device,queue, surface_format);

        let render_settings = RenderSettings {
            clipping_aabb: None,
            time: 0.,
            step_size: 2. / 1000.,
            spatial_filter: wgpu::FilterMode::Linear,
            temporal_filter: wgpu::FilterMode::Linear,
            distance_scale: render_config.distance_scale,
            vmin: render_config.vmin,
            vmax: render_config.vmax,
            gamma_correction: !surface_format.is_srgb(),
            axis_scale: render_config.axis_scale,
            cmap
        };

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = volume.aabb.center();

        let radius = volume.aabb.radius();
        let ratio = size.width as f32 / size.height as f32;
        let camera = Camera::new_aabb_iso(
            volume.aabb.clone(),
            OrthographicProjection::new(Vector2::new(ratio, 1.) * 2. * radius, 1e-4, 100.),
        );

        let animation_duration = render_config
            .duration
            .unwrap_or(Duration::from_secs_f32(5.));

        let num_columns = volume.channels() as u32;
        let volumes_gpu = VolumeGPU::new(device, queue, volume);

        Ok(Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            controller,
            ui_renderer,
            ui_visible: true,
            background_color: render_config.background_color,
            camera,

            volume: volumes_gpu,
            renderer,
            render_settings,
            animation_duration,
            playing: true,
            num_columns,
            selected_channel: None,
            colormap_editor_visible: render_config.show_colormap_editor,

            cmap_select_visible: render_config.show_cmap_select,
        })
    }

    fn load_file(&mut self, path: &PathBuf) -> anyhow::Result<()> {
        let reader = std::fs::File::open(path)?;
        let volume = Volume::load_numpy(reader, true)?;
        let volume_gpu =
            VolumeGPU::new(&self.wgpu_context.device, &self.wgpu_context.queue, volume);
        self.volume = volume_gpu;
        // self.controller.center = volume.aabb.center();
        self.camera
            .projection
            .resize(self.config.width, self.config.height);
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            let new_width = new_size.width;
            let new_height = new_size.height;
            self.config.width = new_width;
            self.config.height = new_height;
            self.camera.projection.resize(new_width, new_height);
            self.surface
                .configure(&self.wgpu_context.device, &self.config);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    fn update(&mut self, dt: Duration) -> bool {
        let mut requires_redraw = false;
        let old_camera = self.camera.clone();
        self.controller.update_camera(&mut self.camera, dt);
        if self.camera != old_camera {
            requires_redraw = true;
        }

        if self.playing && self.volume.volume.timesteps() > 1 {
            self.render_settings.time += dt.as_secs_f32() / self.animation_duration.as_secs_f32();
            self.render_settings.time = self.render_settings.time.fract();

            requires_redraw = true;
        }
        return requires_redraw;
    }

    /// returns whether redraw is required
    fn ui(&mut self) -> (bool, egui::FullOutput) {
        self.ui_renderer.begin_frame(&self.window);
        let request_redraw = ui::ui(self);

        let shapes = self.ui_renderer.end_frame(&self.window);

        return (request_redraw, shapes);
    }

    fn render(&mut self, shapes: Option<FullOutput>) -> Result<(), wgpu::SurfaceError> {
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
        let mut frame_data = Vec::new();

        let columns = self.num_columns as usize;
        let rows = (self.volume.volume.channels() as f32 / columns as f32).ceil() as usize;
        let cell_width = self.config.width as f32 / columns as f32;
        let cell_height = self.config.height as f32 / rows as f32;

        let ui_state = shapes.map(|shapes| {
            self.ui_renderer.prepare(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &mut encoder,
                shapes,
            )
        });

        if let Some(selected_channel) = self.selected_channel {
            let camera = self.camera.clone();
            frame_data.push(self.renderer.prepare(
                &self.wgpu_context.device,
                &self.volume,
                &camera,
                &self.render_settings,
                selected_channel,
            ));
        } else {
            for channel in 0..self.volume.volume.channels() {
                let mut camera = self.camera.clone();
                camera
                    .projection
                    .resize(cell_width as u32, cell_height as u32);
                frame_data.push(self.renderer.prepare(
                    &self.wgpu_context.device,
                    &self.volume,
                    &camera,
                    &self.render_settings,
                    channel,
                ));
            }
        }

        {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view_rgb,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            for (i, v) in frame_data.iter().enumerate() {
                if self.selected_channel.is_none() {
                    let column = i % columns;
                    let row = i / columns;
                    render_pass.set_viewport(
                        column as f32 * cell_width,
                        row as f32 * cell_height,
                        cell_width,
                        cell_height,
                        0.,
                        1.,
                    );
                }
                self.renderer.render(&self.wgpu_context.queue,&mut render_pass, &v);
            }

            if let Some(state) = &ui_state {
                // ui rendering

                self.ui_renderer.render(&mut render_pass, state);
            }
        }
        if let Some(ui_state) = ui_state {
            self.ui_renderer.cleanup(ui_state)
        }
        self.wgpu_context
            .queue
            .submit(std::iter::once(encoder.finish()));

        output.present();
        Ok(())
    }
}

pub struct App {
    state: Option<WindowContext>,
    volume: Volume,
    config: RenderConfig,
    cmap: ColorMap,

    last_touch_position: PhysicalPosition<f64>,
    last_draw: Instant,

    event_loop_proxy: Option<EventLoopProxy<WindowContext>>,
    #[cfg(target_arch = "wasm32")]
    canvas_id: String,
}

impl App {
    fn new(
        event_loop_proxy: EventLoopProxy<WindowContext>,
        volume: Volume,
        cmap: ColorMap,
        config: RenderConfig,
        #[cfg(target_arch = "wasm32")] canvas_id: String,
    ) -> Self {
        Self {
            state: None,
            volume,
            config,
            last_touch_position: PhysicalPosition::new(0., 0.),
            last_draw: Instant::now(),
            event_loop_proxy: Some(event_loop_proxy),
            cmap,
            #[cfg(target_arch = "wasm32")]
            canvas_id,
        }
    }
}

impl ApplicationHandler<WindowContext> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(event_loop_proxy) = self.event_loop_proxy.take() else {
            // event_loop_proxy is already spent - we already constructed Graphics
            return;
        };
        if self.state.is_none() {
            let version = env!("CARGO_PKG_VERSION");
            let name = env!("CARGO_PKG_NAME");

            #[allow(unused_mut)]
            let mut window_attributes = WindowAttributes::default()
                .with_title(format!("{name} {version}"))
                .with_inner_size(winit::dpi::Size::Logical(LogicalSize::new(800., 600.)));

            #[cfg(target_arch = "wasm32")]
            {
                use web_sys::wasm_bindgen::JsCast;
                use winit::platform::web::WindowAttributesExtWebSys;

                let window = web_sys::window().unwrap();
                let document = window.document().unwrap();
                let canvas = document.get_element_by_id(&self.canvas_id).unwrap();
                let html_canvas_element: HtmlCanvasElement = canvas.unchecked_into();
                let width = html_canvas_element.client_width() as u32;
                let height = html_canvas_element.client_height() as u32;
                window_attributes = window_attributes
                    .with_canvas(Some(html_canvas_element))
                    .with_inner_size(winit::dpi::Size::Physical(PhysicalSize::new(width, height)));
            }

            let window = event_loop.create_window(window_attributes).unwrap();

            #[cfg(target_arch = "wasm32")]
            {
                let window_context = WindowContext::new(
                    window,
                    self.volume.clone(),
                    self.cmap.clone(),
                    self.config.clone(),
                );
                wasm_bindgen_futures::spawn_local(async move {
                    let gfx = window_context.await.unwrap();
                    assert!(event_loop_proxy.send_event(gfx).is_ok());
                });
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                let window_context = WindowContext::new(
                    window,
                    self.volume.clone(),
                    self.cmap.clone(),
                    self.config.clone(),
                );
                let context = pollster::block_on(window_context).unwrap();
                assert!(event_loop_proxy.send_event(context).is_ok());
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = &mut self.state {
            if window_id == state.window.id()
                && !(state.ui_visible && state.ui_renderer.on_event(&state.window, &event))
            {
                match event {
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size, None);
                    }
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        state.scale_factor = scale_factor as f32;
                    }
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::ModifiersChanged(m) => {
                        state.controller.alt_pressed = m.state().alt_key();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let PhysicalKey::Code(key) = event.physical_key {
                            state
                                .controller
                                .process_keyboard(key, event.state == ElementState::Pressed);
                            if key == KeyCode::KeyU && event.state == ElementState::Released {
                                state.ui_visible = !state.ui_visible;
                            }
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                            state.controller.process_scroll(dy)
                        }
                        winit::event::MouseScrollDelta::PixelDelta(p) => {
                            state.controller.process_scroll(p.y as f32 / 100.)
                        }
                    },
                    WindowEvent::Touch(touch) => match touch.phase {
                        winit::event::TouchPhase::Started => {
                            state.controller.left_mouse_pressed = true;
                            self.last_touch_position = touch.location;
                        }
                        winit::event::TouchPhase::Ended => {
                            state.controller.left_mouse_pressed = false;
                        }
                        winit::event::TouchPhase::Moved => {
                            state.controller.process_mouse(
                                (touch.location.x - self.last_touch_position.x) as f32,
                                (touch.location.y - self.last_touch_position.y) as f32,
                            );
                            self.last_touch_position = touch.location;
                        }
                        _ => {}
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        let delta_x = position.x - self.last_touch_position.x;
                        let delta_y = position.y - self.last_touch_position.y;
                        state
                            .controller
                            .process_mouse(delta_x as f32, delta_y as f32);
                        self.last_touch_position = position;
                    }
                    WindowEvent::MouseInput {
                        state: button_state,
                        button,
                        ..
                    } => match button {
                        winit::event::MouseButton::Left => {
                            state.controller.left_mouse_pressed =
                                button_state == ElementState::Pressed
                        }
                        winit::event::MouseButton::Right => {
                            state.controller.right_mouse_pressed =
                                button_state == ElementState::Pressed
                        }
                        _ => {}
                    },
                    WindowEvent::DroppedFile(file) => {
                        if let Err(e) = state.load_file(&file) {
                            log::error!("failed to load file: {:?}", e)
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        if !self.config.no_vsync {
                            // make sure the next redraw is called with a small delay
                            event_loop.set_control_flow(ControlFlow::wait_duration(
                                Duration::from_millis(1000 / 60),
                            ));
                        }
                        let now = Instant::now();
                        let dt = now - self.last_draw;
                        self.last_draw = now;
                        let request_redraw = state.update(dt);

                        let (redraw_ui, ui_shapes) = state.ui();

                        if request_redraw || redraw_ui {
                            match state.render(state.ui_visible.then_some(ui_shapes)) {
                                Ok(_) => {}
                                // Reconfigure the surface if lost
                                Err(wgpu::SurfaceError::Lost) => {
                                    log::error!("lost surface!");
                                    state.resize(state.window.inner_size(), None)
                                }
                                // The system is out of memory, we should probably quit
                                Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                                // All other errors (Outdated, Timeout) should be resolved by the next frame
                                Err(e) => println!("error: {:?}", e),
                            }
                        }
                        if self.config.no_vsync {
                            state.window.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        log::info!("exit!");
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {
        match cause {
            winit::event::StartCause::ResumeTimeReached { .. } => {
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, #[allow(unused_variables)] event_loop: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        use winit::platform::web::WindowExtWebSys;
        #[cfg(target_arch = "wasm32")]
        if let Some(Some(canvas)) = self.state.as_ref().map(|s| s.window.canvas()) {
            if canvas.parent_node().is_none() {
                // The canvas has been removed from the DOM, we should exit
                event_loop.exit();
                return;
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, window_context: WindowContext) {
        // window was created and is ready to be used
        // we have to this workaround for wasm because we can't create the window in the main thread as it is async
        self.state = Some(window_context);
    }
}

pub async fn open_window(
    volumes: Volume,
    cmap: ColorMap,
    config: RenderConfig,
    #[cfg(target_arch = "wasm32")] canvas_id: String,
) {
    let event_loop: EventLoop<WindowContext> = EventLoop::with_user_event().build().unwrap();
    let mut app = App::new(
        event_loop.create_proxy(),
        volumes,
        cmap,
        config,
        #[cfg(target_arch = "wasm32")]
        canvas_id,
    );
    event_loop.run_app(&mut app).unwrap();
}
