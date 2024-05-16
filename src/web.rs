use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsCast;
use winit::platform::web::WindowBuilderExtWebSys;
use winit::window::WindowBuilder;

use crate::cmap::{GenericColorMap, COLORMAP_RESOLUTION};
use crate::volume::Volume;
use crate::{open_window, RenderConfig};

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

#[wasm_bindgen]
impl Color {
    #[wasm_bindgen(constructor)]
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

impl From<Color> for wgpu::Color {
    fn from(color: Color) -> Self {
        Self {
            r: color.r as f64,
            g: color.g as f64,
            b: color.b as f64,
            a: color.a as f64,
        }
    }
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct InlineViewerConfig {
    pub background_color: Color,
    pub show_colormap_editor: bool,
    pub show_volume_info: bool,
    pub show_cmap_select: bool,
    // for normalization
    pub vmin: Option<f32>,
    pub vmax: Option<f32>,
    pub distance_scale: f32,
}

#[wasm_bindgen]
impl InlineViewerConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        background_color: Color,
        show_colormap_editor: bool,
        show_volume_info: bool,
        show_cmap_select: bool,
        vmin: Option<f32>,
        vmax: Option<f32>,
        distance_scale: f32,
    ) -> Self {
        Self {
            background_color,
            show_colormap_editor,
            show_volume_info,
            show_cmap_select,
            vmin,
            vmax,
            distance_scale,
        }
    }
}
#[wasm_bindgen]
pub async fn viewer_inline(
    npz_file: Vec<u8>,
    colormap: Vec<u8>,
    canvas_id: String,
    settings: InlineViewerConfig,
) {
    use std::io::Cursor;
    #[cfg(debug_assertions)]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let reader = Cursor::new(npz_file);
    let volumes = Volume::load_numpy(reader, true).expect("Failed to load volumes");

    let reader_colormap = Cursor::new(colormap);

    let cmap = GenericColorMap::read(reader_colormap)
        .unwrap()
        .into_linear_segmented(COLORMAP_RESOLUTION);

    let (canvas, spinner) = web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let canvas = doc
                .get_element_by_id(&canvas_id)
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .ok();
            let spinner = doc
                .get_element_by_id("spinner")
                .unwrap()
                .dyn_into::<web_sys::HtmlElement>()
                .unwrap();
            Some((canvas, spinner))
        })
        .unwrap();
    let window_builder = WindowBuilder::new().with_canvas(canvas);
    spinner.set_attribute("style", "display:none;").unwrap();

    wasm_bindgen_futures::spawn_local(open_window(
        window_builder,
        volumes,
        cmap,
        RenderConfig {
            no_vsync: false,
            background_color: settings.background_color.into(),
            show_colormap_editor: settings.show_colormap_editor,
            show_volume_info: settings.show_volume_info,
            vmin: settings.vmin,
            vmax: settings.vmax,
            #[cfg(feature = "colormaps")]
            show_cmap_select: settings.show_cmap_select,
        },
    ));
}

#[cfg(feature = "colormaps")]
#[wasm_bindgen]
pub async fn viewer_wasm(canvas_id: String) {
    use std::io::Cursor;

    use web_sys::{HtmlCanvasElement, HtmlElement};
    use winit::dpi::PhysicalSize;

    use crate::cmap;
    #[cfg(debug_assertions)]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");

    let cmap = cmap::COLORMAPS["seaborn"]["icefire"].clone();

    let (canvas, spinner): (HtmlCanvasElement, HtmlElement) = web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let canvas = doc
                .get_element_by_id(&canvas_id)
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            let spinner = doc
                .get_element_by_id("spinner")
                .unwrap()
                .dyn_into::<web_sys::HtmlElement>()
                .unwrap();
            Some((canvas, spinner))
        })
        .unwrap();

    let size = (canvas.width() as u32, canvas.height() as u32);
    let window_builder = WindowBuilder::new()
        .with_canvas(Some(canvas))
        .with_inner_size(PhysicalSize::new(size.0, size.1));

    loop {
        if let Some(reader) = rfd::AsyncFileDialog::new()
            .set_title("Select npy file")
            .add_filter("numpy file", &["npy", "npz"])
            .pick_file()
            .await
        {
            spinner.set_attribute("style", "display:flex;").unwrap();
            let data = reader.read().await;
            let reader_v = Cursor::new(data);
            let volumes = Volume::load_numpy(reader_v, true).expect("Failed to load volumes");

            spinner.set_attribute("style", "display:none;").unwrap();
            wasm_bindgen_futures::spawn_local(open_window(
                window_builder,
                volumes,
                cmap.into_linear_segmented(COLORMAP_RESOLUTION),
                RenderConfig {
                    no_vsync: false,
                    background_color: wgpu::Color::BLACK,
                    show_colormap_editor: true,
                    show_volume_info: true,
                    show_cmap_select: true,
                    vmin: None,
                    vmax: None,
                },
            ));
            break;
        }
    }
}
