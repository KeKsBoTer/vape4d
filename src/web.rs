use std::io::Cursor;

use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsError, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::js_sys::{ArrayBuffer, Uint8Array};
use web_sys::{Request, RequestInit, RequestMode, Response};
use web_time::Duration;
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::WindowAttributes;

use crate::cmap::{self, GenericColorMap, COLORMAP_RESOLUTION};
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
    pub duration: Option<f32>,
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
        duration: Option<f32>,
    ) -> Self {
        Self {
            background_color,
            show_colormap_editor,
            show_volume_info,
            show_cmap_select,
            vmin,
            vmax,
            distance_scale,
            duration,
        }
    }
}

#[wasm_bindgen]
pub fn wasm_setup() {
    #[cfg(debug_assertions)]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
}

/// Start the viewer with the given canvas id and optional volume data and colormap.
#[wasm_bindgen]
pub async fn viewer_wasm(
    canvas_id: String,
    volume_data: Option<Vec<u8>>,
    colormap: Option<Vec<u8>>,
    settings: Option<InlineViewerConfig>,
) -> Result<(), JsValue> {
    let render_config = match settings {
        Some(settings) => RenderConfig {
            no_vsync: false,
            background_color: settings.background_color.into(),
            show_colormap_editor: settings.show_colormap_editor,
            show_volume_info: settings.show_volume_info,
            vmin: settings.vmin,
            vmax: settings.vmax,
            distance_scale: settings.distance_scale,
            #[cfg(feature = "colormaps")]
            show_cmap_select: settings.show_cmap_select,
            duration: settings.duration.map(Duration::from_secs_f32),
        },
        None => RenderConfig {
            no_vsync: false,
            background_color: wgpu::Color::BLACK,
            show_colormap_editor: true,
            show_volume_info: true,
            show_cmap_select: true,
            vmin: None,
            vmax: None,
            duration: None,
            distance_scale: 1.0,
        },
    };

    start_viewer(canvas_id, render_config, volume_data, colormap).await
}

/// Download a file from a given url
/// returns body bytes
pub async fn download_file(window: web_sys::Window, url: String) -> Result<Vec<u8>, JsValue> {
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(&url, &opts)?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    // `resp_value` is a `Response` object.
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into()?;
    if !resp.ok() {
        if resp.status() == 404 {
            return Err(JsError::new(&format!("File not found (404): {url}")).into());
        } else {
            return Err(
                JsError::new(&format!("Failed to download file: {}", resp.status_text())).into(),
            );
        }
    }

    // Convert this other `Promise` into a rust `Future`.
    let data = JsFuture::from(resp.array_buffer()?).await?;
    let abuffer = data.dyn_into::<ArrayBuffer>()?;
    let byte_buffer = Uint8Array::new(&abuffer);

    Ok(byte_buffer.to_vec())
}

async fn load_colormap() -> Result<Option<Vec<u8>>, JsValue> {
    let window = web_sys::window().ok_or(JsError::new("cannot access window"))?;
    let search_string = window.location().search()?;
    let file_param = web_sys::UrlSearchParams::new_with_str(&search_string)?.get("colormap");

    return match file_param {
        Some(url) => download_file(window, url).await.map(|v| Some(v)),
        None => Ok(None),
    };
}

/// load volume data from a file file promt or url (HTTP Get parameter "file")
async fn load_data() -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or(JsError::new("cannot access window"))?;
    let search_string = window.location().search()?;
    let file_param = web_sys::UrlSearchParams::new_with_str(&search_string)?.get("file");

    let file_data: Vec<u8> = if let Some(url) = file_param {
        // download file from url
        download_file(window, url).await?
    } else {
        // loop until file is selected
        let mut data = None;
        loop {
            if let Some(reader) = rfd::AsyncFileDialog::new()
                .set_title("Select npy file")
                .add_filter("numpy file", &["npy", "npz"])
                .pick_file()
                .await
            {
                data.replace(reader.read().await);
                break;
            }
        }
        data.unwrap()
    };
    return Ok(file_data);
}

fn show_error(document: &web_sys::Document, error: anyhow::Error) -> Result<(), JsValue> {
    let error_div = document
        .get_element_by_id("error-message")
        .ok_or(JsError::new("cannot find error message div"))?
        .dyn_into::<web_sys::HtmlElement>()?;
    error_div.set_inner_html(&format!("Error: {:?}", error));
    document
        .get_element_by_id("loading-error")
        .ok_or(JsError::new("cannot find loading error div"))?
        .set_attribute("style", "display:block;")?;
    Ok(())
}

/// Start the viewer with the given canvas id and optional volume data and colormap.
/// If volume data and colormap are not provided, the viewer will prompt the user to select a file (or load from a url provided in the page url).
async fn start_viewer(
    canvas_id: String,
    render_config: RenderConfig,
    volume_data: Option<Vec<u8>>,
    colormap: Option<Vec<u8>>,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or(JsError::new("cannot access window"))?;
    let document = window
        .document()
        .ok_or(JsError::new("cannot access document"))?;

    document.set_title(&format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")));

    let canvas = document
        .get_element_by_id(&canvas_id)
        .ok_or(JsError::new("cannot find canvas"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let window_attributes = WindowAttributes::default().with_canvas(Some(canvas));

    let spinner = document
        .get_element_by_id("spinner")
        .ok_or(JsError::new("cannot find loading spinner"))?
        .dyn_into::<web_sys::HtmlElement>()?;

    let overlay = document
        .get_element_by_id("overlay")
        .ok_or(JsError::new("cannot find overlay item"))?
        .dyn_into::<web_sys::HtmlElement>()?;

    spinner.set_attribute("style", "display:flex;")?;
    let volume_data = match volume_data {
        Some(data) => data,
        None => load_data().await?,
    };
    // load colormap from url if present
    let colormap = match colormap {
        Some(data) => Some(data),
        None => load_colormap().await?,
    };
    let colormap = match colormap {
        Some(data) => GenericColorMap::read(Cursor::new(data))
            .map_err(|e| JsError::new(&format!("Failed to load colormap: {}", e)))?
            .into_linear_segmented(COLORMAP_RESOLUTION),
        None => cmap::COLORMAPS["seaborn"]["icefire"]
            .clone()
            .into_linear_segmented(COLORMAP_RESOLUTION),
    };

    wasm_bindgen_futures::spawn_local(async move {
        let reader_v = Cursor::new(volume_data);
        let volumes = match Volume::load_numpy(reader_v, true) {
            Ok(volumes) => volumes,
            Err(err) => {
                show_error(&document, err).ok();
                return;
            }
        };
        overlay.set_attribute("style", "display:none;").ok();

        let err = open_window(window_attributes, volumes, colormap, render_config).await;
        if let Err(err) = err {
            show_error(&document, err).unwrap();
        }
    });
    Ok(())
}
