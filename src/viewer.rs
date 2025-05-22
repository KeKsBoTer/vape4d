use clap::Parser;
#[cfg(not(target_arch = "wasm32"))]
use std::{ffi::OsString, fs::File, io::BufReader};
use std::{fmt::Debug, path::PathBuf};
use cgmath::Vector3;

use crate::{cmap, open_window, volume::Volume, ViewerSettings};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    #[arg(long, default_value_t = false)]
    no_vsync: bool,

    #[arg(long, default_value_t = false)]
    channel_first: bool,

    colormap: Option<PathBuf>,
}

pub async fn viewer<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    env_logger::init();
    let opt = Opt::try_parse_from(args)?;

    let data_file = File::open(&opt.input)?;

    let volume = Volume::load_numpy(BufReader::new(data_file), !opt.channel_first)
        .expect("Failed to load volume");

    let cmap = opt
        .colormap
        .map_or(Ok(cmap::COLORMAPS["seaborn"]["icefire"].clone()), |path| {
            let reader = File::open(path)?;
            cmap::ColorMap::read(reader)
        })?;

    open_window(
        volume,
        ViewerSettings {
            no_vsync: opt.no_vsync,
            render_settings: crate::RenderSettings {
                time: 0.,
                step_size: None,
                spatial_filter: wgpu::FilterMode::Linear,
                temporal_filter: wgpu::FilterMode::Linear,
                distance_scale: 1.0,
                vmin: None,
                vmax: None,
                gamma_correction: false,
                background_color: wgpu::Color::BLACK.into(),
                cmap,
                axis_scale: Vector3::new(1.0, 1.0, 1.0),
            },
            ui_settings: crate::UISettings {
                show_colormap_editor: true,
                show_cmap_select: true,
            },
            duration: None,
        },
    )
    .await;
    Ok(())
}
