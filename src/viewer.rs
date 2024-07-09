use clap::Parser;
use std::{ffi::OsString, fmt::Debug, fs::File, io::BufReader, path::PathBuf};

use winit::{dpi::PhysicalSize, window::WindowBuilder};

use crate::{cmap, open_window, volume::Volume, RenderConfig};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    #[arg(long, default_value_t = false)]
    no_vsync: bool,

    #[cfg(not(feature = "colormaps"))]
    colormap: PathBuf,
}

pub async fn viewer<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    env_logger::init();
    let opt = Opt::try_parse_from(args)?;

    let data_file = File::open(&opt.input)?;

    let window_builder = WindowBuilder::new().with_inner_size(PhysicalSize::new(800, 600));

    let volumes = Volume::load(BufReader::new(data_file)).expect("Failed to load volume");

    #[cfg(feature = "colormaps")]
    let cmap = cmap::COLORMAPS["seaborn"]["icefire"].clone();
    #[cfg(not(feature = "colormaps"))]
    let cmap = {
        let reader = File::open(&opt.colormap)?;
        cmap::GenericColorMap::read(reader)?
    };

    open_window(
        window_builder,
        volumes,
        cmap.into_linear_segmented(cmap::COLORMAP_RESOLUTION),
        RenderConfig {
            no_vsync: opt.no_vsync,
            background_color: wgpu::Color::BLACK,
            show_colormap_editor: true,
            show_volume_info: true,
            vmin: None,
            vmax: None,
            #[cfg(feature = "colormaps")]
            show_cmap_select: true,
            duration: None,
            distance_scale: 1.0,
        },
    )
    .await;
    Ok(())
}
