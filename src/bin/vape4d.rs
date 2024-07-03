use clap::Parser;
use std::{fmt::Debug, fs::File, io::BufReader, path::PathBuf};

use vape4d::cmap;
use vape4d::{open_window, volume::Volume, RenderConfig};
use winit::{dpi::PhysicalSize, window::WindowBuilder};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    #[arg(long, default_value_t = false)]
    no_vsync: bool,

    #[arg(long, default_value_t = false)]
    channel_first: bool,

    #[cfg(not(feature = "colormaps"))]
    colormap: PathBuf,
}

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let opt = Opt::parse();

    let data_file = File::open(&opt.input)?;

    let window_builder = WindowBuilder::new().with_inner_size(PhysicalSize::new(800, 600));

    let volumes = Volume::load_numpy(BufReader::new(data_file), !opt.channel_first)
        .expect("Failed to load volume");

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
        },
    )
    .await;
    Ok(())
}
