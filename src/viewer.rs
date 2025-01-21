use clap::Parser;
#[cfg(not(target_arch = "wasm32"))]
use std::{ffi::OsString, fs::File, io::BufReader};
use std::{fmt::Debug, path::PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use crate::{cmap, open_window, volume::Volume, RenderConfig};

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

    let volume= Volume::load_numpy(BufReader::new(data_file), !opt.channel_first)
        .expect("Failed to load volume");

    let cmap = opt
        .colormap
        .map_or(Ok(cmap::COLORMAPS["seaborn"]["icefire"].clone()), |path| {
            let reader = File::open(path)?;
            cmap::GenericColorMap::read(reader)
        })?;

    open_window(
        volume,
        cmap.into_linear_segmented(cmap::COLORMAP_RESOLUTION),
        RenderConfig {
            no_vsync: opt.no_vsync,
            ..Default::default()
        },
    )
    .await;
    Ok(())
}
