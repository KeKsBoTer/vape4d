use clap::Parser;
use std::{
    collections::HashMap, fmt::Debug, fs::File, io::{BufReader, Cursor}, path::PathBuf
};
use v4dv::{cmap::{self, ColorMap}, open_window, volume::Volume, RenderConfig};
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
}


#[pollster::main]
async fn main() {
    let opt = Opt::parse();

    let data_file = File::open(&opt.input).unwrap();


    let window_builder = WindowBuilder::new().with_inner_size(PhysicalSize::new(800, 600));

    let volumes = Volume::load_npz(BufReader::new(data_file),!opt.channel_first).expect("Failed to load volume");

  
    let cmap = cmap::COLORMAPS.get("viridis").unwrap().clone();

    open_window(
        window_builder,
        volumes,
        cmap,
        RenderConfig {
            no_vsync: opt.no_vsync,
        },
    )
    .await;
}
