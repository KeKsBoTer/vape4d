use clap::Parser;
use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};
use v4dv::{open_window, RenderConfig};
use winit::{dpi::PhysicalSize, window::WindowBuilder};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    #[arg(long, default_value_t = false)]
    no_vsync: bool,
}

#[pollster::main]
async fn main() {
    let opt = Opt::parse();

    let data_file = File::open(&opt.input).unwrap();

    let reader_colormap = Cursor::new(colormap);
    let cmap = ColorMap::from_npy(reader_colormap).unwrap();

    let window_builder = WindowBuilder::new().with_inner_size(PhysicalSize::new(800, 600));

    let volume = Volume::load_npz(BufReader::new(data_file)).unwrap();

    let cmap = ColorMap::from_npy(reader_colormap).unwrap();

    open_window(
        window_builder,
        data_file,
        RenderConfig {
            no_vsync: opt.no_vsync,
        },
    )
    .await;
}
