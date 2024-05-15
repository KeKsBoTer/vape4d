use cgmath::Vector2;
use clap::Parser;

use std::{fs::File, path::PathBuf};
use v4dv::cmap::ColorMapType;
use v4dv::volume::Volume;

#[derive(Debug, Parser)]
#[command(author, version)]
#[command(about = "Offline renderer. Renders to PNG files", long_about = None)]
struct Opt {
    /// input file
    input: PathBuf,

    colormap: PathBuf,

    /// image output directory
    img_out: PathBuf,

    #[arg(long, default_value = "1024")]
    width: u32,
    #[arg(long, default_value = "1024")]
    height: u32,

    #[arg(long, default_value_t = false)]
    channel_first: bool,

    #[arg(long, short)]
    time: f32,

    #[arg(long, short, num_args = 4, default_values_t = [0, 0, 0, 255])]
    background_color: Vec<u8>,

    vmin: Option<f32>,
    vmax: Option<f32>,

    #[arg(long, default_value = "1")]
    distance_scale: f32,
}

#[cfg(not(target_arch = "wasm32"))]
#[pollster::main]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let opt = Opt::parse();

    anyhow::ensure!(
        opt.time >= 0. && opt.time <= 1.,
        "time must be in the range [0, 1]"
    );

    // TODO this is suboptimal as it is never closed
    let volume_file = File::open(&opt.input)?;
    let cmap_file = File::open(opt.colormap)?;
    let volumes = Volume::load_numpy(volume_file, opt.channel_first)?;
    let cmap = ColorMapType::read(cmap_file)?;

    let resolution = Vector2::new(opt.width, opt.height);

    let background_color = wgpu::Color {
        r: opt.background_color[0] as f64 / 255.,
        g: opt.background_color[1] as f64 / 255.,
        b: opt.background_color[2] as f64 / 255.,
        a: opt.background_color[3] as f64 / 255.,
    };

    let img = v4dv::offline::render_volume(
        volumes,
        cmap,
        resolution,
        opt.time,
        background_color,
        opt.vmin,
        opt.vmax,
        opt.distance_scale,
    )
    .await?;

    img.save(opt.img_out)?;

    println!("done!");
    Ok(())
}
#[cfg(target_arch = "wasm32")]
fn main() {
    todo!("not implemented")
}
