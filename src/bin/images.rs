use cgmath::{InnerSpace, Vector3};
use clap::Parser;
use vape4d::{cmap::ColorMap, renderer::{RenderSettings, UpscalingMethod}, volume::Volume};
#[cfg(not(target_arch = "wasm32"))]
use std::{fs::File, io::BufReader};
use std::{fmt::Debug, path::PathBuf};


#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    output: PathBuf,

    colormap: PathBuf,

    #[arg(long, default_value_t = UpscalingMethod::Nearest)]
    upscaling: UpscalingMethod,

    #[arg(long, default_value_t = 1.0)]
    render_scale: f32,
}


pub fn points_on_sphere(n:u32)->Vec<Vector3<f32>>{
    let mut points = Vec::new();
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    for i in 0..n {
        let y = 1.0 - (i as f32) / (n as f32 - 1.0) * 2.0;
        let radius = (1.0 - y * y).sqrt();
        let theta = phi * i as f32;
        points.push(Vector3::new(
            theta.cos() * radius,
            y,
            theta.sin() * radius,
        ).normalize());
    }
    points
}


#[pollster::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let opt = Opt::parse();

    let data_file = File::open(&opt.input)?;

    let volume = Volume::load(BufReader::new(data_file))
        .expect("Failed to load volume");

    let cmap_file = File::open(&opt.colormap)?;
    let cmap = ColorMap::read(cmap_file)?;

    let render_settings = RenderSettings {
        time: 0.0,
        cmap,
        upscaling_method: opt.upscaling,
        render_scale: opt.render_scale,
        distance_scale:20.,
        .. Default::default()
    };

    let (images, render_stats) = vape4d::offline::render_volume_directions(
        vec![volume],
        [1024, 1024].into(),
        wgpu::Color::TRANSPARENT,
        points_on_sphere(16),
        &render_settings
    )
    .await?;

    if !opt.output.exists() {
        std::fs::create_dir_all(&opt.output)?;
    }

    let stats_file = File::create(opt.output.join("renders_stats.json"))?;
    serde_json::to_writer_pretty(stats_file, &render_stats)?;
    
    for (i,img) in images.iter().enumerate(){
        let out_file =opt.output.join(format!("output_{i}.png"));
        img.save(out_file)?;
    }
    Ok(())
}

