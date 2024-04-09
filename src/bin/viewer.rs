use clap::Parser;
use std::{fmt::Debug, fs::File, path::PathBuf};
use v4dv::{open_window, RenderConfig};

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

    open_window(
        data_file,
        RenderConfig {
            no_vsync: opt.no_vsync,
        },
    )
    .await;
}
