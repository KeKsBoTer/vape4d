bash build_wasm.sh &&\
cargo build --release --bin vape4d --features colormaps &&\
cargo build --release --bin vape4d &&\
maturin develop