bash build_wasm.sh &&\
cargo build --release --bin viewer --bin viewer --features colormaps &&\
cargo build --release --bin viewer --bin viewer &&\
maturin develop