cargo build \
    --no-default-features \
    --target wasm32-unknown-unknown \
    --lib \
    --profile web-release \
&& wasm-bindgen \
    --out-dir public \
    --web target/wasm32-unknown-unknown/web-release/v4dv.wasm \
    --no-typescript     