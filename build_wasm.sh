cargo build \
    --no-default-features \
    --target wasm32-unknown-unknown \
    --lib \
    --profile web-release\
&& wasm-bindgen \
    --out-dir public \
    --no-typescript \
    --target web \
    target/wasm32-unknown-unknown/web-release/vape4d.wasm \
    