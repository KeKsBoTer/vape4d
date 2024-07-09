use std::env;

use vape4d::viewer;

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    viewer(env::args()).await
}
