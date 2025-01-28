use cgmath::{Vector2, Vector3};
use half::f16;
use image::{ImageBuffer, Rgba};
use numpy::{ndarray::StrideShape, IntoPyArray, PyArray4, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::env::{self};

use crate::{
    cmap::{self, ListedColorMap},
    offline::render_volume,
    viewer,
    volume::Volume,
};

#[pymodule]
fn vape4d<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (volume, cmap, width, height, time, background, distance_scale, vmin=None, vmax=None, spatial_interpolation=None, temporal_interpolation=None, axis_scale=None))]
    fn render_video<'py>(
        py: Python<'py>,
        volume: PyReadonlyArrayDyn<'py, f16>,
        cmap: PyReadonlyArrayDyn<'py, f32>,
        width: u32,
        height: u32,
        time: Vec<f32>,
        background: (f32, f32, f32, f32),
        distance_scale: f32,
        vmin: Option<f32>,
        vmax: Option<f32>,
        spatial_interpolation: Option<String>,
        temporal_interpolation: Option<String>,
        axis_scale: Option<(f32, f32, f32)>,
    ) -> Bound<'py, PyArray4<u8>> {
        let volume = Volume::from_array(volume.as_array()).unwrap();
        let cmap = ListedColorMap::from_array(cmap.as_array());
        let img: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> = pollster::block_on(render_volume(
            vec![volume],
            cmap::GenericColorMap::Listed(cmap),
            Vector2::new(width, height),
            &time,
            wgpu::Color {
                r: background.0 as f64,
                g: background.1 as f64,
                b: background.2 as f64,
                a: background.3 as f64,
            },
            vmin,
            vmax,
            distance_scale,
            spatial_interpolation
                .map(|s| parse_interpolation(&s).unwrap())
                .unwrap_or_default(),
            temporal_interpolation
                .map(|s| parse_interpolation(&s).unwrap())
                .unwrap_or_default(),
            axis_scale.map(|(x, y, z)| Vector3::new(x, y, z)),
        ))
        .unwrap();

        let shape = StrideShape::from((time.len(), width as usize, height as usize, 4 as usize));
        let arr = numpy::ndarray::Array4::from_shape_vec(
            shape,
            img.iter().flat_map(|img| img.to_vec()).collect(),
        )
        .unwrap();
        return arr.into_pyarray(py);
    }

    #[pyfn(m)]
    fn standalone<'py>(_py: Python<'py>) -> PyResult<()> {
        // donts pass first argument (binary name) to parser
        match pollster::block_on(viewer(env::args().skip(1))) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                "{:?}",
                e
            ))),
        }
    }
    Ok(())
}

fn parse_interpolation(text: &str) -> anyhow::Result<wgpu::FilterMode> {
    match text.to_lowercase().as_str() {
        "nearest" => Ok(wgpu::FilterMode::Nearest),
        "linear" => Ok(wgpu::FilterMode::Linear),
        _ => Err(anyhow::format_err!("Invalid interpolation mode")),
    }
}
