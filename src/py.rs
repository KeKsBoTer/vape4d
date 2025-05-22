use cgmath::Vector2;
use half::f16;
use image::{ImageBuffer, Rgba};
use numpy::{ndarray::StrideShape, IntoPyArray, PyArray4, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyTypeError, prelude::*};
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
        camera_angle: Option<(f32, f32)>,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let volume = Volume::from_array(volume.as_array());
        let cmap = ListedColorMap::from_array(cmap.as_array());

        let spatial_interpolation = spatial_interpolation
            .map(|s| parse_interpolation(&s))
            .transpose()
            .map_err(|e| PyTypeError::new_err(format!("{:?}", e)))?
            .unwrap_or_default();

        let temporal_interpolation = temporal_interpolation
            .map(|s| parse_interpolation(&s))
            .transpose()
            .map_err(|e| PyTypeError::new_err(format!("{:?}", e)))?
            .unwrap_or_default();

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
            spatial_interpolation,
            temporal_interpolation,
            camera_angle,
        ))
        .map_err(|e| PyTypeError::new_err(format!("{:?}", e)))?;

        let shape = StrideShape::from((time.len(), width as usize, height as usize, 4 as usize));
        let arr = numpy::ndarray::Array4::from_shape_vec(
            shape,
            img.iter().flat_map(|img| img.to_vec()).collect(),
        )
        .map_err(|e| PyTypeError::new_err(format!("{:?}", e)))?;
        return Ok(arr.into_pyarray(py));
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
