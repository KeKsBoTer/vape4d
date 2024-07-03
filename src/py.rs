use cgmath::Vector2;
use image::{ImageBuffer, Rgba};
use numpy::{ndarray::StrideShape, IntoPyArray, PyArray3, PyArray4, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::{
    cmap::{self, ListedColorMap},
    offline::render_volume,
    volume::Volume,
};

#[pymodule]
fn vape4d<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    // example using generic PyObject
    #[pyfn(m)]
    fn render_img<'py>(
        py: Python<'py>,
        volume: PyReadonlyArrayDyn<'py, f32>,
        cmap: PyReadonlyArrayDyn<'py, f32>,
        width: u32,
        height: u32,
        time: f32,
        background: (f32, f32, f32, f32),
        distance_scale: f32,
        vmin: Option<f32>,
        vmax: Option<f32>,
        spatial_interpolation: Option<String>,
        temporal_interpolation: Option<String>,
    ) -> Bound<'py, PyArray3<u8>> {
        let volume = Volume::from_array(volume.as_array());
        let cmap = ListedColorMap::from_array(cmap.as_array());
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = pollster::block_on(render_volume(
            vec![volume],
            cmap::GenericColorMap::Listed(cmap),
            Vector2::new(width, height),
            &vec![time],
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
                .unwrap_or(wgpu::FilterMode::Linear),
            temporal_interpolation
                .map(|s| parse_interpolation(&s).unwrap())
                .unwrap_or(wgpu::FilterMode::Linear),
        ))
        .unwrap()
        .pop()
        .unwrap();

        let shape = StrideShape::from((width as usize, height as usize, 4 as usize));
        let arr = numpy::ndarray::Array3::from_shape_vec(shape, img.into_vec()).unwrap();
        return arr.into_pyarray_bound(py);
    }

    #[pyfn(m)]
    fn render_video<'py>(
        py: Python<'py>,
        volume: PyReadonlyArrayDyn<'py, f32>,
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
    ) -> Bound<'py, PyArray4<u8>> {
        let volume = Volume::from_array(volume.as_array());
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
        ))
        .unwrap();

        let shape = StrideShape::from((time.len(), width as usize, height as usize, 4 as usize));
        let arr = numpy::ndarray::Array4::from_shape_vec(
            shape,
            img.iter().flat_map(|img| img.to_vec()).collect(),
        )
        .unwrap();
        return arr.into_pyarray_bound(py);
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
