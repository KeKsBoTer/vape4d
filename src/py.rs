use cgmath::Vector2;
use numpy::{ndarray::StrideShape, IntoPyArray, PyArray3, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::{
    cmap::{self, ListedColorMap},
    offline::render_volume,
    volume::Volume,
};

#[pymodule]
fn v4dv<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
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
    ) -> Bound<'py, PyArray3<u8>> {
        let volume = Volume::from_array(volume.as_array());
        let cmap = ListedColorMap::from_array(cmap.as_array());
        let img = pollster::block_on(render_volume(
            vec![volume],
            cmap::ColorMapType::Listed(cmap),
            Vector2::new(width, height),
            time,
            wgpu::Color {
                r: background.0 as f64,
                g: background.1 as f64,
                b: background.2 as f64,
                a: background.3 as f64,
            },
        ))
        .unwrap();

        let shape = StrideShape::from((width as usize, height as usize, 4 as usize));
        let arr = numpy::ndarray::Array3::from_shape_vec(shape, img.into_vec()).unwrap();
        return arr.into_pyarray_bound(py);
    }

    Ok(())
}
