use cgmath::*;

use crate::volume::Aabb;

#[derive(Debug, Clone, Copy, PartialEq,egui_probe::EguiProbe)]
pub struct Camera {
    pub position: Point3<f32>,
    pub rotation: Quaternion<f32>,
    pub projection: Projection,
}



impl Camera {
    pub fn new(position: Point3<f32>, rotation: Quaternion<f32>, projection: Projection) -> Self {
        Camera {
            position,
            rotation,
            projection: projection,
        }
    }

    pub fn new_aabb_iso(aabb: Aabb<f32>, projection: Projection) -> Self {
        let r = aabb.radius();
        let corner = vec3(1., -1., 1.);
        let view_dir = Quaternion::look_at(-corner, Vector3::unit_y());
        Camera::new(
            aabb.center() + corner.normalize() * r * 2.8,
            view_dir,
            projection,
        )
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        world2view(self.rotation, self.position)
    }
}


#[derive(Debug, Clone, Copy, PartialEq,egui_probe::EguiProbe)]
pub enum Projection {
    Perspective(PerspectiveProjection),
    Orthographic(OrthographicProjection),
}

impl Projection {
    pub fn resize(&mut self, width: u32, height: u32) {
        match self {
            Projection::Perspective(p) => p.resize(width, height),
            Projection::Orthographic(p) => p.resize(width, height),
        }
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        match self {
            Projection::Perspective(p) => p.projection_matrix(),
            Projection::Orthographic(p) => p.projection_matrix(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq,egui_probe::EguiProbe)]
pub struct PerspectiveProjection {
    pub fovy: Rad<f32>,
    pub fovx: Rad<f32>,
    pub znear: f32,
    pub zfar: f32,
    pub aspect_ratio: f32,
}

impl Default for PerspectiveProjection {
    fn default() -> Self {
        Self {
            fovy: Deg(45.).into(),
            fovx: Deg(45.).into(),
            znear: 0.1,
            zfar: 100.,
            aspect_ratio: 1.0,
        }
    }
}

#[rustfmt::skip]
pub const VIEWPORT_Y_FLIP: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.,
    0.0, 0.0, 0., 1.0,
);

impl PerspectiveProjection {
    pub fn new<F: Into<Rad<f32>>>(viewport: Vector2<u32>, fovy: F, znear: f32, zfar: f32) -> Self {
        let vr = viewport.x as f32 / viewport.y as f32;
        let fovyr = fovy.into();
        Self {
            fovy: fovyr,
            fovx: fovyr * vr,
            znear,
            zfar,
            aspect_ratio: vr,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let ratio = width as f32 / height as f32;
        if width > height {
            self.fovy = self.fovx / ratio;
        } else {
            self.fovx = self.fovy * ratio;
        }
        self.aspect_ratio = ratio;
    }

    fn projection_matrix(&self) -> Matrix4<f32> {
        build_proj(self.znear, self.zfar, self.fovx, self.fovy)
    }
}


pub fn world2view(r: impl Into<Matrix3<f32>>, t: Point3<f32>) -> Matrix4<f32> {
    let mut rt = Matrix4::from(r.into());
    rt[0].w = t.x;
    rt[1].w = t.y;
    rt[2].w = t.z;
    rt[3].w = 1.;
    return rt.inverse_transform().unwrap().transpose();
}

pub fn build_proj(znear: f32, zfar: f32, fov_x: Rad<f32>, fov_y: Rad<f32>) -> Matrix4<f32> {
    let tan_half_fov_y = (fov_y / 2.).tan();
    let tan_half_fov_x = (fov_x / 2.).tan();

    let top = tan_half_fov_y * znear;
    let bottom = -top;
    let right = tan_half_fov_x * znear;
    let left = -right;

    let mut p = Matrix4::zero();
    p[0][0] = 2.0 * znear / (right - left);
    p[1][1] = 2.0 * znear / (top - bottom);
    p[0][2] = (right + left) / (right - left);
    p[1][2] = (top + bottom) / (top - bottom);
    p[3][2] = 1.;
    p[2][2] = zfar / (zfar - znear);
    p[2][3] = -(zfar * znear) / (zfar - znear);
    return p.transpose();
}

#[derive(Debug, Clone, Copy, PartialEq,egui_probe::EguiProbe)]
pub struct OrthographicProjection {
    pub znear: f32,
    pub zfar: f32,
    pub viewport: Vector2<f32>,
}

impl Default for OrthographicProjection {
    fn default() -> Self {
        Self {
            znear: 0.1,
            zfar: 100.,
            viewport: Vector2::new(1., 1.),
        }
    }
}

impl OrthographicProjection {
    #[allow(unused)]
    pub fn new(viewport: Vector2<f32>, znear: f32, zfar: f32) -> Self {
        Self {
            viewport,
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let ratio = width as f32 / height as f32;
        if width > height {
            self.viewport.x = self.viewport.y * ratio;
        } else {
            self.viewport.y = self.viewport.x / ratio;
        }
    }

    fn projection_matrix(&self) -> Matrix4<f32> {
        let width = self.viewport.x;
        let right = width / 2.;
        let top = self.viewport.y / 2.;

        let mut p = Matrix4::zero();
        p[0][0] = 1. / right;
        p[1][1] = 1. / top;
        p[2][2] = 1.0 / (self.zfar - self.znear);
        p[2][3] = -self.znear / (self.zfar - self.znear);
        p[3][3] = 1.0;
        return p.transpose();
    }
}
