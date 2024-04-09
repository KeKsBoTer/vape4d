use cgmath::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveCamera {
    pub position: Point3<f32>,
    pub rotation: Quaternion<f32>,
    pub projection: PerspectiveProjection,
}

impl PerspectiveCamera {
    pub fn new(
        position: Point3<f32>,
        rotation: Quaternion<f32>,
        projection: PerspectiveProjection,
    ) -> Self {
        PerspectiveCamera {
            position,
            rotation,
            projection: projection,
        }
    }
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            position: Point3::new(0., 0., -1.),
            rotation: Quaternion::new(1., 0., 0., 0.),
            projection: PerspectiveProjection {
                fovy: Deg(45.).into(),
                znear: 0.1,
                zfar: 100.,
                aspect_ratio: 1.0,
            },
        }
    }
}

impl Camera for PerspectiveCamera {
    fn view_matrix(&self) -> Matrix4<f32> {
        world2view(self.rotation, self.position)
    }

    fn proj_matrix(&self) -> Matrix4<f32> {
        self.projection.projection_matrix()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveProjection {
    pub fovy: Rad<f32>,
    pub znear: f32,
    pub zfar: f32,
    pub aspect_ratio: f32,
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
        Self {
            fovy: fovy.into(),
            znear,
            zfar,
            aspect_ratio: vr,
        }
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        build_proj(
            self.znear,
            self.zfar,
            self.fovy * self.aspect_ratio,
            self.fovy,
        )
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let ratio = width as f32 / height as f32;
        let fovx = self.fovy * ratio;
        if width > height {
            self.fovy = fovx / ratio;
        } else {
            self.fovy = self.fovy * ratio;
        }
    }
}

pub struct FrustumPlanes {
    pub near: Vector4<f32>,
    pub far: Vector4<f32>,
    pub left: Vector4<f32>,
    pub right: Vector4<f32>,
    pub top: Vector4<f32>,
    pub bottom: Vector4<f32>,
}

pub trait Camera {
    fn view_matrix(&self) -> Matrix4<f32>;
    fn proj_matrix(&self) -> Matrix4<f32>;

    fn position(&self) -> Point3<f32> {
        Point3::from_homogeneous(self.view_matrix().inverse_transform().unwrap().w)
    }

    fn frustum_planes(&self) -> FrustumPlanes {
        let p = self.proj_matrix();
        let v = self.view_matrix();
        let pv = p * v;
        let mut planes = [Vector4::zero(); 6];
        planes[0] = pv.row(3) + pv.row(0);
        planes[1] = pv.row(3) - pv.row(0);
        planes[2] = pv.row(3) + pv.row(1);
        planes[3] = pv.row(3) - pv.row(1);
        planes[4] = pv.row(3) + pv.row(2);
        planes[5] = pv.row(3) - pv.row(2);
        for i in 0..6 {
            planes[i] = planes[i].normalize();
        }
        return FrustumPlanes {
            near: planes[4],
            far: planes[5],
            left: planes[0],
            right: planes[1],
            top: planes[3],
            bottom: planes[2],
        };
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
