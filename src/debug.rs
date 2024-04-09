use std::collections::HashMap;

use bytemuck::Zeroable;
use cgmath::{
    BaseNum, EuclideanSpace, Matrix, Matrix4, Point3, Quaternion, SquareMatrix, Transform, Vector3,
    Vector4,
};
use num_traits::Float;
use wgpu::Color;

use crate::{
    camera,
    lines::Line,
    volume::{Aabb, Volume},
};

pub struct LineGroup {
    pub lines: Vec<Line>,
    pub visible: bool,
}

impl LineGroup {
    fn new(lines: impl Into<Vec<Line>>, visible: bool) -> Self {
        Self {
            lines: lines.into(),
            visible,
        }
    }

    fn aabb(&self) -> Aabb<f32> {
        let mut aabb: Aabb<f32> = Aabb::zeroed();
        for line in &self.lines {
            aabb.grow(&line.start);
            aabb.grow(&line.end);
        }
        aabb
    }
}

pub struct DebugLines {
    line_groups: HashMap<String, LineGroup>,
    aabb: Aabb<f32>,
}

impl DebugLines {
    pub fn new(pc: &Volume) -> Self {
        let mut lines = HashMap::new();

        let aabb = pc.aabb;
        let pc_size = aabb.size() * 0.5;
        let center = aabb.center();
        let t =
            Matrix4::from_translation(center.to_vec()) * Matrix4::from_diagonal(pc_size.extend(1.));
        let volume_aabb = box_lines(t, wgpu::Color::BLUE).to_vec();

        lines.insert(
            "volume_aabb".to_string(),
            LineGroup::new(volume_aabb.clone(), false),
        );

        let origin = axes_lines(Matrix4::identity()).to_vec();
        lines.insert("origin".to_string(), LineGroup::new(origin, false));

        let center_up = axes_lines(t).to_vec();
        lines.insert("center_up".to_string(), LineGroup::new(center_up, false));
        let clipping_box: Vec<Line> = volume_aabb
            .iter()
            .map(|l| {
                let mut l2 = l.clone();
                l2.color = Vector4::new(255, 255, 0, 255);
                l2
            })
            .collect();

        lines.insert(
            "clipping_box".to_string(),
            LineGroup::new(clipping_box, false),
        );

        let mut me = Self {
            line_groups: lines,
            aabb: Aabb::zeroed(),
        };
        me.update_aabb();
        return me;
    }

    pub fn all_lines(&mut self) -> Vec<(&String, &mut LineGroup)> {
        self.line_groups.iter_mut().collect()
    }

    fn update_aabb(&mut self) {
        let mut aabb = Aabb::zeroed();
        for lg in self.line_groups.values() {
            aabb.grow_union(&lg.aabb());
        }
        self.aabb = aabb;
    }

    pub fn any_visible(&self) -> bool {
        self.line_groups.values().any(|lg| lg.visible)
    }

    pub fn update_clipping_box(&mut self, clipping_box: &Aabb<f32>) {
        let t = Matrix4::from_translation(clipping_box.center().to_vec())
            * Matrix4::from_diagonal((clipping_box.size() * 0.5).extend(1.));
        self.update_lines(
            "clipping_box",
            box_lines(
                t,
                wgpu::Color {
                    r: 1.,
                    g: 1.,
                    b: 0.,
                    a: 1.,
                },
            )
            .to_vec(),
        );
    }

    pub fn visible_lines(&self) -> Vec<Line> {
        let mut lines = vec![];
        for set in self.line_groups.values() {
            if set.visible {
                lines.extend_from_slice(&set.lines);
            }
        }
        lines
    }

    fn update_lines(&mut self, name: &str, lines: Vec<Line>) {
        if let Some(lg) = self.line_groups.get_mut(name) {
            lg.lines = lines;
        } else {
            self.line_groups
                .insert(name.to_string(), LineGroup::new(lines, false));
            self.update_aabb();
        }
    }

    pub fn bbox(&self) -> &Aabb<f32> {
        &self.aabb
    }
}

fn box_lines(t: Matrix4<f32>, color: wgpu::Color) -> [Line; 12] {
    let p = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
        t.transform_point(Point3::new(
            (i & 1) as f32 * 2. - 1.,
            ((i >> 1) & 1) as f32 * 2. - 1.,
            ((i >> 2) & 1) as f32 * 2. - 1.,
        ))
    });
    [
        // front
        Line::new(p[0], p[1], color),
        Line::new(p[1], p[3], color),
        Line::new(p[2], p[0], color),
        Line::new(p[3], p[2], color),
        // back
        Line::new(p[4], p[5], color),
        Line::new(p[5], p[7], color),
        Line::new(p[6], p[4], color),
        Line::new(p[7], p[6], color),
        // sides
        Line::new(p[0], p[4], color),
        Line::new(p[1], p[5], color),
        Line::new(p[2], p[6], color),
        Line::new(p[3], p[7], color),
    ]
}

fn axes_lines(t: Matrix4<f32>) -> [Line; 3] {
    let c = t.transform_point(Point3::origin());
    let x = t.transform_point(Point3::new(1., 0., 0.));
    let y = t.transform_point(Point3::new(0., 1., 0.));
    let z = t.transform_point(Point3::new(0., 0., 1.));
    [
        Line::new(c, x, Color::RED),
        Line::new(c, y, Color::GREEN),
        Line::new(c, z, Color::BLUE),
    ]
}

fn blend(a: wgpu::Color, b: wgpu::Color, t: f64) -> wgpu::Color {
    wgpu::Color {
        r: a.r * (1. - t) + b.r * t,
        g: a.g * (1. - t) + b.g * t,
        b: a.b * (1. - t) + b.b * t,
        a: a.a * (1. - t) + b.a * t,
    }
}

fn three_plane_intersection(a: Vector4<f32>, b: Vector4<f32>, c: Vector4<f32>) -> Point3<f32> {
    let m = Matrix4::from_cols(a, b, c, Vector4::new(0., 0., 0., 1.)).transpose();

    let intersection = m.inverse_transform().unwrap() * Vector4::new(0., 0., 0., 1.);
    return Point3::from_homogeneous(intersection);
}
