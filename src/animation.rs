use std::time::{Duration, Instant};

use cgmath::{InnerSpace, MetricSpace, One, Point3, Quaternion, Vector3};

pub trait Animation {
    type State;

    /// takes a value between 0 and 1 and returns the state at that time 
    fn sample(&self,dt:Instant)-> Self::State;

    // fn end(&self) -> Option<Instant>;
}


pub struct TurntableAnimation {
    pub start: Instant,
    pub duration: Duration, 
    pub center: Point3<f32>,
    pub zero_pos: Point3<f32>,
    pub radius: f32,
    pub up: Vector3<f32>,
}

impl TurntableAnimation{
    pub fn new(start:Point3<f32>,duration: Duration, center: Point3<f32>,up:Vector3<f32>) -> Self {
        let radius = start.distance(center);
        let start_time = Instant::now();
        TurntableAnimation {
            start:start_time,
            zero_pos: start,
            duration,
            center,
            radius,
            up,
        }
    }
}

impl Animation for TurntableAnimation {
    type State = (Point3<f32>, Quaternion<f32>);

    fn sample(&self, dt: Instant) -> Self::State {
        let v = (dt - self.start).as_secs_f32() / self.duration.as_secs_f32();
        let angle = v * std::f32::consts::PI * 2.0;
        let up = self.up;
        let front = (self.center-self.zero_pos).normalize();
        let right_vector = front.cross(up).normalize();

        let position_on_plane = front * angle.sin() * self.radius +
                            right_vector * angle.cos() * self.radius;

        // let rot:Quaternion<f32> = Matrix3::from_cols(right_vector, up , front).into();
        // let rot:Quaternion<f32> = Matrix3::look_to_lh(-front, self.up).invert().unwrap().into();
        let rot = Quaternion::one();
        return (
            self.center + position_on_plane,
            rot
        )
    }
    
    // fn end(&self) -> Option<Instant> {
    //     None
    // }
}