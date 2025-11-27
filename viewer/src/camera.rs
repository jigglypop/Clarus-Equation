use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw: 0.8,
            pitch: 0.45,
            distance: 12.0,
            target: Vec3::ZERO,
        }
    }

    pub fn update(&mut self) {
        self.position = self.target + Vec3::new(
            self.distance * self.pitch.cos() * self.yaw.sin(),
            self.distance * self.pitch.sin(),
            self.distance * self.pitch.cos() * self.yaw.cos(),
        );
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 300.0) 
            * Mat4::look_at_rh(self.position, self.target, Vec3::Y)
    }
}

