use glam::Vec3;

pub struct SceneObject {
    pub mesh_idx: usize,
    pub position: Vec3,
    pub scale: f32,
    pub color: [f32; 3],
    pub emission: f32,
}

impl SceneObject {
    pub fn new(mesh_idx: usize, position: Vec3, scale: f32, color: [f32; 3]) -> Self {
        Self { mesh_idx, position, scale, color, emission: 0.0 }
    }

    pub fn with_emission(mut self, emission: f32) -> Self {
        self.emission = emission;
        self
    }
}

pub fn create_demo_scene() -> Vec<SceneObject> {
    vec![
        SceneObject::new(1, Vec3::new(2.5, 0.5, 0.0), 1.0, [0.1, 0.5, 0.95]).with_emission(0.8),
        SceneObject::new(1, Vec3::new(-2.5, 0.5, 0.0), 1.0, [0.2, 0.95, 0.3]).with_emission(0.6),
        SceneObject::new(0, Vec3::new(0.0, 0.35, 2.5), 0.7, [0.95, 0.6, 0.1]).with_emission(1.0),
        SceneObject::new(0, Vec3::new(0.0, 0.35, -2.5), 0.7, [0.8, 0.2, 0.95]).with_emission(0.9),
        SceneObject::new(1, Vec3::new(4.0, 0.3, 3.0), 0.6, [0.95, 0.95, 0.3]).with_emission(1.2),
        SceneObject::new(1, Vec3::new(-4.0, 0.3, -3.0), 0.6, [0.3, 0.95, 0.95]).with_emission(1.0),
    ]
}

