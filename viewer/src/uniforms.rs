#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
    pub camera_pos: [f32; 4],
    pub fog_color: [f32; 4],
    pub time: f32,
    pub fog_start: f32,
    pub fog_end: f32,
    pub sfe_lambda: f32,
    pub sfe_tau: f32,
    pub shadow_size: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SfeParams {
    pub lambda: f32,
    pub tau: f32,
    pub size: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BloomParams {
    pub sfe_lambda: f32,
    pub sfe_tau: f32,
    pub width: f32,
    pub height: f32,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub _pad: [f32; 2],
}

