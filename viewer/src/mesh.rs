use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout { 
            array_stride: std::mem::size_of::<Vertex>() as u64, 
            step_mode: wgpu::VertexStepMode::Vertex, 
            attributes: &Self::ATTRIBS 
        }
    }
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

pub fn create_cube(device: &wgpu::Device, size: f32) -> Mesh {
    let s = size * 0.5;
    let v = |p: [f32; 3], n: [f32; 3]| Vertex { position: p, normal: n, color: [1.0, 1.0, 1.0] };
    let vertices = vec![
        v([-s,-s,s],[0.0,0.0,1.0]),v([s,-s,s],[0.0,0.0,1.0]),v([s,s,s],[0.0,0.0,1.0]),v([-s,s,s],[0.0,0.0,1.0]),
        v([-s,-s,-s],[0.0,0.0,-1.0]),v([-s,s,-s],[0.0,0.0,-1.0]),v([s,s,-s],[0.0,0.0,-1.0]),v([s,-s,-s],[0.0,0.0,-1.0]),
        v([-s,s,-s],[0.0,1.0,0.0]),v([-s,s,s],[0.0,1.0,0.0]),v([s,s,s],[0.0,1.0,0.0]),v([s,s,-s],[0.0,1.0,0.0]),
        v([-s,-s,-s],[0.0,-1.0,0.0]),v([s,-s,-s],[0.0,-1.0,0.0]),v([s,-s,s],[0.0,-1.0,0.0]),v([-s,-s,s],[0.0,-1.0,0.0]),
        v([s,-s,-s],[1.0,0.0,0.0]),v([s,s,-s],[1.0,0.0,0.0]),v([s,s,s],[1.0,0.0,0.0]),v([s,-s,s],[1.0,0.0,0.0]),
        v([-s,-s,-s],[-1.0,0.0,0.0]),v([-s,-s,s],[-1.0,0.0,0.0]),v([-s,s,s],[-1.0,0.0,0.0]),v([-s,s,-s],[-1.0,0.0,0.0]),
    ];
    let indices: Vec<u32> = vec![0,1,2,2,3,0,4,5,6,6,7,4,8,9,10,10,11,8,12,13,14,14,15,12,16,17,18,18,19,16,20,21,22,22,23,20];
    Mesh {
        vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: None, 
            contents: bytemuck::cast_slice(&vertices), 
            usage: wgpu::BufferUsages::VERTEX 
        }),
        index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: None, 
            contents: bytemuck::cast_slice(&indices), 
            usage: wgpu::BufferUsages::INDEX 
        }),
        num_indices: 36,
    }
}

pub fn create_sphere(device: &wgpu::Device, radius: f32, seg: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for lat in 0..=seg {
        let theta = lat as f32 * std::f32::consts::PI / seg as f32;
        let (st, ct) = (theta.sin(), theta.cos());
        for lon in 0..=seg {
            let phi = lon as f32 * 2.0 * std::f32::consts::PI / seg as f32;
            let (sp, cp) = (phi.sin(), phi.cos());
            let (x, y, z) = (cp * st, ct, sp * st);
            vertices.push(Vertex { 
                position: [x * radius, y * radius, z * radius], 
                normal: [x, y, z], 
                color: [1.0, 1.0, 1.0] 
            });
        }
    }
    for lat in 0..seg {
        for lon in 0..seg {
            let f = lat * (seg + 1) + lon;
            let s = f + seg + 1;
            indices.extend_from_slice(&[f, s, f + 1, s, s + 1, f + 1]);
        }
    }
    Mesh {
        vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: None, 
            contents: bytemuck::cast_slice(&vertices), 
            usage: wgpu::BufferUsages::VERTEX 
        }),
        index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: None, 
            contents: bytemuck::cast_slice(&indices), 
            usage: wgpu::BufferUsages::INDEX 
        }),
        num_indices: indices.len() as u32,
    }
}

