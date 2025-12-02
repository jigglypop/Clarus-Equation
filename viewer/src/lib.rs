mod shaders;
mod mesh;
mod camera;
mod uniforms;
mod scene;
mod gltf_loader;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::{window, HtmlCanvasElement, MouseEvent, WheelEvent};
use wgpu::util::DeviceExt;
use glam::{Mat4, Vec3};

use crate::shaders::*;
use crate::mesh::{Mesh, Vertex, create_cube, create_sphere, create_island};
use crate::camera::Camera;
use crate::uniforms::*;
use crate::scene::{SceneObject, create_demo_scene};
use crate::gltf_loader::{GltfModel, SkinnedVertex, SkinUniforms};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

const SHADOW_SIZE: u32 = 128;
const MAX_INSTANCES: usize = 128;
const GRASS_INSTANCES: u32 = 120000;
const SAND_INSTANCES: u32 = 80000;
const SNOW_INSTANCES: u32 = 5000000;
const FIRE_INSTANCES: u32 = 8000;

struct InputState { 
    mouse_down: bool, 
    last_x: f32, 
    last_y: f32 
}

struct Benchmark {
    frame_count: u32,
    total_frames: u64,
    last_time: f64,
    fps: f32,
    frame_time_ms: f32,
    avg_fps: f32,
    fps_samples: Vec<f32>,
}

impl Benchmark {
    fn new() -> Self {
        Self { 
            frame_count: 0, 
            total_frames: 0,
            last_time: 0.0, 
            fps: 0.0, 
            frame_time_ms: 0.0,
            avg_fps: 0.0,
            fps_samples: Vec::with_capacity(60),
        }
    }

    fn update(&mut self, now: f64) {
        self.frame_count += 1;
        self.total_frames += 1;
        let elapsed = now - self.last_time;
        if elapsed >= 1000.0 {
            self.fps = self.frame_count as f32 / (elapsed as f32 / 1000.0);
            self.frame_time_ms = elapsed as f32 / self.frame_count as f32;
            if self.fps_samples.len() >= 60 { self.fps_samples.remove(0); }
            self.fps_samples.push(self.fps);
            self.avg_fps = self.fps_samples.iter().sum::<f32>() / self.fps_samples.len() as f32;
            self.frame_count = 0;
            self.last_time = now;
        }
    }
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    
    shadow_pipeline: wgpu::RenderPipeline,
    skinned_shadow_pipeline: wgpu::RenderPipeline,
    sfe_compute_pipeline: wgpu::ComputePipeline,
    mesh_pipeline: wgpu::RenderPipeline,
    floor_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    grass_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    bloom_pipeline: wgpu::RenderPipeline,
    skinned_pipeline: wgpu::RenderPipeline,
    sand_pipeline: wgpu::RenderPipeline,
    flag_pipeline: wgpu::RenderPipeline,
    snow_pipeline: wgpu::RenderPipeline,
    fire_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    volumetric_pipeline: wgpu::RenderPipeline,
    atmosphere_pipeline: wgpu::RenderPipeline,
    
    uniform_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    sfe_params_buffer: wgpu::Buffer,
    bloom_params_buffer: wgpu::Buffer,
    skin_buffer: wgpu::Buffer,
    
    main_bind_group: wgpu::BindGroup,
    skinned_bind_group: wgpu::BindGroup,
    sfe_compute_bind_group: wgpu::BindGroup,
    sfe_shadow_bind_group: wgpu::BindGroup,
    bloom_bind_group: wgpu::BindGroup,
    
    shadow_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    render_view: wgpu::TextureView,
    
    meshes: Vec<Mesh>,
    objects: Vec<SceneObject>,
    gltf_model: Option<GltfModel>,
    camera: Camera,
    input: Rc<RefCell<InputState>>,
    start_time: f64,
    light_dir: Vec3,
    sfe_lambda: f32,
    sfe_tau: f32,
    benchmark: Benchmark,
}

impl GpuState {
    async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let (w, h) = (canvas.width(), canvas.height());
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { 
            backends: wgpu::Backends::BROWSER_WEBGPU, 
            ..Default::default() 
        });
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|_| "surface")?;
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { 
            power_preference: wgpu::PowerPreference::HighPerformance, 
            compatible_surface: Some(&surface), 
            force_fallback_adapter: false 
        }).await.ok_or("adapter")?;
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
            .await.map_err(|e| format!("{:?}", e))?;
        
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration { 
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, 
            format, width: w, height: h, 
            present_mode: wgpu::PresentMode::Fifo, 
            alpha_mode: caps.alpha_modes[0], 
            view_formats: vec![], 
            desired_maximum_frame_latency: 2 
        };
        surface.configure(&device, &config);

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor { 
            label: Some("shadow_raw"), 
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 }, 
            mip_level_count: 1, sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::R32Float, 
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, 
            view_formats: &[] 
        });
        let shadow_view = shadow_tex.create_view(&Default::default());

        let sfe_shadow_tex = device.create_texture(&wgpu::TextureDescriptor { 
            label: Some("sfe_shadow"), 
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 }, 
            mip_level_count: 1, sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::R32Float, 
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, 
            view_formats: &[] 
        });
        let sfe_shadow_view = sfe_shadow_tex.create_view(&Default::default());

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor { 
            label: None, 
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, 
            mip_level_count: 1, sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::Depth32Float, 
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, 
            view_formats: &[] 
        });
        let depth_view = depth_tex.create_view(&Default::default());

        let render_tex = device.create_texture(&wgpu::TextureDescriptor { 
            label: None, 
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, 
            mip_level_count: 1, sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format, 
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, 
            view_formats: &[] 
        });
        let render_view = render_tex.create_view(&Default::default());

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: None, 
            contents: &[0u8; std::mem::size_of::<GlobalUniforms>()], 
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST 
        });
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: None, 
            contents: &vec![0u8; std::mem::size_of::<InstanceData>() * MAX_INSTANCES], 
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST 
        });
        let sfe_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: None, 
            contents: bytemuck::cast_slice(&[SfeParams { lambda: 0.5, tau: 0.8, size: SHADOW_SIZE as f32, _pad: 0.0 }]), 
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST 
        });
        let bloom_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: None, 
            contents: bytemuck::cast_slice(&[BloomParams { 
                sfe_lambda: 0.5, sfe_tau: 0.4, 
                width: w as f32, height: h as f32,
                bloom_intensity: 1.8, bloom_threshold: 0.35,
                time: 0.0, light_dir_y: -0.7,
            }]), 
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST 
        });
        let skin_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skin_uniforms"),
            contents: bytemuck::cast_slice(&[SkinUniforms::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: None, 
            entries: &[
                wgpu::BindGroupLayoutEntry { 
                    binding: 0, 
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, 
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    }, 
                    count: None 
                },
                wgpu::BindGroupLayoutEntry { 
                    binding: 1, 
                    visibility: wgpu::ShaderStages::VERTEX, 
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    }, 
                    count: None 
                },
            ]
        });
        let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: None, 
            layout: &main_bgl, 
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: instance_buffer.as_entire_binding() },
            ]
        });

        let sfe_compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: None, 
            entries: &[
                wgpu::BindGroupLayoutEntry { 
                    binding: 0, 
                    visibility: wgpu::ShaderStages::COMPUTE, 
                    ty: wgpu::BindingType::Texture { 
                        sample_type: wgpu::TextureSampleType::Float { filterable: false }, 
                        view_dimension: wgpu::TextureViewDimension::D2, 
                        multisampled: false 
                    }, 
                    count: None 
                },
                wgpu::BindGroupLayoutEntry { 
                    binding: 1, 
                    visibility: wgpu::ShaderStages::COMPUTE, 
                    ty: wgpu::BindingType::StorageTexture { 
                        access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::R32Float, 
                        view_dimension: wgpu::TextureViewDimension::D2 
                    }, 
                    count: None 
                },
                wgpu::BindGroupLayoutEntry { 
                    binding: 2, 
                    visibility: wgpu::ShaderStages::COMPUTE, 
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    }, 
                    count: None 
                },
            ]
        });
        let sfe_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: None, 
            layout: &sfe_compute_bgl, 
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sfe_shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: sfe_params_buffer.as_entire_binding() },
            ]
        });

        let sfe_shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: None, 
            entries: &[
                wgpu::BindGroupLayoutEntry { 
                    binding: 0, 
                    visibility: wgpu::ShaderStages::FRAGMENT, 
                    ty: wgpu::BindingType::Texture { 
                        sample_type: wgpu::TextureSampleType::Float { filterable: false }, 
                        view_dimension: wgpu::TextureViewDimension::D2, 
                        multisampled: false 
                    }, 
                    count: None 
                },
            ]
        });
        let sfe_shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: None, 
            layout: &sfe_shadow_bgl, 
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&sfe_shadow_view) },
            ]
        });

        let bloom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: None, 
            entries: &[
                wgpu::BindGroupLayoutEntry { 
                    binding: 0, 
                    visibility: wgpu::ShaderStages::FRAGMENT, 
                    ty: wgpu::BindingType::Texture { 
                        sample_type: wgpu::TextureSampleType::Float { filterable: false }, 
                        view_dimension: wgpu::TextureViewDimension::D2, 
                        multisampled: false 
                    }, 
                    count: None 
                },
                wgpu::BindGroupLayoutEntry { 
                    binding: 1, 
                    visibility: wgpu::ShaderStages::FRAGMENT, 
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Uniform, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    }, 
                    count: None 
                },
            ]
        });
        let bloom_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: None, 
            layout: &bloom_bgl, 
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&render_view) },
                wgpu::BindGroupEntry { binding: 1, resource: bloom_params_buffer.as_entire_binding() },
            ]
        });

        let skinned_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let skinned_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &skinned_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: skin_buffer.as_entire_binding() },
            ],
        });

        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &[&main_bgl], 
            push_constant_ranges: &[] 
        });
        let sfe_compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &[&sfe_compute_bgl], 
            push_constant_ranges: &[] 
        });
        let mesh_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &[&main_bgl, &sfe_shadow_bgl], 
            push_constant_ranges: &[] 
        });
        let skinned_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&skinned_bgl, &sfe_shadow_bgl],
            push_constant_ranges: &[],
        });
        let skinned_shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned_shadow_pl"),
            bind_group_layouts: &[&skinned_bgl],
            push_constant_ranges: &[],
        });
        let bloom_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &[&bloom_bgl], 
            push_constant_ranges: &[] 
        });

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()) 
        });
        let sfe_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Wgsl(SFE_SHADOW_COMPUTE.into()) 
        });
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()) 
        });
        let floor_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Wgsl(FLOOR_SHADER.into()) 
        });
        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Wgsl(WATER_SHADER.into()) 
        });
        let grass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Wgsl(GRASS_SHADER.into()) 
        });
        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()) 
        });
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Wgsl(BLOOM_SHADER.into()) 
        });
        let skinned_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SKINNED_MESH_SHADER.into()),
        });
        let skinned_shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SKINNED_SHADOW_SHADER.into()),
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, 
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState { 
                module: &shadow_shader, 
                entry_point: Some("vs_shadow"), 
                buffers: &[Vertex::desc()], 
                compilation_options: Default::default() 
            },
            fragment: Some(wgpu::FragmentState { 
                module: &shadow_shader, 
                entry_point: Some("fs_shadow"), 
                targets: &[Some(wgpu::ColorTargetState { 
                    format: wgpu::TextureFormat::R32Float, 
                    blend: None, 
                    write_mask: wgpu::ColorWrites::ALL 
                })], 
                compilation_options: Default::default() 
            }),
            primitive: wgpu::PrimitiveState { 
                topology: wgpu::PrimitiveTopology::TriangleList, 
                cull_mode: Some(wgpu::Face::Back), 
                ..Default::default() 
            },
            depth_stencil: None, 
            multisample: Default::default(), 
            multiview: None, 
            cache: None,
        });

        let skinned_shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned_shadow"),
            layout: Some(&skinned_shadow_pl),
            vertex: wgpu::VertexState {
                module: &skinned_shadow_shader,
                entry_point: Some("vs_skinned_shadow"),
                buffers: &[SkinnedVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &skinned_shadow_shader,
                entry_point: Some("fs_skinned_shadow"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sfe_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, 
            layout: Some(&sfe_compute_pl), 
            module: &sfe_compute_shader, 
            entry_point: Some("sfe_blur"), 
            compilation_options: Default::default(), 
            cache: None,
        });

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, 
            layout: Some(&mesh_pl),
            vertex: wgpu::VertexState { 
                module: &mesh_shader, 
                entry_point: Some("vs_main"), 
                buffers: &[Vertex::desc()], 
                compilation_options: Default::default() 
            },
            fragment: Some(wgpu::FragmentState { 
                module: &mesh_shader, 
                entry_point: Some("fs_main"), 
                targets: &[Some(wgpu::ColorTargetState { 
                    format, 
                    blend: Some(wgpu::BlendState::REPLACE), 
                    write_mask: wgpu::ColorWrites::ALL 
                })], 
                compilation_options: Default::default() 
            }),
            primitive: wgpu::PrimitiveState { 
                topology: wgpu::PrimitiveTopology::TriangleList, 
                cull_mode: Some(wgpu::Face::Back), 
                ..Default::default() 
            },
            depth_stencil: Some(wgpu::DepthStencilState { 
                format: wgpu::TextureFormat::Depth32Float, 
                depth_write_enabled: true, 
                depth_compare: wgpu::CompareFunction::Less, 
                stencil: Default::default(), 
                bias: Default::default() 
            }),
            multisample: Default::default(), 
            multiview: None, 
            cache: None,
        });

        let floor_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, 
            layout: Some(&mesh_pl),
            vertex: wgpu::VertexState { 
                module: &floor_shader, 
                entry_point: Some("vs_floor"), 
                buffers: &[], 
                compilation_options: Default::default() 
            },
            fragment: Some(wgpu::FragmentState { 
                module: &floor_shader, 
                entry_point: Some("fs_floor"), 
                targets: &[Some(wgpu::ColorTargetState { 
                    format, 
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), 
                    write_mask: wgpu::ColorWrites::ALL 
                })], 
                compilation_options: Default::default() 
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { 
                format: wgpu::TextureFormat::Depth32Float, 
                depth_write_enabled: false, 
                depth_compare: wgpu::CompareFunction::Less, 
                stencil: Default::default(), 
                bias: Default::default() 
            }),
            multisample: Default::default(), 
            multiview: None, 
            cache: None,
        });

        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, 
            layout: Some(&mesh_pl),
            vertex: wgpu::VertexState { 
                module: &water_shader, 
                entry_point: Some("vs_water"), 
                buffers: &[], 
                compilation_options: Default::default() 
            },
            fragment: Some(wgpu::FragmentState { 
                module: &water_shader, 
                entry_point: Some("fs_water"), 
                targets: &[Some(wgpu::ColorTargetState { 
                    format, 
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), 
                    write_mask: wgpu::ColorWrites::ALL 
                })], 
                compilation_options: Default::default() 
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { 
                format: wgpu::TextureFormat::Depth32Float, 
                depth_write_enabled: false, 
                depth_compare: wgpu::CompareFunction::Less, 
                stencil: Default::default(), 
                bias: Default::default() 
            }),
            multisample: Default::default(), 
            multiview: None, 
            cache: None,
        });

        let grass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, 
            layout: Some(&mesh_pl),
            vertex: wgpu::VertexState { 
                module: &grass_shader, 
                entry_point: Some("vs_grass"), 
                buffers: &[], 
                compilation_options: Default::default() 
            },
            fragment: Some(wgpu::FragmentState { 
                module: &grass_shader, 
                entry_point: Some("fs_grass"), 
                targets: &[Some(wgpu::ColorTargetState { 
                    format, 
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), 
                    write_mask: wgpu::ColorWrites::ALL 
                })], 
                compilation_options: Default::default() 
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { 
                format: wgpu::TextureFormat::Depth32Float, 
                depth_write_enabled: false, 
                depth_compare: wgpu::CompareFunction::Less, 
                stencil: Default::default(), 
                bias: Default::default() 
            }),
            multisample: Default::default(), 
            multiview: None, 
            cache: None,
        });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, 
            layout: Some(&mesh_pl),
            vertex: wgpu::VertexState { 
                module: &particle_shader, 
                entry_point: Some("vs_particle"), 
                buffers: &[], 
                compilation_options: Default::default() 
            },
            fragment: Some(wgpu::FragmentState { 
                module: &particle_shader, 
                entry_point: Some("fs_particle"), 
                targets: &[Some(wgpu::ColorTargetState { 
                    format, 
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), 
                    write_mask: wgpu::ColorWrites::ALL 
                })], 
                compilation_options: Default::default() 
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { 
                format: wgpu::TextureFormat::Depth32Float, 
                depth_write_enabled: false, 
                depth_compare: wgpu::CompareFunction::Less, 
                stencil: Default::default(), 
                bias: Default::default() 
            }),
            multisample: Default::default(), 
            multiview: None, 
            cache: None,
        });

        let bloom_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, 
            layout: Some(&bloom_pl),
            vertex: wgpu::VertexState { 
                module: &bloom_shader, 
                entry_point: Some("vs_bloom"), 
                buffers: &[], 
                compilation_options: Default::default() 
            },
            fragment: Some(wgpu::FragmentState { 
                module: &bloom_shader, 
                entry_point: Some("fs_bloom"), 
                targets: &[Some(wgpu::ColorTargetState { 
                    format, 
                    blend: Some(wgpu::BlendState::REPLACE), 
                    write_mask: wgpu::ColorWrites::ALL 
                })], 
                compilation_options: Default::default() 
            }),
            primitive: Default::default(), 
            depth_stencil: None, 
            multisample: Default::default(), 
            multiview: None, 
            cache: None,
        });

        let skinned_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned"),
            layout: Some(&skinned_pl),
            vertex: wgpu::VertexState {
                module: &skinned_shader,
                entry_point: Some("vs_skinned"),
                buffers: &[SkinnedVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &skinned_shader,
                entry_point: Some("fs_skinned"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sand_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SAND_SHADER.into()),
        });
        let sand_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sand"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &sand_shader,
                entry_point: Some("vs_sand"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sand_shader,
                entry_point: Some("fs_sand"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let flag_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(FLAG_SHADER.into()),
        });
        let flag_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flag"),
            layout: Some(&mesh_pl),
            vertex: wgpu::VertexState {
                module: &flag_shader,
                entry_point: Some("vs_flag"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &flag_shader,
                entry_point: Some("fs_flag"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let snow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SNOW_SHADER.into()),
        });
        let snow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("snow"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &snow_shader,
                entry_point: Some("vs_snow"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &snow_shader,
                entry_point: Some("fs_snow"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let fire_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(FIRE_SHADER.into()),
        });
        let fire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fire"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &fire_shader,
                entry_point: Some("vs_fire"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fire_shader,
                entry_point: Some("fs_fire"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            ],
        });
        let ssao_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_pl"),
            bind_group_layouts: &[&ssao_bgl],
            push_constant_ranges: &[],
        });
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SSAO_SHADER.into()),
        });
        let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao"),
            layout: Some(&ssao_pl),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: Some("vs_ssao"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: Some("fs_ssao"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let volumetric_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(VOLUMETRIC_SHADER.into()),
        });
        let volumetric_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric"),
            layout: Some(&mesh_pl),
            vertex: wgpu::VertexState {
                module: &volumetric_shader,
                entry_point: Some("vs_volumetric"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &volumetric_shader,
                entry_point: Some("fs_volumetric"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let atmosphere_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(ATMOSPHERIC_SHADER.into()),
        });
        let atmosphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("atmosphere"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &atmosphere_shader,
                entry_point: Some("vs_atmosphere"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &atmosphere_shader,
                entry_point: Some("fs_atmosphere"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let meshes = vec![create_cube(&device, 1.0), create_sphere(&device, 0.5, 20), create_island(&device, 1.0, 0.1, 32)];
        let objects = create_demo_scene();

        Ok(Self {
            surface, device, queue, config,
            shadow_pipeline, skinned_shadow_pipeline, sfe_compute_pipeline, mesh_pipeline, floor_pipeline, 
            water_pipeline, grass_pipeline, particle_pipeline, bloom_pipeline, skinned_pipeline, sand_pipeline, flag_pipeline, snow_pipeline, fire_pipeline,
            ssao_pipeline, volumetric_pipeline, atmosphere_pipeline,
            uniform_buffer, instance_buffer, sfe_params_buffer, bloom_params_buffer, skin_buffer,
            main_bind_group, skinned_bind_group, sfe_compute_bind_group, sfe_shadow_bind_group, bloom_bind_group,
            shadow_view, depth_view, render_view, meshes, objects,
            gltf_model: None,
            camera: Camera::new(), 
            input: Rc::new(RefCell::new(InputState { mouse_down: false, last_x: 0.0, last_y: 0.0 })),
            start_time: window().and_then(|w| w.performance()).map(|p| p.now()).unwrap_or(0.0),
            light_dir: Vec3::new(1.0, -1.5, 0.8).normalize(),
            sfe_lambda: 0.5,
            sfe_tau: 0.8,
            benchmark: Benchmark::new(),
        })
    }

    pub fn load_gltf(&mut self, data: &[u8]) -> Result<(), String> {
        let model = GltfModel::load_embedded(&self.device, &self.queue, data)?;
        log(&format!("[glTF] Loaded: {} meshes, {} animations", 
            model.meshes.len(),
            model.animations.len()));
        if let Some(ref skel) = model.skeleton {
            log(&format!("[glTF] Skeleton: {} joints", skel.joints.len()));
        }
        self.gltf_model = Some(model);
        Ok(())
    }

    fn render(&mut self) -> Result<(), JsValue> {
        let now = window().and_then(|w| w.performance()).map(|p| p.now()).unwrap_or(0.0);
        let t = ((now - self.start_time) / 1000.0) as f32;
        
        self.benchmark.update(now);
        if self.benchmark.frame_count == 1 && self.benchmark.total_frames % 5 == 0 {
            let total = GRASS_INSTANCES + SAND_INSTANCES + SNOW_INSTANCES + FIRE_INSTANCES;
            log(&format!("[SFE] FPS:{:.0} avg:{:.0} | {:.1}ms | {}M particles", 
                self.benchmark.fps, self.benchmark.avg_fps, self.benchmark.frame_time_ms, total / 1_000_000));
        }
        
        self.camera.update();
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view_proj = self.camera.view_proj(aspect);
        let light_pos = -self.light_dir * 18.0;
        let light_view_proj = Mat4::orthographic_rh(-12.0, 12.0, -12.0, 12.0, 0.1, 50.0) 
            * Mat4::look_at_rh(light_pos, Vec3::ZERO, Vec3::Y);

        let uniforms = GlobalUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            light_view_proj: light_view_proj.to_cols_array_2d(),
            light_dir: [self.light_dir.x, self.light_dir.y, self.light_dir.z, 0.0],
            camera_pos: [self.camera.position.x, self.camera.position.y, self.camera.position.z, 1.0],
            fog_color: self.get_dawn_sky_color(t),
            time: t, 
            fog_start: 20.0, 
            fog_end: 60.0,
            sfe_lambda: self.sfe_lambda, 
            sfe_tau: self.sfe_tau, 
            shadow_size: SHADOW_SIZE as f32, 
            _pad: [0.0; 2],
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let sfe_params = SfeParams { 
            lambda: self.sfe_lambda, 
            tau: self.sfe_tau, 
            size: SHADOW_SIZE as f32, 
            _pad: 0.0 
        };
        self.queue.write_buffer(&self.sfe_params_buffer, 0, bytemuck::cast_slice(&[sfe_params]));

        let bloom_params = BloomParams {
            sfe_lambda: self.sfe_lambda,
            sfe_tau: self.sfe_tau,
            width: self.config.width as f32,
            height: self.config.height as f32,
            bloom_intensity: 1.8,
            bloom_threshold: 0.35,
            time: t,
            light_dir_y: self.light_dir.y,
        };
        self.queue.write_buffer(&self.bloom_params_buffer, 0, bytemuck::cast_slice(&[bloom_params]));

        if let Some(ref mut model) = self.gltf_model {
            model.update(t, self.sfe_lambda);
            if let Some(matrices) = model.get_joint_matrices() {
                let mut skin_uniforms = SkinUniforms::new();
                skin_uniforms.update(matrices);
                self.queue.write_buffer(&self.skin_buffer, 0, bytemuck::cast_slice(&[skin_uniforms]));
            }
        }

        let mut instances: Vec<InstanceData> = Vec::with_capacity(self.objects.len());
        for (i, obj) in self.objects.iter().enumerate() {
            let rot = t * 0.4 + i as f32 * 0.5;
            let model = Mat4::from_translation(obj.position) 
                * Mat4::from_rotation_y(rot) 
                * Mat4::from_scale(Vec3::splat(obj.scale));
            instances.push(InstanceData { 
                model: model.to_cols_array_2d(), 
                color: [obj.color[0], obj.color[1], obj.color[2], obj.emission] 
            });
        }
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));

        let frame = self.surface.get_current_texture().map_err(|_| "frame")?;
        let frame_view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                    view: &self.shadow_view, 
                    resolve_target: None, 
                    ops: wgpu::Operations { 
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }), 
                        store: wgpu::StoreOp::Store 
                    } 
                })],
                depth_stencil_attachment: None, 
                ..Default::default()
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            for (i, obj) in self.objects.iter().enumerate() {
                let mesh = &self.meshes[obj.mesh_idx];
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.num_indices, 0, i as u32..i as u32 + 1);
            }
            
            if let Some(ref model) = self.gltf_model {
                pass.set_pipeline(&self.skinned_shadow_pipeline);
                pass.set_bind_group(0, &self.skinned_bind_group, &[]);
                for mesh in &model.meshes {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
                label: Some("sfe_blur"), 
                timestamp_writes: None 
            });
            pass.set_pipeline(&self.sfe_compute_pipeline);
            pass.set_bind_group(0, &self.sfe_compute_bind_group, &[]);
            pass.dispatch_workgroups((SHADOW_SIZE + 7) / 8, (SHADOW_SIZE + 7) / 8, 1);
        }

        let sky = self.get_dawn_sky_color(t);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                    view: &self.render_view, 
                    resolve_target: None, 
                    ops: wgpu::Operations { 
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: sky[0] as f64, g: sky[1] as f64, b: sky[2] as f64, a: 1.0 }), 
                        store: wgpu::StoreOp::Store 
                    } 
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { 
                    view: &self.depth_view, 
                    depth_ops: Some(wgpu::Operations { 
                        load: wgpu::LoadOp::Clear(1.0), 
                        store: wgpu::StoreOp::Store 
                    }), 
                    stencil_ops: None 
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.floor_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.sfe_shadow_bind_group, &[]);
            pass.draw(0..6, 0..1);

            pass.set_pipeline(&self.water_pipeline);
            pass.draw(0..6, 0..1);

            pass.set_pipeline(&self.grass_pipeline);
            pass.draw(0..15, 0..GRASS_INSTANCES);

            pass.set_pipeline(&self.mesh_pipeline);
            for (i, obj) in self.objects.iter().enumerate() {
                let mesh = &self.meshes[obj.mesh_idx];
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.num_indices, 0, i as u32..i as u32 + 1);
            }

            pass.set_pipeline(&self.sand_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.draw(0..6, 0..SAND_INSTANCES);

            pass.set_pipeline(&self.flag_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.sfe_shadow_bind_group, &[]);
            pass.draw(0..864, 0..1);

            pass.set_pipeline(&self.snow_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.draw(0..1, 0..SNOW_INSTANCES);

            pass.set_pipeline(&self.fire_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.draw(0..6, 0..FIRE_INSTANCES);

            if let Some(ref model) = self.gltf_model {
                pass.set_pipeline(&self.skinned_pipeline);
                pass.set_bind_group(0, &self.skinned_bind_group, &[]);
                pass.set_bind_group(1, &self.sfe_shadow_bind_group, &[]);
                for mesh in &model.meshes {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                    view: &frame_view, 
                    resolve_target: None, 
                    ops: wgpu::Operations { 
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), 
                        store: wgpu::StoreOp::Store 
                    } 
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_pipeline);
            pass.set_bind_group(0, &self.bloom_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn on_mouse_down(&mut self, x: f32, y: f32) { 
        let mut i = self.input.borrow_mut(); 
        i.mouse_down = true; 
        i.last_x = x; 
        i.last_y = y; 
    }
    
    fn on_mouse_up(&mut self) { 
        self.input.borrow_mut().mouse_down = false; 
    }
    
    fn on_mouse_move(&mut self, x: f32, y: f32) {
        let mut i = self.input.borrow_mut();
        if i.mouse_down { 
            self.camera.yaw -= (x - i.last_x) * 0.005; 
            self.camera.pitch = (self.camera.pitch + (y - i.last_y) * 0.005).clamp(-1.5, 1.5); 
        }
        i.last_x = x; 
        i.last_y = y;
    }
    
    fn on_wheel(&mut self, d: f32) { 
        self.camera.distance = (self.camera.distance + d * 0.01).clamp(3.0, 50.0); 
    }

    fn get_dawn_sky_color(&self, t: f32) -> [f32; 4] {
        let cycle = (t * 0.05).sin() * 0.5 + 0.5;
        
        let horizon_gold = [1.0, 0.65, 0.25];
        let sky_rose = [0.95, 0.45, 0.55];
        let sky_blue = [0.3, 0.5, 0.75];
        let deep_purple = [0.15, 0.1, 0.3];
        
        let dawn_factor = (cycle * 3.14159).sin();
        
        let r = horizon_gold[0] * dawn_factor * 0.4 
              + sky_rose[0] * (1.0 - dawn_factor) * 0.3 
              + sky_blue[0] * 0.2 
              + deep_purple[0] * 0.1;
        let g = horizon_gold[1] * dawn_factor * 0.4 
              + sky_rose[1] * (1.0 - dawn_factor) * 0.3 
              + sky_blue[1] * 0.2 
              + deep_purple[1] * 0.1;
        let b = horizon_gold[2] * dawn_factor * 0.4 
              + sky_rose[2] * (1.0 - dawn_factor) * 0.3 
              + sky_blue[2] * 0.2 
              + deep_purple[2] * 0.1;
        
        [r, g, b, 1.0]
    }
}

fn start_loop(state: Rc<RefCell<Option<GpuState>>>) {
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let s = state.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        if let Some(ref mut gpu) = *s.borrow_mut() { let _ = gpu.render(); }
        if let Some(w) = window() { 
            if let Some(cb) = f.borrow().as_ref() { 
                let _ = w.request_animation_frame(cb.as_ref().unchecked_ref()); 
            } 
        }
    }) as Box<dyn FnMut()>));
    if let Some(w) = window() { 
        if let Some(cb) = g.borrow().as_ref() { 
            let _ = w.request_animation_frame(cb.as_ref().unchecked_ref()); 
        } 
    }
    std::mem::forget(g);
}

thread_local! {
    static GLOBAL_STATE: RefCell<Option<Rc<RefCell<Option<GpuState>>>>> = RefCell::new(None);
}

fn setup_input(canvas: &HtmlCanvasElement, state: Rc<RefCell<Option<GpuState>>>) {
    let s = state.clone(); 
    let cb = Closure::wrap(Box::new(move |e: MouseEvent| { 
        if let Some(ref mut g) = *s.borrow_mut() { 
            g.on_mouse_down(e.client_x() as f32, e.client_y() as f32); 
        } 
    }) as Box<dyn FnMut(_)>); 
    canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref()).ok(); 
    cb.forget();
    
    let s = state.clone(); 
    let cb = Closure::wrap(Box::new(move |_: MouseEvent| { 
        if let Some(ref mut g) = *s.borrow_mut() { 
            g.on_mouse_up(); 
        } 
    }) as Box<dyn FnMut(_)>); 
    canvas.add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref()).ok(); 
    cb.forget();
    
    let s = state.clone(); 
    let cb = Closure::wrap(Box::new(move |e: MouseEvent| { 
        if let Some(ref mut g) = *s.borrow_mut() { 
            g.on_mouse_move(e.client_x() as f32, e.client_y() as f32); 
        } 
    }) as Box<dyn FnMut(_)>); 
    canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref()).ok(); 
    cb.forget();
    
    let s = state.clone(); 
    let cb = Closure::wrap(Box::new(move |e: WheelEvent| { 
        if let Some(ref mut g) = *s.borrow_mut() { 
            g.on_wheel(e.delta_y() as f32); 
        } 
    }) as Box<dyn FnMut(_)>); 
    canvas.add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref()).ok(); 
    cb.forget();
}

#[wasm_bindgen]
pub fn run_viewer(canvas_id: String) {
    log("=== SFE Engine v0.3 ===");
    log("[SFE] Core: exp(-(k^2 + lambda*k^4)*tau)");
    log(&format!("[SFE] Shadow: {}x{} + SFE compute blur", SHADOW_SIZE, SHADOW_SIZE));
    log(&format!("[SFE] Grass: {} | Sand: {} | Snow: {} | Fire: {}", GRASS_INSTANCES, SAND_INSTANCES, SNOW_INSTANCES, FIRE_INSTANCES));
    log("[SFE] Effects: SSAO, Volumetric, Atmosphere, Bloom, Water, Fog");
    
    let state: Rc<RefCell<Option<GpuState>>> = Rc::new(RefCell::new(None));
    let sc = state.clone();
    spawn_local(async move {
        let w = match window() { Some(w) => w, None => return };
        let doc = match w.document() { Some(d) => d, None => return };
        let elem = match doc.get_element_by_id(&canvas_id) { Some(e) => e, None => return };
        let canvas: HtmlCanvasElement = match elem.dyn_into() { Ok(c) => c, Err(_) => return };
        match GpuState::new(canvas.clone()).await {
            Ok(gpu) => { 
                sc.borrow_mut().replace(gpu); 
                setup_input(&canvas, sc.clone()); 
                GLOBAL_STATE.with(|g| {
                    *g.borrow_mut() = Some(sc.clone());
                });
                log("[SFE Engine] Ready."); 
                start_loop(sc); 
            }
            Err(e) => log(&format!("[SFE Engine] Failed: {:?}", e)),
        }
    });
}

#[wasm_bindgen]
pub fn load_gltf(data: &[u8]) {
    log(&format!("[glTF] Loading {} bytes...", data.len()));
    GLOBAL_STATE.with(|g| {
        if let Some(ref state) = *g.borrow() {
            if let Some(ref mut gpu) = *state.borrow_mut() {
                match gpu.load_gltf(data) {
                    Ok(_) => log("[glTF] Model loaded successfully!"),
                    Err(e) => log(&format!("[glTF] Error: {}", e)),
                }
            }
        }
    });
}
