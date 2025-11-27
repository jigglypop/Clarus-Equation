use wgpu::util::DeviceExt;
use glam::{Mat4, Vec3, Quat};
use std::collections::HashMap;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

impl SkinnedVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x2,
            3 => Uint32x4,
            4 => Float32x4
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SkinnedVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBS,
        }
    }
}

pub struct GltfMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub has_skin: bool,
    pub base_color: [f32; 4],
    pub texture: Option<wgpu::Texture>,
    pub texture_view: Option<wgpu::TextureView>,
}

pub struct Joint {
    pub name: String,
    pub local_transform: Mat4,
    pub inverse_bind: Mat4,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
}

pub struct Skeleton {
    pub joints: Vec<Joint>,
    pub joint_matrices: Vec<Mat4>,
}

impl Skeleton {
    pub fn new() -> Self {
        Self {
            joints: Vec::new(),
            joint_matrices: Vec::new(),
        }
    }

    pub fn update_matrices(&mut self, local_transforms: &[Mat4]) {
        let n = self.joints.len();
        self.joint_matrices.resize(n, Mat4::IDENTITY);
        
        let mut world_matrices = vec![Mat4::IDENTITY; n];
        
        fn compute_world(
            idx: usize,
            joints: &[Joint],
            local_transforms: &[Mat4],
            world_matrices: &mut [Mat4],
            computed: &mut [bool],
        ) {
            if computed[idx] {
                return;
            }
            
            let local = if idx < local_transforms.len() {
                local_transforms[idx]
            } else {
                joints[idx].local_transform
            };
            
            if let Some(parent) = joints[idx].parent {
                if !computed[parent] {
                    compute_world(parent, joints, local_transforms, world_matrices, computed);
                }
                world_matrices[idx] = world_matrices[parent] * local;
            } else {
                world_matrices[idx] = local;
            }
            computed[idx] = true;
        }
        
        let mut computed = vec![false; n];
        for i in 0..n {
            compute_world(i, &self.joints, local_transforms, &mut world_matrices, &mut computed);
        }
        
        for i in 0..n {
            self.joint_matrices[i] = world_matrices[i] * self.joints[i].inverse_bind;
        }
    }
}

#[derive(Clone)]
pub struct AnimationChannel {
    pub joint_index: usize,
    pub times: Vec<f32>,
    pub translations: Option<Vec<Vec3>>,
    pub rotations: Option<Vec<Quat>>,
    pub scales: Option<Vec<Vec3>>,
}

pub struct Animation {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

impl Animation {
    pub fn sample(&self, time: f32, skeleton: &Skeleton, sfe_lambda: f32) -> Vec<Mat4> {
        let t = if self.duration > 0.0 { time % self.duration } else { 0.0 };
        let mut transforms: Vec<Mat4> = skeleton.joints.iter()
            .map(|j| j.local_transform)
            .collect();
        
        for channel in &self.channels {
            if channel.joint_index >= transforms.len() || channel.times.is_empty() {
                continue;
            }
            
            let (idx, frac) = self.find_keyframe(&channel.times, t);
            let smooth_frac = sfe_smooth(frac, sfe_lambda);
            
            let base = skeleton.joints[channel.joint_index].local_transform;
            let (base_scale, base_rot, base_trans) = base.to_scale_rotation_translation();
            
            let translation = if let Some(ref trans) = channel.translations {
                if trans.is_empty() {
                    base_trans
                } else if idx + 1 < trans.len() {
                    trans[idx].lerp(trans[idx + 1], smooth_frac)
                } else {
                    trans[idx.min(trans.len() - 1)]
                }
            } else {
                base_trans
            };
            
            let rotation = if let Some(ref rots) = channel.rotations {
                if rots.is_empty() {
                    base_rot
                } else if idx + 1 < rots.len() {
                    rots[idx].slerp(rots[idx + 1], smooth_frac)
                } else {
                    rots[idx.min(rots.len() - 1)]
                }
            } else {
                base_rot
            };
            
            let scale = if let Some(ref scls) = channel.scales {
                if scls.is_empty() {
                    base_scale
                } else if idx + 1 < scls.len() {
                    scls[idx].lerp(scls[idx + 1], smooth_frac)
                } else {
                    scls[idx.min(scls.len() - 1)]
                }
            } else {
                base_scale
            };
            
            transforms[channel.joint_index] = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        }
        
        transforms
    }
    
    fn find_keyframe(&self, times: &[f32], t: f32) -> (usize, f32) {
        if times.is_empty() {
            return (0, 0.0);
        }
        
        for i in 0..times.len() - 1 {
            if t >= times[i] && t < times[i + 1] {
                let frac = (t - times[i]) / (times[i + 1] - times[i]);
                return (i, frac);
            }
        }
        
        (times.len().saturating_sub(1), 0.0)
    }
}

fn sfe_smooth(t: f32, lambda: f32) -> f32 {
    let k = t * std::f32::consts::PI;
    let filter = (-lambda * k * k).exp();
    t * filter + t * (1.0 - filter)
}

pub struct GltfModel {
    pub meshes: Vec<GltfMesh>,
    pub skeleton: Option<Skeleton>,
    pub animations: Vec<Animation>,
    pub current_animation: usize,
}

impl GltfModel {
    pub fn load_embedded(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> Result<Self, String> {
        let gltf = gltf::Gltf::from_slice(data).map_err(|e| format!("glTF parse error: {:?}", e))?;
        let buffers = Self::load_buffers(&gltf, data)?;
        
        let images: Vec<gltf::image::Data> = gltf::import_slice(data)
            .map(|(_, _, images)| images)
            .unwrap_or_default();
        
        let mut meshes = Vec::new();
        let mut skeleton = None;
        let mut animations = Vec::new();
        
        for mesh in gltf.meshes() {
            for primitive in mesh.primitives() {
                let gltf_mesh = Self::load_primitive(device, queue, &primitive, &buffers, &images)?;
                meshes.push(gltf_mesh);
            }
        }
        
        if let Some(skin) = gltf.skins().next() {
            skeleton = Some(Self::load_skeleton(&skin, &buffers)?);
        }
        
        for anim in gltf.animations() {
            if let Some(skel) = &skeleton {
                if let Ok(animation) = Self::load_animation(&anim, &buffers, skel) {
                    animations.push(animation);
                }
            }
        }
        
        Ok(Self {
            meshes,
            skeleton,
            animations,
            current_animation: 0,
        })
    }
    
    fn load_buffers(gltf: &gltf::Gltf, data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
        let mut buffers = Vec::new();
        
        for buffer in gltf.buffers() {
            match buffer.source() {
                gltf::buffer::Source::Bin => {
                    if let Some(blob) = gltf.blob.as_ref() {
                        buffers.push(blob.clone());
                    }
                }
                gltf::buffer::Source::Uri(uri) => {
                    if uri.starts_with("data:") {
                        use base64::Engine;
                        let encoded = uri.split(',').nth(1).ok_or("Invalid data URI")?;
                        let decoded = base64::engine::general_purpose::STANDARD
                            .decode(encoded)
                            .map_err(|e| format!("Base64 error: {:?}", e))?;
                        buffers.push(decoded);
                    } else {
                        return Err(format!("External URI not supported: {}", uri));
                    }
                }
            }
        }
        
        if buffers.is_empty() && !data.is_empty() {
            let json_end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
            if json_end < data.len() {
                buffers.push(data[json_end..].to_vec());
            }
        }
        
        Ok(buffers)
    }
    
    fn load_primitive(device: &wgpu::Device, queue: &wgpu::Queue, primitive: &gltf::Primitive, buffers: &[Vec<u8>], images: &[gltf::image::Data]) -> Result<GltfMesh, String> {
        let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|b| b.as_slice()));
        
        let positions: Vec<[f32; 3]> = reader.read_positions()
            .ok_or("No positions")?
            .collect();
        
        let normals: Vec<[f32; 3]> = reader.read_normals()
            .map(|n| n.collect())
            .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
        
        let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
            .map(|tc| tc.into_f32().collect())
            .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
        
        let joints: Vec<[u16; 4]> = reader.read_joints(0)
            .map(|j| j.into_u16().collect())
            .unwrap_or_else(|| vec![[0, 0, 0, 0]; positions.len()]);
        
        let weights: Vec<[f32; 4]> = reader.read_weights(0)
            .map(|w| w.into_f32().collect())
            .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 0.0]; positions.len()]);
        
        let has_skin = reader.read_joints(0).is_some();
        
        let mut base_color = [0.8, 0.6, 0.4, 1.0];
        let mut texture = None;
        let mut texture_view = None;
        
        if let Some(mat) = primitive.material().pbr_metallic_roughness().base_color_texture() {
            let tex_idx = mat.texture().source().index();
            if tex_idx < images.len() {
                let img = &images[tex_idx];
                let (tex, view) = Self::create_texture(device, queue, img);
                texture = Some(tex);
                texture_view = Some(view);
            }
        } else {
            let c = primitive.material().pbr_metallic_roughness().base_color_factor();
            base_color = c;
        }
        
        let vertices: Vec<SkinnedVertex> = (0..positions.len())
            .map(|i| SkinnedVertex {
                position: positions[i],
                normal: normals[i],
                uv: uvs[i],
                joints: [
                    joints[i][0] as u32,
                    joints[i][1] as u32,
                    joints[i][2] as u32,
                    joints[i][3] as u32,
                ],
                weights: weights[i],
            })
            .collect();
        
        let indices: Vec<u32> = reader.read_indices()
            .map(|idx| idx.into_u32().collect())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gltf_vertex"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gltf_index"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        Ok(GltfMesh {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            has_skin,
            base_color,
            texture,
            texture_view,
        })
    }
    
    fn create_texture(device: &wgpu::Device, queue: &wgpu::Queue, img: &gltf::image::Data) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: img.width,
            height: img.height,
            depth_or_array_layers: 1,
        };
        
        let rgba_data: Vec<u8> = match img.format {
            gltf::image::Format::R8G8B8 => {
                img.pixels.chunks(3)
                    .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                    .collect()
            }
            gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
            _ => {
                vec![128u8; (img.width * img.height * 4) as usize]
            }
        };
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gltf_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.width),
                rows_per_image: Some(img.height),
            },
            size,
        );
        
        let view = texture.create_view(&Default::default());
        (texture, view)
    }
    
    fn load_skeleton(skin: &gltf::Skin, buffers: &[Vec<u8>]) -> Result<Skeleton, String> {
        let mut skeleton = Skeleton::new();
        let joints: Vec<_> = skin.joints().collect();
        
        let inverse_binds: Vec<Mat4> = if let Some(accessor) = skin.inverse_bind_matrices() {
            let view = accessor.view().ok_or("No buffer view")?;
            let buffer = &buffers[view.buffer().index()];
            let start = view.offset() + accessor.offset();
            let data = &buffer[start..];
            
            (0..accessor.count())
                .map(|i| {
                    let offset = i * 64;
                    let mat: [f32; 16] = [
                        f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]),
                        f32::from_le_bytes([data[offset+4], data[offset+5], data[offset+6], data[offset+7]]),
                        f32::from_le_bytes([data[offset+8], data[offset+9], data[offset+10], data[offset+11]]),
                        f32::from_le_bytes([data[offset+12], data[offset+13], data[offset+14], data[offset+15]]),
                        f32::from_le_bytes([data[offset+16], data[offset+17], data[offset+18], data[offset+19]]),
                        f32::from_le_bytes([data[offset+20], data[offset+21], data[offset+22], data[offset+23]]),
                        f32::from_le_bytes([data[offset+24], data[offset+25], data[offset+26], data[offset+27]]),
                        f32::from_le_bytes([data[offset+28], data[offset+29], data[offset+30], data[offset+31]]),
                        f32::from_le_bytes([data[offset+32], data[offset+33], data[offset+34], data[offset+35]]),
                        f32::from_le_bytes([data[offset+36], data[offset+37], data[offset+38], data[offset+39]]),
                        f32::from_le_bytes([data[offset+40], data[offset+41], data[offset+42], data[offset+43]]),
                        f32::from_le_bytes([data[offset+44], data[offset+45], data[offset+46], data[offset+47]]),
                        f32::from_le_bytes([data[offset+48], data[offset+49], data[offset+50], data[offset+51]]),
                        f32::from_le_bytes([data[offset+52], data[offset+53], data[offset+54], data[offset+55]]),
                        f32::from_le_bytes([data[offset+56], data[offset+57], data[offset+58], data[offset+59]]),
                        f32::from_le_bytes([data[offset+60], data[offset+61], data[offset+62], data[offset+63]]),
                    ];
                    Mat4::from_cols_array(&mat)
                })
                .collect()
        } else {
            vec![Mat4::IDENTITY; joints.len()]
        };
        
        let mut node_to_joint: HashMap<usize, usize> = HashMap::new();
        for (i, joint) in joints.iter().enumerate() {
            node_to_joint.insert(joint.index(), i);
        }
        
        let mut parent_map: HashMap<usize, usize> = HashMap::new();
        for joint_node in joints.iter() {
            for child in joint_node.children() {
                if let Some(&child_idx) = node_to_joint.get(&child.index()) {
                    if let Some(&parent_idx) = node_to_joint.get(&joint_node.index()) {
                        parent_map.insert(child_idx, parent_idx);
                    }
                }
            }
        }
        
        for (i, joint_node) in joints.iter().enumerate() {
            let (translation, rotation, scale) = joint_node.transform().decomposed();
            let local = Mat4::from_scale_rotation_translation(
                Vec3::from(scale),
                Quat::from_array(rotation),
                Vec3::from(translation),
            );
            
            let parent = parent_map.get(&i).copied();
            
            skeleton.joints.push(Joint {
                name: joint_node.name().unwrap_or("").to_string(),
                local_transform: local,
                inverse_bind: inverse_binds.get(i).copied().unwrap_or(Mat4::IDENTITY),
                parent,
                children: Vec::new(),
            });
        }
        
        for i in 0..skeleton.joints.len() {
            if let Some(parent) = skeleton.joints[i].parent {
                skeleton.joints[parent].children.push(i);
            }
        }
        
        skeleton.joint_matrices = vec![Mat4::IDENTITY; skeleton.joints.len()];
        
        Ok(skeleton)
    }
    
    fn load_animation(anim: &gltf::Animation, buffers: &[Vec<u8>], skeleton: &Skeleton) -> Result<Animation, String> {
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        
        let joint_name_to_index: HashMap<&str, usize> = skeleton.joints
            .iter()
            .enumerate()
            .filter_map(|(i, j)| if !j.name.is_empty() { Some((j.name.as_str(), i)) } else { None })
            .collect();
        
        for channel in anim.channels() {
            let target = channel.target();
            let node = target.node();
            
            let joint_index = if let Some(name) = node.name() {
                joint_name_to_index.get(name).copied()
            } else {
                None
            };
            
            let joint_index = match joint_index {
                Some(idx) => idx,
                None => continue,
            };
            
            let sampler = channel.sampler();
            let input = sampler.input();
            let output = sampler.output();
            
            let times = Self::read_accessor_f32(&input, buffers)?;
            if let Some(&max_time) = times.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                duration = duration.max(max_time);
            }
            
            let mut anim_channel = AnimationChannel {
                joint_index,
                times,
                translations: None,
                rotations: None,
                scales: None,
            };
            
            match target.property() {
                gltf::animation::Property::Translation => {
                    let data = Self::read_accessor_vec3(&output, buffers)?;
                    anim_channel.translations = Some(data);
                }
                gltf::animation::Property::Rotation => {
                    let data = Self::read_accessor_quat(&output, buffers)?;
                    anim_channel.rotations = Some(data);
                }
                gltf::animation::Property::Scale => {
                    let data = Self::read_accessor_vec3(&output, buffers)?;
                    anim_channel.scales = Some(data);
                }
                _ => {}
            }
            
            channels.push(anim_channel);
        }
        
        Ok(Animation {
            name: anim.name().unwrap_or("").to_string(),
            duration,
            channels,
        })
    }
    
    fn read_accessor_f32(accessor: &gltf::Accessor, buffers: &[Vec<u8>]) -> Result<Vec<f32>, String> {
        let view = accessor.view().ok_or("No buffer view")?;
        let buffer = &buffers[view.buffer().index()];
        let start = view.offset() + accessor.offset();
        let data = &buffer[start..];
        
        Ok((0..accessor.count())
            .map(|i| {
                let offset = i * 4;
                f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]])
            })
            .collect())
    }
    
    fn read_accessor_vec3(accessor: &gltf::Accessor, buffers: &[Vec<u8>]) -> Result<Vec<Vec3>, String> {
        let view = accessor.view().ok_or("No buffer view")?;
        let buffer = &buffers[view.buffer().index()];
        let start = view.offset() + accessor.offset();
        let data = &buffer[start..];
        
        Ok((0..accessor.count())
            .map(|i| {
                let offset = i * 12;
                Vec3::new(
                    f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]),
                    f32::from_le_bytes([data[offset+4], data[offset+5], data[offset+6], data[offset+7]]),
                    f32::from_le_bytes([data[offset+8], data[offset+9], data[offset+10], data[offset+11]]),
                )
            })
            .collect())
    }
    
    fn read_accessor_quat(accessor: &gltf::Accessor, buffers: &[Vec<u8>]) -> Result<Vec<Quat>, String> {
        let view = accessor.view().ok_or("No buffer view")?;
        let buffer = &buffers[view.buffer().index()];
        let start = view.offset() + accessor.offset();
        let data = &buffer[start..];
        
        Ok((0..accessor.count())
            .map(|i| {
                let offset = i * 16;
                Quat::from_xyzw(
                    f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]),
                    f32::from_le_bytes([data[offset+4], data[offset+5], data[offset+6], data[offset+7]]),
                    f32::from_le_bytes([data[offset+8], data[offset+9], data[offset+10], data[offset+11]]),
                    f32::from_le_bytes([data[offset+12], data[offset+13], data[offset+14], data[offset+15]]),
                )
            })
            .collect())
    }
    
    pub fn update(&mut self, time: f32, sfe_lambda: f32) {
        if let Some(skeleton) = self.skeleton.as_mut() {
            if let Some(animation) = self.animations.get(self.current_animation) {
                let local_transforms = animation.sample(time, skeleton, sfe_lambda);
                skeleton.update_matrices(&local_transforms);
            } else {
                let identity_transforms: Vec<Mat4> = skeleton.joints.iter()
                    .map(|j| j.local_transform)
                    .collect();
                skeleton.update_matrices(&identity_transforms);
            }
        }
    }
    
    pub fn get_joint_matrices(&self) -> Option<&[Mat4]> {
        self.skeleton.as_ref().map(|s| {
            if s.joint_matrices.is_empty() {
                &[][..]
            } else {
                s.joint_matrices.as_slice()
            }
        })
    }
}

pub const MAX_JOINTS: usize = 128;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinUniforms {
    pub joint_matrices: [[[f32; 4]; 4]; MAX_JOINTS],
    pub num_joints: u32,
    pub _pad: [u32; 3],
}

impl SkinUniforms {
    pub fn new() -> Self {
        let mut uniforms = Self {
            joint_matrices: [[[0.0; 4]; 4]; MAX_JOINTS],
            num_joints: 1,
            _pad: [0; 3],
        };
        uniforms.joint_matrices[0] = Mat4::IDENTITY.to_cols_array_2d();
        uniforms
    }
    
    pub fn update(&mut self, matrices: &[Mat4]) {
        if matrices.is_empty() {
            self.num_joints = 1;
            self.joint_matrices[0] = Mat4::IDENTITY.to_cols_array_2d();
        } else {
            self.num_joints = matrices.len().min(MAX_JOINTS) as u32;
            for (i, mat) in matrices.iter().take(MAX_JOINTS).enumerate() {
                self.joint_matrices[i] = mat.to_cols_array_2d();
            }
        }
    }
}

