pub const SFE_SHADOW_COMPUTE: &str = r#"
@group(0) @binding(0) var shadow_in: texture_2d<f32>;
@group(0) @binding(1) var shadow_out: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: SfeParams;

struct SfeParams {
    lambda: f32,
    tau: f32,
    size: f32,
    _pad: f32,
}

fn sfe_kernel(k2: f32) -> f32 {
    return exp(-(k2 + params.lambda * k2 * k2) * params.tau);
}

@compute
@workgroup_size(8, 8)
fn sfe_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = u32(params.size);
    if gid.x >= size || gid.y >= size { return; }
    
    var sum = 0.0;
    var weight_sum = 0.0;
    
    let radius = 4;
    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let px = clamp(i32(gid.x) + dx, 0, i32(size) - 1);
            let py = clamp(i32(gid.y) + dy, 0, i32(size) - 1);
            let val = textureLoad(shadow_in, vec2<i32>(px, py), 0).r;
            
            let dist2 = f32(dx * dx + dy * dy);
            let k2 = dist2 / (params.size * params.size) * 100.0;
            let w = sfe_kernel(k2);
            
            sum += val * w;
            weight_sum += w;
        }
    }
    
    textureStore(shadow_out, vec2<i32>(gid.xy), vec4<f32>(sum / weight_sum, 0.0, 0.0, 1.0));
}
"#;

pub const SHADOW_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;
struct InstanceData { model: mat4x4<f32>, color: vec4<f32>, }
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;

struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) depth: f32, }

@vertex
fn vs_shadow(@location(0) position: vec3<f32>, @builtin(instance_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = instances[idx].model * vec4<f32>(position, 1.0);
    out.position = u.light_view_proj * world_pos;
    out.depth = out.position.z / out.position.w;
    return out;
}

@fragment
fn fs_shadow(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.depth, 0.0, 0.0, 1.0);
}
"#;

pub const MESH_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;
struct InstanceData { model: mat4x4<f32>, color: vec4<f32>, }
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(0) var sfe_shadow: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) shadow_pos: vec4<f32>,
    @location(4) view_dist: f32,
    @location(5) emission: f32,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) color: vec3<f32>, @builtin(instance_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let inst = instances[idx];
    let world_pos = inst.model * vec4<f32>(position, 1.0);
    out.clip_position = u.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.world_normal = normalize((inst.model * vec4<f32>(normal, 0.0)).xyz);
    out.color = inst.color.rgb;
    out.shadow_pos = u.light_view_proj * world_pos;
    out.view_dist = length(u.camera_pos.xyz - world_pos.xyz);
    out.emission = inst.color.a;
    return out;
}

fn sfe_shadow_sample(shadow_pos: vec4<f32>) -> f32 {
    let proj = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));
    let size = i32(u.shadow_size);
    let px = vec2<i32>(i32(uv.x * f32(size)), i32(uv.y * f32(size)));
    let stored_depth = textureLoad(sfe_shadow, clamp(px, vec2(0), vec2(size - 1)), 0).r;
    let current_depth = proj.z - 0.002;
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    return select(1.0, select(0.2, 1.0, current_depth <= stored_depth), in_bounds);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(-u.light_dir.xyz);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);
    let shadow = sfe_shadow_sample(in.shadow_pos);
    let ambient = 0.12;
    let diffuse = max(dot(N, L), 0.0) * 0.5 * shadow;
    let specular = pow(max(dot(N, H), 0.0), 64.0) * 0.6 * shadow;
    var color = in.color * (ambient + diffuse) + vec3<f32>(specular);
    color += in.color * in.emission * 2.0;
    let fog = clamp((in.view_dist - u.fog_start) / (u.fog_end - u.fog_start), 0.0, 1.0);
    color = mix(color, u.fog_color.rgb, fog);
    return vec4<f32>(color, 1.0);
}
"#;

pub const FLOOR_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;
@group(1) @binding(0) var sfe_shadow: texture_2d<f32>;

struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) shadow_pos: vec4<f32>, }

@vertex
fn vs_floor(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    let p = pos[idx] * 80.0;
    var out: VertexOutput;
    let wp = vec4<f32>(p.x, 0.0, p.y, 1.0);
    out.clip_position = u.view_proj * wp;
    out.world_pos = wp.xyz;
    out.shadow_pos = u.light_view_proj * wp;
    return out;
}

fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

fn sfe_shadow_floor(shadow_pos: vec4<f32>) -> f32 {
    let proj = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));
    let size = i32(u.shadow_size);
    let px = vec2<i32>(i32(uv.x * f32(size)), i32(uv.y * f32(size)));
    let stored_depth = textureLoad(sfe_shadow, clamp(px, vec2(0), vec2(size - 1)), 0).r;
    let current_depth = proj.z - 0.001;
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    return select(1.0, select(0.25, 1.0, current_depth <= stored_depth), in_bounds);
}

@fragment
fn fs_floor(in: VertexOutput) -> @location(0) vec4<f32> {
    let tile = 2.0;
    let checker = (i32(floor(in.world_pos.x / tile)) + i32(floor(in.world_pos.z / tile))) % 2;
    let base = select(vec3<f32>(0.08, 0.09, 0.12), vec3<f32>(0.12, 0.13, 0.16), checker == 0);
    
    let bump_scale = 5.0;
    let e = 0.08;
    let h0 = noise(in.world_pos.xz * bump_scale);
    let hx = noise((in.world_pos.xz + vec2(e, 0.0)) * bump_scale);
    let hz = noise((in.world_pos.xz + vec2(0.0, e)) * bump_scale);
    let N = normalize(vec3<f32>(-(hx - h0) * 0.15, 1.0, -(hz - h0) * 0.15));
    
    let L = normalize(-u.light_dir.xyz);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);
    let shadow = sfe_shadow_floor(in.shadow_pos);
    
    let ambient = 0.1;
    let diffuse = max(dot(N, L), 0.0) * 0.4 * shadow;
    let spec = pow(max(dot(N, H), 0.0), 64.0) * 0.5 * shadow;
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0) * 0.15;
    
    var color = base * (ambient + diffuse) + vec3<f32>(spec + fresnel);
    
    let grid = 1.0; let lw = 0.008;
    let gx = abs(fract(in.world_pos.x / grid + 0.5) - 0.5) * grid;
    let gz = abs(fract(in.world_pos.z / grid + 0.5) - 0.5) * grid;
    let line = 1.0 - smoothstep(0.0, lw, min(gx, gz));
    color = mix(color, vec3<f32>(0.15, 0.4, 0.6), line * 0.4);
    
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog = clamp((dist - u.fog_start) / (u.fog_end - u.fog_start), 0.0, 1.0);
    color = mix(color, u.fog_color.rgb, fog);
    
    let fade = 1.0 - smoothstep(25.0, 60.0, length(in.world_pos.xz));
    return vec4<f32>(color, fade);
}
"#;

pub const WATER_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;

struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) world_pos: vec3<f32>, }

@vertex
fn vs_water(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    let p = pos[idx] * 80.0;
    var out: VertexOutput;
    let wp = vec4<f32>(p.x, -0.25, p.y, 1.0);
    out.clip_position = u.view_proj * wp;
    out.world_pos = wp.xyz;
    return out;
}

fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

fn sfe_water_wave(p: vec2<f32>, t: f32, lambda: f32) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var freq = 1.0;
    var pp = p;
    for (var i = 0; i < 4; i++) {
        let k2 = freq * freq;
        let sfe_damp = exp(-lambda * k2 * k2 * 0.01);
        sum += noise(pp + vec2(t * 0.3, t * 0.2)) * amp * sfe_damp;
        amp *= 0.5;
        freq *= 2.0;
        pp *= 2.0;
    }
    return sum;
}

@fragment
fn fs_water(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = u.time;
    let wave = sfe_water_wave(in.world_pos.xz * 0.12, t, u.sfe_lambda);
    
    let e = 0.1;
    let wx = sfe_water_wave((in.world_pos.xz + vec2(e, 0.0)) * 0.12, t, u.sfe_lambda);
    let wz = sfe_water_wave((in.world_pos.xz + vec2(0.0, e)) * 0.12, t, u.sfe_lambda);
    let N = normalize(vec3<f32>(-(wx - wave) * 1.5, 1.0, -(wz - wave) * 1.5));
    
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let L = normalize(-u.light_dir.xyz);
    let R = reflect(-L, N);
    
    let deep = vec3<f32>(0.02, 0.06, 0.12);
    let shallow = vec3<f32>(0.08, 0.25, 0.35);
    let water_color = mix(deep, shallow, wave * 0.5 + 0.5);
    
    let spec = pow(max(dot(R, V), 0.0), 128.0) * 2.5;
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.5) * 0.6;
    let sky_reflect = vec3<f32>(0.3, 0.4, 0.5) * fresnel;
    
    var color = water_color + vec3<f32>(spec) + sky_reflect;
    
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog = clamp((dist - u.fog_start) / (u.fog_end - u.fog_start), 0.0, 1.0);
    color = mix(color, u.fog_color.rgb, fog);
    
    let fade = 1.0 - smoothstep(30.0, 70.0, length(in.world_pos.xz));
    return vec4<f32>(color, 0.85 * fade);
}
"#;

pub const GRASS_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;
@group(1) @binding(0) var sfe_shadow: texture_2d<f32>;

struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec3<f32>, }

fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }

@vertex
fn vs_grass(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let blade_verts = array<vec2<f32>, 6>(vec2(0.0,0.0),vec2(0.04,0.0),vec2(0.02,0.5),vec2(0.04,0.0),vec2(0.03,0.5),vec2(0.02,0.5));
    let local = blade_verts[vid % 6u];
    
    let grid_size = 40u;
    let gx = iid % grid_size;
    let gz = iid / grid_size;
    let base_x = (f32(gx) - f32(grid_size) / 2.0) * 0.4;
    let base_z = (f32(gz) - f32(grid_size) / 2.0) * 0.4;
    let rand_offset = vec2<f32>(hash(vec2(base_x, base_z)) - 0.5, hash(vec2(base_z, base_x)) - 0.5) * 0.3;
    
    let height = 0.3 + hash(vec2(base_x * 7.0, base_z * 11.0)) * 0.4;
    let wind = sin(u.time * 2.0 + base_x * 0.5 + base_z * 0.3) * local.y * 0.15;
    
    let world_x = base_x + rand_offset.x + local.x + wind;
    let world_y = local.y * height;
    let world_z = base_z + rand_offset.y;
    
    var out: VertexOutput;
    let wp = vec4<f32>(world_x, world_y, world_z, 1.0);
    out.clip_position = u.view_proj * wp;
    out.world_pos = wp.xyz;
    out.uv = local;
    
    let base_green = vec3<f32>(0.15, 0.35, 0.1);
    let tip_green = vec3<f32>(0.3, 0.55, 0.15);
    out.color = mix(base_green, tip_green, local.y);
    return out;
}

@fragment
fn fs_grass(in: VertexOutput) -> @location(0) vec4<f32> {
    let L = normalize(-u.light_dir.xyz);
    let N = vec3<f32>(0.0, 1.0, 0.0);
    let diffuse = max(dot(N, L), 0.0) * 0.6 + 0.4;
    var color = in.color * diffuse;
    
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog = clamp((dist - u.fog_start) / (u.fog_end - u.fog_start), 0.0, 1.0);
    color = mix(color, u.fog_color.rgb, fog);
    
    let fade = 1.0 - smoothstep(8.0, 15.0, dist);
    return vec4<f32>(color, fade);
}
"#;

pub const PARTICLE_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;

struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32>, }

fn hash(p: f32) -> f32 { return fract(sin(p * 127.1) * 43758.5453); }

@vertex
fn vs_particle(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    let local = quad[vid % 6u];
    
    let seed = f32(iid) * 0.1;
    let life = fract(u.time * 0.2 + hash(seed));
    let px = (hash(seed + 1.0) - 0.5) * 10.0;
    let pz = (hash(seed + 2.0) - 0.5) * 10.0;
    let py = life * 5.0;
    
    let size = 0.05 + (1.0 - life) * 0.1;
    let right = vec3<f32>(u.view_proj[0][0], u.view_proj[1][0], u.view_proj[2][0]);
    let up = vec3<f32>(u.view_proj[0][1], u.view_proj[1][1], u.view_proj[2][1]);
    let world_pos = vec3<f32>(px, py, pz) + right * local.x * size + up * local.y * size;
    
    var out: VertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = local * 0.5 + 0.5;
    out.color = vec4<f32>(0.8, 0.6, 0.3, (1.0 - life) * 0.6);
    return out;
}

@fragment
fn fs_particle(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv - vec2(0.5));
    let alpha = smoothstep(0.5, 0.2, dist) * in.color.a;
    return vec4<f32>(in.color.rgb, alpha);
}
"#;

pub const SKINNED_MESH_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;

struct SkinUniforms {
    joint_matrices: array<mat4x4<f32>, 128>,
    num_joints: u32,
}
@group(0) @binding(2) var<uniform> skin: SkinUniforms;

@group(1) @binding(0) var sfe_shadow: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) shadow_pos: vec4<f32>,
}

@vertex
fn vs_skinned(in: VertexInput) -> VertexOutput {
    let scale = 0.015;
    let model_matrix = mat4x4<f32>(
        vec4<f32>(scale, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, scale, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
    
    var skin_matrix = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
    
    let total_weight = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    if total_weight > 0.001 && skin.num_joints > 0u {
        skin_matrix = mat4x4<f32>(
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
        );
        skin_matrix += skin.joint_matrices[in.joints.x] * in.weights.x;
        skin_matrix += skin.joint_matrices[in.joints.y] * in.weights.y;
        skin_matrix += skin.joint_matrices[in.joints.z] * in.weights.z;
        skin_matrix += skin.joint_matrices[in.joints.w] * in.weights.w;
    }
    
    let world_pos = model_matrix * skin_matrix * vec4<f32>(in.position, 1.0);
    let world_normal = normalize((model_matrix * skin_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    
    var out: VertexOutput;
    out.clip_position = u.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.world_normal = world_normal;
    out.uv = in.uv;
    out.shadow_pos = u.light_view_proj * world_pos;
    return out;
}

fn sfe_shadow_skinned(shadow_pos: vec4<f32>) -> f32 {
    let proj = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));
    let size = i32(u.shadow_size);
    let px = vec2<i32>(i32(uv.x * f32(size)), i32(uv.y * f32(size)));
    let stored_depth = textureLoad(sfe_shadow, clamp(px, vec2(0), vec2(size - 1)), 0).r;
    let current_depth = proj.z - 0.003;
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    return select(1.0, select(0.3, 1.0, current_depth <= stored_depth), in_bounds);
}

@fragment
fn fs_skinned(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(-u.light_dir.xyz);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);
    
    let shadow = sfe_shadow_skinned(in.shadow_pos);
    
    let uv = in.uv;
    let checker = floor(uv.x * 8.0) + floor(uv.y * 8.0);
    let pattern = fract(checker * 0.5) * 2.0;
    
    let orange = vec3<f32>(0.95, 0.5, 0.2);
    let cream = vec3<f32>(0.98, 0.92, 0.85);
    let base_color = mix(orange, cream, pattern * 0.3);
    
    let ambient = 0.25;
    let diffuse = max(dot(N, L), 0.0) * 0.65 * shadow;
    let specular = pow(max(dot(N, H), 0.0), 16.0) * 0.15 * shadow;
    
    var color = base_color * (ambient + diffuse) + vec3<f32>(specular);
    
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog = clamp((dist - u.fog_start) / (u.fog_end - u.fog_start), 0.0, 1.0);
    color = mix(color, u.fog_color.rgb, fog);
    
    return vec4<f32>(color, 1.0);
}
"#;

pub const BLOOM_SHADER: &str = r#"
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: BloomParams;

struct BloomParams { sfe_lambda: f32, sfe_tau: f32, width: f32, height: f32, bloom_intensity: f32, bloom_threshold: f32, _pad: vec2<f32>, }

struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, }

@vertex
fn vs_bloom(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VertexOutput;
    out.position = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = pos[idx] * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

fn sfe_bloom_kernel(k2: f32, lambda: f32, tau: f32) -> f32 {
    return exp(-(k2 + lambda * k2 * k2) * tau);
}

@fragment
fn fs_bloom(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<i32>(i32(params.width), i32(params.height));
    let px = vec2<i32>(i32(in.uv.x * params.width), i32(in.uv.y * params.height));
    let color = textureLoad(input_tex, clamp(px, vec2(0), size - vec2(1)), 0);
    
    var bloom = vec3<f32>(0.0);
    var weight_sum = 0.0;
    
    let radius = 8;
    for (var y = -radius; y <= radius; y++) {
        for (var x = -radius; x <= radius; x++) {
            let sample_px = clamp(px + vec2(x * 2, y * 2), vec2(0), size - vec2(1));
            let sample_color = textureLoad(input_tex, sample_px, 0).rgb;
            
            let luminance = dot(sample_color, vec3<f32>(0.299, 0.587, 0.114));
            let bright = max(vec3(0.0), sample_color - vec3(params.bloom_threshold)) * smoothstep(params.bloom_threshold, 1.2, luminance);
            
            let dist2 = f32(x * x + y * y);
            let k2 = dist2 / f32(radius * radius);
            let w = sfe_bloom_kernel(k2, params.sfe_lambda, params.sfe_tau);
            
            bloom += bright * w;
            weight_sum += w;
        }
    }
    
    bloom /= weight_sum;
    
    var final_color = color.rgb + bloom * params.bloom_intensity;
    final_color = final_color / (final_color + vec3(1.0));
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(final_color, 1.0);
}
"#;

