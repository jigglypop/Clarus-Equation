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
    
    var shadow_sum = 0.0;
    let pcf_radius = 3;
    var sample_count = 0.0;
    
    for (var dy = -pcf_radius; dy <= pcf_radius; dy++) {
        for (var dx = -pcf_radius; dx <= pcf_radius; dx++) {
            let offset = vec2<i32>(dx, dy);
            let px = vec2<i32>(i32(uv.x * f32(size)), i32(uv.y * f32(size))) + offset;
            let stored_depth = textureLoad(sfe_shadow, clamp(px, vec2(0), vec2(size - 1)), 0).r;
            let current_depth = proj.z - 0.003;
            shadow_sum += select(0.0, 1.0, current_depth <= stored_depth);
            sample_count += 1.0;
        }
    }
    
    let shadow = shadow_sum / sample_count;
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    return select(1.0, mix(0.15, 1.0, shadow), in_bounds);
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
    
    var shadow_sum = 0.0;
    let pcf_radius = 4;
    var sample_count = 0.0;
    
    for (var dy = -pcf_radius; dy <= pcf_radius; dy++) {
        for (var dx = -pcf_radius; dx <= pcf_radius; dx++) {
            let offset = vec2<i32>(dx, dy);
            let px = vec2<i32>(i32(uv.x * f32(size)), i32(uv.y * f32(size))) + offset;
            let stored_depth = textureLoad(sfe_shadow, clamp(px, vec2(0), vec2(size - 1)), 0).r;
            let current_depth = proj.z - 0.003;
            shadow_sum += select(0.0, 1.0, current_depth <= stored_depth);
            sample_count += 1.0;
        }
    }
    
    let shadow = shadow_sum / sample_count;
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    return select(1.0, mix(0.1, 1.0, shadow), in_bounds);
}

@fragment
fn fs_floor(in: VertexOutput) -> @location(0) vec4<f32> {
    let tile = 2.0;
    let checker = (i32(floor(in.world_pos.x / tile)) + i32(floor(in.world_pos.z / tile))) % 2;
    let base = select(vec3<f32>(0.15, 0.16, 0.2), vec3<f32>(0.2, 0.21, 0.25), checker == 0);
    
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
    
    let ambient = 0.15;
    let diffuse = max(dot(N, L), 0.0) * 0.5 * shadow;
    let spec = pow(max(dot(N, H), 0.0), 64.0) * 0.4 * shadow;
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0) * 0.1;
    
    var color = base * (ambient + diffuse) + vec3<f32>(spec + fresnel);
    
    let shadow_tint = vec3<f32>(0.01, 0.01, 0.02);
    color = mix(shadow_tint, color, shadow * 0.85 + 0.15);
    
    let grid = 1.0; let lw = 0.008;
    let gx = abs(fract(in.world_pos.x / grid + 0.5) - 0.5) * grid;
    let gz = abs(fract(in.world_pos.z / grid + 0.5) - 0.5) * grid;
    let line = 1.0 - smoothstep(0.0, lw, min(gx, gz));
    color = mix(color, vec3<f32>(0.15, 0.4, 0.6), line * 0.3 * shadow);
    
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
fn hash2(p: vec2<f32>) -> vec2<f32> { return vec2(hash(p), hash(p + vec2(57.0, 113.0))); }

@vertex
fn vs_grass(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let blade_verts = array<vec2<f32>, 15>(
        vec2(0.0, 0.0), vec2(0.012, 0.0), vec2(0.01, 0.25),
        vec2(0.012, 0.0), vec2(0.014, 0.25), vec2(0.01, 0.25),
        vec2(0.01, 0.25), vec2(0.014, 0.25), vec2(0.008, 0.5),
        vec2(0.014, 0.25), vec2(0.012, 0.5), vec2(0.008, 0.5),
        vec2(0.008, 0.5), vec2(0.012, 0.5), vec2(0.006, 1.0)
    );
    let local = blade_verts[vid % 15u];
    
    let grid_size = 200u;
    let blades_per_cell = 3u;
    let cell_idx = iid / blades_per_cell;
    let blade_in_cell = iid % blades_per_cell;
    
    let gx = cell_idx % grid_size;
    let gz = cell_idx / grid_size;
    let cell_size = 0.08;
    let base_x = (f32(gx) - f32(grid_size) / 2.0) * cell_size;
    let base_z = (f32(gz) - f32(grid_size) / 2.0) * cell_size;
    
    let cell_offset = hash2(vec2(f32(cell_idx), f32(blade_in_cell) * 7.0)) * cell_size * 0.9;
    let pos_x = base_x + cell_offset.x;
    let pos_z = base_z + cell_offset.y;
    
    let rotation = hash(vec2(pos_x * 13.0, pos_z * 17.0)) * 6.28;
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    
    let height = 0.15 + hash(vec2(pos_x * 7.0, pos_z * 11.0)) * 0.25;
    let bend = hash(vec2(pos_x * 23.0, pos_z * 29.0)) * 0.3;
    
    let wind_phase = pos_x * 0.8 + pos_z * 0.6 + hash(vec2(pos_x, pos_z)) * 2.0;
    let wind = sin(u.time * 2.5 + wind_phase) * local.y * local.y * 0.08;
    let wind2 = sin(u.time * 4.0 + wind_phase * 1.5) * local.y * local.y * 0.03;
    
    var blade_x = local.x * cos_r + bend * local.y * local.y;
    var blade_z = local.x * sin_r;
    blade_x += wind + wind2;
    
    let world_x = pos_x + blade_x;
    let world_y = local.y * height;
    let world_z = pos_z + blade_z;
    
    var out: VertexOutput;
    let wp = vec4<f32>(world_x, world_y, world_z, 1.0);
    out.clip_position = u.view_proj * wp;
    out.world_pos = wp.xyz;
    out.uv = local;
    
    let base_green = vec3<f32>(0.08, 0.22, 0.05);
    let mid_green = vec3<f32>(0.12, 0.35, 0.08);
    let tip_green = vec3<f32>(0.25, 0.5, 0.12);
    let t = local.y;
    var col = mix(base_green, mid_green, smoothstep(0.0, 0.4, t));
    col = mix(col, tip_green, smoothstep(0.4, 1.0, t));
    col *= 0.85 + hash(vec2(pos_x * 31.0, pos_z * 37.0)) * 0.3;
    out.color = col;
    return out;
}

@fragment
fn fs_grass(in: VertexOutput) -> @location(0) vec4<f32> {
    let L = normalize(-u.light_dir.xyz);
    let N = normalize(vec3<f32>(in.uv.x * 0.5, 1.0, 0.0));
    let diffuse = max(dot(N, L), 0.0) * 0.5 + 0.5;
    
    let sss = pow(max(dot(-L, normalize(u.camera_pos.xyz - in.world_pos)), 0.0), 2.0) * 0.3;
    var color = in.color * diffuse + vec3<f32>(0.1, 0.15, 0.05) * sss;
    
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog = clamp((dist - u.fog_start) / (u.fog_end - u.fog_start), 0.0, 1.0);
    color = mix(color, u.fog_color.rgb, fog);
    
    let fade = 1.0 - smoothstep(5.0, 10.0, dist);
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

pub const SKINNED_SHADOW_SHADER: &str = r#"
struct GlobalUniforms { 
    view_proj: mat4x4<f32>, 
    light_view_proj: mat4x4<f32>, 
    light_dir: vec4<f32>, 
    camera_pos: vec4<f32>, 
    fog_color: vec4<f32>, 
    time: f32, 
    fog_start: f32, 
    fog_end: f32, 
    sfe_lambda: f32, 
    sfe_tau: f32, 
    shadow_size: f32,
}
@group(0) @binding(0) var<uniform> u: GlobalUniforms;

struct SkinUniforms {
    joint_matrices: array<mat4x4<f32>, 128>,
    num_joints: u32,
}
@group(0) @binding(2) var<uniform> skin: SkinUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
}

@vertex
fn vs_skinned_shadow(in: VertexInput) -> VertexOutput {
    let scale = 0.018;
    let model_matrix = mat4x4<f32>(
        vec4<f32>(scale, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, scale, 0.0),
        vec4<f32>(0.0, 0.25, 0.0, 1.0)
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
    
    var out: VertexOutput;
    out.clip_position = u.light_view_proj * world_pos;
    out.depth = out.clip_position.z / out.clip_position.w;
    return out;
}

@fragment
fn fs_skinned_shadow(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.depth, 0.0, 0.0, 1.0);
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
    let scale = 0.018;
    let model_matrix = mat4x4<f32>(
        vec4<f32>(scale, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, scale, 0.0),
        vec4<f32>(0.0, 0.25, 0.0, 1.0)
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
    
    var shadow_sum = 0.0;
    let pcf_radius = 3;
    var sample_count = 0.0;
    
    for (var dy = -pcf_radius; dy <= pcf_radius; dy++) {
        for (var dx = -pcf_radius; dx <= pcf_radius; dx++) {
            let offset = vec2<i32>(dx, dy);
            let px = vec2<i32>(i32(uv.x * f32(size)), i32(uv.y * f32(size))) + offset;
            let stored_depth = textureLoad(sfe_shadow, clamp(px, vec2(0), vec2(size - 1)), 0).r;
            let current_depth = proj.z - 0.004;
            shadow_sum += select(0.0, 1.0, current_depth <= stored_depth);
            sample_count += 1.0;
        }
    }
    
    let shadow = shadow_sum / sample_count;
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    return select(1.0, mix(0.2, 1.0, shadow), in_bounds);
}

fn sfe_contact_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let ground_y = 0.0;
    let shadow_pos = world_pos - light_dir * (world_pos.y - ground_y) / max(light_dir.y, 0.001);
    let dist_to_shadow = length(shadow_pos.xz - world_pos.xz);
    let height_factor = clamp(world_pos.y * 2.0, 0.0, 1.0);
    return 1.0 - exp(-dist_to_shadow * 0.5) * height_factor * 0.3;
}

fn sfe_sss(NdotL: f32, thickness: f32, lambda: f32) -> f32 {
    let wrap = 0.5;
    let scatter = max(0.0, (NdotL + wrap) / (1.0 + wrap));
    let sss_falloff = exp(-lambda * thickness * thickness);
    return scatter * sss_falloff * 0.4;
}

fn sfe_fresnel(NdotV: f32, lambda: f32) -> f32 {
    let base = 1.0 - NdotV;
    let fresnel = base * base * base;
    let sfe_smooth = exp(-lambda * fresnel * fresnel * 4.0);
    return fresnel * sfe_smooth * 0.6;
}

fn sfe_rim(NdotV: f32, NdotL: f32, lambda: f32) -> f32 {
    let rim = pow(1.0 - NdotV, 3.0);
    let light_rim = max(0.0, -NdotL + 0.3);
    let sfe_damp = exp(-lambda * rim * 2.0);
    return rim * light_rim * sfe_damp * 1.5;
}

@fragment
fn fs_skinned(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(-u.light_dir.xyz);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);
    
    let NdotL = dot(N, L);
    let NdotV = max(dot(N, V), 0.001);
    let NdotH = max(dot(N, H), 0.0);
    
    let shadow = sfe_shadow_skinned(in.shadow_pos);
    
    let uv = in.uv;
    let fur_pattern = sin(uv.x * 40.0) * sin(uv.y * 40.0) * 0.1 + 0.9;
    
    let orange = vec3<f32>(0.95, 0.55, 0.15);
    let cream = vec3<f32>(0.98, 0.9, 0.8);
    let dark = vec3<f32>(0.3, 0.15, 0.05);
    
    let height_blend = clamp(in.world_pos.y * 0.5 + 0.5, 0.0, 1.0);
    var base_color = mix(dark, orange, height_blend);
    base_color = mix(base_color, cream, clamp(uv.y * 0.3, 0.0, 0.3));
    base_color *= fur_pattern;
    
    let sss = sfe_sss(NdotL, 0.3, u.sfe_lambda);
    let fresnel = sfe_fresnel(NdotV, u.sfe_lambda);
    let rim = sfe_rim(NdotV, NdotL, u.sfe_lambda);
    
    let ambient = 0.15;
    let diffuse = max(NdotL, 0.0) * 0.5 * shadow;
    let specular = pow(NdotH, 32.0) * 0.25 * shadow;
    
    var color = base_color * (ambient + diffuse + sss);
    color += vec3<f32>(specular);
    color += base_color * fresnel * 0.3;
    color += vec3<f32>(1.0, 0.8, 0.5) * rim * shadow;
    
    let emission = max(0.0, color.r - 0.7) * 0.5;
    color += vec3<f32>(emission * 0.8, emission * 0.4, emission * 0.1);
    
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog_factor = 1.0 - exp(-u.sfe_tau * dist * dist * 0.001);
    let fog = clamp(fog_factor, 0.0, 0.8);
    color = mix(color, u.fog_color.rgb * 1.2, fog);
    
    return vec4<f32>(color, 1.0);
}
"#;

pub const BLOOM_SHADER: &str = r#"
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: BloomParams;

struct BloomParams { sfe_lambda: f32, sfe_tau: f32, width: f32, height: f32, bloom_intensity: f32, bloom_threshold: f32, time: f32, light_dir_y: f32, }

struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, }

fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

@vertex
fn vs_bloom(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VertexOutput;
    out.position = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = pos[idx] * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

fn sfe_kernel(k2: f32, lambda: f32, tau: f32) -> f32 {
    return exp(-(k2 + lambda * k2 * k2) * tau);
}

fn sfe_adaptive_radius(lambda: f32, tau: f32, threshold: f32) -> f32 {
    return sqrt(-log(threshold) / tau) / (1.0 + lambda);
}

fn sfe_lod_factor(dist: f32, lambda: f32) -> f32 {
    let k2 = dist * dist * 0.01;
    return max(1.0, 1.0 + log2(1.0 + k2 * lambda));
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 0.75 * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

fn sfe_ssao(uv: vec2<f32>, size: vec2<i32>, lambda: f32) -> f32 {
    let px = vec2<i32>(uv * vec2<f32>(size));
    let center_lum = dot(textureLoad(input_tex, px, 0).rgb, vec3(0.299, 0.587, 0.114));
    
    var occlusion = 0.0;
    var weight_sum = 0.0;
    
    let adaptive_r = sfe_adaptive_radius(lambda, 0.15, 0.05);
    let radius = min(adaptive_r, 0.02);
    
    let rand_angle = hash(uv * 50.0) * 6.28;
    let threshold = 0.05;
    
    for (var i = 0; i < 6; i++) {
        let t = f32(i) / 5.0;
        let k2 = t * t * 4.0;
        let sfe_w = sfe_kernel(k2, lambda, 0.15);
        
        if (sfe_w < threshold) { break; }
        
        let angle = rand_angle + f32(i) * 1.047;
        let offset = vec2(cos(angle), sin(angle)) * radius * t;
        let sample_uv = clamp(uv + offset, vec2(0.0), vec2(1.0));
        let sample_px = vec2<i32>(sample_uv * vec2<f32>(size));
        let sample_lum = dot(textureLoad(input_tex, sample_px, 0).rgb, vec3(0.299, 0.587, 0.114));
        
        let diff = center_lum - sample_lum;
        if (diff > 0.015 && diff < 0.12) {
            occlusion += sfe_w;
        }
        weight_sum += sfe_w;
    }
    
    if (weight_sum < 0.01) { return 1.0; }
    return 1.0 - (occlusion / weight_sum) * 0.35;
}

fn sfe_god_rays(uv: vec2<f32>, size: vec2<i32>, lambda: f32, time: f32) -> vec3<f32> {
    let light_pos = vec2(0.5, 0.12);
    let delta = light_pos - uv;
    let dist = length(delta);
    
    let cutoff = sfe_adaptive_radius(lambda, 0.3, 0.02);
    if (dist > cutoff) { return vec3(0.0); }
    
    let dir = normalize(delta);
    var ray = vec3(0.0);
    var weight_sum = 0.0;
    
    let max_t = min(dist, cutoff * 0.5);
    let threshold = 0.02;
    
    for (var i = 0; i < 10; i++) {
        let t = f32(i) / 9.0 * max_t;
        let k2 = (t / max_t) * (t / max_t) * 4.0;
        let sfe_w = sfe_kernel(k2, lambda, 0.3);
        
        if (sfe_w < threshold) { break; }
        
        let sample_uv = uv + dir * t;
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) { continue; }
        
        let sample_px = vec2<i32>(sample_uv * vec2<f32>(size));
        let sample_color = textureLoad(input_tex, sample_px, 0).rgb;
        let lum = dot(sample_color, vec3(0.299, 0.587, 0.114));
        
        if (lum > 0.6) {
            ray += sample_color * sfe_w * 0.04;
        }
        weight_sum += sfe_w;
    }
    
    let dist_fade = 1.0 - smoothstep(0.0, cutoff, dist);
    return ray * dist_fade;
}

fn sfe_chromatic(uv: vec2<f32>, size: vec2<i32>, lambda: f32) -> vec3<f32> {
    let center = vec2(0.5, 0.5);
    let dist = length(uv - center);
    let dir = normalize(uv - center);
    
    let aberration = dist * dist * 0.003 * (1.0 - lambda * 0.5);
    
    let r_uv = clamp(uv + dir * aberration * 1.0, vec2(0.0), vec2(1.0));
    let g_uv = uv;
    let b_uv = clamp(uv - dir * aberration * 1.0, vec2(0.0), vec2(1.0));
    
    let r = textureLoad(input_tex, vec2<i32>(r_uv * vec2<f32>(size)), 0).r;
    let g = textureLoad(input_tex, vec2<i32>(g_uv * vec2<f32>(size)), 0).g;
    let b = textureLoad(input_tex, vec2<i32>(b_uv * vec2<f32>(size)), 0).b;
    
    return vec3(r, g, b);
}

fn sfe_vignette(uv: vec2<f32>, lambda: f32) -> f32 {
    let center = vec2(0.5, 0.5);
    let dist = length(uv - center);
    let k2 = dist * dist * 4.0;
    return 1.0 - sfe_kernel(k2, lambda * 0.3, 0.8) * 0.3;
}

@fragment
fn fs_bloom(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<i32>(i32(params.width), i32(params.height));
    let px = vec2<i32>(in.uv * vec2<f32>(size));
    
    var color = sfe_chromatic(in.uv, size, params.sfe_lambda);
    
    var bloom = vec3<f32>(0.0);
    var weight_sum = 0.0;
    
    let adaptive_r = i32(sfe_adaptive_radius(params.sfe_lambda, params.sfe_tau, 0.01) * 10.0);
    let radius = clamp(adaptive_r, 3, 5);
    let threshold = 0.01;
    
    for (var y = -radius; y <= radius; y++) {
        for (var x = -radius; x <= radius; x++) {
            let dist2 = f32(x * x + y * y);
            let k2 = dist2 / f32(radius * radius);
            let w = sfe_kernel(k2, params.sfe_lambda, params.sfe_tau);
            
            if (w < threshold) { continue; }
            
            let sample_px = clamp(px + vec2(x * 2, y * 2), vec2(0), size - vec2(1));
            let sample_color = textureLoad(input_tex, sample_px, 0).rgb;
            
            let luminance = dot(sample_color, vec3<f32>(0.299, 0.587, 0.114));
            let bright = max(vec3(0.0), sample_color - vec3(params.bloom_threshold)) * smoothstep(params.bloom_threshold, 1.2, luminance);
            
            bloom += bright * w;
            weight_sum += w;
        }
    }
    
    if (weight_sum > 0.01) {
        bloom /= weight_sum;
    }
    
    let ao = sfe_ssao(in.uv, size, params.sfe_lambda);
    let god_rays = sfe_god_rays(in.uv, size, params.sfe_lambda, params.time);
    let vignette = sfe_vignette(in.uv, params.sfe_lambda);
    
    var final_color = color * ao;
    final_color += bloom * params.bloom_intensity;
    final_color += god_rays;
    final_color *= vignette;
    
    final_color = final_color / (final_color + vec3(1.0));
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));
    
    let grain = (hash(in.uv * 1000.0 + params.time) - 0.5) * 0.012;
    final_color += vec3(grain);
    
    return vec4<f32>(final_color, 1.0);
}
"#;

pub const SUNRAY_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;
@group(0) @binding(1) var input_tex: texture_2d<f32>;

struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) uv: vec2<f32>, }

@vertex
fn vs_sunray(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = pos[idx] * 0.5 + 0.5;
    return out;
}

fn sfe_scatter(theta: f32, lambda: f32) -> f32 {
    let cos_theta = cos(theta);
    let rayleigh = (1.0 + cos_theta * cos_theta) * 0.75;
    let mie = pow(max(cos_theta, 0.0), 8.0);
    let sfe_damp = exp(-lambda * theta * theta);
    return (rayleigh * 0.3 + mie * 0.7) * sfe_damp;
}

@fragment
fn fs_sunray(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureLoad(input_tex, vec2<i32>(in.uv * vec2<f32>(textureDimensions(input_tex))), 0);
    
    let sun_dir = normalize(-u.light_dir.xyz);
    let sun_screen = vec2<f32>(0.7, 0.85);
    
    let ray_dir = normalize(vec3<f32>(in.uv * 2.0 - 1.0, 1.0));
    let theta = acos(clamp(dot(ray_dir, sun_dir), -1.0, 1.0));
    
    let scatter = sfe_scatter(theta, u.sfe_lambda);
    
    let dawn_color = vec3<f32>(1.0, 0.6, 0.3);
    let day_color = vec3<f32>(1.0, 0.95, 0.85);
    let time_factor = sin(u.time * 0.1) * 0.5 + 0.5;
    let sun_color = mix(dawn_color, day_color, time_factor);
    
    let to_sun = sun_screen - in.uv;
    let sun_dist = length(to_sun);
    let sun_glow = exp(-sun_dist * sun_dist * 8.0) * 0.4;
    
    var rays = 0.0;
    let num_samples = 16;
    let ray_step = to_sun / f32(num_samples);
    var sample_pos = in.uv;
    for (var i = 0; i < num_samples; i++) {
        let sample_color = textureLoad(input_tex, vec2<i32>(sample_pos * vec2<f32>(textureDimensions(input_tex))), 0);
        let luminance = dot(sample_color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        rays += luminance * exp(-f32(i) * 0.15);
        sample_pos += ray_step;
    }
    rays /= f32(num_samples);
    
    let ray_intensity = rays * scatter * 0.3;
    var final_color = color.rgb + sun_color * (sun_glow + ray_intensity);
    
    return vec4<f32>(final_color, 1.0);
}
"#;

pub const SAND_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;

struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) life: f32, @location(2) turbulence: f32, }

fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
fn hash3(p: vec3<f32>) -> f32 { return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453); }

fn sfe_turbulence(p: vec3<f32>, t: f32, lambda: f32) -> vec3<f32> {
    var turb = vec3<f32>(0.0);
    var amp = 1.0;
    var freq = 1.0;
    for (var i = 0; i < 4; i++) {
        let k2 = freq * freq;
        let sfe_damp = exp(-lambda * k2 * 0.1);
        turb.x += sin(p.x * freq + t * 2.0 + p.z) * amp * sfe_damp;
        turb.y += sin(p.y * freq * 0.5 + t * 1.5) * amp * sfe_damp * 0.3;
        turb.z += cos(p.z * freq + t * 1.8 + p.x) * amp * sfe_damp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return turb;
}

@vertex
fn vs_sand(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    
    let seed = vec3<f32>(f32(iid), f32(iid) * 1.7, f32(iid) * 2.3);
    let layer = f32(iid % 12u);
    
    let spawn_radius = 35.0;
    let base_x = (hash3(seed) - 0.5) * spawn_radius * 2.0;
    let base_z = (hash3(seed + vec3(100.0, 0.0, 0.0)) - 0.5) * spawn_radius * 2.0;
    let base_y = hash3(seed + vec3(150.0, 0.0, 0.0)) * 5.0 + layer * 0.4;
    
    let cycle = 1.5 + hash3(seed + vec3(200.0, 0.0, 0.0)) * 2.5;
    let phase = hash3(seed + vec3(250.0, 0.0, 0.0)) * cycle;
    let t = fract((u.time * 1.5 + phase) / cycle);
    
    let wind_strength = 15.0 + hash3(seed + vec3(300.0, 0.0, 0.0)) * 10.0;
    let wind_angle = sin(u.time * 0.15) * 0.5;
    let wind_dir = vec3<f32>(cos(wind_angle), 0.0, sin(wind_angle));
    
    let turb = sfe_turbulence(seed * 0.03, u.time * 2.0, u.sfe_lambda) * 0.6;
    
    var world_pos = vec3<f32>(base_x, base_y, base_z);
    world_pos += wind_dir * wind_strength * t;
    world_pos += turb;
    world_pos.y += sin(t * 3.14) * 0.2;
    
    let size = 0.003 + hash3(seed + vec3(500.0, 0.0, 0.0)) * 0.004;
    let billboard_offset = quad[vid % 6u] * size;
    
    let view_right = vec3<f32>(u.view_proj[0][0], u.view_proj[1][0], u.view_proj[2][0]);
    let view_up = vec3<f32>(u.view_proj[0][1], u.view_proj[1][1], u.view_proj[2][1]);
    world_pos += view_right * billboard_offset.x + view_up * billboard_offset.y;
    
    var out: VertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.life = t;
    out.turbulence = length(turb);
    return out;
}

@fragment
fn fs_sand(in: VertexOutput) -> @location(0) vec4<f32> {
    let sand_light = vec3<f32>(0.95, 0.82, 0.6);
    let sand_dark = vec3<f32>(0.8, 0.65, 0.45);
    let sand_color = mix(sand_dark, sand_light, in.turbulence * 0.5 + 0.5);
    
    let fade_in = smoothstep(0.0, 0.05, in.life);
    let fade_out = 1.0 - smoothstep(0.9, 1.0, in.life);
    let alpha = fade_in * fade_out * 0.4;
    
    let L = normalize(-u.light_dir.xyz);
    let light = 0.5 + 0.5 * dot(vec3<f32>(0.0, 1.0, 0.0), L);
    
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog = clamp((dist - u.fog_start) / (u.fog_end - u.fog_start), 0.0, 1.0);
    var color = mix(sand_color * light, u.fog_color.rgb, fog);
    
    return vec4<f32>(color, alpha);
}
"#;

pub const FLAG_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;
@group(1) @binding(0) var sfe_shadow: texture_2d<f32>;

struct VertexOutput { 
    @builtin(position) clip_position: vec4<f32>, 
    @location(0) world_pos: vec3<f32>, 
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) shadow_pos: vec4<f32>,
}

fn sfe_cloth_wave(uv: vec2<f32>, t: f32, lambda: f32) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var freq = 1.0;
    for (var i = 0; i < 4; i++) {
        let k2 = freq * freq;
        let sfe_damp = exp(-lambda * k2 * k2 * 0.005);
        let phase = uv.x * freq * 3.14159 * 2.0 + t * (2.0 + f32(i) * 0.5);
        sum += sin(phase) * amp * sfe_damp * uv.x;
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum;
}

@vertex
fn vs_flag(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let seg_x = 16u;
    let seg_y = 10u;
    let total_verts = seg_x * seg_y * 6u;
    
    let quad_idx = vid / 6u;
    let vert_in_quad = vid % 6u;
    let qx = quad_idx % (seg_x - 1u);
    let qy = quad_idx / (seg_x - 1u);
    
    let corner_offsets = array<vec2<u32>, 6>(vec2(0u,0u),vec2(1u,0u),vec2(1u,1u),vec2(0u,0u),vec2(1u,1u),vec2(0u,1u));
    let corner = corner_offsets[vert_in_quad];
    let vx = qx + corner.x;
    let vy = qy + corner.y;
    
    let uv = vec2<f32>(f32(vx) / f32(seg_x - 1u), f32(vy) / f32(seg_y - 1u));
    
    let flag_width = 1.5;
    let flag_height = 1.0;
    let pole_x = -3.0 + f32(iid) * 6.0;
    let pole_y = 2.0;
    let pole_z = 0.0;
    
    let wave = sfe_cloth_wave(uv, u.time * 3.0, u.sfe_lambda) * 0.2;
    let wave_y = sfe_cloth_wave(uv + vec2(0.5, 0.0), u.time * 2.5, u.sfe_lambda) * 0.05;
    
    let local_x = uv.x * flag_width;
    let local_y = (1.0 - uv.y) * flag_height + wave_y;
    let local_z = wave;
    
    let world_pos = vec3<f32>(pole_x + local_x, pole_y + local_y, pole_z + local_z);
    
    let e = 0.01;
    let wave_dx = sfe_cloth_wave(uv + vec2(e, 0.0), u.time * 3.0, u.sfe_lambda) * 0.2;
    let wave_dy = sfe_cloth_wave(uv + vec2(0.0, e), u.time * 3.0, u.sfe_lambda) * 0.2;
    let tangent = normalize(vec3<f32>(flag_width * e, 0.0, wave_dx - wave));
    let bitangent = normalize(vec3<f32>(0.0, -flag_height * e, wave_dy - wave));
    let normal = normalize(cross(tangent, bitangent));
    
    var out: VertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normal;
    out.uv = uv;
    out.shadow_pos = u.light_view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

fn sfe_shadow_flag(shadow_pos: vec4<f32>) -> f32 {
    let proj = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));
    let size = i32(u.shadow_size);
    let px = vec2<i32>(i32(uv.x * f32(size)), i32(uv.y * f32(size)));
    let stored_depth = textureLoad(sfe_shadow, clamp(px, vec2(0), vec2(size - 1)), 0).r;
    let current_depth = proj.z - 0.005;
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    return select(1.0, select(0.3, 1.0, current_depth <= stored_depth), in_bounds);
}

@fragment
fn fs_flag(in: VertexOutput) -> @location(0) vec4<f32> {
    let stripe_count = 3.0;
    let stripe = floor(in.uv.y * stripe_count);
    let stripe_colors = array<vec3<f32>, 3>(
        vec3<f32>(0.9, 0.1, 0.1),
        vec3<f32>(0.95, 0.95, 0.95),
        vec3<f32>(0.1, 0.2, 0.8)
    );
    var base_color = stripe_colors[i32(stripe) % 3];
    
    let N = normalize(in.world_normal);
    let L = normalize(-u.light_dir.xyz);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);
    
    let shadow = sfe_shadow_flag(in.shadow_pos);
    
    let ambient = 0.2;
    let NdotL = dot(N, L);
    let diffuse = max(NdotL, 0.0) * 0.6 * shadow;
    let back_light = max(-NdotL, 0.0) * 0.2;
    let spec = pow(max(dot(N, H), 0.0), 32.0) * 0.3 * shadow;
    
    let cloth_scatter = (1.0 - abs(NdotL)) * 0.15;
    
    var color = base_color * (ambient + diffuse + back_light + cloth_scatter) + vec3<f32>(spec);
    
    let dist = length(u.camera_pos.xyz - in.world_pos);
    let fog = clamp((dist - u.fog_start) / (u.fog_end - u.fog_start), 0.0, 1.0);
    color = mix(color, u.fog_color.rgb, fog);
    
    return vec4<f32>(color, 1.0);
}
"#;

pub const SNOW_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;

struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) alpha: f32, }

fn pcg(n: u32) -> u32 {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}
fn rand(id: u32, offset: u32) -> f32 {
    return f32(pcg(id + offset * 19482u)) / 4294967295.0;
}

@vertex
fn vs_snow(@builtin(instance_index) iid: u32) -> VertexOutput {
    let r = u.sfe_lambda;
    let spawn = 60.0;
    let px = (rand(iid, 0u) - 0.5) * spawn * 2.0;
    let pz = (rand(iid, 1u) - 0.5) * spawn * 2.0;
    let h0 = 8.0 + rand(iid, 2u) * 20.0;
    
    let spd = 0.6 + rand(iid, 3u) * 0.8;
    let cyc = h0 / spd;
    let ph = rand(iid, 4u) * cyc;
    let t = fract((u.time * 0.5 + ph) / cyc);
    
    let wx = sin(u.time * 0.3 + rand(iid, 5u) * 6.28) * 1.5;
    let wz = cos(u.time * 0.25 + rand(iid, 6u) * 6.28) * 1.2;
    
    let k = 1.0 + rand(iid, 7u) * 2.0;
    let sfe = exp(-r * k * k * 0.1);
    let drift_x = sin(t * 3.14 * k + rand(iid, 8u) * 6.28) * sfe * 2.0;
    let drift_z = cos(t * 3.14 * k * 0.7 + rand(iid, 9u) * 6.28) * sfe * 1.5;
    
    let x = px + wx * t + drift_x;
    let y = h0 * (1.0 - t);
    let z = pz + wz * t + drift_z;
    
    var out: VertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(x, y, z, 1.0);
    let fade = smoothstep(0.0, 0.02, t) * (1.0 - smoothstep(0.98, 1.0, t));
    let dist = length(vec3(x, y, z) - u.camera_pos.xyz);
    let fog = 1.0 - clamp(dist / 60.0, 0.0, 0.9);
    out.alpha = fade * fog * 0.9;
    return out;
}

@fragment
fn fs_snow(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, in.alpha);
}
"#;

pub const FIRE_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;

struct VertexOutput { 
    @builtin(position) clip_position: vec4<f32>, 
    @location(0) uv: vec2<f32>,
    @location(1) life: f32, 
    @location(2) heat: f32,
    @location(3) world_pos: vec3<f32>,
    @location(4) size: f32,
}

fn pcg(n: u32) -> u32 {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}
fn rand(id: u32, off: u32) -> f32 { return f32(pcg(id + off * 7919u)) / 4294967295.0; }

fn hash2(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let sm = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2(1.0, 0.0)), sm.x),
               mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), sm.x), sm.y);
}

fn sfe_fbm(p: vec2<f32>, t: f32, lambda: f32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    var pos = p;
    for (var i = 0; i < 5; i++) {
        let k2 = freq * freq;
        let sfe = exp(-lambda * k2 * 0.005);
        sum += noise(pos + vec2(t * 0.8, -t * 1.2)) * amp * sfe;
        amp *= 0.5;
        freq *= 2.0;
        pos = pos * 2.0 + vec2(1.7, 9.2);
    }
    return sum;
}

@vertex
fn vs_fire(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    let uvs = array<vec2<f32>, 6>(vec2(0.0,0.0),vec2(1.0,0.0),vec2(1.0,1.0),vec2(0.0,0.0),vec2(1.0,1.0),vec2(0.0,1.0));
    
    let fire_pos = vec3<f32>(5.0, 0.25, -3.0);
    let r0 = rand(iid, 0u);
    let r1 = rand(iid, 1u);
    let r2 = rand(iid, 2u);
    let r3 = rand(iid, 3u);
    let r4 = rand(iid, 4u);
    
    let is_spark = iid > 6000u;
    let is_ember = iid > 7000u;
    
    let angle = r0 * 6.28318;
    var rad: f32;
    if (is_ember) { rad = pow(r1, 0.3) * 0.6; }
    else if (is_spark) { rad = pow(r1, 0.5) * 0.25; }
    else { rad = pow(r1, 0.7) * 0.12; }
    
    var life_span: f32;
    if (is_ember) { life_span = 1.5 + r2 * 1.0; }
    else if (is_spark) { life_span = 0.4 + r2 * 0.3; }
    else { life_span = 0.25 + r2 * 0.35; }
    
    let phase = r3 * life_span;
    let t = fract((u.time * 1.5 + phase) / life_span);
    
    let turb_str = select(0.15, select(0.3, 0.08, is_spark), is_ember);
    let turb_x = (sfe_fbm(vec2(r0 * 10.0, t * 2.0), u.time * 2.0, u.sfe_lambda) - 0.5) * turb_str;
    let turb_z = (sfe_fbm(vec2(r1 * 10.0 + 50.0, t * 2.0), u.time * 2.0, u.sfe_lambda) - 0.5) * turb_str;
    
    var rise_speed: f32;
    if (is_ember) { rise_speed = 0.3 + r4 * 0.2; }
    else if (is_spark) { rise_speed = 1.5 + r4 * 1.0; }
    else { rise_speed = 0.8 + r4 * 0.5; }
    
    let spread = select(t * t * 0.2, select(t * 0.15, t * t * 0.5, is_spark), is_ember);
    
    var pos = fire_pos;
    pos.x += cos(angle) * rad * (1.0 + spread) + turb_x * (1.0 + t);
    pos.y += t * rise_speed;
    pos.z += sin(angle) * rad * (1.0 + spread) + turb_z * (1.0 + t);
    
    var base_size: f32;
    if (is_ember) { base_size = 0.008 + r2 * 0.006; }
    else if (is_spark) { base_size = 0.012 + r2 * 0.015; }
    else { base_size = 0.025 + r2 * 0.04; }
    
    let flicker = 0.8 + 0.2 * sin(u.time * 25.0 + r0 * 100.0);
    let size = base_size * (1.0 - t * 0.6) * flicker;
    
    let local = quad[vid % 6u] * size;
    let view_right = vec3<f32>(u.view_proj[0][0], u.view_proj[1][0], u.view_proj[2][0]);
    let view_up = vec3<f32>(u.view_proj[0][1], u.view_proj[1][1], u.view_proj[2][1]);
    pos += view_right * local.x + view_up * local.y;
    
    var out: VertexOutput;
    out.clip_position = u.view_proj * vec4<f32>(pos, 1.0);
    out.uv = uvs[vid % 6u];
    out.life = t;
    out.heat = select(pow(1.0 - t, 1.2), select(pow(1.0 - t, 0.8), pow(1.0 - t, 2.0), is_spark), is_ember);
    out.world_pos = pos;
    out.size = size;
    return out;
}

@fragment
fn fs_fire(in: VertexOutput) -> @location(0) vec4<f32> {
    let center = in.uv - vec2(0.5);
    let dist = length(center);
    
    let flame_distort = sfe_fbm(in.uv * 3.0 + vec2(0.0, -u.time * 3.0), u.time * 2.0, u.sfe_lambda) * 0.3;
    let dist_mod = dist + flame_distort * (1.0 - in.uv.y);
    
    var shape: f32;
    if (in.size < 0.015) {
        shape = 1.0 - smoothstep(0.0, 0.4, dist);
    } else {
        let tear = 1.0 - in.uv.y * 0.5 - abs(center.x) * 0.8;
        shape = (1.0 - smoothstep(0.0, 0.45, dist_mod)) * max(tear, 0.3);
    }
    
    let h = in.heat;
    
    let white = vec3<f32>(1.0, 0.98, 0.92);
    let bright_yellow = vec3<f32>(1.0, 0.9, 0.4);
    let yellow = vec3<f32>(1.0, 0.7, 0.15);
    let orange = vec3<f32>(1.0, 0.35, 0.02);
    let red = vec3<f32>(0.85, 0.12, 0.0);
    let dark = vec3<f32>(0.3, 0.03, 0.0);
    
    var color: vec3<f32>;
    if (h > 0.9) { color = mix(bright_yellow, white, (h - 0.9) / 0.1); }
    else if (h > 0.7) { color = mix(yellow, bright_yellow, (h - 0.7) / 0.2); }
    else if (h > 0.45) { color = mix(orange, yellow, (h - 0.45) / 0.25); }
    else if (h > 0.2) { color = mix(red, orange, (h - 0.2) / 0.25); }
    else { color = mix(dark, red, h / 0.2); }
    
    let inner_glow = exp(-dist * 4.0) * h * h;
    color += vec3<f32>(inner_glow * 0.4, inner_glow * 0.15, 0.0);
    
    let flicker = 0.9 + 0.1 * sin(u.time * 35.0 + in.world_pos.x * 50.0 + in.world_pos.y * 30.0);
    color *= flicker;
    
    let fade_in = smoothstep(0.0, 0.08, in.life);
    let fade_out = 1.0 - smoothstep(0.7, 1.0, in.life);
    var alpha = shape * fade_in * fade_out;
    
    if (in.size < 0.015) {
        alpha *= 0.7 + h * 0.3;
    } else {
        alpha *= 0.5 + h * 0.5;
    }
    
    return vec4<f32>(color, alpha);
}
"#;

pub const SSAO_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, }

fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

@vertex
fn vs_ssao(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    let uvs = array<vec2<f32>, 6>(vec2(0.0,1.0),vec2(1.0,1.0),vec2(1.0,0.0),vec2(0.0,1.0),vec2(1.0,0.0),vec2(0.0,0.0));
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fs_ssao(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(depth_tex));
    let px = vec2<i32>(in.uv * tex_size);
    let depth = textureLoad(depth_tex, px, 0);
    
    if (depth >= 1.0) { return vec4(1.0); }
    
    let radius = 0.02;
    let samples = 16;
    var occlusion = 0.0;
    
    let rand_angle = hash(in.uv * tex_size * 0.25) * 6.28318;
    
    for (var i = 0; i < samples; i++) {
        let angle = rand_angle + f32(i) * 6.28318 / f32(samples);
        let dist = (f32(i) + 0.5) / f32(samples);
        
        let k2 = dist * dist * 4.0;
        let sfe_weight = exp(-u.sfe_lambda * k2 * 0.1);
        
        let offset = vec2(cos(angle), sin(angle)) * radius * dist;
        let sample_uv = in.uv + offset;
        
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) { continue; }
        
        let sample_px = vec2<i32>(sample_uv * tex_size);
        let sample_depth = textureLoad(depth_tex, sample_px, 0);
        
        let depth_diff = depth - sample_depth;
        if (depth_diff > 0.0001 && depth_diff < 0.01) {
            occlusion += sfe_weight;
        }
    }
    
    occlusion = 1.0 - (occlusion / f32(samples));
    occlusion = pow(occlusion, 2.0);
    
    return vec4(occlusion, occlusion, occlusion, 1.0);
}
"#;

pub const VOLUMETRIC_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;
@group(1) @binding(0) var sfe_shadow: texture_2d<f32>;

struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, }

fn hash3(p: vec3<f32>) -> f32 { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let sm = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i), hash3(i + vec3(1.0, 0.0, 0.0)), sm.x),
                   mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), sm.x), sm.y),
               mix(mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), sm.x),
                   mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), sm.x), sm.y), sm.z);
}

fn sfe_fbm3d(p: vec3<f32>, lambda: f32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    var pos = p;
    for (var i = 0; i < 4; i++) {
        let k2 = freq * freq;
        let sfe = exp(-lambda * k2 * 0.02);
        sum += noise3d(pos) * amp * sfe;
        amp *= 0.5;
        freq *= 2.0;
        pos = pos * 2.0 + vec3(1.7, 9.2, 5.3);
    }
    return sum;
}

@vertex
fn vs_volumetric(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    let uvs = array<vec2<f32>, 6>(vec2(0.0,1.0),vec2(1.0,1.0),vec2(1.0,0.0),vec2(0.0,1.0),vec2(1.0,0.0),vec2(0.0,0.0));
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fs_volumetric(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = normalize(vec3((in.uv - 0.5) * 2.0, -1.0));
    let L = normalize(-u.light_dir.xyz);
    
    let steps = 24;
    var accumulated = 0.0;
    var transmittance = 1.0;
    
    for (var i = 0; i < steps; i++) {
        let t = f32(i) * 1.5;
        let pos = u.camera_pos.xyz + ray_dir * t;
        
        if (pos.y < -5.0 || pos.y > 20.0) { continue; }
        
        let density = sfe_fbm3d(pos * 0.08 + vec3(u.time * 0.05, 0.0, 0.0), u.sfe_lambda);
        let height_fade = exp(-abs(pos.y - 5.0) * 0.1);
        let final_density = max(density - 0.3, 0.0) * height_fade * 0.3;
        
        if (final_density > 0.001) {
            let scatter = final_density * transmittance;
            accumulated += scatter;
            transmittance *= exp(-final_density * 1.5);
        }
        
        if (transmittance < 0.01) { break; }
    }
    
    return vec4(u.fog_color.rgb, accumulated * 0.15);
}
"#;

pub const ATMOSPHERIC_SHADER: &str = r#"
struct GlobalUniforms { view_proj: mat4x4<f32>, light_view_proj: mat4x4<f32>, light_dir: vec4<f32>, camera_pos: vec4<f32>, fog_color: vec4<f32>, time: f32, fog_start: f32, fog_end: f32, sfe_lambda: f32, sfe_tau: f32, shadow_size: f32, }
@group(0) @binding(0) var<uniform> u: GlobalUniforms;

struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) world_dir: vec3<f32>, }

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 0.75 * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = (1.0 - g2);
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return num / (4.0 * 3.14159 * denom);
}

@vertex
fn vs_atmosphere(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    let uvs = array<vec2<f32>, 6>(vec2(0.0,1.0),vec2(1.0,1.0),vec2(1.0,0.0),vec2(0.0,1.0),vec2(1.0,0.0),vec2(0.0,0.0));
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    out.world_dir = normalize(vec3((uvs[vid] - 0.5) * 2.0, -1.0));
    return out;
}

@fragment
fn fs_atmosphere(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_dir = normalize(-u.light_dir.xyz);
    let ray_dir = normalize(in.world_dir);
    
    let cos_theta = dot(ray_dir, sun_dir);
    let rayleigh = rayleigh_phase(cos_theta);
    let mie = mie_phase(cos_theta, 0.76);
    
    let beta_r = vec3(5.8e-3, 13.5e-3, 33.1e-3);
    let beta_m = vec3(21e-3);
    
    let k2_r = 1.0 / (0.650 * 0.650);
    let k2_g = 1.0 / (0.570 * 0.570);
    let k2_b = 1.0 / (0.475 * 0.475);
    let sfe_scatter = vec3(
        exp(-u.sfe_lambda * k2_r * 0.0005),
        exp(-u.sfe_lambda * k2_g * 0.0005),
        exp(-u.sfe_lambda * k2_b * 0.0005)
    );
    
    let scatter_r = beta_r * rayleigh * sfe_scatter;
    let scatter_m = beta_m * mie;
    
    let height = max(ray_dir.y, 0.0);
    let optical_depth = 1.0 / (height + 0.15);
    
    var sky_color = (scatter_r + scatter_m) * optical_depth * 12.0;
    
    let sun_disk = smoothstep(0.9995, 0.9999, cos_theta);
    sky_color += vec3(1.0, 0.95, 0.85) * sun_disk * 8.0;
    
    let horizon_glow = exp(-height * 4.0) * max(cos_theta + 0.2, 0.0);
    sky_color += vec3(1.0, 0.5, 0.2) * horizon_glow * 0.4;
    
    sky_color = 1.0 - exp(-sky_color);
    
    return vec4(sky_color, 1.0);
}
"#;

