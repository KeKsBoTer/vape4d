const FILTER_NEAREST:u32 = 0;
const FILTER_LINEAR:u32 = 1;

const PI:f32 = 3.1415926535897932384626433832795;
const TWO_PI:f32 = 6.283185307179586476925286766559;

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};

struct Settings {
    volume_aabb: Aabb,
    time: f32,
    step_size: f32,
    temporal_filter: u32,
    spatial_filter: u32,
    
    distance_scale: f32,
    vmin: f32,
    vmax: f32,
    gamma_correction: u32
}


struct Aabb {
    @align(16) min: vec3<f32>,
    @align(16) max: vec3<f32>,
}

struct Ray {
    orig: vec3<f32>,
    dir: vec3<f32>
};

/// adapted from https://github.com/evanw/webgl-path-tracing/blob/master/webgl-path-tracing.js
fn intersectAABB(ray: Ray, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let tMin = (box_min - ray.orig) / ray.dir;
    let tMax = (box_max - ray.orig) / ray.dir;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2<f32>(tNear, tFar);
}

// ray is created based on view and proj matrix so
// that it matches the rasterizer used for drawing other stuff
fn create_ray(view_inv: mat4x4<f32>, proj_inv: mat4x4<f32>, px: vec2<f32>) -> Ray {
    var far = vec4<f32>((px * 2. - (1.)), -1., 1.);
    far.y *= -1.;
    // depth prepass location
    var far_w = view_inv * proj_inv * far;
    far_w /= far_w.w + 1e-4;


    var near = vec4<f32>((px * 2. - (1.)), 1., 1.);
    near.y *= -1.;
    // depth prepass location
    var near_w = view_inv * proj_inv * near;
    near_w /= near_w.w + 1e-4;

    return Ray(
        near_w.xyz,
        normalize(far_w.xyz - near_w.xyz),
    );
}

@group(0) @binding(0)
var volume : texture_3d<f32>;
@group(0) @binding(1)
var volume_next : texture_3d<f32>;
@group(0) @binding(2)
var volume_sampler: sampler;

@group(0) @binding(3)
var<uniform> camera: CameraUniforms;

@group(0) @binding(4)
var<uniform> settings: Settings;

@group(1) @binding(0)
var cmap : texture_2d<f32>;
@group(1) @binding(1)
var cmap_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOut {

    // creates two vertices that cover the whole screen
    let xy = vec2<f32>(
        f32(in_vertex_index % 2u == 0u),
        f32(in_vertex_index < 2u)
    );
    return VertexOut(vec4<f32>(xy * 2. - (1.), 0., 1.), vec2<f32>(xy.x, 1. - xy.y));
}

// performs a step and returns the NDC cooidinate for the volume sampling
fn next_pos(pos: ptr<function,vec3<f32>>, step_size: f32, ray_dir: vec3<f32>) -> vec3<f32> {
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    let sample_pos = ((*pos) - aabb.min) / aabb_size;
    *pos += ray_dir * step_size;
    return sample_pos;
}

fn sample_volume(pos: vec3<f32>) -> f32 {
    //  origin is in bottom left corner so we need to flip y 
    let pos_m = vec3<f32>(pos.x, 1. - pos.y, pos.z);
    let sample_curr = textureSampleLevel(volume, volume_sampler, pos_m, 0.).r;
    let sample_next = textureSampleLevel(volume_next, volume_sampler, pos_m, 0.).r;
    if settings.temporal_filter == FILTER_NEAREST {
        return sample_curr;
    } else {
        return mix(sample_curr, sample_next, settings.time);
    }
}

fn sample_cmap(value: f32) -> vec4<f32> {
    let value_n = (value - settings.vmin) / (settings.vmax - settings.vmin);
    return textureSampleLevel(cmap, cmap_sampler, vec2<f32>(value_n, 0.5), 0.);
}

/// random float between 0 and 1
fn rand(co:vec2<f32> ) -> f32{
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}


struct DDAState {
    t_max: vec3<f32>,
    t_delta: vec3<f32>,
    step: vec3<i32>,
    voxel_index: vec3<i32>,
    step_dir_curr: vec3<i32>,
}

fn get_voxel_segment_length(ray_origin: vec3<f32>, ray_direction: vec3<f32>, voxel_index: vec3<i32>) -> f32 {
    let lower = vec3<f32>(voxel_index);
    let upper = vec3<f32>(voxel_index + 1);
    var t_near = -1e7;
    var t_far = 1e7;
    for (var i: i32 = 0; i < 3; i += 1) {
        if abs(ray_direction[i]) < 1e-4 {
            if ray_origin[i] < lower[i] || ray_origin[i] > upper[i] {
                return 0.0;
            }
        } else {
            var t0 = (lower[i] - ray_origin[i]) / ray_direction[i];
            var t1 = (upper[i] - ray_origin[i]) / ray_direction[i];
            if (t0 > t1) {
                let tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            if t0 > t_near {
                t_near = t0;
            }
            if t1 < t_far {
                t_far = t1;
            }
            if t_near > t_far || t_far < 0 {
                return 0.0;
            }
        }
    }
    return t_far - t_near;
}

fn next_pos_dda(
        pos: ptr<function,vec3<f32>>, step_size: ptr<function,f32>, ray_orig: vec3<f32>, ray_dir: vec3<f32>,
        state: ptr<function,DDAState>) -> vec3<f32> {
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;

    let sample_pos = (vec3<f32>((*state).voxel_index) + 0.5) / vec3<f32>(textureDimensions(volume));

    *step_size = get_voxel_segment_length(ray_orig, ray_dir, (*state).voxel_index) / f32(textureDimensions(volume).x) * aabb_size.x;

    if ((*state).t_max.x < (*state).t_max.y) {
        if ((*state).t_max.x < (*state).t_max.z) {
            (*state).voxel_index.x += (*state).step.x;
            (*state).t_max.x += (*state).t_delta.x;
            (*state).step_dir_curr = vec3<i32>((*state).step.x, 0, 0);
          } else {
            (*state).voxel_index.z += (*state).step.z;
            (*state).t_max.z += (*state).t_delta.z;
            (*state).step_dir_curr = vec3<i32>(0, 0, (*state).step.z);
        }
    } else {
        if ((*state).t_max.y < (*state).t_max.z) {
            (*state).voxel_index.y += (*state).step.y;
            (*state).t_max.y += (*state).t_delta.y;
            (*state).step_dir_curr = vec3<i32>(0, (*state).step.y, 0);
        } else {
            (*state).voxel_index.z += (*state).step.z;
            (*state).t_max.z += (*state).t_delta.z;
            (*state).step_dir_curr = vec3<i32>(0, 0, (*state).step.z);
        }
    }

    *pos += ray_dir * (*step_size);

    return sample_pos;
}

fn intersect_aabb_dir(ray_origin: vec3<f32>, ray_direction: vec3<f32>, lower: vec3<f32>, upper: vec3<f32>) -> i32 {
    var dir = 0;
    var t_near_best = -1e7;
    for (var i: i32 = 0; i < 3; i += 1) {
        var t0 = (lower[i] - ray_origin[i]) / ray_direction[i];
        var t1 = (upper[i] - ray_origin[i]) / ray_direction[i];
        if (t0 > t1) {
            let tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        let t_near = t0;
        if t_near > t_near_best {
            t_near_best = t_near;
            dir = i;
        }
    }
    return dir;
}

// traces ray trough volume and returns color
fn trace_ray(ray_in: Ray,pixel_pos:vec2<f32>) -> vec4<f32> {
    
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    var ray = ray_in;
    // find closest point on volume
    let aabb_min = (aabb.min + aabb_size); //  zxy for tensorf alignment
    let aabb_max = (aabb.max - aabb_size); //  zxy for tensorf alignment
    let intersec = intersectAABB(ray, aabb_min, aabb_max);

    if intersec.x > intersec.y {
        return vec4<f32>(0.);
    }

    let start_cam_pos = ray.orig;
    let start = max(0., intersec.x) + 1e-4;
    ray.orig += start * ray.dir;

    var iters = 0u;
    var color = vec3<f32>(0.);
    var transmittance = 1.;

    let volume_size = textureDimensions(volume);

    var distance_scale = settings.distance_scale;

    var pos = ray.orig;

    let early_stopping_t = 1. / 255.;

    var state: DDAState;
    let step_size_g = (settings.step_size) ;//+ (100.*settings.step_size) * rand(pixel_pos);
    var sample_pos: vec3<f32>;
    var start_point: vec3<f32>;

    if settings.spatial_filter == FILTER_NEAREST  {
        start_point = (ray.orig - aabb.min) / aabb_size * vec3<f32>(volume_size);
        let end_point = (start_cam_pos + (intersec.y - 1e-4) * ray.dir - aabb.min) / aabb_size * vec3<f32>(volume_size);

        for (var i: i32 = 0; i < 3; i += 1) {
            state.step[i] = i32(sign(end_point[i] - start_point[i]));
            if state.step[i] != 0 {
                state.t_delta[i] = min(f32(state.step[i]) / (end_point[i] - start_point[i]), 1e7);
            } else {
                state.t_delta[i] = 1e7; // inf
            }
            if state.step[i] > 0 {
                state.t_max[i] = state.t_delta[i] * (1.0 - fract(start_point[i]));
            } else {
                state.t_max[i] = state.t_delta[i] * fract(start_point[i]);
            }
            state.voxel_index[i] = i32(floor(start_point[i]));
        }

        state.step_dir_curr = vec3<i32>(0, 0, 0);
        let dir_init = intersect_aabb_dir(start_point, ray.dir, aabb_min, aabb_max);
        state.step_dir_curr[dir_init] = state.step[dir_init];
    }


    var step_size: f32;
    loop{
        if settings.spatial_filter == FILTER_NEAREST {
            sample_pos = next_pos_dda(&pos, &step_size, start_point, ray.dir, &state);
        } else {
            step_size = step_size_g;
            sample_pos = next_pos(&pos, step_size_g, ray.dir);
        }

        let slice_test = any(sample_pos < vec3<f32>(0.)) || any(sample_pos > vec3<f32>(1.)) ;
        if slice_test || iters > 10000 {
            break;
        }

        let sample = sample_volume(sample_pos.xyz);
        let color_tf = sample_cmap(sample);
        // we try to avoid values that are exactly one as this can cause artifacts
        let sigma = color_tf.a * (1. - 1e-6);

        if sigma > 0. {
            var sample_color = color_tf.rgb;
            let a_i = 1. - pow(1. - sigma, step_size * distance_scale);
            color += transmittance * a_i * sample_color;
            transmittance *= 1. - a_i;

            if transmittance <= early_stopping_t {
                break;
            }
        }
        iters += 1u;
    }
    return vec4<f32>(color, 1. - transmittance);
}

fn gamma_correction(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(pow(color.rgb, vec3<f32>(1. / 2.2)), color.a);
}


@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    let r_pos = vec2<f32>(vertex_in.tex_coord.x, 1. - vertex_in.tex_coord.y);
    let ray = create_ray(camera.view_inv, camera.proj_inv, r_pos);
    var color = trace_ray(ray,r_pos);
    if settings.gamma_correction == 1u {
        color = fromLinear(color);
    }
    return color;
}


fn fromLinear(color: vec4<f32>) -> vec4<f32> {
    let cutoff = color.rgb < vec3<f32>(0.0031308);
    let higher = vec3<f32>(1.055) * pow(color.rgb, vec3<f32>(1.0 / 2.4)) - 0.055;
    let lower = color.rgb * 12.92;

    return vec4<f32>(mix(higher, lower, vec3<f32>(cutoff)), color.a);
}
