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
    clipping: Aabb,
    time: f32,
    time_steps: u32,
    step_size: f32,
    temporal_filter: u32,
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
fn next_pos(pos: ptr<function,vec3<f32>>, step_size: f32, ray_dir: vec3<f32>) -> vec4<f32> {
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    let sample_pos = ((*pos) - aabb.min) / aabb_size;
    *pos += ray_dir * step_size;
    return vec4<f32>(
        sample_pos,
        step_size,
    );
}

fn sample_volume(pos: vec3<f32>) -> f32 {
    //  origin is in bottom left corner so we need to flip y 
    let pos_m = vec3<f32>(pos.x, 1. - pos.y, pos.z);
    let sample_curr = textureSampleLevel(volume, volume_sampler, pos_m, 0.).r;
    let sample_next = textureSampleLevel(volume_next, volume_sampler, pos_m, 0.).r;
    if settings.temporal_filter == FILTER_NEAREST {
        return sample_curr;
    } else {
        let time_fraction = fract(settings.time * f32(settings.time_steps - (1)));
        return mix(sample_curr, sample_next, time_fraction);
    }
}

fn sample_cmap(value: f32) -> vec4<f32> {
    let value_n = (value - settings.vmin) / (settings.vmax - settings.vmin);
    return textureSampleLevel(cmap, cmap_sampler, vec2<f32>(value_n, 0.5), 0.);
}


// traces ray trough volume and returns color
fn trace_ray(ray_in: Ray) -> vec4<f32> {
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    var ray = ray_in;
    let slice_min = settings.clipping.min;
    let slice_max = settings.clipping.max;
    // find closest point on volume
    let aabb_min = (aabb.min + (slice_min * aabb_size)); //  zxy for tensorf alignment
    let aabb_max = (aabb.max - ((1. - slice_max) * aabb_size)); //  zxy for tensorf alignment
    let intersec = intersectAABB(ray, aabb_min, aabb_max);

    if intersec.x > intersec.y {
        return vec4<f32>(0.);
    }

    let start = max(0., intersec.x) + 1e-4;
    ray.orig += start * ray.dir;

    var iters = 0u;
    var color = vec3<f32>(0.);
    var transmittance = 0.;

    let volume_size = textureDimensions(volume);

    var distance_scale = settings.distance_scale;

    var pos = ray.orig;

    let early_stopping_t = 1. / 255.;
    let step_size_g = settings.step_size;
    var sample_pos: vec4<f32>;
    loop{
        sample_pos = next_pos(&pos, step_size_g, ray.dir);
        let step_size = sample_pos.w;

        let sample = sample_volume(sample_pos.xyz);
        let color_tf = sample_cmap(sample);
        let sigma = color_tf.a;

        if sigma > 0. {
            var sample_color = color_tf.rgb;
            let a_i = log(1. / (1. - sigma + 1e-4)) * step_size * distance_scale;
            color += exp(-transmittance) * (1. - exp(-a_i)) * sample_color;
            transmittance += a_i;

            if exp(-transmittance) <= early_stopping_t {
                break;
            }
        }
        // check if within slice
        let slice_test = any(sample_pos.xyz < settings.clipping.min) || any(sample_pos.xyz > settings.clipping.max) ;

        if slice_test || iters > 10000 {
            break;
        }
        iters += 1u;
    }
    let a = 1. - exp(-transmittance);
    return vec4<f32>(color, a);
}

fn gamma_correction(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(pow(color.rgb, vec3<f32>(1. / 2.2)), color.a);
}


@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    let r_pos = vec2<f32>(vertex_in.tex_coord.x, 1. - vertex_in.tex_coord.y);
    let ray = create_ray(camera.view_inv, camera.proj_inv, r_pos);
    var color = trace_ray(ray);
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
