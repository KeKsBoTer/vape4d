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
    use_dda: u32,
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
fn create_ray(view_inv: mat4x4<f32>, proj_inv: mat4x4<f32>, px: vec2<f32>, depth: f32) -> Ray {
   // TODO use depth
    // [0,1] -> [-1,1]
    var point_screen = vec4<f32>(px * 2. - (1.), 0., 1.);
    point_screen.y *= -1.;
    // depth prepass location
    let screen_ray = normalize((proj_inv * point_screen).xyz);
    let dir = view_inv * vec4<f32>(screen_ray, 0.);

    let orig = view_inv[3].xyz;
    return Ray(
        orig,
        dir.xyz,
    );
}

struct DDAState {
    mask: vec3<bool>,
    side_dist: vec3<f32>,
    delta_dist: vec3<f32>,
    ray_step: vec3<f32>,
    last_distance: f32,
}

@group(0) @binding(0)
var volume : texture_3d<f32>;
@group(0) @binding(1)
var volume_next : texture_3d<f32>;
@group(1) @binding(0)
var<uniform> camera: CameraUniforms;
@group(2) @binding(0)
var<uniform> settings: Settings;
@group(3) @binding(0)
var volume_sampler: sampler;

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

// performs a dda step and returns the next sampling position (xyz component) and step size (w component)
fn next_pos_dda(pos: ptr<function,vec3<f32>>, state: ptr<function,DDAState>) -> vec4<f32> {
    let curr_pos = *pos;
    var st = *state;

    st.mask = st.side_dist.xyz <= min(st.side_dist.yzx, st.side_dist.zxy);
    st.side_dist += vec3<f32>(st.mask) * st.delta_dist;

    let d = length(vec3<f32>(st.mask) * (st.side_dist - st.delta_dist));

    let step_size = d - st.last_distance;
    st.last_distance = d;

    (*pos) += vec3<f32>(st.mask) * st.ray_step;
    *state = st;

    // we want to sample at the middle of each voxel
    let inv_size = 1. / vec3<f32>(textureDimensions(volume));
    let sample_pos = curr_pos * inv_size;
    return vec4<f32>(
        sample_pos,
        step_size
    );
}

// performs a step and returns the NDC cooidinate for the volume sampling
fn next_pos(pos: ptr<function,vec3<f32>>, step_size: f32, ray_dir: vec3<f32>) -> vec4<f32> {
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    let sample_pos = ((*pos) - aabb.min) / aabb_size; // zxy for tensorf alignment
    *pos += ray_dir * step_size;
    return vec4<f32>(
        sample_pos,
        step_size,
    );
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

    let start = max(0., intersec.x);
    ray.orig += start * ray.dir;

    var iters = 0u;
    var color = vec3<f32>(0.);
    var transmittance = 0.;

    let volume_size = textureDimensions(volume);

    var distance_scale = 1.;
    let use_dda = settings.use_dda;
    if bool(use_dda) {
        let inv_size = 1. / vec3<f32>(volume_size);
        distance_scale *= length(inv_size);
    }


    var pos = ray.orig;
    var state: DDAState;

    if bool(use_dda) {
        let pos_f = (ray.orig - aabb.min) / aabb_size * (vec3<f32>(volume_size) - (1.));
        pos = round(pos_f);
        state.mask = vec3<bool>(false);
        state.delta_dist = 1. / abs(ray.dir);
        state.ray_step = sign(ray.dir);
        state.side_dist = ((0.5 - (pos_f - pos)) * state.ray_step) * state.delta_dist;
        state.last_distance = 0.;
    }
    let early_stopping_t = 1. / 255.;
    let step_size_g = settings.step_size;
    var sample_pos: vec4<f32>;
    loop{

        if bool(use_dda) {
            sample_pos = next_pos_dda(&pos, &state);
        } else {
            sample_pos = next_pos(&pos, step_size_g, ray.dir);
        }
        let step_size = sample_pos.w;

        let sample_curr = textureSample(volume, volume_sampler, sample_pos.xyz).r;
        let sample_next = textureSample(volume, volume_sampler, sample_pos.xyz).r;
        let time_fraction = fract(settings.time * f32(settings.time_steps));
        let sample = mix(sample_curr, sample_next, time_fraction);
        var sigma = saturate((sample + 15.) / 30.);
        if sigma < 0.5 {
            sigma = 0.;
        }

        if sigma > 0. {
            var sample_color = mix(vec3<f32>(1., 0., 0.), vec3<f32>(0., 0., 1.), sigma);
            color += exp(-transmittance) * (1. - exp(-sigma * step_size * distance_scale)) * sample_color ;
            transmittance += sigma * step_size * distance_scale;

            if exp(-transmittance) <= early_stopping_t {
                // scale value to full "energy"
                color /= 1. - exp(-transmittance);
                transmittance = 1e6;
                break;
            }
        }
        // check if within slice
        let e = 1e-4;
        let slice_test = any(sample_pos.xyz < settings.clipping.min - e) || any(sample_pos.xyz > settings.clipping.max + e) ;


        if slice_test || iters > 10000 {
            break;
        }
        iters += 1u;
    }
    let a = 1. - exp(-transmittance);
    return vec4<f32>(color / (a + 1e-6), a);
}


    @fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {

    let r_pos = vec2<f32>(vertex_in.tex_coord.x, 1. - vertex_in.tex_coord.y);
    let ray = create_ray(camera.view_inv, camera.proj_inv, r_pos, 0.);
    var color = trace_ray(ray);
    return color;
}