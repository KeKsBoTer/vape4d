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
    step_size: f32,
    temporal_filter: u32,
    spatial_filter: u32,
    
    distance_scale: f32,
    vmin: f32,
    vmax: f32,
    gamma_correction: u32,
    upscaling_method:u32,
    selected_channel:u32,
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
    var far = vec4<f32>(px, 1., 1.);
    far.y *= -1.;
    // depth prepass location
    var far_w = view_inv * proj_inv * far;
    far_w /= far_w.w + 1e-4;


    var near = vec4<f32>(px, 0., 1.);
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

@group(1) @binding(0)
var<uniform> settings: Settings;

@group(2) @binding(0)
var cmap : texture_2d<f32>;
@group(2) @binding(1)
var cmap_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) grad_x: vec4<f32>,
    @location(2) grad_y: vec4<f32>,
    @location(3) grad_xy: vec4<f32>,
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
    return VertexOut(vec4<f32>(xy * 2. - (1.), 0., 1.), xy*2.-1.);
}

// performs a step and returns the NDC cooidinate for the volume sampling
fn next_pos(pos: ptr<function,vec3<f32>>, step_size: f32, ray_dir: vec3<f32>) -> vec3<f32> {
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    let sample_pos = ((*pos) - aabb.min) / aabb_size;
    *pos += ray_dir * step_size;
    return sample_pos;
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
fn trace_ray_simple(
    pixel_pos:vec2<f32>,
    color_out:ptr<function,vec4<f32>>,
    color_out_dx:ptr<function,vec4<f32>>,
    color_out_dy:ptr<function,vec4<f32>>,
    color_out_dxy:ptr<function,vec4<f32>>
){
    
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    var ray = create_ray(camera.view_inv, camera.proj_inv, pixel_pos);
    let slice_min = settings.clipping.min;
    let slice_max = settings.clipping.max;
    // find closest point on volume
    let aabb_min = (aabb.min + (slice_min * aabb_size)); //  zxy for tensorf alignment
    let aabb_max = (aabb.max - ((1. - slice_max) * aabb_size)); //  zxy for tensorf alignment
    let intersec = intersectAABB(ray, aabb_min, aabb_max);

    if intersec.x > intersec.y {
        return;
    }

    let start_cam_pos = ray.orig;
    let start = max(0., intersec.x) + 1e-4;
    ray.orig += start * ray.dir;


    let cam_bounds = extract_znear_zfar(camera.proj);
    let znear = cam_bounds.x;
    let zfar = cam_bounds.y;
    let cam_pos = start_cam_pos - znear*ray.dir;
    // let depth = length(start * ray.dir);

    var iters = 0u;
    var color = vec4<f32>(0.);
    var color_dx = vec4<f32>(0.);
    var color_dy = vec4<f32>(0.);
    var color_dxy = vec4<f32>(0.);


    var distance_scale = settings.distance_scale;

    var pos = ray.orig;

    let early_stopping_t = 1. / 255.;

    let step_size_g = (settings.step_size);
    var sample_pos: vec3<f32>;
    var start_point: vec3<f32>;


    var step_size: f32;
    loop{
        step_size = step_size_g;
        sample_pos = next_pos(&pos, step_size_g, ray.dir);

        let slice_test = any(sample_pos < slice_min) || any(sample_pos > slice_max) ;
        if slice_test || iters > 10000 {
            break;
        }

        let depth = length(pos - cam_pos);
        let depth_ndc = (depth - znear) / (zfar - znear);
    
        // let depth = length(ray.orig - cam_pos);
        var pos3d = ndc_to_world(
            vec3<f32>(pixel_pos.x,-pixel_pos.y, depth_ndc),
            camera.view_inv,
            camera.proj_inv
        );
        
        let pos_n = (pos3d - aabb.min) / aabb_size;
        let sample = sample_volume(pos_n);
        let color_tf = sample_cmap(sample);


        // var color_tf_grad = vec4<f32>(0.);
        // sample_cmap_grad(sample, &color_tf_grad);

        
        // var sample_volume_dx = 0.;
        // var sample_volume_dy = 0.;
        // var sample_volume_dxy = 0.;
        // sample_volume_grad(pos_n, &sample_volume_dx, &sample_volume_dy, &sample_volume_dxy);

        // var proj_dx = vec3<f32>(0.);
        // var proj_dy = vec3<f32>(0.);
        // var proj_dxy = vec3<f32>(0.);
        // ndc_to_world_grad(
        //     vec3<f32>(pixel_pos.x,-pixel_pos.y, depth_ndc),
        //     camera.view_inv,
        //     camera.proj_inv,
        //     &proj_dx, &proj_dy, &proj_dxy
        // );

        var sample_dx = dpdxFine(color_tf);
        var sample_dy = dpdyFine(color_tf);
        var sample_dxy = dpdxFine(sample_dy);

        // we try to avoid values that are exactly one as this can cause artifacts
        let sigma = min(color_tf.a,1.-1e-6);

        var sample_color = color_tf.rgb;
        let a_i = 1. - pow(1. - sigma, step_size * distance_scale);
        color = blend(color, vec4<f32>(sample_color, a_i));
        
        let color_dx_c = color_dx;
        let color_dy_c = color_dy;
        let color_dxy_c = color_dxy;
        blend_gradient(
            color,color_dx_c,color_dxy_c,color_dy_c,
            vec4<f32>(sample_color, a_i),sample_dx,sample_dxy,sample_dy,
            &color_dx, &color_dxy, &color_dy,
        );

        if color.a > 1.- early_stopping_t {
            break;
        }
        iters += 1u;
    }
    *color_out = color;
    *color_out_dx = color_dx;
    *color_out_dy = color_dy;
    *color_out_dxy = color_dxy;

}




@fragment
fn fs_main(vertex_in: VertexOut) -> FragmentOut {
    let r_pos = vertex_in.tex_coord.xy;
    
    var color = vec4<f32>(0.);
    var grad_x = vec4<f32>(0.);
    var grad_y = vec4<f32>(0.);
    var grad_xy = vec4<f32>(0.);
    trace_ray_simple(r_pos, &color, &grad_x, &grad_y, &grad_xy);

    var frag_out: FragmentOut;

    // grad_x = dpdxFine(color);
    // grad_y = dpdyFine(color);
    // grad_xy = dpdxFine(grad_y);

    frag_out.color = color;
    frag_out.grad_x = grad_x;
    frag_out.grad_y = grad_y;
    frag_out.grad_xy = grad_xy;
    return frag_out;
}


//
/// generated with sympy
fn blend(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {

    var blend_result:vec4<f32>;
    blend_result[0] = (1 - a[3])*b[0]*b[3] + a[0];
    blend_result[1] = (1 - a[3])*b[1]*b[3] + a[1];
    blend_result[2] = (1 - a[3])*b[2]*b[3] + a[2];
    blend_result[3] = (1 - a[3])*b[3] + a[3];
    return blend_result;
 
}

/// generated with sympy
fn blend_gradient(a: vec4<f32>, a_dx: vec4<f32>, a_dxy: vec4<f32>, a_dy: vec4<f32>, b: vec4<f32>, b_dx: vec4<f32>, b_dxy: vec4<f32>, b_dy: vec4<f32>, dblend_dx: ptr<function,vec4<f32>>, dblend_dxy: ptr<function,vec4<f32>>, dblend_dy: ptr<function,vec4<f32>>) {

    (*dblend_dx)[0] = (1 - a[3])*b[0]*b_dx[3] + (1 - a[3])*b[3]*b_dx[0] + a_dx[0] - a_dx[3]*b[0]*b[3];
    (*dblend_dx)[1] = (1 - a[3])*b[1]*b_dx[3] + (1 - a[3])*b[3]*b_dx[1] + a_dx[1] - a_dx[3]*b[1]*b[3];
    (*dblend_dx)[2] = (1 - a[3])*b[2]*b_dx[3] + (1 - a[3])*b[3]*b_dx[2] + a_dx[2] - a_dx[3]*b[2]*b[3];
    (*dblend_dx)[3] = (1 - a[3])*b_dx[3] - a_dx[3]*b[3] + a_dx[3];
    (*dblend_dxy)[0] = -(a[3] - 1)*b[0]*b_dxy[3] - (a[3] - 1)*b[3]*b_dxy[0] - (a[3] - 1)*b_dx[0]*b_dy[3] - (a[3] - 1)*b_dx[3]*b_dy[0] - a_dx[3]*b[0]*b_dy[3] - a_dx[3]*b[3]*b_dy[0] + a_dxy[0] - a_dxy[3]*b[0]*b[3] - a_dy[3]*b[0]*b_dx[3] - a_dy[3]*b[3]*b_dx[0];
    (*dblend_dxy)[1] = -(a[3] - 1)*b[1]*b_dxy[3] - (a[3] - 1)*b[3]*b_dxy[1] - (a[3] - 1)*b_dx[1]*b_dy[3] - (a[3] - 1)*b_dx[3]*b_dy[1] - a_dx[3]*b[1]*b_dy[3] - a_dx[3]*b[3]*b_dy[1] + a_dxy[1] - a_dxy[3]*b[1]*b[3] - a_dy[3]*b[1]*b_dx[3] - a_dy[3]*b[3]*b_dx[1];
    (*dblend_dxy)[2] = -(a[3] - 1)*b[2]*b_dxy[3] - (a[3] - 1)*b[3]*b_dxy[2] - (a[3] - 1)*b_dx[2]*b_dy[3] - (a[3] - 1)*b_dx[3]*b_dy[2] - a_dx[3]*b[2]*b_dy[3] - a_dx[3]*b[3]*b_dy[2] + a_dxy[2] - a_dxy[3]*b[2]*b[3] - a_dy[3]*b[2]*b_dx[3] - a_dy[3]*b[3]*b_dx[2];
    (*dblend_dxy)[3] = -(a[3] - 1)*b_dxy[3] - a_dx[3]*b_dy[3] - a_dxy[3]*b[3] + a_dxy[3] - a_dy[3]*b_dx[3];
    (*dblend_dy)[0] = (1 - a[3])*b[0]*b_dy[3] + (1 - a[3])*b[3]*b_dy[0] + a_dy[0] - a_dy[3]*b[0]*b[3];
    (*dblend_dy)[1] = (1 - a[3])*b[1]*b_dy[3] + (1 - a[3])*b[3]*b_dy[1] + a_dy[1] - a_dy[3]*b[1]*b[3];
    (*dblend_dy)[2] = (1 - a[3])*b[2]*b_dy[3] + (1 - a[3])*b[3]*b_dy[2] + a_dy[2] - a_dy[3]*b[2]*b[3];
    (*dblend_dy)[3] = (1 - a[3])*b_dy[3] - a_dy[3]*b[3] + a_dy[3];
 
}

fn extract_znear_zfar(proj: mat4x4<f32>) -> vec2<f32> {
    let p22 = proj[2][2];
    let p23 = proj[2][3];
    
    let z_far = (1. -p23) / p22;
    let z_near = - p23 / p22;
    
    return vec2<f32>(z_near, z_far);
}

fn ndc_to_world(ndc: vec3<f32>, inv_view: mat4x4<f32>, inv_proj: mat4x4<f32>) -> vec3<f32> {
    
    // Transform NDC to clip space (homogeneous coordinates)
    let clip_space = vec4<f32>(ndc, 1.0);
    
    // Transform to world space
    var world_space = inv_view * inv_proj * clip_space;
    // world_space /= world_space.w; # TODO we assume orthographic projection right now
    
    return world_space.xyz;
}

/// calculates the gradient for world position for a change in screen space (x,y)
fn ndc_to_world_grad(ndc: vec3<f32>, inv_view: mat4x4<f32>, inv_proj: mat4x4<f32>, dx:ptr<function,vec3<f32>>,dy:ptr<function,vec3<f32>>,dxy:ptr<function,vec3<f32>>,){
    let vp = inv_view*inv_proj;
    *dx = vp[0].xyz;         // first column of the inverse projection matrix
    *dy = vp[1].xyz;         // second column of the inverse projection matrix
    *dxy = vec3<f32>(0.);// linear transform is zero in second order derivative
}


fn sample_volume(pos: vec3<f32>) -> f32 {
    //  origin is in bottom left corner so we need to flip y 
    let pos_m = vec3<f32>(pos.x, pos.y, pos.z);
    let size = textureDimensions(volume);
    let pos_f = pos_m * (vec3<f32>(size)-1.)+0.5;
    let pos_i = vec3<i32>(pos_f);
    if settings.temporal_filter == FILTER_NEAREST {
        return textureLoad(volume, pos_i, 0).r;
    }

    // let sample_curr = textureSampleLevel(volume, volume_sampler,pos_m, 0.).r;
    // if settings.temporal_filter == FILTER_NEAREST {
    //     return sample_curr;
    // } else {
    //     let sample_next = textureSampleLevel(volume_next, volume_sampler,pos_m, 0.).r;
    //     return mix(sample_curr, sample_next, settings.time);
    // }
    let corners = array<f32,8>(
        textureLoad(volume, pos_i + vec3<i32>(0,0,0), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(0,0,1), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(0,1,0), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(0,1,1), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(1,0,0), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(1,0,1), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(1,1,0), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(1,1,1), 0).r
    );

    let p = fract(pos_f);
    let weights = array<f32,8>(
        (1. - p.x) * (1. - p.y) * (1. - p.z),
        (1. - p.x) * (1. - p.y) * (p.z),
        (1. - p.x) * (p.y) * (1. - p.z),
        (1. - p.x) * (p.y) * (p.z),
        (p.x) * (1. - p.y) * (1. - p.z),
        (p.x) * (1. - p.y) * (p.z),
        (p.x) * (p.y) * (1. - p.z),
        (p.x) * (p.y) * (p.z)
    );
    var result = 0.;
    for (var i: i32 = 0; i < 8; i += 1) {
        result += corners[i] * weights[i];
    }
    return result;
    
}



fn sample_volume_grad(pos: vec3<f32>, dx:ptr<function,f32>,dy:ptr<function,f32>,dxy:ptr<function,f32>)  {
    *dx = 0.;
    *dy = 0.;
    *dxy = 0.;
    //  origin is in bottom left corner so we need to flip y 
    let pos_m = vec3<f32>(pos.x, pos.y, pos.z);
    let size = textureDimensions(volume);
    let pos_f = pos_m * (vec3<f32>(size)-1.)+0.5;
    let pos_i = vec3<i32>(pos_f);
    if settings.temporal_filter == FILTER_NEAREST {
        return;
    }

    let corners = array<f32,8>(
        textureLoad(volume, pos_i + vec3<i32>(0,0,0), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(0,0,1), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(0,1,0), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(0,1,1), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(1,0,0), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(1,0,1), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(1,1,0), 0).r,
        textureLoad(volume, pos_i + vec3<i32>(1,1,1), 0).r
    );

    let p = fract(pos_f);
    let x = p.x;
    let y = p.y;
    let z = p.z;

    let dweights_dx = array<f32,8>(-(1.0 - y)*(1.0 - z), -z*(1.0 - y), -y*(1.0 - z), -y*z, (1.0 - y)*(1.0 - z), z*(1.0 - y), y*(1.0 - z), y*z) ;
    let dweights_dy = array<f32,8>(-(1.0 - x)*(1.0 - z), -z*(1.0 - x), (1.0 - x)*(1.0 - z), z*(1.0 - x), -x*(1.0 - z), -x*z, x*(1.0 - z), x*z) ;
    let dweights_dxy = array<f32,8>(1.0 - z, z, z - 1.0, -z, z - 1.0, -z, 1.0 - z, z) ;

    // dooing this in a loop causes naga to freak out...
    *dx += corners[0] * dweights_dx[0];
    *dy += corners[0] * dweights_dy[0];
    *dxy += corners[0] * dweights_dxy[0];
    *dx += corners[1] * dweights_dx[1];
    *dy += corners[1] * dweights_dy[1];
    *dxy += corners[1] * dweights_dxy[1];
    *dx += corners[2] * dweights_dx[2];
    *dy += corners[2] * dweights_dy[2];
    *dxy += corners[2] * dweights_dxy[2];
    *dx += corners[3] * dweights_dx[3];
    *dy += corners[3] * dweights_dy[3];
    *dxy += corners[3] * dweights_dxy[3];
    *dx += corners[4] * dweights_dx[4];
    *dy += corners[4] * dweights_dy[4];
    *dxy += corners[4] * dweights_dxy[4];
    *dx += corners[5] * dweights_dx[5];
    *dy += corners[5] * dweights_dy[5];
    *dxy += corners[5] * dweights_dxy[5];
    *dx += corners[6] * dweights_dx[6];
    *dy += corners[6] * dweights_dy[6];
    *dxy += corners[6] * dweights_dxy[6];
    *dx += corners[7] * dweights_dx[7];
    *dy += corners[7] * dweights_dy[7];
    *dxy += corners[7] * dweights_dxy[7];

}

fn sample_cmap(value: f32) -> vec4<f32> {
    // let value_n = (value - settings.vmin) / (settings.vmax - settings.vmin);
    // return textureSampleLevel(cmap, cmap_sampler, vec2<f32>(value_n, 0.5), 0.);
    let value_n = (value - settings.vmin) / (settings.vmax - settings.vmin);
    let n = f32(textureDimensions(cmap).x); // number of elements in the color map
    let pos_n = value_n * (n-1) + 0.5;
    let pos_f = fract(pos_n);
    let pos_left = max(i32(floor(pos_n)),0);
    let pos_right = min(i32(ceil(pos_n)),i32(n-1));
    let left_v  = textureLoad(cmap,vec2<i32>(pos_left,0),0);
    let right_v = textureLoad(cmap,vec2<i32>(pos_right,0),0);
    let result = left_v*(1.-pos_f) + right_v*pos_f;
    return result;
}


fn sample_cmap_grad(x: f32, dcolor_dx:ptr<function,vec4<f32>>) {
    let value_n = (x - settings.vmin) / (settings.vmax - settings.vmin);
    let n = f32(textureDimensions(cmap).x); // number of elements in the color map
    let pos_n = value_n * (n-1) + 0.5;
    let pos_f = fract(pos_n);
    let pos_left = max(i32(floor(pos_n)),0);
    let pos_right = min(i32(ceil(pos_n)),i32(n-1));
    let left_v  = textureLoad(cmap,vec2<i32>(pos_left,0),0);
    let right_v = textureLoad(cmap,vec2<i32>(pos_right,0),0);
    
    let range = settings.vmax - settings.vmin;
    *dcolor_dx =( (n-1)/range * (right_v - left_v));
}

fn generate_derivative_terms(
    u: f32, 
    v: f32, 
    d: f32,
    proj_view_inv: mat4x4<f32>,
    corners: array<f32, 8>,
    tf0: vec4<f32>,
    tf1: vec4<f32>,
    tf_res: f32,
    vmin: f32,
    vmax: f32,
    out_du: ptr<function, vec4<f32>>,
    out_dv: ptr<function, vec4<f32>>,
    out_duv: ptr<function, vec4<f32>>
) {
    // --- Common Calculations ---
    let coords = vec4<f32>(u, v, d, 1.0);
    let world_coords_h = proj_view_inv * coords;

    // Renamed for clarity (world coords before homogeneous division)
    let wx = world_coords_h.x;
    let wy = world_coords_h.y;
    let wz = world_coords_h.z;

    // Precompute factors (wx-1), etc.
    let wxm1 = wx - 1.0;
    let wym1 = wy - 1.0;
    let wzm1 = wz - 1.0;

    // Extract relevant matrix elements (first two columns, first 3 rows)
    let px0 = proj_view_inv[0][0]; // P[:, 0] related - for du
    let px1 = proj_view_inv[0][1];
    let px2 = proj_view_inv[0][2];
    
    let py0 = proj_view_inv[1][0]; // P[:, 1] related - for dv
    let py1 = proj_view_inv[1][1];
    let py2 = proj_view_inv[1][2];

    // --- Calculate Sum Term for du (using P[:, 0]) ---
    let sum_term_du = 
        corners[0] * (wxm1*wym1*px2 + wxm1*wzm1*px1 + wym1*wzm1*px0)
      - corners[1] * (wz*wxm1*px1   + wz*wym1*px0   + wxm1*wym1*px2)
      - corners[2] * (wy*wxm1*px2   + wy*wzm1*px0   + wxm1*wzm1*px1)
      + corners[3] * (wy*wz*px0     + wy*wxm1*px2   + wz*wxm1*px1)
      - corners[4] * (wx*wym1*px2   + wx*wzm1*px1   + wym1*wzm1*px0)
      + corners[5] * (wx*wz*px1     + wx*wym1*px2   + wz*wym1*px0)
      + corners[6] * (wx*wy*px2     + wx*wzm1*px1   + wy*wzm1*px0)
      - corners[7] * (wx*wy*px2     + wx*wz*px1     + wy*wz*px0);

    // --- Calculate Sum Term for dv (using P[:, 1]) ---
    // Structure is identical to du, just replace px with py
    let sum_term_dv = 
        corners[0] * (wxm1*wym1*py2 + wxm1*wzm1*py1 + wym1*wzm1*py0)
      - corners[1] * (wz*wxm1*py1   + wz*wym1*py0   + wxm1*wym1*py2)
      - corners[2] * (wy*wxm1*py2   + wy*wzm1*py0   + wxm1*wzm1*py1)
      + corners[3] * (wy*wz*py0     + wy*wxm1*py2   + wz*wxm1*py1)
      - corners[4] * (wx*wym1*py2   + wx*wzm1*py1   + wym1*wzm1*py0)
      + corners[5] * (wx*wz*py1     + wx*wym1*py2   + wz*wym1*py0)
      + corners[6] * (wx*wy*py2     + wx*wzm1*py1   + wy*wzm1*py0)
      - corners[7] * (wx*wy*py2     + wx*wz*py1     + wy*wz*py0);

    // --- Calculate Sum Term for duv (mixed derivative structure) ---
    let sum_term_duv = 
      // c0 terms (use wxm1, wym1, wzm1)
        corners[0] * (wxm1 * (px1*py2 + py1*px2) +
                wym1 * (px0*py2 + py0*px2) +
                wzm1 * (px0*py1 + py0*px1))
      // c1 terms (use wxm1, wym1, wz) -> swap wzm1 with wz
      - corners[1] * (wz   * (px0*py1 + py0*px1) +
                wxm1 * (px1*py2 + py1*px2) +
                wym1 * (px0*py2 + py0*px2))
      // c2 terms (use wxm1, wy, wzm1) -> swap wym1 with wy
      - corners[2] * (wy   * (px0*py2 + py0*px2) +
                wxm1 * (px1*py2 + py1*px2) +
                wzm1 * (px0*py1 + py0*px1))
      // c3 terms (use wxm1, wy, wz) -> swap wym1, wzm1
      + corners[3] * (wy   * (px0*py2 + py0*px2) +
                wz   * (px0*py1 + py0*px1) +
                wxm1 * (px1*py2 + py1*px2))
      // c4 terms (use wx, wym1, wzm1) -> swap wxm1 with wx
      - corners[4] * (wx   * (px1*py2 + py1*px2) +
                wym1 * (px0*py2 + py0*px2) +
                wzm1 * (px0*py1 + py0*px1))
      // c5 terms (use wx, wym1, wz) -> swap wxm1, wzm1
      + corners[5] * (wx   * (px1*py2 + py1*px2) +
                wz   * (px0*py1 + py0*px1) +
                wym1 * (px0*py2 + py0*px2))
      // c6 terms (use wx, wy, wzm1) -> swap wxm1, wym1
      + corners[6] * (wx   * (px1*py2 + py1*px2) +
                wy   * (px0*py2 + py0*px2) +
                wzm1 * (px0*py1 + py0*px1))
      // c7 terms (use wx, wy, wz) -> swap wxm1, wym1, wzm1
      - corners[7] * (wx   * (px1*py2 + py1*px2) +
                wy   * (px0*py2 + py0*px2) +
                wz   * (px0*py1 + py0*px1));

    // --- Calculate Final Expressions ---
    let common_factor = (tf0 - tf1) * (tf_res - 1.0) / (vmax - vmin);

    *out_du = common_factor * sum_term_du;
    *out_dv = common_factor * sum_term_dv;
    *out_duv = common_factor * sum_term_duv;
}