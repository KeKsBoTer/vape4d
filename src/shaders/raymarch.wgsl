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
    hardware_interpolation:u32,
    clamp_gradients:u32,
    gradient_vis_scale:f32
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
        
        let pos_ndc = vec3<f32>(pixel_pos.x,-pixel_pos.y, depth_ndc);
        let sample_color = rescale_alpha(sample_cmap(sample_volume(ndc_to_world(
            camera.proj_inv,
            camera.view_inv,
            pos_ndc
        ))));


        var sample_dx = vec4<f32>(0.);
        var sample_dy = vec4<f32>(0.);
        var sample_dxy = vec4<f32>(0.);


        color_grad(camera.proj_inv, camera.view_inv, pos_ndc, &sample_dx, &sample_dxy, &sample_dy);

        if bool(settings.clamp_gradients) {
            sample_dx = clamp(sample_dx, vec4<f32>(-1000.), vec4<f32>(1000.));
            sample_dy = clamp(sample_dy, vec4<f32>(-1000.), vec4<f32>(1000.));
            sample_dxy = clamp(sample_dxy, vec4<f32>(-1000.), vec4<f32>(1000.));
        }

        blend_gradient(
            color,color_dx,color_dxy,color_dy,
            sample_color,sample_dx,sample_dxy,sample_dy,
            &color,&color_dx, &color_dxy, &color_dy,
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

fn extract_znear_zfar(proj: mat4x4<f32>) -> vec2<f32> {
    let p22 = proj[2][2];
    let p23 = proj[2][3];
    
    let z_far = (1. -p23) / p22;
    let z_near = - p23 / p22;
    
    return vec2<f32>(z_near, z_far);
}

/// generated with sympy
fn blend_gradient(a: vec4<f32>, a_dx: vec4<f32>, a_dxy: vec4<f32>, a_dy: vec4<f32>, b: vec4<f32>, b_dx: vec4<f32>, b_dxy: vec4<f32>, b_dy: vec4<f32>, color: ptr<function,vec4<f32>>, dblend_dx: ptr<function,vec4<f32>>, dblend_dxy: ptr<function,vec4<f32>>, dblend_dy: ptr<function,vec4<f32>>) {

    (*color)[0] = (1 - a[3])*b[0]*b[3] + a[0];
    (*color)[1] = (1 - a[3])*b[1]*b[3] + a[1];
    (*color)[2] = (1 - a[3])*b[2]*b[3] + a[2];
    (*color)[3] = (1 - a[3])*b[3] + a[3];
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


// fn ndc_to_world(ndc: vec3<f32>, inv_view: mat4x4<f32>, inv_proj: mat4x4<f32>) -> vec3<f32> {
    
//     // Transform NDC to clip space (homogeneous coordinates)
//     let clip_space = vec4<f32>(ndc, 1.0);
    
//     // Transform to world space
//     var world_space = inv_view * inv_proj * clip_space;
//     // world_space /= world_space.w; # TODO we assume orthographic projection right now
    
//     return world_space.xyz;
// }


fn volumeRead3D(volume: texture_3d<f32>, loc:vec3<i32>,channel:i32)->f32{
    return textureLoad(volume,loc,0)[channel];
}

fn volumeRead1D(tex: texture_2d<f32>, loc:i32,channel:i32)->f32{
    return textureLoad(tex,vec2<i32>(loc,0),0)[channel];
}

fn textureSize3D(v: texture_3d<f32>,dim:i32) -> u32 {
    return textureDimensions(v)[dim];
}
fn textureSize2D(v: texture_2d<f32>,dim:i32) -> u32 {
    return textureDimensions(v)[dim];
}


/// generated with sympy
fn sample_volume(p: vec3<f32>) -> f32 {
    let x0 = -settings.volume_aabb.min[2];
    let x1 = (x0 + p[2])*f32(textureSize3D(volume, 2))/(x0 + settings.volume_aabb.max[2]) - 0.5;
    let x2 = fract(x1);
    let x3 = -settings.volume_aabb.min[1];
    let x4 = (x3 + p[1])*f32(textureSize3D(volume, 1))/(x3 + settings.volume_aabb.max[1]) - 0.5;
    let x5 = fract(x4);
    let x6 = -settings.volume_aabb.min[0];
    let x7 = (x6 + p[0])*f32(textureSize3D(volume, 0))/(x6 + settings.volume_aabb.max[0]) - 0.5;
    let x8 = fract(x7);
    let x9 = x5*x8;
    let x10 = floor(x7);
    let x11 = x10 + 1;
    let x12 = floor(x4);
    let x13 = x12 + 1;
    let x14 = floor(x1);
    let x15 = x14 + 1;
    let x16 = 1 - x8;
    let x17 = x16*x5;
    let x18 = 1 - x5;
    let x19 = x18*x8;
    let x20 = 1 - x2;
    let x21 = x16*x18;
 
    var sample_volume_result:f32;
    sample_volume_result = x17*x2*volumeRead3D(volume, vec3<i32>(vec3<f32>(x10, x13, x15)), 0) + x17*x20*volumeRead3D(volume, vec3<i32>(vec3<f32>(x10, x13, x14)), 0) + x19*x2*volumeRead3D(volume, vec3<i32>(vec3<f32>(x11, x12, x15)), 0) + x19*x20*volumeRead3D(volume, vec3<i32>(vec3<f32>(x11, x12, x14)), 0) + x2*x21*volumeRead3D(volume, vec3<i32>(vec3<f32>(x10, x12, x15)), 0) + x2*x9*volumeRead3D(volume, vec3<i32>(vec3<f32>(x11, x13, x15)), 0) + x20*x21*volumeRead3D(volume, vec3<i32>(vec3<f32>(x10, x12, x14)), 0) + x20*x9*volumeRead3D(volume, vec3<i32>(vec3<f32>(x11, x13, x14)), 0);
    return sample_volume_result;
 
 }
 
 /// generated with sympy
 fn sample_cmap(value: f32) -> vec4<f32> {
    let x0 = -0.5 + (-settings.vmin + value)*f32(textureSize2D(cmap, 0))/(settings.vmax - settings.vmin);
    let x1 = fract(x0);
    let x2 = i32(floor(x0));
    let x3 = x2 + 1;
    let x4 = 1.0 - x1;
 
    var sample_cmap_result:vec4<f32>;
    sample_cmap_result[0] = x1*volumeRead1D(cmap, x3, 0) + x4*volumeRead1D(cmap, x2, 0);
    sample_cmap_result[1] = x1*volumeRead1D(cmap, x3, 1) + x4*volumeRead1D(cmap, x2, 1);
    sample_cmap_result[2] = x1*volumeRead1D(cmap, x3, 2) + x4*volumeRead1D(cmap, x2, 2);
    sample_cmap_result[3] = x1*volumeRead1D(cmap, x3, 3) + x4*volumeRead1D(cmap, x2, 3);
    return sample_cmap_result;
 
 }
 
 /// generated with sympy
 fn ndc_to_world(inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>, pos: vec3<f32>) -> vec3<f32> {
    let x0 = 1.0*inv_proj[3][0];
    let x1 = 1.0*inv_proj[3][1];
    let x2 = 1.0*inv_proj[3][2];
    let x3 = 1.0*inv_proj[3][3];
 
    var ndc_to_world_result:vec3<f32>;
    ndc_to_world_result[0] = x0*inv_view[0][0] + x1*inv_view[1][0] + x2*inv_view[2][0] + x3*inv_view[3][0] + (inv_proj[0][0]*inv_view[0][0] + inv_proj[0][1]*inv_view[1][0] + inv_proj[0][2]*inv_view[2][0] + inv_proj[0][3]*inv_view[3][0])*pos[0] + (inv_proj[1][0]*inv_view[0][0] + inv_proj[1][1]*inv_view[1][0] + inv_proj[1][2]*inv_view[2][0] + inv_proj[1][3]*inv_view[3][0])*pos[1] + (inv_proj[2][0]*inv_view[0][0] + inv_proj[2][1]*inv_view[1][0] + inv_proj[2][2]*inv_view[2][0] + inv_proj[2][3]*inv_view[3][0])*pos[2];
    ndc_to_world_result[1] = x0*inv_view[0][1] + x1*inv_view[1][1] + x2*inv_view[2][1] + x3*inv_view[3][1] + (inv_proj[0][0]*inv_view[0][1] + inv_proj[0][1]*inv_view[1][1] + inv_proj[0][2]*inv_view[2][1] + inv_proj[0][3]*inv_view[3][1])*pos[0] + (inv_proj[1][0]*inv_view[0][1] + inv_proj[1][1]*inv_view[1][1] + inv_proj[1][2]*inv_view[2][1] + inv_proj[1][3]*inv_view[3][1])*pos[1] + (inv_proj[2][0]*inv_view[0][1] + inv_proj[2][1]*inv_view[1][1] + inv_proj[2][2]*inv_view[2][1] + inv_proj[2][3]*inv_view[3][1])*pos[2];
    ndc_to_world_result[2] = x0*inv_view[0][2] + x1*inv_view[1][2] + x2*inv_view[2][2] + x3*inv_view[3][2] + (inv_proj[0][0]*inv_view[0][2] + inv_proj[0][1]*inv_view[1][2] + inv_proj[0][2]*inv_view[2][2] + inv_proj[0][3]*inv_view[3][2])*pos[0] + (inv_proj[1][0]*inv_view[0][2] + inv_proj[1][1]*inv_view[1][2] + inv_proj[1][2]*inv_view[2][2] + inv_proj[1][3]*inv_view[3][2])*pos[1] + (inv_proj[2][0]*inv_view[0][2] + inv_proj[2][1]*inv_view[1][2] + inv_proj[2][2]*inv_view[2][2] + inv_proj[2][3]*inv_view[3][2])*pos[2];
    return ndc_to_world_result;
 
 }
 
 /// generated with sympy
 fn rescale_alpha(color: vec4<f32>) -> vec4<f32> {
 
    var rescale_alpha_result:vec4<f32>;
    rescale_alpha_result[0] = color[0];
    rescale_alpha_result[1] = color[1];
    rescale_alpha_result[2] = color[2];
    rescale_alpha_result[3] = 1 - pow(1 - 0.996078431372549*color[3], settings.distance_scale*settings.step_size);
    return rescale_alpha_result;
 
 }
 
 /// generated with sympy
 fn color_grad(inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>, pos_ndc: vec3<f32>, color_dx: ptr<function,vec4<f32>>, color_dxy: ptr<function,vec4<f32>>, color_dy: ptr<function,vec4<f32>>) {
    let x0 = -settings.vmin;
    let x1 = f32(textureSize2D(cmap, 0))/(settings.vmax + x0);
    let x2 = f32(textureSize3D(volume, 2));
    let x3 = -settings.volume_aabb.min[2];
    let x4 = 1.0/(x3 + settings.volume_aabb.max[2]);
    let x5 = 1.0*inv_proj[3][0];
    let x6 = 1.0*inv_proj[3][1];
    let x7 = 1.0*inv_proj[3][2];
    let x8 = 1.0*inv_proj[3][3];
    let x9 = inv_proj[0][0]*inv_view[0][2] + inv_proj[0][1]*inv_view[1][2] + inv_proj[0][2]*inv_view[2][2] + inv_proj[0][3]*inv_view[3][2];
    let x10 = inv_proj[1][0]*inv_view[0][2] + inv_proj[1][1]*inv_view[1][2] + inv_proj[1][2]*inv_view[2][2] + inv_proj[1][3]*inv_view[3][2];
    let x11 = x2*x4*(x10*pos_ndc[1] + x3 + x5*inv_view[0][2] + x6*inv_view[1][2] + x7*inv_view[2][2] + x8*inv_view[3][2] + x9*pos_ndc[0] + (inv_proj[2][0]*inv_view[0][2] + inv_proj[2][1]*inv_view[1][2] + inv_proj[2][2]*inv_view[2][2] + inv_proj[2][3]*inv_view[3][2])*pos_ndc[2]) - 0.5;
    let x12 = fract(x11);
    let x13 = f32(textureSize3D(volume, 1));
    let x14 = -settings.volume_aabb.min[1];
    let x15 = 1.0/(x14 + settings.volume_aabb.max[1]);
    let x16 = inv_proj[0][0]*inv_view[0][1] + inv_proj[0][1]*inv_view[1][1] + inv_proj[0][2]*inv_view[2][1] + inv_proj[0][3]*inv_view[3][1];
    let x17 = inv_proj[1][0]*inv_view[0][1] + inv_proj[1][1]*inv_view[1][1] + inv_proj[1][2]*inv_view[2][1] + inv_proj[1][3]*inv_view[3][1];
    let x18 = x13*x15*(x14 + x16*pos_ndc[0] + x17*pos_ndc[1] + x5*inv_view[0][1] + x6*inv_view[1][1] + x7*inv_view[2][1] + x8*inv_view[3][1] + (inv_proj[2][0]*inv_view[0][1] + inv_proj[2][1]*inv_view[1][1] + inv_proj[2][2]*inv_view[2][1] + inv_proj[2][3]*inv_view[3][1])*pos_ndc[2]) - 0.5;
    let x19 = fract(x18);
    let x20 = f32(textureSize3D(volume, 0));
    let x21 = -settings.volume_aabb.min[0];
    let x22 = 1.0/(x21 + settings.volume_aabb.max[0]);
    let x23 = inv_proj[0][0]*inv_view[0][0] + inv_proj[0][1]*inv_view[1][0] + inv_proj[0][2]*inv_view[2][0] + inv_proj[0][3]*inv_view[3][0];
    let x24 = inv_proj[1][0]*inv_view[0][0] + inv_proj[1][1]*inv_view[1][0] + inv_proj[1][2]*inv_view[2][0] + inv_proj[1][3]*inv_view[3][0];
    let x25 = x20*x22*(x21 + x23*pos_ndc[0] + x24*pos_ndc[1] + x5*inv_view[0][0] + x6*inv_view[1][0] + x7*inv_view[2][0] + x8*inv_view[3][0] + (inv_proj[2][0]*inv_view[0][0] + inv_proj[2][1]*inv_view[1][0] + inv_proj[2][2]*inv_view[2][0] + inv_proj[2][3]*inv_view[3][0])*pos_ndc[2]) - 0.5;
    let x26 = fract(x25);
    let x27 = x19*x26;
    let x28 = floor(x25);
    let x29 = x28 + 1;
    let x30 = floor(x18);
    let x31 = x30 + 1;
    let x32 = floor(x11);
    let x33 = x32 + 1;
    let x34 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x29, x31, x33)), 0);
    let x35 = x26 - 1;
    let x36 = -x35;
    let x37 = x19*x36;
    let x38 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x28, x31, x33)), 0);
    let x39 = x19 - 1;
    let x40 = -x39;
    let x41 = x26*x40;
    let x42 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x29, x30, x33)), 0);
    let x43 = x12 - 1;
    let x44 = -x43;
    let x45 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x29, x31, x32)), 0);
    let x46 = x36*x40;
    let x47 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x28, x30, x33)), 0);
    let x48 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x28, x31, x32)), 0);
    let x49 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x29, x30, x32)), 0);
    let x50 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x28, x30, x32)), 0);
    let x51 = x1*(x0 + x12*x27*x34 + x12*x37*x38 + x12*x41*x42 + x12*x46*x47 + x27*x44*x45 + x37*x44*x48 + x41*x44*x49 + x44*x46*x50) - 0.5;
    let x52 = i32(floor(x51));
    let x53 = volumeRead1D(cmap, x52, 0);
    let x54 = x20*x22;
    let x55 = x23*x54;
    let x56 = x40*x55;
    let x57 = x44*x56;
    let x58 = x12*x56;
    let x59 = x19*x55;
    let x60 = x44*x59;
    let x61 = x13*x15;
    let x62 = x16*x61;
    let x63 = x36*x62;
    let x64 = x44*x63;
    let x65 = x12*x63;
    let x66 = x26*x62;
    let x67 = x44*x66;
    let x68 = x2*x4;
    let x69 = x68*x9;
    let x70 = x46*x69;
    let x71 = x37*x69;
    let x72 = x41*x69;
    let x73 = x12*x59;
    let x74 = x12*x66;
    let x75 = x27*x69;
    let x76 = x34*x73 + x34*x74 + x34*x75 - x38*x73 - x42*x74 - x45*x75;
    let x77 = x38*x65 + x38*x71 + x42*x58 + x42*x72 + x45*x60 + x45*x67 - x47*x58 - x47*x65 + x47*x70 - x48*x60 + x48*x64 - x48*x71 + x49*x57 - x49*x67 - x49*x72 - x50*x57 - x50*x64 - x50*x70 + x76;
    let x78 = x52 + 1;
    let x79 = volumeRead1D(cmap, x78, 0);
    let x80 = volumeRead1D(cmap, x52, 1);
    let x81 = volumeRead1D(cmap, x78, 1);
    let x82 = volumeRead1D(cmap, x52, 2);
    let x83 = volumeRead1D(cmap, x78, 2);
    let x84 = settings.distance_scale*settings.step_size;
    let x85 = 0.996078431372549*x84;
    let x86 = volumeRead1D(cmap, x78, 3);
    let x87 = fract(x51);
    let x88 = volumeRead1D(cmap, x52, 3);
    let x89 = -0.996078431372549*x86*x87 - 0.996078431372549*x88*(1.0 - x87) + 1;
    let x90 = pow(x89, x84);
    let x91 = 1.0/x89;
    let x92 = x24*x54;
    let x93 = x40*x92;
    let x94 = x44*x93;
    let x95 = x12*x93;
    let x96 = x19*x92;
    let x97 = x44*x96;
    let x98 = x17*x61;
    let x99 = x36*x98;
    let x100 = x44*x99;
    let x101 = x12*x99;
    let x102 = x26*x98;
    let x103 = x102*x44;
    let x104 = x10*x68;
    let x105 = x104*x46;
    let x106 = x104*x37;
    let x107 = x104*x41;
    let x108 = x12*x96;
    let x109 = x102*x12;
    let x110 = x104*x27;
    let x111 = x108*x34 - x108*x38 + x109*x34 - x109*x42 + x110*x34 - x110*x45;
    let x112 = x100*x48 - x100*x50 + x101*x38 - x101*x47 + x103*x45 - x103*x49 + x105*x47 - x105*x50 + x106*x38 - x106*x48 + x107*x42 - x107*x49 + x111 + x42*x95 + x45*x97 - x47*x95 - x48*x97 + x49*x94 - x50*x94;
    let x113 = x35*x39;
    let x114 = x19*x35;
    let x115 = x26*x39;
    let x116 = x1*(-settings.vmin - x113*x43*x50 - x114*x12*x38 - x115*x12*x42 + x12*x19*x26*x34 + x12*x35*x39*x47 + x19*x35*x43*x48 + x26*x39*x43*x49 - x27*x43*x45) - 0.5;
    let x117 = i32(floor(x116));
    let x118 = x55*x98;
    let x119 = x118*x12;
    let x120 = x62*x92;
    let x121 = x12*x120;
    let x122 = x104*x59;
    let x123 = x69*x96;
    let x124 = x104*x66;
    let x125 = x102*x69;
    let x126 = x35*x62;
    let x127 = x104*x126;
    let x128 = x35*x98;
    let x129 = x128*x69;
    let x130 = x39*x55;
    let x131 = x104*x130;
    let x132 = x39*x92;
    let x133 = x132*x69;
    let x134 = x118*x43;
    let x135 = x120*x43;
    let x136 = x119*x34 - x119*x38 - x119*x42 + x119*x47 + x121*x34 - x121*x38 - x121*x42 + x121*x47 + x122*x34 - x122*x38 - x122*x45 + x122*x48 + x123*x34 - x123*x38 - x123*x45 + x123*x48 + x124*x34 - x124*x42 - x124*x45 + x124*x49 + x125*x34 - x125*x42 - x125*x45 + x125*x49 - x127*x38 + x127*x47 + x127*x48 - x127*x50 - x129*x38 + x129*x47 + x129*x48 - x129*x50 - x131*x42 + x131*x47 + x131*x49 - x131*x50 - x133*x42 + x133*x47 + x133*x49 - x133*x50 - x134*x45 + x134*x48 + x134*x49 - x134*x50 - x135*x45 + x135*x48 + x135*x49 - x135*x50;
    let x137 = x117 + 1;
    let x138 = volumeRead1D(cmap, x137, 3);
    let x139 = fract(x116);
    let x140 = volumeRead1D(cmap, x117, 3);
    let x141 = -0.996078431372549*x138*x139 - 0.996078431372549*x140*(1.0 - x139) + 1;
    let x142 = pow(x141, x84);
    let x143 = 1.0/x141;
    let x144 = 0.992172241445598*x1;
    let x145 = x130*x43;
    let x146 = x12*x130;
    let x147 = x43*x59;
    let x148 = x126*x43;
    let x149 = x12*x126;
    let x150 = x43*x66;
    let x151 = x113*x69;
    let x152 = x114*x69;
    let x153 = x115*x69;
    let x154 = x145*x49 - x145*x50 - x146*x42 + x146*x47 - x147*x45 + x147*x48 + x148*x48 - x148*x50 - x149*x38 + x149*x47 - x150*x45 + x150*x49 + x151*x47 - x151*x50 - x152*x38 + x152*x48 - x153*x42 + x153*x49 + x76;
    let x155 = x138*x154 - x140*x154;
    let x156 = x132*x43;
    let x157 = x12*x132;
    let x158 = x43*x96;
    let x159 = x128*x43;
    let x160 = x12*x128;
    let x161 = x102*x43;
    let x162 = x104*x113;
    let x163 = x104*x114;
    let x164 = x104*x115;
    let x165 = x111 + x156*x49 - x156*x50 - x157*x42 + x157*x47 - x158*x45 + x158*x48 + x159*x48 - x159*x50 - x160*x38 + x160*x47 - x161*x45 + x161*x49 + x162*x47 - x162*x50 - x163*x38 + x163*x48 - x164*x42 + x164*x49;
    let x166 = x138*x165 - x140*x165;
 
    (*color_dx)[0] = -x1*x53*x77 + x1*x77*x79;
    (*color_dx)[1] = -x1*x77*x80 + x1*x77*x81;
    (*color_dx)[2] = -x1*x77*x82 + x1*x77*x83;
    (*color_dx)[3] = x85*x90*x91*(x1*x77*x86 - x1*x77*x88);
    (*color_dxy)[0] = x1*(-x136*volumeRead1D(cmap, x117, 0) + x136*volumeRead1D(cmap, x137, 0));
    (*color_dxy)[1] = x1*(-x136*volumeRead1D(cmap, x117, 1) + x136*volumeRead1D(cmap, x137, 1));
    (*color_dxy)[2] = x1*(-x136*volumeRead1D(cmap, x117, 2) + x136*volumeRead1D(cmap, x137, 2));
    (*color_dxy)[3] = x1*x84*(-x142*pow(x143, 2.0)*x144*x155*x166*x84 + x142*pow(x143, 2.0)*x144*x155*x166 + 0.996078431372549*x142*x143*(x136*x138 - x136*x140));
    (*color_dy)[0] = -x1*x112*x53 + x1*x112*x79;
    (*color_dy)[1] = -x1*x112*x80 + x1*x112*x81;
    (*color_dy)[2] = -x1*x112*x82 + x1*x112*x83;
    (*color_dy)[3] = x85*x90*x91*(x1*x112*x86 - x1*x112*x88);
 
 }
 
 