const FILTER_NEAREST:u32 = 0;
const FILTER_LINEAR:u32 = 1;

const PI:f32 = 3.1415926535897932384626433832795;
const TWO_PI:f32 = 6.283185307179586476925286766559;

const INTERPOLATION_NEAREST:u32 = 0u;
const INTERPOLATION_BILINEAR:u32 = 1u;
const INTERPOLATION_BICUBIC:u32 = 2u;
const INTERPOLATION_SPLINE:u32 = 3u;
const INTERPOLATION_LANCZOS:u32 = 4u;

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    resolution: vec2<u32>,
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
    gradient_vis_scale:f32,

    clamp_gradients:u32,
    gradient_clamp_value_x: f32,
    gradient_clamp_value_y: f32,
    gradient_clamp_value_xy: f32,

    random_ray_start_offset:u32,
    cmap_fd_h: f32,
    cmap_grad_clip: f32,
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

fn rand(co:vec2<f32>)->f32{
    return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
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
    var start = max(0., intersec.x) + 1e-4;
    if bool(settings.random_ray_start_offset){
        start += rand(pixel_pos) * settings.step_size;
    }
    ray.orig += start * ray.dir;

    // *color_out = vec4<f32>((ray.orig-aabb.min)/aabb_size,1.);
    // if true {
    // return;
    // }


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

        var sample_color = vec4<f32>(0.);
        var sample_dx = vec4<f32>(0.);
        var sample_dy = vec4<f32>(0.);
        var sample_dxy = vec4<f32>(0.);


        let inv_proj_inv_view = camera.view_inv * camera.proj_inv;
        if settings.upscaling_method == INTERPOLATION_SPLINE {
            sample_color_grad(inv_proj_inv_view, pos_ndc, &sample_color, &sample_dx, &sample_dxy, &sample_dy);
        } else {
            sample_color(inv_proj_inv_view, pos_ndc, &sample_color);
        }

        // if bool(settings.clamp_gradients) {
        //     let inv_width = 0.25*f32(camera.resolution.x);
        //     let inv_height = 0.25*f32(camera.resolution.y);
        //     let cutoff_x = settings.gradient_clamp_value_x;
        //     let cutoff_y = settings.gradient_clamp_value_y;
        //     let cutoff_xy = settings.gradient_clamp_value_xy;
        //     sample_dx = clamp(sample_dx, vec4<f32>(-cutoff_x), vec4<f32>(cutoff_x));
        //     sample_dy = clamp(sample_dy, vec4<f32>(-cutoff_y), vec4<f32>(cutoff_y));
        //     sample_dxy = clamp(sample_dxy, vec4<f32>(-cutoff_xy), vec4<f32>(cutoff_xy));
        // }

        blend_gradient(
            color,color_dx,color_dxy,color_dy,
            sample_color,sample_dx,sample_dxy,sample_dy,
            &color,&color_dx, &color_dxy, &color_dy,
        );

        if bool(settings.clamp_gradients) {
            let inv_width = 0.25*f32(camera.resolution.x);
            let inv_height = 0.25*f32(camera.resolution.y);
            let cutoff_x = settings.gradient_clamp_value_x;
            let cutoff_y = settings.gradient_clamp_value_y;
            let cutoff_xy = settings.gradient_clamp_value_xy;
            color_dx = clamp(color_dx, vec4<f32>(-cutoff_x), vec4<f32>(cutoff_x));
            color_dy = clamp(color_dy, vec4<f32>(-cutoff_y), vec4<f32>(cutoff_y));
            color_dxy = clamp(color_dxy, vec4<f32>(-cutoff_xy), vec4<f32>(cutoff_xy));
        }


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




fn volumeRead3D(volume: texture_3d<f32>, loc:vec3<i32>,channel:i32)->f32{
    return textureLoad(volume,loc,0)[channel];
}

fn volumeRead1D(tex: texture_2d<f32>, loc:i32,channel:i32)->f32{
    return textureLoad(tex,vec2<i32>(loc,0),0)[channel];
}

/// _h is just a dummy value to make the function signature match the one in the original code
fn SampleCmap(cmap: texture_2d<f32>, value:f32,channel:i32)->f32{
    return textureSample(cmap,cmap_sampler, vec2<f32>(value, 0.5))[channel];
}

fn sample_volume(volume: texture_3d<f32>, pos:vec3<f32>) -> f32 {
    let pos_n = (pos-settings.volume_aabb.min) / (settings.volume_aabb.max-settings.volume_aabb.min);
    return textureSample(volume, volume_sampler, pos_n).r;
}

fn textureSize3D(v: texture_3d<f32>,dim:i32) -> u32 {
    return textureDimensions(v)[dim];
}
fn textureSize2D(v: texture_2d<f32>,dim:i32) -> u32 {
    return textureDimensions(v)[dim];
}



/// generated with sympy
fn sample_color_grad(inv_proj_inv_view: mat4x4<f32>, pos_ndc: vec3<f32>, color: ptr<function,vec4<f32>>, color_dx: ptr<function,vec4<f32>>, color_dxy: ptr<function,vec4<f32>>, color_dy: ptr<function,vec4<f32>>) {
   let x0 = -settings.vmin;
   let x1 = 1.0/(settings.vmax + x0);
   let x2 = f32(textureSize3D(volume, 2));
   let x3 = -settings.volume_aabb.min[2];
   let x4 = 1.0/(x3 + settings.volume_aabb.max[2]);
   let x5 = x2*x4*(x3 + inv_proj_inv_view[0][2]*pos_ndc[0] + inv_proj_inv_view[1][2]*pos_ndc[1] + inv_proj_inv_view[2][2]*pos_ndc[2] + 1.0*inv_proj_inv_view[3][2]) - 0.5;
   let x6 = fract(x5);
   let x7 = f32(textureSize3D(volume, 1));
   let x8 = -settings.volume_aabb.min[1];
   let x9 = 1.0/(x8 + settings.volume_aabb.max[1]);
   let x10 = x7*x9*(x8 + inv_proj_inv_view[0][1]*pos_ndc[0] + inv_proj_inv_view[1][1]*pos_ndc[1] + inv_proj_inv_view[2][1]*pos_ndc[2] + 1.0*inv_proj_inv_view[3][1]) - 0.5;
   let x11 = fract(x10);
   let x12 = f32(textureSize3D(volume, 0));
   let x13 = -settings.volume_aabb.min[0];
   let x14 = 1.0/(x13 + settings.volume_aabb.max[0]);
   let x15 = x12*x14*(x13 + inv_proj_inv_view[0][0]*pos_ndc[0] + inv_proj_inv_view[1][0]*pos_ndc[1] + inv_proj_inv_view[2][0]*pos_ndc[2] + 1.0*inv_proj_inv_view[3][0]) - 0.5;
   let x16 = fract(x15);
   let x17 = x11*x16;
   let x18 = floor(x15);
   let x19 = x18 + 1;
   let x20 = floor(x10);
   let x21 = x20 + 1;
   let x22 = floor(x5);
   let x23 = x22 + 1;
   let x24 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x19, x21, x23)), 0);
   let x25 = x16 - 1;
   let x26 = -x25;
   let x27 = x11*x26;
   let x28 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x18, x21, x23)), 0);
   let x29 = x11 - 1;
   let x30 = -x29;
   let x31 = x16*x30;
   let x32 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x19, x20, x23)), 0);
   let x33 = x6 - 1;
   let x34 = -x33;
   let x35 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x19, x21, x22)), 0);
   let x36 = x26*x30;
   let x37 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x18, x20, x23)), 0);
   let x38 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x18, x21, x22)), 0);
   let x39 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x19, x20, x22)), 0);
   let x40 = volumeRead3D(volume, vec3<i32>(vec3<f32>(x18, x20, x22)), 0);
   let x41 = x1*(x0 + x17*x24*x6 + x17*x34*x35 + x27*x28*x6 + x27*x34*x38 + x31*x32*x6 + x31*x34*x39 + x34*x36*x40 + x36*x37*x6);
   let x42 = 1 - 0.996078431372549*SampleCmap(cmap, x41, 3);
   let x43 = settings.distance_scale*settings.step_size;
   let x44 = pow(x42, x43);
   let x45 = -settings.cmap_grad_clip;
   let x46 = 1.0/settings.cmap_fd_h;
   let x47 = (1.0/2.0)*settings.cmap_fd_h;
   let x48 = x41 + x47;
   let x49 = x41 - x47;
   let x50 = x1*min(settings.cmap_grad_clip, max(x45, f32(x46*(SampleCmap(cmap, x48, 0) - SampleCmap(cmap, x49, 0)))));
   let x51 = x12*x14;
   let x52 = x51*inv_proj_inv_view[0][0];
   let x53 = x30*x52;
   let x54 = x34*x53;
   let x55 = x53*x6;
   let x56 = x11*x52;
   let x57 = x34*x56;
   let x58 = x7*x9;
   let x59 = x58*inv_proj_inv_view[0][1];
   let x60 = x26*x59;
   let x61 = x34*x60;
   let x62 = x6*x60;
   let x63 = x16*x59;
   let x64 = x34*x63;
   let x65 = x2*x4;
   let x66 = x65*inv_proj_inv_view[0][2];
   let x67 = x36*x66;
   let x68 = x27*x66;
   let x69 = x31*x66;
   let x70 = x56*x6;
   let x71 = x6*x63;
   let x72 = x17*x66;
   let x73 = x24*x70 + x24*x71 + x24*x72 - x28*x70 - x32*x71 - x35*x72;
   let x74 = x28*x62 + x28*x68 + x32*x55 + x32*x69 + x35*x57 + x35*x64 - x37*x55 - x37*x62 + x37*x67 - x38*x57 + x38*x61 - x38*x68 + x39*x54 - x39*x64 - x39*x69 - x40*x54 - x40*x61 - x40*x67 + x73;
   let x75 = x1*min(settings.cmap_grad_clip, max(x45, f32(x46*(SampleCmap(cmap, x48, 1) - SampleCmap(cmap, x49, 1)))));
   let x76 = x1*min(settings.cmap_grad_clip, max(x45, f32(x46*(SampleCmap(cmap, x48, 2) - SampleCmap(cmap, x49, 2)))));
   let x77 = x1*x43;
   let x78 = 0.996078431372549*x77*min(settings.cmap_grad_clip, max(x45, f32(x46*(SampleCmap(cmap, x48, 3) - SampleCmap(cmap, x49, 3)))));
   let x79 = 1.0/x42;
   let x80 = x51*inv_proj_inv_view[1][0];
   let x81 = x30*x80;
   let x82 = x34*x81;
   let x83 = x6*x81;
   let x84 = x11*x80;
   let x85 = x34*x84;
   let x86 = x58*inv_proj_inv_view[1][1];
   let x87 = x26*x86;
   let x88 = x34*x87;
   let x89 = x6*x87;
   let x90 = x16*x86;
   let x91 = x34*x90;
   let x92 = x65*inv_proj_inv_view[1][2];
   let x93 = x36*x92;
   let x94 = x27*x92;
   let x95 = x31*x92;
   let x96 = x6*x84;
   let x97 = x6*x90;
   let x98 = x17*x92;
   let x99 = x24*x96 + x24*x97 + x24*x98 - x28*x96 - x32*x97 - x35*x98;
   let x100 = x28*x89 + x28*x94 + x32*x83 + x32*x95 + x35*x85 + x35*x91 - x37*x83 - x37*x89 + x37*x93 - x38*x85 + x38*x88 - x38*x94 + x39*x82 - x39*x91 - x39*x95 - x40*x82 - x40*x88 - x40*x93 + x99;
   let x101 = x25*x29;
   let x102 = x11*x25;
   let x103 = x16*x29;
   let x104 = -settings.vmin - x101*x33*x40 - x102*x28*x6 - x103*x32*x6 + x11*x16*x24*x6 + x11*x25*x33*x38 + x16*x29*x33*x39 - x17*x33*x35 + x25*x29*x37*x6;
   let x105 = 2*x1*x104;
   let x106 = (1.0/2.0)*settings.cmap_fd_h + (1.0/2.0)*x105;
   let x107 = -settings.cmap_fd_h;
   let x108 = (1.0/2.0)*x105 + (1.0/2.0)*x107;
   let x109 = f32(x46*(SampleCmap(cmap, x106, 0) - SampleCmap(cmap, x108, 0)));
   let x110 = max(x109, x45);
   let x111 = x52*x86;
   let x112 = x111*x6;
   let x113 = x59*x80;
   let x114 = x113*x6;
   let x115 = x56*x92;
   let x116 = x66*x84;
   let x117 = x63*x92;
   let x118 = x66*x90;
   let x119 = x25*x59;
   let x120 = x119*x92;
   let x121 = x25*x86;
   let x122 = x121*x66;
   let x123 = x29*x52;
   let x124 = x123*x92;
   let x125 = x29*x80;
   let x126 = x125*x66;
   let x127 = x111*x33;
   let x128 = x113*x33;
   let x129 = x112*x24 - x112*x28 - x112*x32 + x112*x37 + x114*x24 - x114*x28 - x114*x32 + x114*x37 + x115*x24 - x115*x28 - x115*x35 + x115*x38 + x116*x24 - x116*x28 - x116*x35 + x116*x38 + x117*x24 - x117*x32 - x117*x35 + x117*x39 + x118*x24 - x118*x32 - x118*x35 + x118*x39 - x120*x28 + x120*x37 + x120*x38 - x120*x40 - x122*x28 + x122*x37 + x122*x38 - x122*x40 - x124*x32 + x124*x37 + x124*x39 - x124*x40 - x126*x32 + x126*x37 + x126*x39 - x126*x40 - x127*x35 + x127*x38 + x127*x39 - x127*x40 - x128*x35 + x128*x38 + x128*x39 - x128*x40;
   let x130 = x1*x46;
   let x131 = x1*x104;
   let x132 = SampleCmap(cmap, x131, 0);
   let x133 = settings.cmap_fd_h + x131;
   let x134 = x125*x33;
   let x135 = x125*x6;
   let x136 = x33*x84;
   let x137 = x121*x33;
   let x138 = x121*x6;
   let x139 = x33*x90;
   let x140 = x101*x92;
   let x141 = x102*x92;
   let x142 = x103*x92;
   let x143 = x134*x39 - x134*x40 - x135*x32 + x135*x37 - x136*x35 + x136*x38 + x137*x38 - x137*x40 - x138*x28 + x138*x37 - x139*x35 + x139*x39 + x140*x37 - x140*x40 - x141*x28 + x141*x38 - x142*x32 + x142*x39 + x99;
   let x144 = x107 + x131;
   let x145 = x123*x33;
   let x146 = x123*x6;
   let x147 = x33*x56;
   let x148 = x119*x33;
   let x149 = x119*x6;
   let x150 = x33*x63;
   let x151 = x101*x66;
   let x152 = x102*x66;
   let x153 = x103*x66;
   let x154 = x145*x39 - x145*x40 - x146*x32 + x146*x37 - x147*x35 + x147*x38 + x148*x38 - x148*x40 - x149*x28 + x149*x37 - x150*x35 + x150*x39 + x151*x37 - x151*x40 - x152*x28 + x152*x38 - x153*x32 + x153*x39 + x73;
   let x155 = f32(x46*(SampleCmap(cmap, x106, 1) - SampleCmap(cmap, x108, 1)));
   let x156 = max(x155, x45);
   let x157 = SampleCmap(cmap, x131, 1);
   let x158 = f32(x46*(SampleCmap(cmap, x106, 2) - SampleCmap(cmap, x108, 2)));
   let x159 = max(x158, x45);
   let x160 = SampleCmap(cmap, x131, 2);
   let x161 = f32(x46*(SampleCmap(cmap, x106, 3) - SampleCmap(cmap, x108, 3)));
   let x162 = max(x161, x45);
   let x163 = min(settings.cmap_grad_clip, x162);
   let x164 = SampleCmap(cmap, x131, 3);
   let x165 = 1 - 0.996078431372549*x164;
   let x166 = pow(x165, x43);
   let x167 = 1.0/x165;
   let x168 = 0.992172241445598*x1*pow(x163, 2.0);

   (*color)[0] = SampleCmap(cmap, x41, 0);
   (*color)[1] = SampleCmap(cmap, x41, 1);
   (*color)[2] = SampleCmap(cmap, x41, 2);
   (*color)[3] = 1 - x44;
   (*color_dx)[0] = x50*x74;
   (*color_dx)[1] = x74*x75;
   (*color_dx)[2] = x74*x76;
   (*color_dx)[3] = x44*x74*x78*x79;
   (*color_dxy)[0] = x1*(x129*min(settings.cmap_grad_clip, x110) + x154*step(0.,-(settings.cmap_grad_clip + x109))*step(0.,-(settings.cmap_grad_clip - x110))*f32(x130*(x143*min(settings.cmap_grad_clip, max(x45, f32(x46*(-x132 + SampleCmap(cmap, x133, 0))))) - x143*min(settings.cmap_grad_clip, max(x45, f32(x46*(x132 - SampleCmap(cmap, x144, 0))))))));
   (*color_dxy)[1] = x1*(x129*min(settings.cmap_grad_clip, x156) + x154*step(0.,-(settings.cmap_grad_clip + x155))*step(0.,-(settings.cmap_grad_clip - x156))*f32(x130*(x143*min(settings.cmap_grad_clip, max(x45, f32(x46*(-x157 + SampleCmap(cmap, x133, 1))))) - x143*min(settings.cmap_grad_clip, max(x45, f32(x46*(x157 - SampleCmap(cmap, x144, 1))))))));
   (*color_dxy)[2] = x1*(x129*min(settings.cmap_grad_clip, x159) + x154*step(0.,-(settings.cmap_grad_clip + x158))*step(0.,-(settings.cmap_grad_clip - x159))*f32(x130*(x143*min(settings.cmap_grad_clip, max(x45, f32(x46*(-x160 + SampleCmap(cmap, x133, 2))))) - x143*min(settings.cmap_grad_clip, max(x45, f32(x46*(x160 - SampleCmap(cmap, x144, 2))))))));
   (*color_dxy)[3] = x77*(0.996078431372549*x129*x163*x166*x167 - x143*x154*x166*pow(x167, 2.0)*x168*x43 + x143*x154*x166*pow(x167, 2.0)*x168 + 0.996078431372549*x154*x166*x167*step(0.,-(settings.cmap_grad_clip + x161))*step(0.,-(settings.cmap_grad_clip - x162))*f32(x130*(x143*min(settings.cmap_grad_clip, max(x45, f32(x46*(-x164 + SampleCmap(cmap, x133, 3))))) - x143*min(settings.cmap_grad_clip, max(x45, f32(x46*(x164 - SampleCmap(cmap, x144, 3))))))));
   (*color_dy)[0] = x100*x50;
   (*color_dy)[1] = x100*x75;
   (*color_dy)[2] = x100*x76;
   (*color_dy)[3] = x100*x44*x78*x79;

}



/// generated with sympy
fn sample_color(inv_proj_inv_view: mat4x4<f32>, pos_ndc: vec3<f32>, color: ptr<function,vec4<f32>>) {
   let x0 = -settings.vmin;
   let x1 = (x0 + sample_volume_manual(volume, vec3<f32>(inv_proj_inv_view[0][0]*pos_ndc[0] + inv_proj_inv_view[1][0]*pos_ndc[1] + inv_proj_inv_view[2][0]*pos_ndc[2] + 1.0*inv_proj_inv_view[3][0], inv_proj_inv_view[0][1]*pos_ndc[0] + inv_proj_inv_view[1][1]*pos_ndc[1] + inv_proj_inv_view[2][1]*pos_ndc[2] + 1.0*inv_proj_inv_view[3][1], inv_proj_inv_view[0][2]*pos_ndc[0] + inv_proj_inv_view[1][2]*pos_ndc[1] + inv_proj_inv_view[2][2]*pos_ndc[2] + 1.0*inv_proj_inv_view[3][2])))/(settings.vmax + x0);

   (*color)[0] = SampleCmap(cmap, x1, 0);
   (*color)[1] = SampleCmap(cmap, x1, 1);
   (*color)[2] = SampleCmap(cmap, x1, 2);
   (*color)[3] = 1 - pow(1 - 0.996078431372549*SampleCmap(cmap, x1, 3), settings.distance_scale*settings.step_size);

}


/// generated with sympy
fn sample_volume_manual(volume: texture_3d<f32>,pos: vec3<f32>) -> f32 {
    // if bool(settings.hardware_interpolation){
    //     return sample_volume(volume, pos);
    // }
   let x0 = -settings.volume_aabb.min[2];
   let x1 = (x0 + pos[2])*f32(textureSize3D(volume, 2))/(x0 + settings.volume_aabb.max[2]) - 0.5;
   let x2 = fract(x1);
   let x3 = -settings.volume_aabb.min[1];
   let x4 = (x3 + pos[1])*f32(textureSize3D(volume, 1))/(x3 + settings.volume_aabb.max[1]) - 0.5;
   let x5 = fract(x4);
   let x6 = -settings.volume_aabb.min[0];
   let x7 = (x6 + pos[0])*f32(textureSize3D(volume, 0))/(x6 + settings.volume_aabb.max[0]) - 0.5;
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

   var sample_volume_manual_result:f32;
   sample_volume_manual_result = x17*x2*volumeRead3D(volume, vec3<i32>(vec3<f32>(x10, x13, x15)), 0) + x17*x20*volumeRead3D(volume, vec3<i32>(vec3<f32>(x10, x13, x14)), 0) + x19*x2*volumeRead3D(volume, vec3<i32>(vec3<f32>(x11, x12, x15)), 0) + x19*x20*volumeRead3D(volume, vec3<i32>(vec3<f32>(x11, x12, x14)), 0) + x2*x21*volumeRead3D(volume, vec3<i32>(vec3<f32>(x10, x12, x15)), 0) + x2*x9*volumeRead3D(volume, vec3<i32>(vec3<f32>(x11, x13, x15)), 0) + x20*x21*volumeRead3D(volume, vec3<i32>(vec3<f32>(x10, x12, x14)), 0) + x20*x9*volumeRead3D(volume, vec3<i32>(vec3<f32>(x11, x13, x14)), 0);
   return sample_volume_manual_result;

}
