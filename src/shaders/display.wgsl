const PI:f32 = 3.1415926535897932384626433832795;
const TWO_PI:f32 = 6.283185307179586476925286766559;

const CHANNEL_COLOR:u32 = 0;
const CHANNEL_GRAD_X:u32 = 1;
const CHANNEL_GRAD_Y:u32 = 2;
const CHANNEL_GRAD_XY:u32 = 3;


const INTERPOLATION_NEAREST:u32 = 0u;
const INTERPOLATION_BILINEAR:u32 = 1u;
const INTERPOLATION_BICUBIC:u32 = 2u;
const INTERPOLATION_SPLINE:u32 = 3u;


struct Aabb {
    @align(16) min: vec3<f32>,
    @align(16) max: vec3<f32>,
}

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


@group(0) @binding(0)
var colorBuffer: texture_2d<f32>;
@group(0) @binding(1)
var gradientBuffer_x: texture_2d<f32>;
@group(0) @binding(2)
var gradientBuffer_y: texture_2d<f32>;
@group(0) @binding(3)
var gradientBuffer_xy: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> settings: Settings;

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

fn sample_nearest(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = vec2<f32>(textureDimensions(colorBuffer));
    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    return textureLoad(colorBuffer, pixel_pos, 0);
}

fn sample_bilinear(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = vec2<f32>(textureDimensions(colorBuffer));

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);

    let z00 = textureLoad(colorBuffer, pixel_pos,0);
    let z10 = textureLoad(colorBuffer, clamp(pixel_pos + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
    let z01 = textureLoad(colorBuffer, clamp(pixel_pos + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
    let z11 = textureLoad(colorBuffer, clamp(pixel_pos + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(tex_size-1)),0);

    return mix(
        mix(z00, z10, p_frac.x),
        mix(z01, z11, p_frac.x),
        p_frac.y
    );
}


fn sample_bicubic(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = vec2<f32>(textureDimensions(colorBuffer));

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,4>;
    var dx: array<mat2x2<f32>,4>;
    var dy: array<mat2x2<f32>,4>;
    var dxy:array<mat2x2<f32>,4>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = clamp(pixel_pos + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(tex_size-1));

            let z_v = textureLoad(colorBuffer, sample_pos,0);
            let z_up = textureLoad(colorBuffer, clamp(sample_pos + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
            let z_left = textureLoad(colorBuffer, clamp(sample_pos + vec2<i32>(-1, 0), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
            let z_right = textureLoad(colorBuffer, clamp(sample_pos + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
            let z_down = textureLoad(colorBuffer, clamp(sample_pos + vec2<i32>(0, -1), vec2<i32>(0), vec2<i32>(tex_size-1)),0);

            for (var c = 0u; c < 4u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = (z_right[c] - z_left[c]) * 0.5;
                dy[c][i][j] = (z_up[c] - z_down[c]) * 0.5;
                dxy[c][i][j] = (z_right[c] + z_left[c] - 2.0 * z_v[c]) * 0.5;
            }

        }
    }

    return vec4<f32>(
        spline_interp(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[3], dx[3], dy[3], dxy[3], vec2<f32>(p_frac.y, p_frac.x))
    ); 
}

fn sample_spline(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = vec2<f32>(textureDimensions(colorBuffer));

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,4>;
    var dx: array<mat2x2<f32>,4>;
    var dy: array<mat2x2<f32>,4>;
    var dxy:array<mat2x2<f32>,4>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = clamp(pixel_pos + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(tex_size-1));

            let z_v = textureLoad(colorBuffer, sample_pos,0);
            let dx_v = textureLoad(gradientBuffer_x, sample_pos,0);
            let dy_v = -textureLoad(gradientBuffer_y, sample_pos,0);
            let dxy_v = -textureLoad(gradientBuffer_xy, sample_pos,0);

            for (var c = 0u; c < 4u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = dx_v[c];
                dy[c][i][j] = dy_v[c];
                dxy[c][i][j] = dxy_v[c];
            }

        }
    }

    return vec4<f32>(
        spline_interp(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[3], dx[3], dy[3], dxy[3], vec2<f32>(p_frac.y, p_frac.x))
    ); 
}


@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> 
{
    let tex_size = vec2<f32>(textureDimensions(colorBuffer));
    let pixel_pos = vec2<i32>(tex_size * vertex_in.tex_coord);
    var grad: vec4<f32>;
    let selected_channel = settings.selected_channel;
    let upscaling_method = settings.upscaling_method;
    var color:vec4<f32>;

    switch (selected_channel) {
        case CHANNEL_COLOR: {
            switch upscaling_method{
                case INTERPOLATION_BILINEAR:{
                    color = sample_bilinear(vertex_in.tex_coord);
                }
                case INTERPOLATION_BICUBIC:{
                    color = sample_bicubic(vertex_in.tex_coord);
                }
                case INTERPOLATION_SPLINE:{
                    color = sample_spline(vertex_in.tex_coord);
                }
                default:{
                    color = sample_nearest(vertex_in.tex_coord);
                }
            }
            if (settings.gamma_correction == 1u) {
                color = fromLinear(color);
            }
            return color;
        }
        case CHANNEL_GRAD_X: {
            grad = textureLoad(gradientBuffer_x, pixel_pos,0);
        }
        case CHANNEL_GRAD_Y: {
            grad = textureLoad(gradientBuffer_y, pixel_pos,0);
        }
        case CHANNEL_GRAD_XY: {
            grad = textureLoad(gradientBuffer_xy, pixel_pos,0);
        }
        default: {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }

    grad *= 10.;

    if grad.r < 0.0 {
        color = mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(1.0, 0.0, 0.0, 1.0), -grad.r);
    } else {
        color = mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 0.0, 1.0, 1.0), grad.r);
    }

    return color;
}

fn spline_interp( z:mat2x2<f32>, dx:mat2x2<f32>, dy: mat2x2<f32>, dxy: mat2x2<f32>, p: vec2<f32>) -> f32
{
    let f = mat4x4<f32>(
        z[0][0], z[0][1], dy[0][0], dy[0][1],
        z[1][0], z[1][1], dy[1][0], dy[1][1],
        dx[0][0], dx[0][1], dxy[0][0], dxy[0][1],
        dx[1][0], dx[1][1], dxy[1][0], dxy[1][1]
    );
    let m = mat4x4<f32>(
        1., 0., 0., 0.,
        0., 0., 1., 0.,
        -3., 3., -2., -1.,
        2., -2., 1., 1.
    );
    let a = transpose(m) * f * (m);

    let tx = vec4<f32>(1., p.x, p.x * p.x, p.x * p.x * p.x);
    let ty = vec4<f32>(1., p.y, p.y * p.y, p.y * p.y * p.y);
    return dot(tx, a * ty);
}


fn fromLinear(color: vec4<f32>) -> vec4<f32> {
    let cutoff = color.rgb < vec3<f32>(0.0031308);
    let higher = vec3<f32>(1.055) * pow(color.rgb, vec3<f32>(1.0 / 2.4)) - 0.055;
    let lower = color.rgb * 12.92;

    return vec4<f32>(mix(higher, lower, vec3<f32>(cutoff)), color.a);
}
