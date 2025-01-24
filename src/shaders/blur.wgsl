// in_texture is the following:
// ssoa_frag: it is a rgba16 with the normal in rgb and the depth in a
// blur_vert_frag / blur_hor_frag: output from the ssoa_frag or the previews blur step
@group(0) @binding(0)
var in_texture : texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

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

// TODO which radius do we need? 
// this has a radius of 4 pixels
fn blur9(uv:vec2<f32>, direction:vec2<f32>) -> f32{
    let resolution = vec2<f32>(textureDimensions(in_texture));
    var color = 0.;
    let off1 = vec2<f32>(1.3846153846) * direction;
    let off2 = vec2<f32>(3.2307692308) * direction;
    color += textureSample(in_texture,texture_sampler, uv).r * 0.2270270270;
    color += textureSample(in_texture,texture_sampler, uv + (off1 / resolution)).r * 0.3162162162;
    color += textureSample(in_texture,texture_sampler, uv - (off1 / resolution)).r * 0.3162162162;
    color += textureSample(in_texture,texture_sampler, uv + (off2 / resolution)).r * 0.0702702703;
    color += textureSample(in_texture,texture_sampler, uv - (off2 / resolution)).r * 0.0702702703;
    return color;
}

// first vertical blur
// we return a gray scale color
@fragment
fn blur_vert_frag(vertex_in: VertexOut) -> @location(0) f32 {
    let uv = vec2<f32>(vertex_in.tex_coord.x, vertex_in.tex_coord.y);
    let c = blur9(uv,vec2<f32>(0.,1.));
    return c;
}



// next horizontal blur
// we return a gray scale color as vec4 and multiply it with the rgb image
@fragment
fn blur_hor_frag(vertex_in: VertexOut) -> @location(0) f32 {
    let uv = vec2<f32>(vertex_in.tex_coord.x, vertex_in.tex_coord.y);
    let c = blur9(uv,vec2<f32>(1.,0.));
    return c;
}