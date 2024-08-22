const SSAO_RADIUS:f32 = 0.05; // = 0.05;
const SSAO_BIAS:f32 = 0.005;
const KERNEL_SIZE:u32 = 64;


struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};


// in_texture is the following:
// ssoa_frag: it is a rgba16 with the normal in rgb and the depth in a
// blur_vert_frag / blur_hor_frag: output from the ssoa_frag or the previews blur step
@group(0) @binding(0)
var in_texture : texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

@group(0) @binding(2)
var<uniform> camera: CameraUniforms;



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

fn rand(co:vec2<f32>) -> f32{
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

@fragment
fn ssao_frag(vertex_in: VertexOut) -> @location(0) f32 {
    let uv = vec2<f32>(vertex_in.tex_coord.x, vertex_in.tex_coord.y);
    let normal_depth = textureSample(in_texture,texture_sampler,uv);
    // normalized normals
    let normal = normal_depth.rgb;
    // depth in world space
    var depth = normal_depth.a;
    if depth == 0.{
        depth = 100.;
    }
    return clamp(1.-(depth*0.4),0.,1.); // this is just a placeholder that renders the depth 

    // TODO get this to work
    // let rotationVec = normalize(vec3<f32>(rand(uv),rand(uv*2.),rand(uv*3.)));

    // // Create basis change matrix converting tangent space to view space.
    // let tangent = normalize(rotationVec - normal * dot(rotationVec, normal));
    // let bitangent = cross(normal, tangent);
    // let frameMatrix = mat3x3<f32>(tangent, bitangent, normal);

    // // Compute occlusion factor as occlusion average over all kernel samples.
    // var occlusion = 0.0;
    // for (var i = 0u; i < KERNEL_SIZE; i+=1u) {
    //     // Convert sample position from tangent space to view space.
    //     // TODO read this from a precomputed array / buffer
    //     let sample = vec3<f32>(rand(uv),rand(uv*2.),rand(uv*3.))*0.01;// samples[i].xyz;
    //     let sampleViewSpace = vec3<f32>(uv,0.) + frameMatrix * sample * vec3<f32>(SSAO_RADIUS); 

    //     // Apply projection matrix to view space sample to get position in clip space.
    //     var screenSpacePosition = vec4<f32>(sampleViewSpace, 1.0);
    //     screenSpacePosition = camera.proj * screenSpacePosition;
    //     screenSpacePosition = vec4<f32>(
    //         screenSpacePosition.x/screenSpacePosition.w * 0.5 + 0.5,
    //         screenSpacePosition.y/screenSpacePosition.y * 0.5 + 0.5,
    //         0.,
    //         0.
    //     );

    //     // Get depth at sample position (of kernel sample).
    //     let sampleDepth = textureSample(in_texture,texture_sampler, screenSpacePosition.xy).a;

    //     // Range check: Make sure only depth differences in the radius contribute to occlusion.
    //     let rangeCheck = smoothstep(0.0, 1.0, SSAO_RADIUS / abs(depth - sampleDepth));

    //     // Check if the sample contributes to occlusion.
    //     if sampleDepth >= sampleViewSpace.z + SSAO_BIAS{
    //         occlusion +=rangeCheck;
    //     }
    // }

    // return  1.0 - (occlusion / f32(KERNEL_SIZE));
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
fn blur_hor_frag(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(vertex_in.tex_coord.x, vertex_in.tex_coord.y);
    let c = blur9(uv,vec2<f32>(1.,0.));
    return vec4<f32>(c);
}