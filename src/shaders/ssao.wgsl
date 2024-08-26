const SSAO_RADIUS: f32 = 0.05; // = 0.05;
const SSAO_BIAS: f32 = 0.005;
const KERNEL_SIZE: u32 = 64;


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

//fn rand(co:vec2<f32>) -> f32{
//    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
//}

/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////// AO //////////////////////////////////////
const EPS = 0.05;
const M_PI = 3.141592653589;

fn ComputeDefaultBasis(normal: vec3<f32>, x: ptr<function,vec3<f32>>, y: ptr<function,vec3<f32>>) {
    // ZAP's default coordinate system for compatibility
    let z = normal;
    let yz = -z.y * z.z;
    if abs(z.z) > 0.99999f {
        *y = vec3(-z.x * z.y, 1.0f - z.y * z.y, yz);
    } else {
        *y = vec3(-z.x * z.z, yz, 1.0f - z.z * z.z);
    }
    *y = normalize(*y);

    *x = cross(*y, z);
}


//-------------------------------------------------------------------------------------------------
// Random
//-------------------------------------------------------------------------------------------------

// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
fn tea(val0: u32, val1: u32) -> u32 {
    var v0 = val0;
    var v1 = val1;
    var s0 = 0u;

    for (var n = 0u; n < 16u; n += 1u) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }

    return v0;
}

// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
fn lcg(prev: ptr<function,u32>) -> u32 {
    let LCG_A = 1664525u;
    let LCG_C = 1013904223u;
    *prev     = (LCG_A * (*prev) + LCG_C);
    return (*prev) & 0x00FFFFFFu;
}

// Generate a random float in [0, 1) given the previous RNG state
fn rnd(seed: ptr<function,u32>) -> f32 {
    return f32(lcg(seed)) / f32(0x01000000);
}

@fragment
fn ssao_frag(vertex_in: VertexOut) -> @location(0) f32 {
    let uv = vec2<f32>(vertex_in.tex_coord.x, vertex_in.tex_coord.y);
    let texture_size = textureDimensions(in_texture);
    let pixel_coords = vec2<u32>(uv * vec2<f32>(texture_size));

    let normal_depth = textureSample(in_texture, texture_sampler, uv);
    // normalized normals
    let normal = normal_depth.rgb;
    // depth in world space
    var depth = normal_depth.a;

    let p = camera.proj;
    let znear = -(p[3][2] + 1.0) / p[2][2];
    let zfar = (1.0 - p[3][2]) / p[2][2];

    if depth == 0.0 {
        depth = zfar;
    }
    let depth_n = (depth - znear) / (zfar - znear);
    //return 1.0 - depth_n; // this is just a placeholder that renders the depth

    // view space position
    let frag_pos_ndc = vec4<f32>(uv * 2.0 - 1.0, depth_n, 1.0);
    let frag_pos_view_hom = camera.proj_inv * frag_pos_ndc;
    let frag_pos_view = frag_pos_view_hom.xyz / frag_pos_view_hom.w;

    //let rotationVec = normalize(vec3<f32>(rand(uv),rand(uv*2.),rand(uv*3.)));

    // Create basis change matrix converting tangent space to view space.
    //let tangent = normalize(rotationVec - normal * dot(rotationVec, normal));
    //let bitangent = cross(normal, tangent);
    var tangent: vec3<f32>;
    var bitangent: vec3<f32>;
    ComputeDefaultBasis(normal, &tangent, &bitangent);
    let frame_matrix = mat3x3<f32>(tangent, bitangent, normal);

    // Initialize random seed.
    //uint seed = tea(u32(time_ms), pixel_coords.x + pixel_coords.y * texture_size.x);
    var seed = tea(19u, pixel_coords.x + pixel_coords.y * texture_size.x);

    // Compute occlusion factor as occlusion average over all kernel samples.
    var occlusion = 0.0;
    for (var i = 0u; i < KERNEL_SIZE; i += 1u) {
        // Convert sample position from tangent space to view space.
        // TODO read this from a precomputed array / buffer
        var sample_vec = vec3<f32>(rnd(&seed) * 2.0 - 1.0, rnd(&seed) * 2.0 - 1.0, rnd(&seed));// samples[i].xyz;
        sample_vec = normalize(sample_vec);
        sample_vec *= rnd(&seed);

        var scale = f32(i) / f32(KERNEL_SIZE);
        scale = mix(0.1, 1.0, scale * scale);
        sample_vec *= scale;

        let sample_view_space = frag_pos_view + frame_matrix * sample_vec * vec3<f32>(SSAO_RADIUS);
        let sample_screen_space_hom = camera.proj * vec4(sample_view_space, 1.0);
        let sample_screen_space = sample_screen_space_hom.xyz / sample_screen_space_hom.w * 0.5 + 0.5;

        // Apply projection matrix to view space sample to get position in clip space.
        /*var screen_space_position = vec4<f32>(sample_view_space, 1.0);
        screen_space_position = camera.proj * screen_space_position;
        screen_space_position = vec4<f32>(
            screen_space_position.x / screen_space_position.w * 0.5 + 0.5,
            screen_space_position.y / screen_space_position.y * 0.5 + 0.5,
            0.0,
            0.0
        );*/

        // Get depth at sample position (of kernel sample).
        let sample_depth = textureSample(in_texture, texture_sampler, sample_screen_space.xy).a;

        // Range check: Make sure only depth differences in the radius contribute to occlusion.
        let range_check = smoothstep(0.0, 1.0, SSAO_RADIUS / abs(depth - sample_depth));

        // Check if the sample contributes to occlusion.
        if sample_depth >= -sample_view_space.z + SSAO_BIAS {
            occlusion += range_check;
        }
    }

    return 1.0 - (occlusion / f32(KERNEL_SIZE));
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