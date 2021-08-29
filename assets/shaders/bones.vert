#version 450
#extension GL_EXT_debug_printf : enable

#define IS_AVAILABLE_BUFFER_MVP
#define IS_AVAILABLE_BUFFER_PARTICLES
#include "render_vertex_descriptors.comp"

layout(location = 0) in vec3 center;
layout(location = 1) in float color1;
layout(location = 2) in vec3 size;
layout(location = 3) in float color2;
layout(location = 0) out vec4 texColor;


void main() {
    const vec3 A = vec3(-1,-1,-1);// left bottom front
    const vec3 B = vec3(1,-1,-1);// right bottom front
    const vec3 C = vec3(1,-1,1);// right bottom back
    const vec3 D = vec3(-1,-1,1);// left bottom back
    const vec3 E = vec3(-1,1,-1);// left top front
    const vec3 F = vec3(1,1,-1);// right top front
    const vec3 G = vec3(1,1,1);// right top back
    const vec3 H = vec3(-1,1,1);// left top back

    const vec3[6*6] direction_per_vertex = vec3[6*6](
        // XPlus ortientation = block's right face
        G, B, F, B, G, C,
        // XMinus ortientation = block's left face
        A, D, H, A, H, E,
        // YPlus ortientation = block's top face
        G, F, E, G, E, H,
        // YMinus ortientation = block's bottom face
        C, A, B, C, D, A,
        // ZPlus ortientation = block's back face
        H, D, C, G, H, C,
        // ZMinus ortientation = block's front face
        F, B, A, F, A, E
    );

    //Next we define texture UV coordinates for this cube
    const vec2 K = vec2(0,0);// left bottom
    const vec2 L = vec2(1,0);// right bottom
    const vec2 M = vec2(1,1);// right top
    const vec2 N = vec2(0,1);// left top

    const vec2[6*6] texture_uv = vec2[6*6](
        // XPlus ortientation = block's right face
        M, K, N, K, M, L,
        // XMinus ortientation = block's left face
        L, K, N, L, N, M,
        // YPlus ortientation = block's top face
        M, L, K, M, K, N,
        // YMinus ortientation = block's bottom face
        M, K, L, M, N, K,
        // ZPlus ortientation = block's back face
        M, L, K, N, M, K,
        // ZMinus ortientation = block's front face
        M, L, K, M, K, N
    );

    gl_Position = MVP * vec4(center + direction_per_vertex[gl_VertexIndex]*size, 1.0);
    gl_Position.y = -gl_Position.y;
//    uint bone_idx = body_part_to_bone_idx[part_variant];
//    float bone_stride = tex_stride[bone_idx];
//    uint face_idx = uint(gl_VertexIndex) / num_faces;
//    uint tex_idx = bone_idx*num_faces*uint(2) + face_idx*uint(2);
//    vec2 tex_offset = tex_offset_and_size[tex_idx];
//    vec2 tex_size = tex_offset_and_size[tex_idx+uint(1)];
//    fragTex = texture_uv[gl_VertexIndex] * tex_size + tex_offset;
//    fragTex.x += bone_stride*texture_variant;
    texColor = vec4(color1, color2, 0, 0);
}