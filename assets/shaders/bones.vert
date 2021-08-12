#version 450
#extension GL_EXT_debug_printf : enable

#include "data.comp"

layout(location = 0) in uint[4] particle_ids;
layout(location = 4) in vec3 center;
layout(location = 5) in uint texture_variant;
layout(location = 6) in uint part_variant;
layout(location = 7) in vec3 normal;
layout(location = 8) in float thickness;


//layout(location = 0) out vec3 fragColor;
layout(location = 0) out vec2 fragTex;

layout (binding = 0) uniform Matrices{
    mat4 MVP;
    mat4 MV;
};

layout(std430, binding = 2) buffer Particles{
    Particle particles[];
};
//
//layout(std430, binding = 3) buffer Bones{
//    Bone bones[];
//};

void main() {
    const float width = 1;
    const vec2 A = vec2(0,width);// left bottom front
    const vec2 B = vec2(1,width);// right bottom front
    const vec2 C = vec2(1,-width);// right bottom back
    const vec2 D = vec2(0,-width);// left bottom back
    const vec2 E = vec2(3,width);// left top front
    const vec2 F = vec2(2,width);// right top front
    const vec2 G = vec2(2,-width);// right top back
    const vec2 H = vec2(3,-width);// left top back

    const vec2[6*6] particle_id_per_vertex = vec2[6*6](
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
    //Now we list a bunch of predefined sizes, that will be used as hands, legs, heads etc.
    //All sizes are specified in a specific unit of minecraft pixels (every block is 16x16 pixels)
    const float pixel = 1./64.; //size of a single texture pixel measured in UV coordinates
    const vec2[6*4*2] tex_offset_and_size = vec2[6*4*2](
        /////// Zombie leg
        // XPlus
        vec2(pixel*8.,pixel*0.),vec2(pixel*4.,pixel*12.),
        // XMinus
        vec2(pixel*0.,pixel*0.),vec2(pixel*4.,pixel*12.),
        // YPlus
        vec2(pixel*4.,pixel*12.),vec2(pixel*4.,pixel*4.),
        // YMinus
        vec2(pixel*8.,pixel*12.),vec2(pixel*4.,pixel*4.),
        // ZPlus
        vec2(pixel*12.,pixel*0.),vec2(pixel*4.,pixel*12.),
        // ZMinus
        vec2(pixel*4.,pixel*0.),vec2(pixel*4.,pixel*12.),
        /////// Zombie arm
        // XPlus
        vec2(pixel*(8.+16.),pixel*0.),vec2(pixel*4.,pixel*12.),
        // XMinus
        vec2(pixel*(0.+16.),pixel*0.),vec2(pixel*4.,pixel*12.),
        // YPlus
        vec2(pixel*(4.+16.),pixel*12.),vec2(pixel*4.,pixel*4.),
        // YMinus
        vec2(pixel*(8.+16.),pixel*12.),vec2(pixel*4.,pixel*4.),
        // ZPlus
        vec2(pixel*(12.+16.),pixel*0.),vec2(pixel*4.,pixel*12.),
        // ZMinus
        vec2(pixel*(4.+16.),pixel*0.),vec2(pixel*4.,pixel*12.),
        /////// Zombie torso
        // XPlus
        vec2(pixel*12.,pixel*16.),vec2(pixel*4.,pixel*12.),
        // XMinus
        vec2(pixel*0.,pixel*16.),vec2(pixel*4.,pixel*12.),
        // YPlus
        vec2(pixel*4.,pixel*28.),vec2(pixel*8.,pixel*4.),
        // YMinus
        vec2(pixel*12.,pixel*28.),vec2(pixel*8.,pixel*4.),
        // ZPlus
        vec2(pixel*16.,pixel*16.),vec2(pixel*8.,pixel*12.),
        // ZMinus
        vec2(pixel*4.,pixel*16.),vec2(pixel*8.,pixel*12.),
        /////// Zombie head
        // XPlus
        vec2(pixel*16.,pixel*32.),vec2(pixel*8.,pixel*8.),
        // XMinus
        vec2(pixel*0.,pixel*32.),vec2(pixel*8.,pixel*8.),
        // YPlus
        vec2(pixel*8.,pixel*40.),vec2(pixel*8.,pixel*8.),
        // YMinus
        vec2(pixel*16.,pixel*40.),vec2(pixel*8.,pixel*8.),
        // ZPlus
        vec2(pixel*0.,pixel*40.),vec2(pixel*8.,pixel*8.),
        // ZMinus
        vec2(pixel*8.,pixel*32.),vec2(pixel*8.,pixel*8.)
    );
    const uint[6] body_part_to_bone_idx = uint[6](
        uint(0), // Zombie left leg
        uint(0), // Zombie right leg
        uint(2), // Zombie torso
        uint(3), // Zombie head
        uint(1), // Zombie left hand
        uint(1) // Zombie right hand
    );
    const float[4] tex_stride = float[4](
        pixel*32., // Zombie leg
        pixel*32., // Zombie arm
        pixel*24., // Zombie torso
        pixel*24. // Zombie head
    );
    const uint num_faces = uint(6);
    const vec2 particle_id_and_normal_direction = particle_id_per_vertex[gl_VertexIndex];
    const vec3 relative = particles[particle_ids[uint(particle_id_and_normal_direction.x)]].new_position - center;
    gl_Position = MVP * vec4(center + relative*1.2 + particle_id_and_normal_direction.y*thickness*normal, 1.0);
    gl_Position.y = -gl_Position.y;
    uint bone_idx = body_part_to_bone_idx[part_variant];
    float bone_stride = tex_stride[bone_idx];
    uint face_idx = uint(gl_VertexIndex) / num_faces;
    uint tex_idx = bone_idx*num_faces*uint(2) + face_idx*uint(2);
    vec2 tex_offset = tex_offset_and_size[tex_idx];
    vec2 tex_size = tex_offset_and_size[tex_idx+uint(1)];
    fragTex = texture_uv[gl_VertexIndex] * tex_size + tex_offset;
    fragTex.x += bone_stride*texture_variant;
}