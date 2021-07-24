#version 330 core

layout (location = 10) in vec3 instance_position;
layout (location = 12) in uvec2 body_part_and_bone_variant;
layout (location = 14) in vec4 rotation;
out vec2 UV;

layout (std140) uniform Matrices
{
    mat4 MVP;
    mat4 MV;
};

vec4 quat_conj(vec4 q)
{
    return vec4(-q.x, -q.y, -q.z, q.w);
}

vec4 quat_mult(vec4 q1, vec4 q2)
{
    vec4 qr;
    qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
    qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
    qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
    qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
    return qr;
}

vec3 quat_rotate_vec(vec4 q, vec3 v){
    return quat_mult(q,quat_mult(vec4(v,0.),quat_conj(q))).xyz;
}

void main()
{

    //First we define a cube. This will be the basis for assembling any mob.
    const vec3 A = vec3(0,0,0);// left bottom front
    const vec3 B = vec3(1,0,0);// right bottom front
    const vec3 C = vec3(1,0,1);// right bottom back
    const vec3 D = vec3(0,0,1);// left bottom back
    const vec3 E = vec3(0,1,0);// left top front
    const vec3 F = vec3(1,1,0);// right top front
    const vec3 G = vec3(1,1,1);// right top back
    const vec3 H = vec3(0,1,1);// left top back

    const vec3[6*6] vertices = vec3[6*6](
        // YPlus ortientation = block's top face
        G, F, E, G, E, H,
         // YMinus ortientation = block's bottom face
        C, A, B, C, D, A,
         // XPlus ortientation = block's right face
        G, B, F, B, G, C,
        // XMinus ortientation = block's left face
        A, D, H, A, H, E,
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
        // YPlus ortientation = block's top face
        M, L, K, M, K, N,
        // YMinus ortientation = block's bottom face
        M, K, L, M, N, K,
        // XPlus ortientation = block's right face
        M, K, N, K, M, L,
        // XMinus ortientation = block's left face
        L, K, N, L, N, M,
        // ZPlus ortientation = block's back face
        M, L, K, N, M, K,
        // ZMinus ortientation = block's front face
        M, L, K, M, K, N
    );

    //Now we list a bunch of predefined sizes, that will be used as hands, legs, heads etc.
    //All sizes are specified in a specific unit of minecraft pixels (every block is 16x16 pixels)
    const float unit = 1./16.;
    const vec3 zombie_leg_size=vec3(unit*4.,unit*12.,unit*4.);
    const vec3 zombie_arm_size=zombie_leg_size;
    const vec3 zombie_torso_size=vec3(unit*8.,unit*12.,unit*4.);
    const vec3 zombie_head_size=vec3(unit*8.,unit*8.,unit*8.);
    const vec3[6*2] body_part_size_and_joint_pos = vec3[6*2](
        zombie_leg_size,   vec3(unit*4., unit*12., unit*2), // Zombie left leg
        zombie_leg_size,   vec3(unit*0., unit*12., unit*2), // Zombie right leg
        zombie_torso_size, zombie_torso_size/2.,   // Zombie torso
        zombie_head_size,  vec3(unit*4,0,unit*4),           // Zombie head
        zombie_arm_size,   vec3(unit*4, unit*10., unit*2),  // Zombie left arm
        zombie_arm_size,   vec3(unit*0, unit*10., unit*2)   // Zombie right arm
    );

    const float pixel = 1./64.; //size of a single texture pixel measured in UV coordinates
    const vec2[6*4*2] tex_offset_and_size = vec2[6*4*2](
        /////// Zombie leg
        // YPlus
        vec2(pixel*4.,pixel*12.),vec2(pixel*4.,pixel*4.),
        // YMinus
        vec2(pixel*8.,pixel*12.),vec2(pixel*4.,pixel*4.),
        // XPlus
        vec2(pixel*8.,pixel*0.),vec2(pixel*4.,pixel*12.),
        // XMinus
        vec2(pixel*0.,pixel*0.),vec2(pixel*4.,pixel*12.),
        // ZPlus
        vec2(pixel*12.,pixel*0.),vec2(pixel*4.,pixel*12.),
        // ZMinus
        vec2(pixel*4.,pixel*0.),vec2(pixel*4.,pixel*12.),
        /////// Zombie arm
        // YPlus
        vec2(pixel*(4.+16.),pixel*12.),vec2(pixel*4.,pixel*4.),
        // YMinus
        vec2(pixel*(8.+16.),pixel*12.),vec2(pixel*4.,pixel*4.),
        // XPlus
        vec2(pixel*(8.+16.),pixel*0.),vec2(pixel*4.,pixel*12.),
        // XMinus
        vec2(pixel*(0.+16.),pixel*0.),vec2(pixel*4.,pixel*12.),
        // ZPlus
        vec2(pixel*(12.+16.),pixel*0.),vec2(pixel*4.,pixel*12.),
        // ZMinus
        vec2(pixel*(4.+16.),pixel*0.),vec2(pixel*4.,pixel*12.),
        /////// Zombie torso
        // YPlus
        vec2(pixel*4.,pixel*28.),vec2(pixel*8.,pixel*4.),
        // YMinus
        vec2(pixel*12.,pixel*28.),vec2(pixel*8.,pixel*4.),
        // XPlus
        vec2(pixel*12.,pixel*16.),vec2(pixel*4.,pixel*12.),
        // XMinus
        vec2(pixel*0.,pixel*16.),vec2(pixel*4.,pixel*12.),
        // ZPlus
        vec2(pixel*16.,pixel*16.),vec2(pixel*8.,pixel*12.),
        // ZMinus
        vec2(pixel*4.,pixel*16.),vec2(pixel*8.,pixel*12.),
        /////// Zombie head
        // YPlus
        vec2(pixel*8.,pixel*40.),vec2(pixel*8.,pixel*8.),
        // YMinus
        vec2(pixel*16.,pixel*40.),vec2(pixel*8.,pixel*8.),
        // XPlus
        vec2(pixel*16.,pixel*32.),vec2(pixel*8.,pixel*8.),
        // XMinus
        vec2(pixel*0.,pixel*32.),vec2(pixel*8.,pixel*8.),
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
    float bone_variant = float(body_part_and_bone_variant.y);
    uint body_part = body_part_and_bone_variant.x;
    uint bone_idx = body_part_to_bone_idx[body_part];
    float bone_stride = tex_stride[bone_idx];
    vec3 bone_size = body_part_size_and_joint_pos[body_part*uint(2)];
    vec3 joint_position = body_part_size_and_joint_pos[body_part*uint(2)+uint(1)];
    uint face_idx = uint(gl_VertexID) / num_faces;
    vec3 local_vertex_pos = vertices[gl_VertexID] * bone_size - joint_position;
    vec3 rotated_vertex_pos = quat_rotate_vec(rotation, local_vertex_pos);
    vec3 absolute_vertex_pos = rotated_vertex_pos + instance_position;
    gl_Position = MVP * vec4(absolute_vertex_pos, 1.0);
    uint tex_idx = bone_idx*num_faces*uint(2) + face_idx*uint(2);
    vec2 tex_offset = tex_offset_and_size[tex_idx];
    vec2 tex_size = tex_offset_and_size[tex_idx+uint(1)];
    UV = texture_uv[gl_VertexID] * tex_size + tex_offset;
    UV.x += bone_stride*bone_variant;
}
