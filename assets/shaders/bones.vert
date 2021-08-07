#version 450
#include "data.comp"


layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTex;

layout (binding = 0) uniform Matrices{
    mat4 MVP;
    mat4 MV;
};

layout(std430, set = 0, binding = 1) buffer Bones{
    Bone bones[];
};

layout(std430, set = 0, binding = 2) buffer Particles{
    Particle particles[];
};

void main() {

//    gl_Position = ubo.mvp * vec4(inPosition-gl_InstanceIndex*0.1, -gl_InstanceIndex*0.1, 1.0);
//    fragColor = inColor;
//    fragTex = inTexCoord;
}