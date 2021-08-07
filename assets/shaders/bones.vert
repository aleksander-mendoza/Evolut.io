#version 450
#include "data.comp"

layout(location = 0) in uvec4 particle_ids;
layout(location = 1) in vec3 center;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTex;

layout (binding = 0) uniform Matrices{
    mat4 MVP;
    mat4 MV;
};


void main() {

//    gl_Position = ubo.mvp * vec4(inPosition-gl_InstanceIndex*0.1, -gl_InstanceIndex*0.1, 1.0);
//    fragColor = inColor;
//    fragTex = inTexCoord;
}