#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTex;

layout(binding = 1) uniform UniformBufferObject {
    mat4 mvp;
} ubo;

void main() {
    gl_Position = ubo.mvp * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
    fragTex = inTexCoord;
}