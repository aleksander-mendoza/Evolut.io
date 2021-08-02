#version 450
layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTex;


void main() {
    outColor = texture(texSampler, fragTex);
}