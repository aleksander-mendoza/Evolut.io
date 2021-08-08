#version 450
layout(binding = 3) uniform sampler2D mobsSampler;

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec2 fragTex;


void main() {
    outColor = texture(mobsSampler, fragTex);
}