#version 450
#define IS_AVAILABLE_SAMPLER_MOBS
#include "render_fragment_descriptors.comp"

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec2 fragTex;


void main() {
    outColor = texture(mobsSampler, fragTex);
}