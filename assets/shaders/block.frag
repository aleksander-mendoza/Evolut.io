#version 450
#define IS_AVAILABLE_SAMPLER_BLOCKS
#include "render_fragment_descriptors.comp"

layout (location = 0) out vec4 FragColor;
layout (location = 0) in vec2 UV;

void main()
{
    FragColor = texture( blocksSampler, UV );
}