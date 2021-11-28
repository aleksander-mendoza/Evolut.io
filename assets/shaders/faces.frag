#version 450
#define IS_AVAILABLE_SAMPLER_BLOCKS
#include "descriptors_render_fragment.comp"

layout (location = 0) out vec4 FragColor;
layout (location = 0) in vec2 UV;

void main()
{
    FragColor = texture( blocksSampler, UV );
}