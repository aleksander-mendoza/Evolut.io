#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 0) in vec2 UV;
layout(binding = 1) uniform sampler2D texSampler;
void main()
{
    FragColor = texture( texSampler, UV );
}