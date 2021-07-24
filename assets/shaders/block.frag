#version 330 core
out vec4 FragColor;
in vec2 UV;
uniform sampler2D myTextureSampler;
void main()
{
    FragColor = texture( myTextureSampler, UV );
}