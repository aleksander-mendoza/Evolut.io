
#version 330 core
out vec4 FragColor;
in vec2 UV;
uniform sampler2D myTextureSampler;
void main()
{
    vec3 MaterialDiffuseColor = texture( myTextureSampler, UV ).rgb;
    FragColor = vec4(MaterialDiffuseColor, 1.0);
}