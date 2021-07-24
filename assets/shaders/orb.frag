#version 330 core
out vec4 FragColor;
in vec4 frag_color;
void main()
{
    FragColor = frag_color;
}