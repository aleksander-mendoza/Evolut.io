#version 450
layout(location=0) out vec4 FragColor;
layout(location=0) in vec2 frag_color;
void main()
{
    FragColor = vec4(frag_color.x, frag_color.y,0,1);
}