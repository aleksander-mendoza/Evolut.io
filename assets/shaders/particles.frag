#version 450
layout(location=0) out vec4 FragColor;
layout(location=0) in float frag_color;
void main()
{
    FragColor = vec4(frag_color, 1-frag_color,0,1);
}