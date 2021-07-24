
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 vertexUV;
layout (location = 10) in vec3 instancePosition;

out vec2 UV;

uniform mat4 MVP;

void main()
{
    gl_Position = MVP * vec4(aPos + instancePosition, 1.0);
    UV = vertexUV;
}  