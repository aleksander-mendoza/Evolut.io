#version 330 core
layout (location = 0) in vec3 point;
layout (location = 6) in float point_size;
layout (location = 1) in vec4 color;
out vec4 frag_color;
layout (std140) uniform Matrices
{
    mat4 MVP;
    mat4 MV;
};

const float eye_distance=0.5;//this is meant to emulate the effect of having eyes
//slightly in front of the camera, rather than directly in the centre.
void main()
{
    vec4 point4 = vec4(point,1);
    gl_Position = MVP * point4;
    float point_distance = length(MV * point4)-eye_distance;
    gl_PointSize = point_size/point_distance;
    frag_color = color;
}
