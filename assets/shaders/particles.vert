#version 450
layout (location = 0) in vec3 point;
layout (location = 1) in float point_size;
layout (location = 2) in float color;
layout (location = 0) out float frag_color;
layout (binding = 0) uniform Matrices{
    mat4 MVP;
    mat4 MV;
};


const float eye_distance=0.5;//this is meant to emulate the effect of having eyes
//slightly in front of the camera, rather than directly in the centre.
void main()
{
    vec4 point4 = vec4(point,1);
    gl_Position = MVP * point4;
    gl_Position.y = -gl_Position.y;
    float point_distance = length(MV * point4)-eye_distance;
    gl_PointSize = point_size/point_distance;
    frag_color = color;
}
