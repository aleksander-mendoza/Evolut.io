#version 450

#define IS_AVAILABLE_BUFFER_MVP
#define IS_AVAILABLE_BUFFER_CONSTANTS
#include "render_vertex_descriptors.comp"

layout (location = 0) in vec3 point;
layout (location = 1) in float point_size;
layout (location = 2) in float color;
layout (location = 0) out vec2 frag_color;

const float eye_distance=0.5;//this is meant to emulate the effect of having eyes
//slightly in front of the camera, rather than directly in the centre.
void main()
{
    vec4 point4 = vec4(point,1);
    gl_Position = MVP * point4;
    gl_Position.y = -gl_Position.y;
    float point_distance = length(MV * point4)-eye_distance;
    gl_PointSize = point_size/point_distance;
    frag_color = vec2(0, color);
}
