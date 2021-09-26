#version 450

#define IS_AVAILABLE_BUFFER_MVP
#include "descriptors_render_vertex.comp"

layout (location = 0) in vec3 position;
layout (location = 1) in float energy;

layout (location = 0) out float frag_color;
const float point_size = 40;
const float eye_distance=0.5;//this is meant to emulate the effect of having eyes
//slightly in front of the camera, rather than directly in the centre.
void main()
{
    vec4 point4 = vec4(position,1);
    gl_Position = MVP * point4;
    gl_Position.y = -gl_Position.y;
    float point_distance = length(MV * point4)-eye_distance;
    gl_PointSize = point_size/point_distance;
    frag_color = energy;
}