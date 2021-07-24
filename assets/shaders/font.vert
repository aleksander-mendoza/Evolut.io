#version 330 core
layout (location = 11) in uint ascii_char;
out vec2 UV;
uniform vec4 text_position_and_size;
#define ASCII_ROWS 8
#define ASCII_COLS 16
void main() {
    const float z = 0.;
    const vec2 A = vec2(0,0);
    const vec2 B = vec2(1,0);
    const vec2 C = vec2(1,1);
    const vec2 D = vec2(0,1);
    const vec2 glyph_size_uv = vec2(1./float(ASCII_COLS), 1./float(ASCII_ROWS));
    const vec2[6] vertices = vec2[6](
        A, B, C, A, C, D
    );
    uint row = ascii_char/uint(ASCII_COLS);
    vec2 ascii_pos_uv = vec2(ascii_char-row*uint(ASCII_COLS),row)*glyph_size_uv; // ASCII has 128 symbols (extended ASCII doesn't count) and is arranged into 16x8 grid.
    // Second (lower) nibble encodes the x position (starting from left), and the first (upper) nibble encodes y position (starting from top)
    vec2 v = vertices[gl_VertexID];
    vec2 t = vec2(v.x,1.-v.y);
    vec2 ascii_size_uv = t*glyph_size_uv;
    UV = ascii_pos_uv+ascii_size_uv;
    gl_Position = vec4(text_position_and_size.xy + v*text_position_and_size.zw, z, 1.0);
    gl_Position.x += gl_InstanceID*text_position_and_size.z;
}
