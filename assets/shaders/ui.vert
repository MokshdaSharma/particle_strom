#version 330

in vec2 in_position;
in vec2 in_texcoord;

// The position and size of the UI rect in NDC
uniform vec2 rect_pos;
uniform vec2 rect_size;

out vec2 v_texcoord;

void main() {
    // scale and translate the quad
    vec2 pos = (in_position * rect_size) + rect_pos;
    gl_Position = vec4(pos, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
