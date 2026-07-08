#version 330

in vec2 in_position;
in vec2 in_velocity;
in vec4 in_color;

out vec4 v_color;

uniform float base_size;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    // Size scales based on speed, but clamp it to prevent massive blobs
    float speed = length(in_velocity);
    gl_PointSize = base_size + clamp(speed * 5.0, 0.0, 5.0);
    v_color = in_color;
}
