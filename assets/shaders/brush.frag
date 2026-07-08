#version 330
out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    // soft circle
    float alpha = smoothstep(0.5, 0.4, dist);
    // Fully opaque white
    fragColor = vec4(1.0, 1.0, 1.0, alpha);
}
