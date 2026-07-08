#version 330

in vec4 v_color;
out vec4 fragColor;

void main() {
    // Make particles circular with a soft edge
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) {
        discard;
    }
    
    // Soft glow - multiply alpha by 0.15 so 50k particles don't blow out to white instantly
    float alpha = smoothstep(0.5, 0.0, dist) * v_color.a * 0.15;
    fragColor = vec4(v_color.rgb * alpha, alpha);
}
