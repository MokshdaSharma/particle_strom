#version 330
out vec4 fragColor;

void main() {
    // Calculate distance from center of the point (for gl_PointSize)
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Smooth soft circle
    float alpha = smoothstep(0.5, 0.0, dist);
    
    // Soft cyan/blue color for the wet screen trail
    fragColor = vec4(0.5, 0.8, 1.0, alpha * 0.5);
}
