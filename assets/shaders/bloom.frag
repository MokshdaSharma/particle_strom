#version 330
in vec2 v_texcoord;
uniform sampler2D base_texture;
uniform sampler2D blur_texture;
uniform float bloom_intensity;
out vec4 fragColor;

void main() {
    vec4 base = texture(base_texture, v_texcoord);
    vec4 bloom = texture(blur_texture, v_texcoord);
    // Additive blending
    fragColor = base + bloom * bloom_intensity;
}
