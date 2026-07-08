#version 330
in vec2 v_texcoord;
uniform sampler2D texture0;
uniform vec2 direction;
uniform float blur_radius;
out vec4 fragColor;

void main() {
    vec2 tex_offset = 1.0 / textureSize(texture0, 0);
    vec4 result = texture(texture0, v_texcoord) * 0.227027;
    vec2 offset1 = direction * tex_offset * blur_radius * 1.384615;
    vec2 offset2 = direction * tex_offset * blur_radius * 3.230769;

    result += texture(texture0, v_texcoord + offset1) * 0.316216;
    result += texture(texture0, v_texcoord - offset1) * 0.316216;
    result += texture(texture0, v_texcoord + offset2) * 0.070270;
    result += texture(texture0, v_texcoord - offset2) * 0.070270;
    
    fragColor = result;
}
