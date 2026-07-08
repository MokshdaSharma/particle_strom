#version 330

in vec2 v_texcoord;
uniform vec4 rect_color;
uniform sampler2D text_texture;
uniform bool use_texture;
out vec4 fragColor;

void main() {
    if (use_texture) {
        vec4 tex_color = texture(text_texture, v_texcoord);
        // Blend the text (white text) with the background button color
        // Use the texture's alpha to mix between button color and white text
        vec3 mixed_color = mix(rect_color.rgb, vec3(1.0), tex_color.a);
        
        // Final alpha is the button's base alpha, but opaque where text is
        float final_alpha = max(rect_color.a, tex_color.a);
        
        fragColor = vec4(mixed_color, final_alpha);
    } else {
        fragColor = rect_color;
    }
}
