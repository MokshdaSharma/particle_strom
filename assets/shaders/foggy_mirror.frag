#version 330
in vec2 v_texcoord;

uniform sampler2D camera_texture;
uniform sampler2D mask_texture;
uniform vec2 resolution;

out vec4 fragColor;

void main() {
    // 1. Read mask
    float mask = texture(mask_texture, v_texcoord).r;
    
    // 2. Read sharp camera
    vec4 sharp_color = texture(camera_texture, v_texcoord);
    
    // 3. Create foggy camera (simple 5-tap fast blur + tint)
    vec2 tex_offset = 1.0 / resolution * 8.0; // heavy blur radius
    vec4 foggy_color = texture(camera_texture, v_texcoord) * 0.380;
    foggy_color += texture(camera_texture, v_texcoord + vec2(tex_offset.x, tex_offset.y)) * 0.155;
    foggy_color += texture(camera_texture, v_texcoord + vec2(-tex_offset.x, tex_offset.y)) * 0.155;
    foggy_color += texture(camera_texture, v_texcoord + vec2(tex_offset.x, -tex_offset.y)) * 0.155;
    foggy_color += texture(camera_texture, v_texcoord + vec2(-tex_offset.x, -tex_offset.y)) * 0.155;
    
    // Desaturate and lighten to look like fog/steam
    float luminance = dot(foggy_color.rgb, vec3(0.299, 0.587, 0.114));
    vec3 foggy_grey = mix(foggy_color.rgb, vec3(luminance), 0.5) + vec3(0.3); // add white tint
    foggy_color = vec4(foggy_grey, 1.0);
    
    // 4. Mix them: where mask > 0, show sharp. Otherwise, show foggy.
    fragColor = mix(foggy_color, sharp_color, mask);
}
