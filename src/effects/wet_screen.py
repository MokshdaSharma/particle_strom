import moderngl
import numpy as np
from pathlib import Path
from loguru import logger

class WetScreenRenderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        self._load_shaders()
        self._create_quad()
        self._create_fbos(width, height)
        
    def _load_shaders(self):
        base_path = Path('assets/shaders')
        
        # Brush shader for drawing points
        with open(base_path / 'brush.vert', 'r') as f:
            brush_vert = f.read()
        with open(base_path / 'brush.frag', 'r') as f:
            brush_frag = f.read()
            
        self.brush_prog = self.ctx.program(vertex_shader=brush_vert, fragment_shader=brush_frag)
        
        # Shader for composite / fading
        with open(base_path / 'quad.vert', 'r') as f:
            quad_vert = f.read()
        with open(base_path / 'quad.frag', 'r') as f:
            quad_frag = f.read()
            
        self.quad_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=quad_frag)
        
        # Simple fade shader
        fade_frag = """
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(0.0, 0.0, 0.0, 0.02); // Small opacity to slowly fade
        }
        """
        self.fade_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=fade_frag)

    def _create_quad(self):
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.quad_vao = self.ctx.vertex_array(self.quad_prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])
        
        # 'in_texcoord' is optimized out in fade_prog because the fragment shader doesn't use it.
        # So we skip the texcoord bytes (8 bytes) and only bind 'in_position'.
        self.fade_vao = self.ctx.vertex_array(self.fade_prog, [(self.vbo, '2f 8x', 'in_position')])
        
        # A dynamic VBO for brush strokes (up to max hands)
        self.brush_vbo = self.ctx.buffer(reserve=10 * 8) # 10 points max, 2 floats each
        self.brush_vao = self.ctx.vertex_array(self.brush_prog, [(self.brush_vbo, '2f', 'in_position')])

    def _create_fbos(self, width: int, height: int):
        self.texture = self.ctx.texture((width, height), 4)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.texture])
        # Clear it completely transparent initially
        self.fbo.use()
        self.fbo.clear(0, 0, 0, 0)

    def resize(self, width: int, height: int):
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height
        self.fbo.release()
        self.texture.release()
        self._create_fbos(width, height)

    def draw(self, positions: list[np.ndarray]):
        """Draws brush strokes at the given NDC positions onto the persistent canvas."""
        if not positions:
            return
            
        self.fbo.use()
        
        # Write points to brush VBO
        data = np.array(positions, dtype='f4').tobytes()
        self.brush_vbo.write(data)
        
        # Setup blending for drawing the brush
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        # Additive blending for the brush to make it glow
        self.ctx.blend_func = moderngl.ADDITIVE_BLENDING
        
        self.brush_vao.render(moderngl.POINTS, vertices=len(positions))

    def fade(self):
        """Slowly fades out the drawn canvas."""
        self.fbo.use()
        self.ctx.enable(moderngl.BLEND)
        # Subtract alpha or blend a dark transparent rect over it
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING
        
        # Ensure we can write to alpha
        self.ctx.blend_equation = moderngl.FUNC_REVERSE_SUBTRACT
        self.fade_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.blend_equation = moderngl.FUNC_ADD # reset

    def render(self):
        """Composites the canvas onto the currently bound framebuffer."""
        self.texture.use(0)
        self.quad_prog['texture0'].value = 0
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)
