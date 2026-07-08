import moderngl
import numpy as np
from pathlib import Path
from loguru import logger

class WetScreenRenderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        self.prev_positions = None
        
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
        
        # Foggy Mirror shader
        with open(base_path / 'foggy_mirror.frag', 'r') as f:
            foggy_frag = f.read()
            
        self.foggy_prog = self.ctx.program(vertex_shader=quad_vert, fragment_shader=foggy_frag)

    def _create_quad(self):
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.quad_vao = self.ctx.vertex_array(self.quad_prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])
        self.foggy_vao = self.ctx.vertex_array(self.foggy_prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])
        
        # A dynamic VBO for brush strokes. We use a larger buffer to hold interpolated points
        self.brush_vbo = self.ctx.buffer(reserve=4096 * 8) # Up to 4096 points per frame
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
            self.prev_positions = None
            return
            
        # Interpolate points between previous and current positions for a continuous line
        points_to_draw = []
        if self.prev_positions is not None and len(self.prev_positions) == len(positions):
            for i in range(len(positions)):
                p1 = self.prev_positions[i]
                p2 = positions[i]
                dist = np.linalg.norm(p2 - p1)
                num_steps = max(1, int(dist * 100)) # roughly a point every 0.01 NDC units
                for t in np.linspace(0, 1, num_steps):
                    points_to_draw.append(p1 * (1 - t) + p2 * t)
        else:
            points_to_draw = positions
            
        self.prev_positions = positions
            
        self.fbo.use()
        
        # Write points to brush VBO
        data = np.array(points_to_draw, dtype='f4').tobytes()
        # if too many points, truncate
        if len(data) <= 4096 * 8:
            self.brush_vbo.write(data)
            
            # Setup blending for drawing the brush
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.ctx.enable(moderngl.BLEND)
            # Additive blending for the brush to make it glow/opaque
            self.ctx.blend_func = moderngl.ADDITIVE_BLENDING
            
            self.brush_vao.render(moderngl.POINTS, vertices=len(points_to_draw))

    def clear(self):
        """Clears the drawn canvas."""
        self.fbo.use()
        self.fbo.clear(0, 0, 0, 0)
        self.prev_positions = None

    def render_foggy_mirror(self, camera_texture: moderngl.Texture, screen_fbo: moderngl.Framebuffer):
        """Renders the foggy mirror effect directly to the screen."""
        screen_fbo.use()
        
        camera_texture.use(0)
        self.texture.use(1) # Mask texture
        
        self.foggy_prog['camera_texture'].value = 0
        self.foggy_prog['mask_texture'].value = 1
        self.foggy_prog['resolution'].value = (self.width, self.height)
        
        self.ctx.disable(moderngl.BLEND)
        self.foggy_vao.render(moderngl.TRIANGLE_STRIP)

    def render(self, screen_fbo: moderngl.Framebuffer):
        """Composites the strokes directly onto the screen (for dark mode)."""
        screen_fbo.use()
        self.texture.use(0)
        self.quad_prog['texture0'].value = 0
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)
