import moderngl
import numpy as np
from pathlib import Path
from loguru import logger
from src.config.config import PostProcessConfig

class PostProcessor:
    def __init__(self, ctx: moderngl.Context, config: PostProcessConfig, width: int, height: int):
        self.ctx = ctx
        self.config = config
        self.width = width
        self.height = height
        
        self._load_shaders()
        self._create_quad()
        self._create_fbos(width, height)

    def _load_shaders(self):
        base_path = Path('assets/shaders')
        
        with open(base_path / 'quad.vert', 'r') as f:
            vert = f.read()
            
        with open(base_path / 'quad.frag', 'r') as f:
            self.quad_prog = self.ctx.program(vertex_shader=vert, fragment_shader=f.read())
            
        with open(base_path / 'blur.frag', 'r') as f:
            self.blur_prog = self.ctx.program(vertex_shader=vert, fragment_shader=f.read())
            
        with open(base_path / 'bloom.frag', 'r') as f:
            self.bloom_prog = self.ctx.program(vertex_shader=vert, fragment_shader=f.read())

    def _create_quad(self):
        # A fullscreen quad
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        
        self.quad_vao = self.ctx.vertex_array(self.quad_prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])
        self.bloom_vao = self.ctx.vertex_array(self.bloom_prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])

    def _create_fbos(self, width: int, height: int):
        # Base render FBO
        self.base_texture = self.ctx.texture((width, height), 4)
        self.base_fbo = self.ctx.framebuffer(color_attachments=[self.base_texture])
        
        # Ping-pong FBOs for blurring 
        self.blur_tex_1 = self.ctx.texture((width, height), 4)
        self.blur_fbo_1 = self.ctx.framebuffer(color_attachments=[self.blur_tex_1])
        
        self.blur_tex_2 = self.ctx.texture((width, height), 4)
        self.blur_fbo_2 = self.ctx.framebuffer(color_attachments=[self.blur_tex_2])

    def resize(self, width: int, height: int):
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height
        
        # Release old FBOs
        self.base_fbo.release()
        self.base_texture.release()
        self.blur_fbo_1.release()
        self.blur_tex_1.release()
        self.blur_fbo_2.release()
        self.blur_tex_2.release()
        
        self._create_fbos(width, height)

    def begin(self):
        """Binds the base FBO so the scene renders into it."""
        self.base_fbo.use()
        self.base_fbo.clear(0.05, 0.05, 0.05, 1.0)
        
    def draw_background(self, texture: moderngl.Texture):
        """Draws a texture as the background into the base FBO."""
        # Ensure we are rendering into base_fbo
        self.base_fbo.use()
        
        # Bind the background texture
        texture.use(0)
        self.quad_prog['texture0'].value = 0
        
        # Disable blending to overwrite the clear color
        self.ctx.disable(moderngl.BLEND)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    def end(self, screen_fbo: moderngl.Framebuffer):
        """Applies blur passes and composites bloom onto the screen FBO."""
        
        # We need to blur the base_texture
        self.blur_prog['blur_radius'].value = self.config.blur_radius
        
        # Pass 1: Horizontal Blur (Base -> FBO 1)
        self.blur_fbo_1.use()
        self.blur_fbo_1.clear(0, 0, 0, 0)
        self.base_texture.use(0)
        self.blur_prog['texture0'].value = 0
        self.blur_prog['direction'].value = (1.0, 0.0)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)
        
        # Ping pong blur iterations
        iterations = self.config.bloom_iterations
        for i in range(iterations):
            # Vertical (FBO 1 -> FBO 2)
            self.blur_fbo_2.use()
            self.blur_fbo_2.clear(0, 0, 0, 0)
            self.blur_tex_1.use(0)
            self.blur_prog['texture0'].value = 0
            self.blur_prog['direction'].value = (0.0, 1.0)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)
            
            # Horizontal (FBO 2 -> FBO 1)
            self.blur_fbo_1.use()
            self.blur_fbo_1.clear(0, 0, 0, 0)
            self.blur_tex_2.use(0)
            self.blur_prog['texture0'].value = 0
            self.blur_prog['direction'].value = (1.0, 0.0)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)
            
        # The final blurred texture is in blur_tex_1
        
        # Final Pass: Bloom composite to Screen
        screen_fbo.use()
        screen_fbo.clear(0.05, 0.05, 0.05, 1.0)
        
        self.base_texture.use(0)
        self.blur_tex_1.use(1)
        
        self.bloom_prog['base_texture'].value = 0
        self.bloom_prog['blur_texture'].value = 1
        self.bloom_prog['bloom_intensity'].value = self.config.bloom_intensity
        
        # We must use proper blending for final screen
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING
        
        self.bloom_vao.render(moderngl.TRIANGLE_STRIP)
