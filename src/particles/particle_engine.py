import moderngl
import numpy as np
import math
from pathlib import Path
from loguru import logger
from src.config.config import ParticleConfig

class ParticleEngine:
    def __init__(self, ctx: moderngl.Context, config: ParticleConfig):
        self.ctx = ctx
        self.config = config
        
        # Load shaders
        self._load_shaders()
        
        # Initialize buffers
        self._init_buffers()
        
        # VBO for rendering
        self.vao = self.ctx.vertex_array(
            self.render_program,
            [
                (self.particle_buffer, '2f 2f 4f 16x', 'in_position', 'in_velocity', 'in_color')
            ]
        )

    def _load_shaders(self):
        base_path = Path('assets/shaders')
        
        try:
            with open(base_path / 'particles.comp', 'r') as f:
                comp_source = f.read()
            with open(base_path / 'particles.vert', 'r') as f:
                vert_source = f.read()
            with open(base_path / 'particles.frag', 'r') as f:
                frag_source = f.read()
                
            self.compute_program = self.ctx.compute_shader(comp_source)
            self.render_program = self.ctx.program(
                vertex_shader=vert_source,
                fragment_shader=frag_source
            )
            logger.info("Particle shaders loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load particle shaders: {e}")
            raise

    def _init_buffers(self):
        num_particles = self.config.count
        
        # Memory layout matches the GLSL std430 struct (12 floats = 48 bytes)
        initial_data = np.zeros((num_particles, 12), dtype=np.float32)
        
        # Randomize initial positions [-1, 1]
        initial_data[:, 0] = np.random.uniform(-1, 1, num_particles)
        initial_data[:, 1] = np.random.uniform(-1, 1, num_particles)
        
        # Color (White with alpha 1)
        initial_data[:, 4:8] = 1.0 
        
        # Age
        initial_data[:, 8] = np.random.uniform(0, 5, num_particles)
        
        # Lifetime
        initial_data[:, 9] = np.random.uniform(2, 5, num_particles)
        
        # Target index
        initial_data[:, 10] = 0
        
        self.particle_buffer = self.ctx.buffer(initial_data.tobytes())
        # SSBO 1: Landmarks Buffer (Store tracked facial/hand points)
        # Reserve enough space for face (478) + hands (21 each) + overhead
        self.max_landmarks = 1024
        self.landmark_buffer = self.ctx.buffer(reserve=self.max_landmarks * 8) # 2 floats per landmark = 8 bytes
        
        # Set static uniforms
        self.render_program['base_size'].value = self.config.base_size
        self.compute_program['attraction_speed'].value = self.config.attraction_speed
        self.compute_program['friction'].value = self.config.friction
        
        # Initialize color theme
        self.current_theme = 0
        if 'color_theme' in self.compute_program:
            self.compute_program['color_theme'].value = self.current_theme

    def set_theme(self, theme: int):
        self.current_theme = theme
        if 'color_theme' in self.compute_program:
            self.compute_program['color_theme'].value = self.current_theme

    def update(self, dt: float, time: float, landmarks: np.ndarray | None):
        # Bind buffers to the shader
        self.particle_buffer.bind_to_storage_buffer(0)
        self.landmark_buffer.bind_to_storage_buffer(1)
        
        # Set dynamic uniforms
        self.compute_program['dt'].value = dt
        self.compute_program['time'].value = time
        
        if landmarks is not None:
            # write landmarks to SSBO
            self.landmark_buffer.write(landmarks.astype(np.float32).tobytes())
            self.compute_program['num_landmarks'].value = landmarks.shape[0]
            self.compute_program['face_detected'].value = True
        else:
            self.compute_program['num_landmarks'].value = 0
            self.compute_program['face_detected'].value = False
            
        # Dispatch compute shader (workgroup size is 256)
        num_groups = math.ceil(self.config.count / 256)
        self.compute_program.run(group_x=num_groups)

    def render(self):
        # Program size settings required for gl_PointSize
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ADDITIVE_BLENDING
        
        self.vao.render(moderngl.POINTS)

    def reset(self):
        """Resets the particles back to random states."""
        num_particles = self.config.count
        initial_data = np.zeros((num_particles, 12), dtype=np.float32)
        initial_data[:, 0] = np.random.uniform(-1, 1, num_particles)
        initial_data[:, 1] = np.random.uniform(-1, 1, num_particles)
        initial_data[:, 4:8] = 1.0 
        initial_data[:, 8] = np.random.uniform(0, 5, num_particles)
        initial_data[:, 9] = np.random.uniform(2, 5, num_particles)
        initial_data[:, 10] = 0
        self.particle_buffer.write(initial_data.tobytes())
