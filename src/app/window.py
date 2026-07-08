import glfw
import moderngl
from loguru import logger
from src.config.config import AppConfig

class Window:
    """Manages the GLFW window and ModernGL context."""
    def __init__(self, config: AppConfig):
        self.config = config
        self.window = None
        self.ctx = None
        self.is_running = False
        self.keys = set()
        self.just_pressed = set()

    def init(self):
        """Initializes GLFW and creates a window with an OpenGL context."""
        if not glfw.init():
            logger.error("Failed to initialize GLFW")
            raise RuntimeError("GLFW initialization failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        
        # Optional: Multi-sampling
        glfw.window_hint(glfw.SAMPLES, 4)

        logger.info(f"Creating window: {self.config.window_width}x{self.config.window_height}")
        self.window = glfw.create_window(
            self.config.window_width, 
            self.config.window_height, 
            self.config.window_title, 
            None, 
            None
        )

        if not self.window:
            glfw.terminate()
            logger.error("Failed to create GLFW window")
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1) # V-Sync ON by default

        # Set callbacks
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)

        # Create ModernGL context
        self.ctx = moderngl.create_context(require=330)
        logger.info(f"ModernGL context created. OpenGL version: {self.ctx.version_code}")
        
        self.is_running = True

    def _key_callback(self, window, key, scancode, action, mods):
        """Handles keyboard input."""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            
        if action == glfw.PRESS:
            self.keys.add(key)
            self.just_pressed.add(key)
        elif action == glfw.RELEASE:
            if key in self.keys:
                self.keys.remove(key)

    def get_just_pressed(self, key) -> bool:
        """Returns True if the key was pressed this frame, then clears the state."""
        if key in self.just_pressed:
            self.just_pressed.remove(key)
            return True
        return False

    def _resize_callback(self, window, width, height):
        """Handles window resize events."""
        if self.ctx:
            self.ctx.viewport = (0, 0, width, height)

    def should_close(self) -> bool:
        """Returns whether the window should close."""
        return glfw.window_should_close(self.window)

    def swap_buffers(self):
        """Swaps the front and back buffers."""
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def terminate(self):
        """Cleans up GLFW resources."""
        logger.info("Terminating window...")
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
