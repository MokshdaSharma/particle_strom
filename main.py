import os
import time
from loguru import logger
from src.config.config import Config
from src.utils.logger import setup_logger
from src.camera.camera import CameraManager
from src.app.window import Window
from src.tracking.face_tracker import MotionTracker
from src.particles.particle_engine import ParticleEngine
from src.effects.post_processor import PostProcessor
import glfw
from PIL import Image

def main():
    # 1. Load Configuration
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    try:
        config = Config.load(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    # 2. Setup Logger
    setup_logger(config.app.debug_mode)
    logger.info("Starting Interactive Face Particle Visualization System")

    # 3. Initialize Camera Manager
    camera = CameraManager(config.camera)
    
    # 4. Initialize Tracker
    tracker = MotionTracker(config.tracker)
    
    # 5. Initialize Window and Renderer
    window = Window(config.app)
    
    try:
        camera.start()
        window.init()
        
        # 6. Initialize Particle Engine
        # Must happen after window.init() to access the context
        particle_engine = ParticleEngine(window.ctx, config.particles)
        
        # 7. Initialize Post Processor
        post_processor = PostProcessor(window.ctx, config.post_process, config.app.window_width, config.app.window_height)
        
        logger.info("Entering main loop")
        
        start_time = time.time()
        last_time = start_time
        frames = 0
        is_paused = False
        
        while not window.should_close():
            current_time = time.time()
            dt = current_time - last_time
            total_time = current_time - start_time
            
            # Simple FPS counter
            frames += 1
            if dt >= 1.0:
                if config.app.debug_mode:
                    logger.debug(f"FPS: {frames}")
                frames = 0
                last_time = current_time
            # For smooth dt usage without stalling, we use small step
            elif dt > 0:
                pass
            
            # To ensure physics doesn't explode if loop hangs
            sim_dt = min(dt, 0.05)

            # Get latest frame
            frame = camera.get_frame()
            landmarks = None
            if frame is not None:
                # Process frame for face tracking
                landmarks = tracker.process_frame(frame)
                
                # Optional: log if we have landmarks in debug mode (once per second to avoid spam)
                if config.app.debug_mode and landmarks is not None and frames == 1:
                    logger.debug(f"Tracking active: {landmarks.shape[0]} landmarks detected")
                    
            # Input handling
            if window.get_just_pressed(glfw.KEY_R):
                particle_engine.reset()
                logger.info("Particles reset.")
                
            if window.get_just_pressed(glfw.KEY_P):
                is_paused = not is_paused
                logger.info(f"Paused: {is_paused}")
                
            if window.get_just_pressed(glfw.KEY_D):
                config.app.debug_mode = not config.app.debug_mode
                logger.info(f"Debug mode: {config.app.debug_mode}")
                
            if window.get_just_pressed(glfw.KEY_S):
                # Take screenshot
                os.makedirs("screenshots", exist_ok=True)
                filename = f"screenshots/capture_{int(time.time())}.png"
                image = Image.frombytes('RGB', (window.ctx.screen.viewport[2], window.ctx.screen.viewport[3]), window.ctx.screen.read(components=3))
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                image.save(filename)
                logger.info(f"Saved screenshot to {filename}")
                
            # Check window resize
            viewport = window.ctx.screen.viewport
            post_processor.resize(viewport[2], viewport[3])
                
            # Update Particles
            if not is_paused:
                particle_engine.update(sim_dt, total_time, landmarks)
                
            # Render Pass
            post_processor.begin()
            
            # Render Particles
            particle_engine.render()
            
            post_processor.end(window.ctx.screen)

            # Swap buffers and poll events
            window.swap_buffers()

    except Exception as e:
        logger.exception(f"Application error: {e}")
    finally:
        logger.info("Shutting down...")
        camera.stop()
        tracker.close()
        window.terminate()

if __name__ == "__main__":
    main()
