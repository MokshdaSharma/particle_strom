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
from src.effects.wet_screen import WetScreenRenderer
from src.ui.ui_renderer import UIRenderer
import glfw
from PIL import Image
import cv2
import numpy as np

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
        
        # 8. Initialize Wet Screen Renderer
        wet_screen = WetScreenRenderer(window.ctx, config.app.window_width, config.app.window_height)
        
        # 9. Initialize UI Renderer
        ui_renderer = UIRenderer(window.ctx)
        
        logger.info("Entering main loop")
        
        start_time = time.time()
        last_time = start_time
        frames = 0
        is_paused = False
        
        # State for new features
        show_camera_bg = False
        current_theme = 0
        wet_screen_mode = False
        camera_texture = None
        
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
            tracked_data = None
            face_landmarks = None
            
            if frame is not None:
                # Update camera texture if background mode is enabled
                if show_camera_bg:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Flip upside down for OpenGL
                    rgb_frame = cv2.flip(rgb_frame, 0)
                    if camera_texture is None or camera_texture.size != (rgb_frame.shape[1], rgb_frame.shape[0]):
                        if camera_texture:
                            camera_texture.release()
                        camera_texture = window.ctx.texture((rgb_frame.shape[1], rgb_frame.shape[0]), 3, rgb_frame.tobytes())
                    else:
                        camera_texture.write(rgb_frame.tobytes())
                
                # Process frame for face tracking
                tracked_data = tracker.process_frame(frame)
                
                if tracked_data:
                    face_landmarks = tracked_data.get('face')
                    
                    # Update UI
                    ui_events = ui_renderer.update(tracked_data.get('hands'))
                    for event in ui_events:
                        if event == "bg":
                            show_camera_bg = not show_camera_bg
                            logger.info(f"UI Toggle -> Camera background: {show_camera_bg}")
                        elif event == "theme":
                            current_theme = (current_theme + 1) % 3
                            particle_engine.set_theme(current_theme)
                            logger.info(f"UI Toggle -> Color theme changed to {current_theme}")
                        elif event == "wet":
                            wet_screen_mode = not wet_screen_mode
                            logger.info(f"UI Toggle -> Wet screen mode: {wet_screen_mode}")
                
                # Optional: log if we have landmarks in debug mode (once per second to avoid spam)
                if config.app.debug_mode and face_landmarks is not None and frames == 1:
                    logger.debug(f"Tracking active: face detected")
                    
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
                
            if window.get_just_pressed(glfw.KEY_B):
                show_camera_bg = not show_camera_bg
                logger.info(f"Camera background: {show_camera_bg}")
                
            if window.get_just_pressed(glfw.KEY_C):
                current_theme = (current_theme + 1) % 3
                particle_engine.set_theme(current_theme)
                logger.info(f"Color theme changed to {current_theme}")
                
            if window.get_just_pressed(glfw.KEY_W):
                wet_screen_mode = not wet_screen_mode
                logger.info(f"Wet screen mode: {wet_screen_mode}")
                
            # Check window resize
            viewport = window.ctx.screen.viewport
            post_processor.resize(viewport[2], viewport[3])
            wet_screen.resize(viewport[2], viewport[3])
                
            # Update Particles
            if not is_paused:
                all_lms = []
                if face_landmarks is not None:
                    all_lms.append(face_landmarks)
                if tracked_data and tracked_data.get('hands'):
                    # Repeat hand landmarks 15 times to increase density around hands
                    for hand in tracked_data['hands']:
                        for _ in range(15):
                            all_lms.append(hand)
                    
                combined_landmarks = np.concatenate(all_lms, axis=0) if all_lms else None
                particle_engine.update(sim_dt, total_time, combined_landmarks)
                
                # Wet screen interaction
                if wet_screen_mode:
                    wet_screen.fade() # slowly fade out previous drawings
                    if tracked_data and tracked_data.get('hands'):
                        brush_pts = []
                        for hand in tracked_data['hands']:
                            # Index finger tip is index 8
                            brush_pts.append(hand[8])
                        wet_screen.draw(brush_pts)
                
            # Render Pass
            post_processor.begin()
            
            # 1. Render Particles to base FBO
            particle_engine.render()
            
            # 2. Composite everything to screen
            # post_processor.end takes care of the screen clear and drawing camera background if provided
            
            # First, check if we need to draw background
            window.ctx.screen.use()
            if show_camera_bg and camera_texture is not None:
                post_processor.draw_camera(camera_texture, window.ctx.screen)
            else:
                window.ctx.screen.clear(0.05, 0.05, 0.05, 1.0)
                
            # Now composite particles
            post_processor.end(window.ctx.screen)
            
            # 3. Wet Screen Canvas (over screen)
            if wet_screen_mode:
                wet_screen.render()
                
            # 4. Render UI over everything
            ui_renderer.render()

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
