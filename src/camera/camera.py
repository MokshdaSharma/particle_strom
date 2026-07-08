import cv2
import threading
import time
from loguru import logger
from src.config.config import CameraConfig

class CameraManager:
    """Asynchronous camera capture manager to avoid blocking the main thread."""
    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture = None
        self.latest_frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Starts the camera capture thread."""
        logger.info(f"Starting camera on device {self.config.device_id}")
        self.capture = cv2.VideoCapture(self.config.device_id)
        
        if not self.capture.isOpened():
            logger.error(f"Failed to open camera device {self.config.device_id}")
            raise RuntimeError(f"Could not open camera {self.config.device_id}")
            
        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # Verify settings
        actual_w = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera opened with resolution: {int(actual_w)}x{int(actual_h)}")

        self.is_running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Internal loop to continuously fetch frames."""
        while self.is_running:
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                logger.warning("Failed to grab frame")
                time.sleep(0.01)

    def get_frame(self):
        """Returns the most recent frame."""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        """Stops the camera capture thread."""
        logger.info("Stopping camera...")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()
        logger.info("Camera stopped.")
