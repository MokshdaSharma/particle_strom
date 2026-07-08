import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from src.config.config import TrackerConfig

class MotionTracker:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.max_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        
        self.alpha = config.smoothing_factor
        self.prev_face = None 
        self.smoothed_landmarks = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Processes a BGR frame, extracts face landmarks, applies EMA smoothing,
        and returns them as normalized device coordinates (NDC) [-1, 1].
        """
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        all_landmarks = []
        
        # Process Face
        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0]
            current_face = np.zeros((478, 2), dtype=np.float32)
            for i, lm in enumerate(face_lms.landmark):
                current_face[i, 0] = (lm.x * 2.0 - 1.0) * 0.75
                current_face[i, 1] = 1.0 - lm.y * 2.0
                
            if self.prev_face is None:
                self.prev_face = current_face
            else:
                self.prev_face = self.alpha * current_face + (1 - self.alpha) * self.prev_face
                
            all_landmarks.append(self.prev_face)
        else:
            self.prev_face = None
            
        # Process Hands
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                current_hand = np.zeros((21, 2), dtype=np.float32)
                for i, lm in enumerate(hand_lms.landmark):
                    current_hand[i, 0] = (lm.x * 2.0 - 1.0) * 0.75
                    current_hand[i, 1] = 1.0 - lm.y * 2.0
                all_landmarks.append(current_hand)

        if len(all_landmarks) > 0:
            return np.concatenate(all_landmarks, axis=0)
            
        return None
        
    def close(self):
        """Releases MediaPipe resources."""
        self.face_mesh.close()
        self.hands.close()
