import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AppConfig:
    window_title: str
    window_width: int
    window_height: int
    target_fps: int
    debug_mode: bool

@dataclass
class CameraConfig:
    device_id: int
    width: int
    height: int
    fps: int

@dataclass
class TrackerConfig:
    max_faces: int
    min_detection_confidence: float
    min_tracking_confidence: float
    smoothing_factor: float
    max_hands: int

@dataclass
class ParticleConfig:
    count: int
    base_size: float
    attraction_speed: float
    friction: float

@dataclass
class PostProcessConfig:
    bloom_iterations: int
    bloom_intensity: float
    blur_radius: float

@dataclass
class Config:
    app: AppConfig
    camera: CameraConfig
    tracker: TrackerConfig
    particles: ParticleConfig
    post_process: PostProcessConfig

    @classmethod
    def load(cls, path: str | Path) -> 'Config':
        """Loads configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        app_config = AppConfig(**data.get('app', {}))
        camera_config = CameraConfig(**data.get('camera', {}))
        tracker_config = TrackerConfig(**data.get('tracker', {}))
        particle_config = ParticleConfig(**data.get('particles', {}))
        post_config = PostProcessConfig(**data.get('post_process', {}))

        return cls(app=app_config, camera=camera_config, tracker=tracker_config, particles=particle_config, post_process=post_config)
