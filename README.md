# Interactive Face Particle Visualization System

A high-performance, real-time interactive visualization system that tracks your face and hands using MediaPipe, and renders up to a million GPU-accelerated particles that react to your movements.

This project uses **ModernGL** for hardware-accelerated compute shaders, resulting in fluid dynamics, glowing bloom effects, and seamless real-time interactions.

## Features

- **Real-Time Tracking**: Fast and robust face and hand tracking using Google's MediaPipe.
- **Compute Shader Physics**: Over 1 million particles simulated entirely on the GPU, reacting dynamically to the magnetic pull of your facial landmarks and hands.
- **Beautiful Bloom Effects**: Custom multi-pass bloom rendering for glowing, neon-like aesthetics.
- **Interactive UI**: On-screen buttons that you can interact with using **Hand Gestures** (hover your finger to trigger) or simply by clicking with your **Mouse**.
- **Foggy Mirror Mode**: A unique interactive drawing mode. The camera feed instantly blurs and frosts over like a bathroom mirror, and you can use your finger to wipe away the fog and draw continuous smooth lines!
- **Color Themes**: Cycle between beautifully curated palettes (Fire, Water/Cyberpunk, Matrix).

## Installation

Ensure you have Python 3.9+ installed.

1. Clone the repository.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

### On-Screen UI Controls
You can interact with the on-screen buttons using:
1. **Mouse Click**: Simply click the buttons.
2. **Hand Tracking**: Move your hand to see the crosshair cursor. Hover your index finger over a button for 0.5 seconds to trigger it!

### Keyboard Controls
- `B`: Toggle Camera Background (switch between Dark Mode and Camera Feed)
- `C`: Cycle Color Themes (Fire, Water, Matrix)
- `W`: Toggle Foggy Mirror (Draw Mode)
- `R`: Reset Particle Simulation
- `P`: Pause/Unpause Simulation
- `D`: Toggle Debug Mode (shows FPS and tracking logs)
- `S`: Take a Screenshot (saved to the `screenshots/` directory)
- `ESC`: Exit Application

## Architecture

- **`main.py`**: The core loop handling input, rendering order, and state.
- **`src/tracking/`**: MediaPipe wrappers for robust motion tracking.
- **`src/particles/`**: GPU compute shaders handling massive particle physics.
- **`src/effects/`**: Framebuffer objects handling Bloom processing and the Foggy Mirror shaders.
- **`src/ui/`**: 2D Orthographic UI renderer supporting hand-hover triggers and texture-mapped buttons.
- **`assets/shaders/`**: All the raw GLSL compute, vertex, and fragment shaders.
