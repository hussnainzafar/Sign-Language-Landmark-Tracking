# Sign Language Landmark Tracker

## Overview
A comprehensive tool for extracting, analyzing, and visualizing 3D landmarks from sign language videos using MediaPipe's holistic tracking. This project enables researchers and developers to capture detailed hand movements, body posture, and facial expressions with high precision in three-dimensional space.

## Features

- **High-Precision 3D Landmark Extraction**: Captures detailed landmarks for body, hands, and face using MediaPipe Holistic
- **Advanced Hand Orientation Tracking**: Quaternion-based rotation tracking for accurate 3D hand orientation
- **Depth Perception Enhancement**: Custom Z-axis filtering for improved spatial accuracy
- **Interactive 3D Visualization**: Real-time viewing of landmarks with PyVista for comprehensive analysis
- **Frame-by-Frame Navigation**: Interactive timeline slider for detailed movement analysis
- **JSON Export**: Structured output format for compatibility with other analysis tools
- **Minimal Landmark Jittering**: Advanced filtering techniques for stable tracking

## System Architecture

![System Architecture](docs/architecture.png "System Architecture Diagram")

The system consists of three main components:
1. **Landmark Extractor**: Processes videos using MediaPipe to extract 3D landmarks
2. **Orientation Calculator**: Computes precise hand orientations using quaternions and rotation matrices
3. **Landmark Visualizer**: Creates interactive 3D visualizations with PyVista

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- MediaPipe
- PyVista (for 3D visualization)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hussnainzafar/Sign-Language-Landmark-Tracking.git
cd Sign-Language-Landmark-Tracking
```

2. Create and activate a conda environment (recommended):
```bash
conda create --name sign_tracker python=3.12
conda activate sign_tracker
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your data path:
```
DATA_PATH=/path/to/your/sign/language/videos
```

## Usage

### Basic Usage

```bash
python main.py
```

This will:
1. Load all videos from the specified DATA_PATH
2. Extract landmarks using MediaPipe
3. Calculate hand orientations
4. Save the results as JSON files in the `output` directory
5. Visualize the landmarks in an interactive 3D viewer (falls back to 2D if 3D fails)

### Command-Line Options

The application supports several command-line options for flexibility:

```bash
# Use the simpler 2D matplotlib visualizer (for systems with OpenGL issues)
python main.py --simple

# Process without any visualization
python main.py --no-vis

# Disable the video preview during processing
python main.py --no-show

# Process a specific video file instead of all files in the data directory
python main.py --file path/to/your/video.mp4

# Combine options as needed
python main.py --simple --file path/to/your/video.mp4
```

### Platform-Specific Notes

#### Linux/Ubuntu
On Linux systems, you might encounter OpenGL rendering issues with the 3D visualization. Options to resolve this:

1. Install necessary OpenGL libraries:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libgl1-mesa-dev xvfb libxrender1 mesa-utils
   ```

2. Use the simple 2D visualizer instead:
   ```bash
   python main.py --simple
   ```

### Processing a Single Video

```python
from src.landmark_extractor import LandmarkExtractor
from src.landmark_visualizer import LandmarkVisualizer

# Extract landmarks
extractor = LandmarkExtractor()
landmarks_data = extractor.process_video('path/to/video.mp4', show_video=True)

# Visualize the results
visualizer = LandmarkVisualizer()
visualizer.visualize_landmarks(landmarks_data)
```

## Output Format

The system generates JSON files with the following structure:

```json
{
  "metadata": {
    "video_path": "path/to/video.mp4",
    "frame_count": 120,
    "fps": 30,
    "duration": 4.0,
    "timestamp": 1645789532.123
  },
  "frames": [
    {
      "frame_idx": 0,
      "pose_landmarks": [...],
      "left_hand_landmarks": [...],
      "right_hand_landmarks": [...],
      "face_landmarks": [...],
      "left_hand_orientation": {
        "quaternion": [w, x, y, z],
        "euler_angles": {"pitch": 45.0, "yaw": 30.0, "roll": 15.0},
        "rotation_matrix": [...]
      },
      "right_hand_orientation": {...}
    },
    // Additional frames...
  ]
}
```

## Visualization Controls

### 3D PyVista Visualization
In the 3D visualization window:
- **Mouse drag**: Rotate the view
- **Scroll wheel**: Zoom in/out
- **Frame slider**: Navigate through video frames
- **Middle mouse button + drag**: Pan the view

### 2D Matplotlib Visualization
In the 2D visualization window:
- **Frame slider**: Navigate through video frames
- **Pan/Zoom buttons**: Standard matplotlib navigation controls
- **Left view**: 2D front view (X-Y plane)
- **Right view**: 3D perspective view with rotation controls

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for the incredible pose, hand, and face tracking technology
- [PyVista](https://github.com/pyvista/pyvista) for the powerful 3D visualization capabilities
- [transforms3d](https://github.com/matthew-brett/transforms3d) for 3D rotation mathematics