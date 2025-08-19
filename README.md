# RLdetector
This Python application implements a comprehensive lane detection system that combines traditional computer vision techniques with deep learning capabilities. Designed with a user-friendly PyQt5 interface, the tool processes both images and videos to identify road lanes under various environmental conditions. Below are its key features and technical innovations:

Core Functionality
1. Multi-Algorithm Detection:
Traditional CV Pipeline: Utilizes OpenCV for Canny edge detection, Gaussian blur, and Hough line transforms
Deep Learning Option: Integrates a simplified SCNN (Spatial CNN) model for neural network-based detection
Hybrid Approach: Implements curve fitting for non-linear lanes and frame-skipping optimization for video processing

2. Environmental Adaptation:
Automatically classifies weather conditions (night/cloudy/sunny/strong light) using image brightness analysis
Dynamically adjusts contrast and brightness through CLAHE histogram equalization
Supports 16-bit images and transparent PNGs with automatic background handling

3. Stability Enhancements:
Motion Smoothing: Applies exponential smoothing to lane positions across frames
Angle Validation: Discards outlier lanes exceeding user-defined angular thresholds
Persistence Mechanism: Retains last valid lanes during detection failures

Technical Architecture
Modular Components:
`ImageProcessor`: Handles cross-format image loading and weather-based preprocessing
`LaneDetector`: Implements core detection logic with parameterized algorithms
`VideoProcessor`: Manages video I/O with temporary file handling for Unicode paths

SCNN Model:
'''
  class SCNN(nn.Module):
      def __init__(self, input_size):
          super(SCNN, self).__init__()
          self.features = nn.Sequential(...)  # Conv layers
          self.classifier = nn.Sequential(...)  # Fully-connected layers
'''
Accepts 320x480 input images
Outputs left/right lane classifications

Advanced ROI Processing:
Adaptive trapezoidal masks scaled to image dimensions
Horizontal line filtering and slope-based lane separation
Semi-transparent lane region filling (BGRα)

User Interface Features
Parameter Controls:
Real-time sliders for Canny thresholds and Hough parameters
Curve fitting and region filling toggles
Frame skipping options (0/1/3/6 frames)

Visualization Modes:
Interactive display of original images, Canny edges, ROIs, and final results
Weather classification and processing statistics in status bar
Multi-step video progress dialog

File Handling:
Unicode path support via temporary file copying
Video encoding with compression settings
Batch processing for video frames

Performance Optimizations
1. Selective Processing:
'''
if skip_frames > 0 and frame_counter % (skip_frames+1) != 0:
use_last_valid_frame()
'''
2. GPU Acceleration: PyTorch-based model inference
3. Memory Management: Frame-by-frame video processing
4. Dynamic Smoothing:
'''
current_line = smooth_factor * current + (1-smooth_factor) * previous
'''
Practical Applications
Autonomous vehicle perception systems
Road infrastructure analysis
Driver assistance tools
Traffic management solutions

The system demonstrates sophisticated integration of computer vision and deep learning, featuring robust error handling for real-world scenarios like low-light conditions and complex road geometries. Its modular design allows easy extension with more advanced models while maintaining accessibility through an intuitive GUI.
