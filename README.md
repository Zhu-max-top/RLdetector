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
  ```
  class SCNN(nn.Module):
      def __init__(self, input_size):
          super(SCNN, self).__init__()
          self.features = nn.Sequential(...)  # Conv layers
          self.classifier = nn.Sequential(...)  # Fully-connected layers
  ```
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
   ```python
   if skip_frames > 0 and frame_counter % (skip_frames+1) != 0:
       use_last_valid_frame()
   ```  
2. GPU Acceleration: PyTorch-based model inference
3. Memory Management: Frame-by-frame video processing
4. Dynamic Smoothing:
   ```python
   current_line = smooth_factor * current + (1-smooth_factor) * previous
   ```
Practical Applications
Autonomous vehicle perception systems
Road infrastructure analysis
Driver assistance tools
Traffic management solutions

The system demonstrates sophisticated integration of computer vision and deep learning, featuring robust error handling for real-world scenarios like low-light conditions and complex road geometries. Its modular design allows easy extension with more advanced models while maintaining accessibility through an intuitive GUI.

该Python应用实现了一套融合传统计算机视觉技术与深度学习能力的综合车道检测系统。通过用户友好的PyQt5界面，该工具可处理图像和视频数据，在各种环境条件下识别道路车道。核心特性与技术革新如下：

核心功能
1.多算法检测  
传统CV流程：采用OpenCV进行Canny边缘检测、高斯模糊和霍夫线变换  
深度学习选项：集成简化版SCNN（空间卷积神经网络）模型实现神经网络检测  
混合方案：通过曲线拟合处理非线性车道，采用跳帧优化技术提升视频处理效率

2.环境自适应 
基于图像亮度分析自动识别天气状况（夜间/多云/晴天/强光）  
通过CLAHE直方图均衡化动态调整对比度与亮度  
支持16位图像及透明PNG格式（自动处理背景）

3. 稳定性增强  
	运动平滑：对连续帧的车道位置应用指数平滑算法  
	角度验证：根据用户定义的阈值过滤超限异常车道  
	持续机制：检测失败时保留最后有效车道数据

技术架构  
模块化组件  
`ImageProcessor`：跨格式图像加载及基于天气的预处理  
	`LaneDetector`：参数化算法实现的核心检测逻辑  
`VideoProcessor`：视频I/O管理（支持Unicode路径的临时文件处理）

SCNN模型
  ```
  class SCNN(nn.Module):
      def __init__(self, input_size):
          super(SCNN, self).__init__()
          self.features = nn.Sequential(...)  # 卷积层
          self.classifier = nn.Sequential(...)  # 全连接层
  ```
输入尺寸：320×480像素  
输出：左/右车道分类结果

高级ROI处理 
自适应梯形掩模（按图像尺寸缩放）  
水平线过滤与基于斜率的车道分离  
半透明车道区域填充（BGRα色彩空间）

用户界面特性 
参数控制  
Canny阈值/霍夫参数的实时滑块调节  
	曲线拟合与区域填充开关  
	跳帧选项（0/1/3/6帧）  

可视化模式 
	原始图像、Canny边缘、ROI区域及最终结果交互式显示  
	状态栏实时显示天气分类与处理统计信息  
	多步骤视频进度弹窗  

文件处理 
	通过临时文件拷贝支持Unicode路径  
	可配置压缩设置的视频编码  
	视频帧批量处理功能  

性能优化
1. 选择性处理
   ```python
   if skip_frames > 0 and frame_counter % (skip_frames+1) != 0:
       use_last_valid_frame()
   ```  
2. **GPU加速**：基于PyTorch的模型推理  
3. **内存管理**：逐帧视频流处理  
4. **动态平滑**：  
   ```python
   current_line = smooth_factor * current + (1-smooth_factor) * previous
   ```

应用场景  
自动驾驶感知系统  
	道路基础设施分析  
	驾驶员辅助工具  
	交通管理解决方案  

该系统展示了计算机视觉与深度学习的深度融合，通过鲁棒的异常处理机制应对低光照、复杂道路几何等现实场景。模块化设计便于集成更先进模型，同时通过直观GUI保持操作便捷性。
