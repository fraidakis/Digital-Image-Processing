# Assignment 2: Edge Detection and Circle Detection

This assignment implements and compares edge detection algorithms followed by circular Hough transform for geometric shape detection. It demonstrates the complete pipeline from low-level feature extraction to high-level object recognition.

## Overview

Edge detection and geometric shape recognition are fundamental components of computer vision systems. This assignment explores two complementary approaches:

1. **Edge Detection**: Identifying intensity discontinuities that correspond to object boundaries
2. **Circle Detection**: Using the Hough transform to detect circular objects in edge maps

The implementation covers both **Sobel edge detection** (gradient-based) and **Laplacian of Gaussian (LoG)** edge detection (second-derivative based), followed by **circular Hough transform** with advanced post-processing techniques.

## Project Structure

```
assignment-2/
├── docs/
│   ├── assignment-2.pdf           # Assignment specification
│   ├── input-image.png            # Primary test image (basketball)
│   ├── report.pdf                 # Technical report and analysis
│   └── report.tex                 # LaTeX source
├── results/
│   ├── hough/                     # Circle detection results
│   ├── log/                       # LoG edge detection results
│   └── sobel/                     # Sobel edge detection results
└── src/
    ├── demo.py                    # Main demonstration script
    ├── circ_hough.py             # Circular Hough transform implementation
    ├── fir_conv.py               # Finite impulse response convolution
    ├── log_edge.py               # Laplacian of Gaussian edge detection
    ├── result_refinement.py      # Non-maximum suppression utilities
    └── sobel_edge.py             # Sobel edge detection implementation
```

## Algorithms Implemented

### 1. Sobel Edge Detection

The Sobel operator detects edges by computing the gradient magnitude at each pixel using separable convolution kernels.

```python
def sobel_edge(in_img_array: np.ndarray, threshold: float) -> np.ndarray:
    """
    Detects edges using Sobel gradient operators.
    
    Args:
        in_img_array: Input grayscale image [0, 1]
        threshold: Gradient magnitude threshold for edge detection
    
    Returns:
        Binary edge map where 1 indicates detected edges
    """
```

**Key Features:**
- **Gradient Computation**: Separable convolution with Sobel kernels
- **Threshold Analysis**: Systematic evaluation of threshold impact
- **Edge Count Metrics**: Quantitative analysis of detected edges vs. threshold

**Sobel Kernels:**
```
Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
      [-2, 0, 2],             [ 0,  0,  0],
      [-1, 0, 1]]             [ 1,  2,  1]]
```

### 2. Laplacian of Gaussian (LoG) Edge Detection

The LoG operator combines Gaussian smoothing with Laplacian second-derivative computation for robust edge detection.

```python
def log_edge(in_img_array: np.ndarray, sigma: float) -> np.ndarray:
    """
    Detects edges using Laplacian of Gaussian operator.
    
    Args:
        in_img_array: Input grayscale image [0, 1]
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Binary edge map from zero-crossing detection
    """
```

**Key Features:**
- **Scale-Space Analysis**: Multi-scale edge detection with varying σ values
- **Zero-Crossing Detection**: Identifying sign changes in the Laplacian response
- **Kernel Size Optimization**: Automatic kernel sizing based on σ parameter

**LoG Mathematical Foundation:**
```
LoG(x,y) = -1/(πσ⁴) * [1 - (x² + y²)/(2σ²)] * exp(-(x² + y²)/(2σ²))
```

### 3. Circular Hough Transform

The circular Hough transform detects circles by mapping edge pixels to parameter space (center coordinates and radius).

```python
def circ_hough(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, 
               V_min: int, R_min: float = 1) -> tuple:
    """
    Detects circles using the Hough transform in parameter space.
    
    Args:
        in_img_array: Binary edge map
        R_max: Maximum circle radius
        dim: Accumulator dimensions [a_bins, b_bins, r_bins]
        V_min: Minimum votes for circle detection
        R_min: Minimum circle radius
    
    Returns:
        Detected centers, radii, and vote counts
    """
```

**Key Features:**
- **3D Parameter Space**: (x_center, y_center, radius) accumulator
- **Voting Mechanism**: Each edge pixel votes for possible circle parameters
- **Non-Maximum Suppression**: Advanced post-processing to eliminate duplicate detections

## Advanced Post-Processing

### Non-Maximum Suppression (NMS)

The implementation includes sophisticated NMS for eliminating redundant circle detections:

```python
def non_maximum_suppression(centers, radii, votes, 
                          center_thresh, radius_thresh_perc):
    """
    Eliminates duplicate circle detections based on spatial and 
    radius proximity criteria.
    """
```

**NMS Criteria:**
- **Spatial Distance**: Euclidean distance between circle centers
- **Radius Similarity**: Percentage difference in circle radii
- **Vote Priority**: Preserves circles with higher vote counts

## Comprehensive Analysis Pipeline

### 1. Sobel Edge Analysis
- **Threshold Sweeping**: Systematic evaluation of threshold values (0.1 to 0.4)
- **Edge Count Analysis**: Relationship between threshold and detected edge pixels
- **Visual Comparison**: Side-by-side threshold comparison plots

### 2. LoG Scale-Space Analysis  
- **Multi-Scale Detection**: Testing σ values from 5 to 13
- **Kernel Size Effects**: Analysis of computational vs. accuracy trade-offs
- **Zero-Crossing Visualization**: Clear edge map generation

### 3. Circle Detection Pipeline
- **Parameter Optimization**: Systematic evaluation of Hough parameters
- **Detection Validation**: Visual overlay of detected circles on original image
- **Performance Metrics**: Vote count analysis and detection accuracy

## Usage Instructions

### Quick Start
```bash
# Navigate to the assignment directory
cd assignment-2

# Run the complete demonstration
python src/demo.py
```

### Custom Image Processing
```python
from src.sobel_edge import sobel_edge
from src.log_edge import log_edge
from src.circ_hough import circ_hough

# Load and preprocess your image
gray_image = load_image('your_image.png')

# Sobel edge detection
sobel_edges = sobel_edge(gray_image, threshold=0.2)

# LoG edge detection  
log_edges = log_edge(gray_image, sigma=7)

# Circle detection on edge map
centers, radii, votes = circ_hough(sobel_edges, R_max=250, 
                                  dim=np.array([100, 100, 50]), 
                                  V_min=100)
```

### Parameter Tuning Guidelines

**Sobel Thresholds:**
- Low (0.1-0.15): Detects more edges, including noise
- Medium (0.2-0.25): Balanced edge detection
- High (0.3-0.4): Only strong edges, may miss details

**LoG Sigma Values:**
- Small σ (5-7): Fine-scale edges, more detail
- Medium σ (8-10): Balanced scale detection
- Large σ (11-13): Coarse-scale edges, smoother results

**Hough Parameters:**
- **R_min/R_max**: Expected circle size range
- **V_min**: Minimum confidence threshold (adjust based on image noise)
- **Accumulator Dimensions**: Balance between resolution and computation time

## Results and Visualizations

### Generated Outputs

1. **Sobel Results** (`results/sobel/`):
   - Individual threshold results
   - Threshold comparison grid
   - Edge count vs. threshold analysis

2. **LoG Results** (`results/log/`):
   - Multi-scale edge detection results  
   - Kernel size comparison
   - Comprehensive scale-space visualization

3. **Circle Detection** (`results/hough/`):
   - Detected circles overlaid on original image
   - Parameter-specific detection results
   - Comparative analysis between edge detectors

### Performance Analysis

The assignment provides quantitative analysis including:
- **Edge Density**: Percentage of pixels classified as edges
- **Computational Efficiency**: Processing time comparisons
- **Detection Accuracy**: Circle detection success rate
- **Parameter Sensitivity**: Robustness analysis

## Technical Requirements

- **Python 3.7+**
- **NumPy**: Numerical computations and array operations
- **OpenCV**: Advanced computer vision operations
- **Matplotlib**: Comprehensive visualization
- **PIL/Pillow**: Image I/O operations
- **SciPy**: Signal processing utilities

## Key Learning Objectives

This assignment demonstrates:

1. **Gradient-Based Edge Detection**: Understanding first-derivative operators
2. **Scale-Space Theory**: Multi-scale image analysis with LoG operator
3. **Parameter Space Voting**: Hough transform methodology
4. **Post-Processing Techniques**: Non-maximum suppression and result refinement
5. **Algorithm Comparison**: Evaluating different approaches objectively
6. **Real-World Applications**: Complete detection pipeline implementation

## Applications and Extensions

The techniques implemented form the foundation for:

- **Object Detection**: Geometric shape recognition in industrial inspection
- **Medical Imaging**: Circular structure detection (cells, tumors, vessels)
- **Autonomous Vehicles**: Traffic sign and obstacle detection
- **Quality Control**: Automated inspection of circular components
- **Robotics**: Visual navigation and object manipulation
- **Astronomy**: Planetary and stellar object detection

## Advanced Concepts Covered

- **Convolution Theory**: Separable kernels and computational optimization
- **Scale-Space Analysis**: Multi-resolution image understanding
- **Parameter Space Transformation**: Mapping between image and feature spaces
- **Statistical Voting**: Robust detection through accumulator arrays
- **Post-Processing Pipelines**: Combining multiple algorithms for improved results

---

**Course:** Digital Image Processing  
**Institution:** Aristotle University of Thessaloniki  
**Semester:** Spring 2025