# Assignment 1: Histogram Processing and Modification

This assignment implements and compares different algorithms for histogram equalization and histogram matching, demonstrating fundamental techniques for image enhancement through intensity distribution manipulation.

## Overview

Histogram processing is a cornerstone technique in digital image processing that modifies the intensity distribution of an image to improve visual quality or match specific statistical properties. This assignment explores three distinct approaches to histogram modification:

1. **Greedy Algorithm**: Sequential bin filling approach
2. **Non-Greedy Algorithm**: Balanced distribution approach  
3. **Post-Disturbance Algorithm**: Tie-breaking through controlled noise injection

## Project Structure

```
assignment-1/
├── docs/
│   ├── assignment-1.pdf           # Assignment specification
│   ├── report.pdf                 # Technical report
│   ├── report.tex                 # LaTeX source
│   └── input-images/
│       ├── input_img.jpg          # Primary test image
│       └── ref_img.jpg            # Reference image for histogram matching
├── results/
│   ├── histograms/                # Generated histogram plots
│   └── images/                    # Processed images and comparisons
└── src/
    ├── demo.py                    # Main demonstration script
    ├── hist_modif.py              # Core histogram modification algorithms
    ├── hist_utils.py              # Histogram calculation utilities
    ├── image_utils.py             # Image I/O and visualization utilities
    └── metrics.py                 # Quantitative evaluation metrics
```

## Key Features

### Histogram Equalization
- **Objective**: Transform image histogram to approximate uniform distribution
- **Benefit**: Enhances contrast by spreading intensity values across full dynamic range
- **Implementation**: Three different mapping strategies with varying characteristics

### Histogram Matching
- **Objective**: Transform image histogram to match a reference histogram
- **Benefit**: Allows precise control over output intensity distribution
- **Application**: Color correction, style transfer, and image standardization

### Algorithm Comparison
- **Greedy**: Fast, sequential approach with complete bin filling
- **Non-Greedy**: Balanced approach avoiding overfilling of intensity bins
- **Post-Disturbance**: Tie-breaking mechanism for improved distribution uniformity

## Algorithms Implemented

### 1. Greedy Histogram Modification
```python
def perform_hist_modification(img_array, hist_ref, mode='greedy'):
    """
    Assigns input intensities to output bins sequentially, completely 
    filling each bin before moving to the next.
    """
```
- **Characteristics**: Simple, deterministic mapping
- **Advantage**: Computationally efficient
- **Limitation**: May create abrupt intensity transitions

### 2. Non-Greedy Histogram Modification
```python
def perform_hist_modification(img_array, hist_ref, mode='non-greedy'):
    """
    Assigns input intensities while avoiding overfilling by checking
    if at least half of the pixels fit in the current bin.
    """
```
- **Characteristics**: Balanced distribution approach
- **Advantage**: Smoother intensity transitions
- **Trade-off**: More complex decision logic

### 3. Post-Disturbance Histogram Modification
```python
def perform_hist_modification(img_array, hist_ref, mode='post-disturbance'):
    """
    Adds small random noise to break ties, then applies greedy algorithm
    for improved histogram approximation.
    """
```
- **Characteristics**: Tie-breaking through controlled randomization
- **Advantage**: Better histogram matching accuracy
- **Application**: When precise statistical matching is required

## Quantitative Evaluation

The implementation includes comprehensive metrics for objective performance assessment:

### Earth Mover's Distance (EMD)
- Measures histogram dissimilarity as optimal transport cost
- Lower values indicate better histogram matching
- Provides intuitive interpretation of distribution differences

### Mean Squared Error (MSE)  
- Quantifies pixel-wise intensity differences
- Evaluates preservation of image structure
- Balances histogram accuracy with visual quality

### Chi-Square Distance
- Statistical measure of histogram dissimilarity
- Sensitive to bin-wise distribution differences
- Useful for comparing histogram matching accuracy

## Usage Instructions

### Quick Start
```bash
# Navigate to the assignment directory
cd assignment-1

# Run the complete demonstration
python src/demo.py
```

### Custom Processing
```python
from src.image_utils import load_image
from src.hist_modif import perform_hist_eq, perform_hist_matching

# Load your image
img = load_image('path/to/your/image.jpg')

# Perform histogram equalization
eq_img = perform_hist_eq(img, mode='non-greedy')

# Perform histogram matching with reference image
ref_img = load_image('path/to/reference/image.jpg')
matched_img = perform_hist_matching(img, ref_img, mode='greedy')
```

## Results and Analysis

The assignment generates comprehensive visualizations including:

### Histogram Equalization Results
- **Input**: Original image with its histogram
- **Output**: Equalized images using all three algorithms
- **Reference**: Ideal uniform distribution overlay
- **Metrics**: Quantitative comparison of histogram approximation quality

### Histogram Matching Results
- **Input**: Source image histogram
- **Target**: Reference image histogram  
- **Output**: Matched images with resulting histograms
- **Analysis**: Statistical evaluation of matching accuracy

### Comparative Analysis
- **Visual Comparison**: Side-by-side image comparisons
- **Quantitative Metrics**: EMD, MSE, and Chi-Square measurements
- **Algorithm Performance**: Strengths and limitations of each approach

## Key Insights

1. **Greedy Algorithm**: Fastest execution but may create artificial intensity clustering
2. **Non-Greedy Algorithm**: Best balance between accuracy and visual quality
3. **Post-Disturbance Algorithm**: Highest histogram matching accuracy but with computational overhead

## Technical Requirements

- **Python 3.7+**
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization and plotting
- **PIL/Pillow**: Image loading and manipulation
- **SciPy**: Statistical distance calculations

## Academic Learning Objectives

This assignment demonstrates:
- **Histogram Analysis**: Understanding intensity distribution characteristics
- **Transform Design**: Creating mappings between input and output intensity spaces
- **Algorithm Comparison**: Evaluating trade-offs between different approaches
- **Quantitative Evaluation**: Using metrics to assess algorithm performance objectively
- **Visual Assessment**: Interpreting results through both statistical and perceptual analysis

## Extensions and Applications

The techniques implemented in this assignment form the foundation for:
- **Adaptive Histogram Equalization**: Local contrast enhancement
- **Color Histogram Matching**: Multi-channel image processing
- **Real-time Enhancement**: Optimized implementations for video processing
- **Medical Imaging**: Contrast improvement in diagnostic images
- **Photography**: Automated exposure and contrast correction

---

**Course:** Digital Image Processing  
**Institution:** Aristotle University of Thessaloniki  
**Semester:** Spring 2025