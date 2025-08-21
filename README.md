# Digital Image Processing Assignments 
**Author:** Fraidakis Ioannis  
**Institution:** Aristotle University of Thessaloniki  
**Date:** 2025  

## Overview
This repository contains three comprehensive assignments covering fundamental concepts and advanced techniques in digital image processing. Each assignment demonstrates key algorithms and methodologies used in computer vision and image analysis.

## Repository Structure

```
Digital-Image-Processing/
├── assignment-1/     # Histogram Processing
├── assignment-2/     # Edge Detection & Circle Detection
├── assignment-3/     # Image Segmentation
└── README.md         # This file
```

## Assignments Overview

### [Assignment 1: Histogram Processing and Modification](./assignment-1/)
- **Focus**: Histogram equalization and histogram matching
- **Algorithms**: Greedy, non-greedy, and post-disturbance histogram modification techniques
- **Applications**: Image enhancement, contrast improvement, and histogram specification
- **Key Concepts**: Cumulative distribution functions, intensity transformations, and quantitative evaluation metrics

### [Assignment 2: Edge Detection and Circle Detection](./assignment-2/)
- **Focus**: Edge detection using Sobel and Laplacian of Gaussian (LoG) filters, followed by circular Hough transform
- **Algorithms**: Sobel edge detection, LoG edge detection, and circular Hough transform with non-maximum suppression
- **Applications**: Object detection, feature extraction, and geometric shape recognition
- **Key Concepts**: Gradient computation, convolution operations, and parameter space voting

### [Assignment 3: Image Segmentation](./assignment-3/)
- **Focus**: Graph-based image segmentation using spectral clustering and normalized cuts
- **Algorithms**: Spectral clustering, normalized cuts (recursive and non-recursive), and image-to-graph conversion
- **Applications**: Object segmentation, region partitioning, and semantic image understanding
- **Key Concepts**: Graph theory, eigenvalue decomposition, and affinity matrix construction

## Technologies Used

- **Python 3.x**: Primary programming language
- **NumPy**: Numerical computations and array operations
- **OpenCV**: Computer vision operations
- **Matplotlib**: Visualization and plotting
- **SciPy**: Scientific computing and optimization
- **PIL/Pillow**: Image processing and manipulation

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fraidakis/Digital-Image-Processing.git
   cd Digital-Image-Processing
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy opencv-python matplotlib scipy pillow
   ```

3. **Navigate to any assignment directory**:
   ```bash
   cd assignment-1  # or assignment-2, assignment-3
   ```

4. **Run the demo scripts**:
   ```bash
   python src/demo.py
   ```

## Key Features

- **Modular Design**: Each assignment is self-contained with clear separation of concerns
- **Comprehensive Documentation**: Detailed docstrings and comments explaining algorithms
- **Visual Results**: Generated plots and images demonstrating algorithm performance
- **Quantitative Evaluation**: Metrics and performance analysis for objective assessment
- **Professional Code Quality**: Clean, readable, and well-structured Python implementations

## Learning Outcomes

By working through these assignments, you will gain hands-on experience with:

1. **Image Enhancement Techniques**: Understanding how to improve image quality through histogram manipulation
2. **Feature Detection**: Learning to identify edges and geometric shapes in images
3. **Advanced Segmentation**: Implementing state-of-the-art graph-based segmentation methods
4. **Algorithm Implementation**: Translating mathematical concepts into efficient code
5. **Performance Analysis**: Evaluating and comparing different algorithmic approaches

## Getting Help

Each assignment folder contains detailed documentation including:
- Assignment specifications
- Implementation guidelines

For technical support or questions, please refer to the individual assignment README files.

---

**Course:** Digital Image Processing  
**Institution:** Aristotle University of Thessaloniki  
**Semester:** Spring 2025