# Assignment 3: Image Segmentation with Spectral Clustering and Normalized Cuts

This assignment implements advanced graph-based image segmentation techniques, including spectral clustering and normalized cuts (N-cuts). It demonstrates how to transform images into graph representations and apply sophisticated mathematical frameworks for semantic region partitioning.

## Overview

Image segmentation is one of the most challenging problems in computer vision, requiring the partitioning of images into semantically meaningful regions. This assignment explores state-of-the-art graph-theoretical approaches that model images as weighted graphs where pixels become vertices and similarity relationships become edges.

The implementation covers three major components:
1. **Image-to-Graph Conversion**: Transforming spatial image data into graph representations
2. **Spectral Clustering**: Eigenvalue-based clustering using graph Laplacians
3. **Normalized Cuts**: Optimization-based segmentation with recursive refinement

## Project Structure

```
assignment-3/
├── docs/
│   ├── assignment-3.pdf          # Assignment specification
│   ├── dip_hw_3.mat              # MATLAB data file with test matrices
│   ├── report.pdf                # Technical report and analysis
│   └── report.tex                # LaTeX source
├── results/
│   ├── demo1/                    # Spectral clustering on pre-built affinity
│   ├── demo2/                    # Image-to-graph conversion results
│   ├── demo2_affinity/           # Affinity matrix visualizations
│   ├── demo2_input/              # Input image processing
│   ├── demo3a/                   # Non-recursive N-cuts results
│   ├── demo3b/                   # Recursive N-cuts results
│   └── demo3c/                   # Advanced N-cuts with initialization
├── requirements.txt              # Python dependencies
└── src/
    ├── demo1.py                  # Spectral clustering validation
    ├── demo2.py                  # Image-to-graph demonstration
    ├── demo3a.py                 # Non-recursive N-cuts
    ├── demo3b.py                 # Recursive N-cuts implementation
    ├── demo3c.py                 # Advanced N-cuts with custom initialization
    ├── demo_utils.py             # Shared utilities and visualization
    ├── image_to_graph.py         # Image-to-graph conversion algorithm
    ├── n_cuts.py                 # Normalized cuts implementation
    ├── spectral_clustering.py    # Spectral clustering algorithm
    └── run_all.py                # Execute all demonstrations
```

## Core Algorithms

### 1. Image-to-Graph Conversion

Transforms images into fully-connected weighted graphs where pixel similarity drives edge weights.

```python
def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    """
    Convert image into fully-connected graph representation.
    
    Args:
        img_array: Input image [M, N, C] with values in [0, 1]
    
    Returns:
        Affinity matrix W where W[i,j] represents similarity between pixels i and j
    """
```

**Key Features:**
- **Spatial-Intensity Affinity**: Combines spatial proximity and intensity similarity
- **Gaussian Weighting**: Exponential decay based on feature distance
- **Multi-Channel Support**: Handles grayscale and color images
- **Computational Optimization**: Efficient vectorized operations

**Affinity Function:**
```
W(i,j) = exp(-γ_I * ||F_I(i) - F_I(j)||²) * exp(-γ_X * ||F_X(i) - F_X(j)||²)
```
Where:
- `F_I(i)`: Intensity feature vector at pixel i
- `F_X(i)`: Spatial coordinate vector at pixel i  
- `γ_I`, `γ_X`: Scaling parameters for intensity and spatial similarity

### 2. Spectral Clustering

Implements spectral clustering using the normalized graph Laplacian for dimensionality reduction followed by k-means clustering.

```python
def spectral_clustering(affinity_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Perform spectral clustering on affinity matrix.
    
    Args:
        affinity_matrix: Symmetric affinity matrix W
        k: Number of clusters
    
    Returns:
        Cluster labels for each data point
    """
```

**Algorithm Steps:**
1. **Graph Laplacian Construction**: L = D - W (where D is degree matrix)
2. **Normalized Laplacian**: L_norm = D^(-1/2) * L * D^(-1/2)  
3. **Eigendecomposition**: Find k smallest eigenvalues and eigenvectors
4. **Embedding**: Project data into k-dimensional spectral space
5. **K-means Clustering**: Cluster embedded points

**Mathematical Foundation:**
- **Graph Cut Theory**: Minimizes normalized cut criterion
- **Spectral Graph Theory**: Leverages eigenstructure of graph Laplacian
- **Relaxation**: Continuous relaxation of discrete partitioning problem

### 3. Normalized Cuts (N-Cuts)

Implements the normalized cuts algorithm for optimal graph partitioning with both non-recursive and recursive variants.

```python
def n_cuts(W: np.ndarray, k: int, recursive: bool = False) -> np.ndarray:
    """
    Perform normalized cuts segmentation.
    
    Args:
        W: Affinity matrix
        k: Number of segments
        recursive: Whether to use recursive bisection
    
    Returns:
        Segment labels for each pixel
    """
```

**Non-Recursive N-Cuts:**
- **Multi-way Partitioning**: Direct k-way segmentation
- **Simultaneous Optimization**: Considers all segments simultaneously
- **Global Solution**: Finds globally optimal partitioning

**Recursive N-Cuts:**
- **Hierarchical Bisection**: Iterative binary partitioning
- **Local Optimization**: Each split optimized independently  
- **Tree Structure**: Creates hierarchical segmentation

**N-Cuts Objective:**
```
NCuts(A,B) = Cut(A,B)/Vol(A) + Cut(A,B)/Vol(B)
```
Where Cut(A,B) is the sum of edge weights between sets A and B, and Vol(A) is the sum of all edge weights from set A.

## Demonstration Programs

### Demo 1: Spectral Clustering Validation
```bash
python src/demo1.py
```
- **Purpose**: Validate spectral clustering on provided affinity matrix
- **Test Cases**: k = 2, 3, 4 clusters on controlled dataset
- **Output**: Cluster visualizations and performance analysis

### Demo 2: Image-to-Graph Conversion
```bash
python src/demo2.py  
```
- **Purpose**: Demonstrate image-to-graph transformation
- **Features**: Affinity matrix visualization, parameter sensitivity analysis
- **Output**: Graphs showing spatial and intensity relationships

### Demo 3a: Non-Recursive N-Cuts
```bash
python src/demo3a.py
```
- **Purpose**: Multi-way normalized cuts segmentation
- **Method**: Direct k-way partitioning
- **Analysis**: Comparison with spectral clustering results

### Demo 3b: Recursive N-Cuts  
```bash
python src/demo3b.py
```
- **Purpose**: Hierarchical binary segmentation
- **Method**: Recursive bisection approach
- **Output**: Tree-structured segmentation results

### Demo 3c: Advanced N-Cuts
```bash
python src/demo3c.py
```
- **Purpose**: N-cuts with custom initialization strategies
- **Features**: Multiple initialization methods, stability analysis
- **Comparison**: Performance evaluation across different approaches

## Complete Pipeline Execution

```bash
# Run all demonstrations sequentially
python src/run_all.py
```

This executes the complete segmentation pipeline, generating comprehensive results for analysis and comparison.

## Usage Instructions

### Quick Start
```bash
# Navigate to assignment directory
cd assignment-3

# Install dependencies
pip install -r requirements.txt

# Run complete demonstration
python src/run_all.py
```

### Custom Image Segmentation
```python
from src.image_to_graph import image_to_graph
from src.spectral_clustering import spectral_clustering
from src.n_cuts import n_cuts

# Load and preprocess image
img = load_image('your_image.png')

# Convert to graph representation
affinity_matrix = image_to_graph(img)

# Apply spectral clustering
spectral_labels = spectral_clustering(affinity_matrix, k=3)

# Apply normalized cuts
ncuts_labels = n_cuts(affinity_matrix, k=3, recursive=False)

# Visualize results
visualize_segmentation(img, spectral_labels, ncuts_labels)
```

### Parameter Tuning

**Image-to-Graph Parameters:**
- **γ_I (Intensity Scale)**: Controls sensitivity to intensity differences
  - Higher values: More sensitive to intensity variations
  - Lower values: More tolerant of intensity differences
- **γ_X (Spatial Scale)**: Controls spatial neighborhood influence
  - Higher values: Stronger spatial locality constraint
  - Lower values: Weaker spatial influence

**Clustering Parameters:**
- **k (Number of Clusters)**: Target number of segments
- **Recursive vs. Non-Recursive**: Trade-off between optimality and hierarchy

## Results and Analysis

### Generated Visualizations

1. **Affinity Matrices**: Heatmaps showing pixel-pixel similarity relationships
2. **Segmentation Results**: Color-coded segment assignments overlaid on original images
3. **Comparative Analysis**: Side-by-side comparison of different algorithms
4. **Parameter Sensitivity**: Analysis of algorithm behavior across parameter ranges

### Quantitative Evaluation

The implementation includes metrics for objective assessment:
- **Normalized Cut Value**: Optimization objective evaluation
- **Silhouette Score**: Cluster quality measurement  
- **Segment Coherence**: Within-segment similarity analysis
- **Boundary Quality**: Edge-preserving segmentation assessment

## Technical Requirements

- **Python 3.8+**
- **NumPy**: Matrix operations and linear algebra
- **SciPy**: Sparse matrices and eigenvalue computations
- **Matplotlib**: Comprehensive visualization capabilities
- **scikit-learn**: K-means clustering and evaluation metrics
- **PIL/Pillow**: Image I/O and basic processing

## Advanced Mathematical Concepts

### Graph Theory Foundations
- **Weighted Graphs**: Representation of image similarity relationships
- **Graph Laplacian**: Fundamental operator encoding graph structure
- **Spectral Properties**: Eigenvalues and eigenvectors for partitioning

### Optimization Theory
- **Quadratic Programming**: N-cuts as generalized eigenvalue problem
- **Relaxation Techniques**: Continuous approximation of discrete problems
- **Global vs. Local Optima**: Trade-offs in optimization strategies

### Linear Algebra
- **Eigendecomposition**: Core computational component
- **Sparse Matrix Operations**: Efficient handling of large affinity matrices
- **Numerical Stability**: Robust computation of small eigenvalues

## Applications and Extensions

The implemented techniques form the foundation for:

- **Medical Image Analysis**: Organ and tissue segmentation
- **Autonomous Driving**: Road scene understanding and object detection
- **Satellite Imagery**: Land cover classification and change detection
- **Manufacturing**: Quality control and defect detection
- **Content-Based Retrieval**: Image database organization and search
- **Augmented Reality**: Real-time scene understanding and object tracking

## Learning Outcomes

Students gain hands-on experience with:
1. **Graph-Based Modeling**: Representing images as mathematical graphs
2. **Spectral Methods**: Using eigenvalue decomposition for data analysis
3. **Optimization Techniques**: Solving complex partitioning problems
4. **Algorithm Implementation**: Translating mathematical theory into efficient code
5. **Performance Analysis**: Evaluating and comparing sophisticated algorithms
6. **Real-World Applications**: Understanding modern computer vision methodologies

---

**Course:** Digital Image Processing  
**Institution:** Aristotle University of Thessaloniki  
**Semester:** Spring 2025