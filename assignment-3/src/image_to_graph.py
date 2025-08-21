import numpy as np
from scipy.spatial.distance import pdist, squareform

def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    """
    Convert an image into a fully-connected graph representation.
    
    This function transforms spatial image data into a mathematical graph where each pixel
    becomes a vertex and edge weights represent pixel similarity. The affinity matrix
    uses exponential weighting to ensure similar pixels have high affinity values.
    
    Args:
        img_array: numpy array of dtype=float, dimensions [M, N, C], 
                  values in range [0, 1], representing a C-channel input image
    
    Returns:
        affinity_mat: 2D numpy array of dtype=float, dimensions [M*N, M*N], 
                     representing the affinity matrix where A(i,j) = e^(-d(i,j))
    """
    # Validate input
    if not isinstance(img_array, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if len(img_array.shape) != 3:
        raise ValueError("Input must be 3D array with dimensions [M, N, C]")
    
    # Get image dimensions
    M, N, C = img_array.shape
    n_pixels = M * N
    
    # Reshape image to (M*N, C) - each row represents a pixel's color values
    pixels = img_array.reshape(n_pixels, C)
    
    # Efficient pairwise distance calculation using SciPy
    # Only the upper triangle is computed, but squareform will convert it to a full matrix
    # distances = squareform(pdist(pixels, metric='euclidean'))
    distances = squareform(pdist(pixels, metric='chebyshev'))
    
    # Apply exponential weighting: A(i,j) = e^(-d(i,j))
    # This ensures similar pixels (small distance) have high affinity (close to 1),
    # and dissimilar pixels (large distance) have low affinity (close to 0).
    affinity_mat = np.exp(-distances)

    # The graph is fully-connected by this construction, as exp(-d) is always > 0.
    return affinity_mat
