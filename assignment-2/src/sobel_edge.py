import numpy as np
from fir_conv import fir_conv
from result_refinement import remove_small_components

def sobel_edge(
    in_img_array: np.ndarray,
    threshold: float,
    min_size: int = 40
    ) -> np.ndarray:

    """
    Detects edges using the Sobel operator.
    
    Parameters:
    -----------        
        in_img_array: ndarray
            2D array representing the input grayscale image.
            The image should be normalized to the range [0, 1].
        
        threshold: float
            Gradient magnitude threshold for edge detection.
            Pixels with gradient magnitude above this value are considered edges.
        
        min_size: int, optional
            Minimum size of connected components to keep in the edge map.
            Smaller components will be removed.            
    
    Returns:
    --------
        ndarray
            2D binary edge map of the input image.
            The output is a binary image where edges are marked as 1 and non-edges as 0.
    """
    
    #  Sobel masks for Gx1 (approximates derivative along n1, columns) and Gx2 (along n2, rows) 
    Gx = np.array([ [-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
    
    Gy = -Gx.T  # Transpose and negate to get the vertical kernel 
       
    # Convolve with kernels
    gx, _ = fir_conv(in_img_array, Gx)
    gy, _ = fir_conv(in_img_array, Gy)
    
    # Compute magnitude and threshold
    magnitude = np.hypot(gx, gy)
    binary_edge_map = (magnitude >= threshold).astype(int) # Convert to binary image by thresholding (0 or 1)

    # Refine edges by removing small components
    refined_edge_map = remove_small_components(binary_edge_map, min_size)
    
    return refined_edge_map
