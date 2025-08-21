import numpy as np
from fir_conv import fir_conv
from result_refinement import remove_small_components

def log_edge(
    in_img_array: np.ndarray,
    kernel_size: int = 5
    ) -> np.ndarray:

    """
    Detect edges using Laplacian of Gaussian and zero crossing detection.
    
    Parameters:
    -----------
        in_img_array : ndarray
            Input grayscale image.
    
        kernel_size : int, optional
            Size of the LoG kernel (must be odd).
        
    Returns:
    --------
        ndarray
            2D binary edge map of the input image.
            The output is a binary image where edges are marked as 1 and non-edges as 0.
    """

    if kernel_size == 5: 
        log_kernel = np.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=float)
        thresh_mult = 0.5
        min_size = 5

    else:
        sigma, thresh_mult, min_size, kernel_size = calculate_parameters(kernel_size)
        log_kernel = create_log_kernel(sigma, kernel_size)
    
    
    # Apply LoG filter
    filtered_img, _ = fir_conv(in_img_array, log_kernel)
    
    # Detect zero crossings with adaptive thresholding
    binary_edge_map = find_zero_crossings(filtered_img, thresh_mult)
    
    # Clean small edge segments
    refined_edge_map = remove_small_components(binary_edge_map, min_size)

    # Crop the image to the original size due to full convolution
    if kernel_size % 2 == 0:
        half_size = kernel_size // 2
        refined_edge_map = refined_edge_map[half_size:-half_size, half_size:-half_size]
    else:
        half_size = kernel_size // 2
        refined_edge_map = refined_edge_map[half_size:-half_size+1, half_size:-half_size+1]

    return refined_edge_map


def create_log_kernel(sigma, size):
    """
    Create a Laplacian of Gaussian (LoG) kernel.
    
    Parameters:
    -----------
    sigma : float
        Standard deviation of the Gaussian.
    size : int
        Size of the kernel.
        
    Returns:
    --------
    ndarray
        LoG kernel.
    """
    # Create mesh grid
    half_size = size // 2
    y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
    
    # Calculate LoG values
    variance = sigma ** 2
    denom = 2 * np.pi * variance
    exp_term = np.exp(-(x**2 + y**2) / (2 * variance))
    kernel = -1 / denom * (1 - (x**2 + y**2) / (2 * variance)) * exp_term
    
    # Ensure kernel sums to approximately zero
    kernel -= np.mean(kernel)
    
    return kernel

def find_zero_crossings(log_response, thresh_mult=0.0):
    """
    Detect zero crossings in the LoG response.
    
    Parameters:
    -----------
    log_response : ndarray
        LoG filter response.
    thresh_mult : float
        thresh_multtiplier for threshold calculation.
        
    Returns:
    --------
    ndarray
        Binary edge map.
    """
    rows, cols = log_response.shape
    edge_map = np.zeros((rows, cols), dtype=np.uint8)
    
    # Create shifted versions for each of the 8 neighbors
    shifts = [
        (0, 1),   # right
        (1, 1),   # bottom-right
        (1, 0),   # bottom
        (1, -1),  # bottom-left
        (0, -1),  # left
        (-1, -1), # top-left
        (-1, 0),  # top
        (-1, 1)   # top-right
    ]
    
    # Calculate adaptive threshold based on response intensity
    diffs = []
    for dy, dx in shifts:
        # Roll the image in both dimensions
        shifted = np.roll(np.roll(log_response, dy, axis=0), dx, axis=1)
        # Compute product (negative means zero crossing)
        product = log_response * shifted
        # Compute absolute difference
        abs_diff = np.abs(log_response - shifted)
        # Get differences at zero crossings
        diffs.extend(abs_diff[product < 0].flatten())
    
    if diffs:  # If there are zero crossings
        threshold = np.mean(diffs) + thresh_mult * np.std(diffs)
    else:
        threshold = 0
    
    # Detect zero crossings with intensity check in one pass
    for dy, dx in shifts:
        shifted = np.roll(np.roll(log_response, dy, axis=0), dx, axis=1)
        product = log_response * shifted
        abs_diff = np.abs(log_response - shifted)
        # Zero crossing with sufficient gradient
        edge_map = np.logical_or(edge_map, (product < 0) & (abs_diff > threshold))
    
    return edge_map.astype(np.uint8)

def calculate_parameters(kernel_size=13):
    """
    Calculate parameters for LoG edge detection.
    
    Parameters:
    -----------
    kernel_size : int
        Size of the LoG kernel (must be odd).

    Returns:
    --------
    tuple
        Parameters for LoG edge detection.
    """
    
    if kernel_size == 7:
        sigma = 2
        thresh_mult = 0.35
        min_size = 50
    elif kernel_size == 9:
        sigma = 2
        thresh_mult = 0.4
        min_size = 30
    elif kernel_size == 10:
        sigma = 2
        thresh_mult = 0.2
        min_size = 50
    elif kernel_size == 11:
        sigma = 2.5
        thresh_mult = 0.3
        min_size = 60
    elif kernel_size == 13:
        sigma = 2.9
        thresh_mult = 0.1
        min_size = 60
    elif kernel_size == 15:
        sigma = 2.9
        thresh_mult = 0.11
        min_size = 65
    else:
        # Default values if dimension doesn't match any in the database
        kernel_size = 13
        sigma = 2.9
        thresh_mult = 0.1
        min_size = 60

    return sigma, thresh_mult, min_size, kernel_size
