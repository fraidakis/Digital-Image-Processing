import numpy as np

def fir_conv(
    in_img_array: np.ndarray,
    h: np.ndarray,
    in_origin: np.ndarray = None,
    mask_origin: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform 'full' convolution between an image and a mask, similar to 
    scipy.signal.convolve2d(mode='full').
    
    Parameters:
    in_img_array: Input image as numpy array
    h: Convolution mask/kernel
    in_origin: Origin coordinates of the input image. Defaults to [0,0].
    mask_origin: Origin coordinates of the mask. Defaults to [1,1] to match
                 the behavior of the reference fir_convFast implementation if not provided.
    
    Returns:
    Tuple of (output image array, output origin coordinates)
    """
    # Flip the kernel for convolution
    h_flipped = np.flipud(np.fliplr(h))

    if in_origin is None:
        # Default in_origin to [0, 0]
        in_origin = np.array([0, 0])
    
    if mask_origin is None:
        # Default mask_origin to center of mask
        mask_origin = np.array([h.shape[0] // 2, h.shape[1] // 2], dtype=int)
    
    # Calculate output origin (this calculation remains the same)
    out_origin = np.array([in_origin[0] + mask_origin[0], in_origin[1] + mask_origin[1]])
    
    M, N = in_img_array.shape
    m, n = h.shape

    # Calculate output dimensions for 'full' convolution
    out_rows = M + m - 1
    out_cols = N + n - 1
    
    # Create output array initialized to zeros
    out_img_array = np.zeros((out_rows, out_cols), dtype=np.float64)
    
    # Perform 'full' convolution using sum of scaled, shifted inputs
    for r_k in range(m):  # Iterate over kernel rows
        for c_k in range(n):  # Iterate over kernel columns
            if h_flipped[r_k, c_k] == 0: # Optimization: skip if kernel element is zero
                continue
            
            out_img_array[r_k : r_k + M, c_k : c_k + N] += in_img_array * h_flipped[r_k, c_k]
    
    return out_img_array, out_origin
