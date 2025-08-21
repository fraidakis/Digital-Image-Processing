import numpy as np
from typing import Dict

def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict:
    """
    Calculate the histogram of a grayscale image.
    
    The histogram represents the distribution of pixel intensities in the image.
    This function counts occurrences of each unique intensity value and returns them as a dictionary.
    
    Args:
        img_array (np.ndarray): A 2D numpy array of type float with values in [0, 1] representing a grayscale image.
        return_normalized (bool): If True, returns normalized histogram (relative frequencies that sum to 1.0), otherwise returns raw counts.
        
    Returns:
        Dict: A dictionary with:
            - keys: intensity levels found in the image
            - values: counts or relative frequencies of each intensity level depending on the return_normalized parameter
    
    Example:
        >>> img = np.array([[0.1, 0.2], [0.1, 0.3]])
        >>> calculate_hist_of_img(img, False) 
        {0.1: 2, 0.2: 1, 0.3: 1}
        >>> calculate_hist_of_img(img, True) 
        {0.1: 0.5, 0.2: 0.25, 0.3: 0.25}
    """
    # Get unique values and their counts in a single operation
    unique_values, counts = np.unique(img_array, return_counts=True)
    
    # Create histogram dictionary directly using the counts
    total_pixels = img_array.size
    
    if return_normalized:
        # Calculate relative frequencies by dividing counts by total number of pixels
        hist = dict(zip(unique_values, counts / total_pixels))
    else:
        # Use raw counts
        hist = dict(zip(unique_values, counts))
    
    return hist


def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict) -> np.ndarray:
    """
    Apply a histogram modification transform to a grayscale image.
    
    This function maps each pixel intensity in the input image to a new intensity
    value according to the provided transformation dictionary.
    
    Args:
        img_array (np.ndarray): A 2D numpy array representing a grayscale image.
        modification_transform (Dict): A dictionary mapping original intensity values
                                      to new intensity values. If an intensity value
                                      is not found in the dictionary, it remains unchanged.
    
    Returns:
        np.ndarray: A new image array with the same shape as the input, where each pixel's intensity has been transformed according to the mapping.
    
    Example:
        >>> img = np.array([[0.1, 0.2], [0.1, 0.3]])
        >>> transform = {0.1: 0.5, 0.2: 0.6}
        >>> apply_hist_modification_transform(img, transform)
        array([[0.5, 0.6], [0.5, 0.3]])
    """
    # Create an empty array with the same shape and data type as the input
    modified_img = np.zeros_like(img_array)
    
    # Iterate through each pixel in the image
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            # Apply transformation by looking up the value in the dictionary
            # If not found, keep the original value (default behavior of .get())
            modified_img[i, j] = modification_transform.get(img_array[i, j], img_array[i, j])
    
    return modified_img