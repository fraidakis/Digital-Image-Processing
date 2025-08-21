from hist_utils import calculate_hist_of_img, apply_hist_modification_transform
import numpy as np
from typing import Dict


##################################  Histogram Modification  ##################################

def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    """
    Modify the histogram of an input image to match a reference histogram using a specified algorithm.
    
    This function transforms pixel intensity values to produce an output image whose intensity
    distribution approximates the provided reference histogram. Three different mapping strategies are 
    available, offering different tradeoffs between histogram accuracy and preservation of local contrasts.

    Args:
        img_array (np.ndarray): 2D numpy array (grayscale image) with values in [0, 1].
        hist_ref (Dict[float, float]): Reference histogram as {intensity_value: probability}, where probabilities sum to 1.
        mode (str): Algorithm to use. One of: 'greedy', 'non-greedy', or 'post-disturbance'.

    Returns:
        np.ndarray: Image array with modified histogram matching the reference.

    Raises:
        ValueError: If input arguments are invalid.

    Notes:
        - 'greedy': Assigns input intensities to output bins in order, completely filling each bin before moving to the next.
        - 'non-greedy': Assigns input intensities to output bins, avoiding overfilling by checking if at least half of the pixels fit.
        - 'post-disturbance': Adds small random noise to break ties, then applies the greedy algorithm.
    """

    # Input validation
    if not isinstance(img_array, np.ndarray) or img_array.size == 0:
        raise ValueError("Input image must be a non-empty numpy array")
    
    if not isinstance(hist_ref, dict) or not hist_ref:
        raise ValueError("Reference histogram must be a non-empty dictionary")
        
    valid_modes = ['greedy', 'non-greedy', 'post-disturbance']
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")

    # Calculate the histogram of the input image (pixel counts, not normalized)
    input_hist = calculate_hist_of_img(img_array, return_normalized=False)
    
    # Get sorted lists of input and output intensity values
    sorted_input_intensity_values = sorted(input_hist.keys())
    N = img_array.size  # Total number of pixels
    sorted_output_intensity_values = sorted(hist_ref.keys())
    
    # Convert reference probabilities to target pixel counts using total pixels N
    target_counts = {g: N * hist_ref[g] for g in sorted_output_intensity_values}
    
    # Mapping from input intensity to output intensity
    histogram_mapping = {}
    
    if mode == 'greedy':
        
        current_input_index = 0
        
        for output_intensity in sorted_output_intensity_values:  # For each output intensity value
            target_pixels = target_counts[output_intensity] # How many pixels should have this output value
            accumulated_pixels = 0 # Tracks pixels assigned so far to this output bin
            
            # Assign input pixels to current output value until we reach the target
            while current_input_index < len(sorted_input_intensity_values) and accumulated_pixels < target_pixels:
                current_input_intensity = sorted_input_intensity_values[current_input_index] # Current intensity value
                pixel_count = input_hist[current_input_intensity]  # Number of pixels with this input value
                
                # Add this input value to current output bin
                histogram_mapping[current_input_intensity] = output_intensity 
                accumulated_pixels += pixel_count
                current_input_index += 1
    
    elif mode == 'non-greedy':
            
            current_input_index = 0      
            
            for output_intensity in sorted_output_intensity_values:
                target_pixels = target_counts[output_intensity]
                accumulated_pixels = 0

                if(target_pixels == 0):
                    # If no pixels should have this output value, skip to next
                    continue
                
                # Process input values: assign at least one, then continue while half or more would fit
                while True:
                    if current_input_index >= len(sorted_input_intensity_values):
                        break

                    # Get current input intensity and pixel count
                    current_input_intensity = sorted_input_intensity_values[current_input_index]
                    pixel_count = input_hist[current_input_intensity]

                    # Calculate deficiency: how many pixels are left to fill in the current output bin
                    deficiency = target_pixels - accumulated_pixels  

                    # Non-greedy decision strategy:
                    # 1. First value for each bin is always assigned (to ensure every bin gets at least one value)
                    # 2. Otherwise, only assign if at least 50% of pixels would fit in remaining bin capacity
                    if accumulated_pixels == 0 or deficiency >= pixel_count / 2:
                        histogram_mapping[current_input_intensity] = output_intensity
                        accumulated_pixels += pixel_count
                        current_input_index += 1
                    else:
                        # Less than half would fit - stop filling current bin to avoid overfilling
                        break

    elif mode == 'post-disturbance':
        
        unique_values = sorted(input_hist.keys())
        
        # Calculate typical intensity step size between adjacent values to determine appropriate noise scale
        d = unique_values[1] - unique_values[0] if len(unique_values) > 1 else 0.01
        
        # Add small random noise to break ties while preserving overall ordering
        noise = np.random.uniform(-d/2, d/2, img_array.shape)
        # Keep noise within valid range [0, 1]
        noisy_image = np.clip(img_array + noise, 0, 1)

        return perform_hist_modification(noisy_image, hist_ref, 'greedy')        
        
    # Apply the transformation to the original image
    return apply_hist_modification_transform(img_array, histogram_mapping)



###################################  Histogram Equalization  ###################################

def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    """
    Perform histogram equalization on a grayscale image using the specified algorithm.
    
    This implementation creates a uniform reference histogram, then applies the chosen histogram modification technique.

    Args:
        img_array (np.ndarray): 2D numpy array of floats in [0, 1] representing a grayscale image.
        mode (str): Histogram modification mode: 'greedy', 'non-greedy', or 'post-disturbance'.

    Returns:
        np.ndarray: Equalized image as a 2D numpy array with values in [0, 1].

    Notes:
        - Uses 256 uniform intensity levels (standard for 8-bit images)
        - Each intensity level gets equal probability in the target histogram
        - The output image can be further processed with min-max stretching to normalize values to [0, 1].
    """

    # Set up 256 discrete intensity levels for the equalized histogram
    Lg = 256
    
    # Create a uniform reference histogram with equal probability for each intensity level
    # Format: {intensity_value: probability}, where:
    #   - intensity_value ranges from 0 to 1 in (Lg-1) equal steps: 0, 1/255, 2/255, ..., 1
    #   - probability is constant at 1/Lg for perfect uniformity (all values equally likely)
    hist_ref = {i/(Lg-1): 1/Lg for i in range(Lg)}
    
    # Apply histogram modification using the specified algorithm to match the uniform target histogram
    equalized_img = perform_hist_modification(img_array, hist_ref, mode)

    # Calculate a version with full [0,1] range for maximum contrast
    stretch_img = apply_min_max_stretch(equalized_img)  # Normalize to [0, 1] range   

    # Change which image to return (equalized_img or stretch_img) based on the desired output
    return equalized_img 



#####################################  Histogram Matching  ######################################

def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    """
    Modify the histogram of an input image to match that of a reference image.
    
    This function extracts the intensity distribution from a reference image and applies it to the source image. 

    Args:
        img_array (np.ndarray): Source image to be modified (2D array, values in [0,1]).
        img_array_ref (np.ndarray): Reference image whose histogram will be matched.
        mode (str): Algorithm to use: 'greedy', 'non-greedy', or 'post-disturbance'.

    Returns:
        np.ndarray: Modified image with histogram matching the reference image.        
    """
    # Calculate normalized histogram of reference image
    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized=True)
    
    # Perform histogram modification using the reference histogram
    return perform_hist_modification(img_array, hist_ref, mode)



#####################################  Min-Max Stretching  ####################################

def apply_min_max_stretch(img_array: np.ndarray) -> np.ndarray:
    """
    Apply min-max stretch to normalize image values to full [0,1] range.
    
    This function linearly stretches the intensity values in an image to span the full [0,1] range.
    
    Args:
        img_array (np.ndarray): Input image array with any range of values
        
    Returns:
        np.ndarray: Normalized image with values stretched to [0,1] range
    """

    min_val = np.min(img_array)
    max_val = np.max(img_array)
    
    # Avoid division by zero if all pixel values are the same
    if max_val != min_val:        
        return (img_array - min_val) / (max_val - min_val)
    else:
        return img_array
    