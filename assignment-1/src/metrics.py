import numpy as np
from scipy.stats import wasserstein_distance
from hist_utils import calculate_hist_of_img

def mean_squared_error(hist1, hist2):
    """
    Calculate Mean Squared Error between two histograms.
    
    Args:
        hist1, hist2: Dictionary histograms with intensity values as keys and frequencies as values
        
    Returns:
        MSE value (lower is better)
    """
    # Create a union of all intensity values
    all_intensities = sorted(set(list(hist1.keys()) + list(hist2.keys())))
    
    # Calculate MSE
    mse = 0
    for intensity in all_intensities:
        val1 = hist1.get(intensity, 0)
        val2 = hist2.get(intensity, 0)
        mse += (val1 - val2) ** 2
    
    return mse / len(all_intensities)

def bhattacharyya_distance(hist1, hist2):
    """
    Calculate Bhattacharyya distance between two histograms.
    
    Args:
        hist1, hist2: Dictionary histograms with intensity values as keys and frequencies as values
        
    Returns:
        Bhattacharyya distance (lower is better, 0 means identical distributions)
    """
    # Create a union of all intensity values
    all_intensities = sorted(set(list(hist1.keys()) + list(hist2.keys())))
    
    # Calculate Bhattacharyya coefficient
    bc = 0
    for intensity in all_intensities:
        val1 = hist1.get(intensity, 0)
        val2 = hist2.get(intensity, 0)
        bc += np.sqrt(val1 * val2)
    
    # Convert coefficient to distance
    return -np.log(bc if bc > 0 else 1e-10)

def histogram_intersection(hist1, hist2):
    """
    Calculate the histogram intersection (higher is better, 1.0 means perfect match).
    
    Args:
        hist1, hist2: Dictionary histograms with intensity values as keys and frequencies as values
        
    Returns:
        Intersection value between 0 and 1
    """
    # Create a union of all intensity values
    all_intensities = sorted(set(list(hist1.keys()) + list(hist2.keys())))
    
    # Calculate intersection
    intersection = 0
    for intensity in all_intensities:
        val1 = hist1.get(intensity, 0)
        val2 = hist2.get(intensity, 0)
        intersection += min(val1, val2)
    
    return intersection

def earth_movers_distance(hist1, hist2):
    """
    Calculate Earth Mover's Distance (Wasserstein distance) between two histograms.
    
    Args:
        hist1, hist2: Dictionary histograms with intensity values as keys and frequencies as values
        
    Returns:
        EMD value (lower is better)
    """
    # Convert dictionary histograms to arrays for wasserstein_distance
    keys1 = sorted(hist1.keys())
    keys2 = sorted(hist2.keys())
    
    values1 = [hist1[k] for k in keys1]
    values2 = [hist2[k] for k in keys2]
    
    return wasserstein_distance(keys1, keys2, values1, values2)

def evaluate_histogram_matching(modified_img, target_hist):
    """
    Evaluate how well the modified image's histogram matches the target histogram.
    
    Args:
        original_img: Original image array
        modified_img: Modified image array after histogram processing
        target_hist: Target histogram (reference or uniform)
        
    Returns:
        Dictionary of metrics
    """
    # Get the histogram of the modified image
    achieved_hist = calculate_hist_of_img(modified_img, return_normalized=True)
    
    # Calculate metrics
    metrics = {
        'MSE': mean_squared_error(achieved_hist, target_hist),
        'Bhattacharyya': bhattacharyya_distance(achieved_hist, target_hist),
        'Intersection': histogram_intersection(achieved_hist, target_hist),
        'EMD': earth_movers_distance(achieved_hist, target_hist)
    }
    
    return metrics