import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat

def get_base_dir():
    """Get the base directory (parent of current script directory)"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    base_dir = get_base_dir()
    results_dir = os.path.join(base_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

def load_demo_images():
    """Load and normalize images d2a and d2b from dip_hw_3.mat"""
    print("Loading images from dip_hw_3.mat...")
    
    # Get the correct path to the data file
    base_dir = get_base_dir()
    data_file_path = os.path.join(base_dir, "docs", "dip_hw_3.mat")
    
    try:
        data = loadmat(data_file_path)
        d2a = data["d2a"]  # RGB image 1
        d2b = data["d2b"]  # RGB image 2
        print(f"Successfully loaded images from {data_file_path}:")
        print(f"  d2a shape: {d2a.shape}")
        print(f"  d2b shape: {d2b.shape}")
        
        # Convert to float and normalize to [0,1] if needed
        d2a = normalize_image(d2a)
        d2b = normalize_image(d2b)
        
        return {'d2a': d2a, 'd2b': d2b}
        
    except FileNotFoundError:
        print(f"ERROR: {data_file_path} file not found!")
        print("Please ensure the data file is in the docs/ directory.")
        return None
    except KeyError as e:
        print(f"ERROR: Key {e} not found in the .mat file!")
        return None
    
def normalize_image(img):
    """Normalize image to [0,1] range regardless of input format"""
    img = img.astype(np.float64)
    img_min, img_max = np.min(img), np.max(img)
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def visualize_segmentation(image, cluster_labels, k, title="Segmentation"):
    """
    Helper function to visualize segmentation results.
    
    Args:
        image: Original image array [M, N, C]
        cluster_labels: 1D array of cluster assignments
        k: Number of clusters
        title: Title for the visualization
    
    Returns:
        segmented_image: Image with average colors per segment
        cluster_map: 2D array showing cluster assignments
    """
    M, N, C = image.shape
    segmented_image = np.zeros_like(image)
    
    # Reshape cluster labels back to image dimensions
    cluster_map = cluster_labels.reshape(M, N)
    
    # Assign average color to each segment
    for cluster_id in range(k):
        mask = (cluster_map == cluster_id)
        if np.any(mask):
            avg_color = np.mean(image[mask], axis=0)
            segmented_image[mask] = avg_color
    
    return segmented_image, cluster_map

def visualize_binary_segmentation(image, cluster_labels, title="Binary Segmentation"):
    """Helper function to visualize binary segmentation results"""
    M, N, C = image.shape
    segmented_image = np.zeros_like(image)
    
    # Reshape cluster labels back to image dimensions
    cluster_map = cluster_labels.reshape(M, N)
    
    # Assign average color to each segment (binary: 0 and 1)
    for cluster_id in [0, 1]:
        mask = (cluster_map == cluster_id)
        if np.any(mask):
            avg_color = np.mean(image[mask], axis=0)
            segmented_image[mask] = avg_color
    
    return segmented_image, cluster_map

def analyze_cluster_statistics(cluster_labels, image_shape=None):
    """Analyze cluster statistics"""
    unique_labels = np.unique(cluster_labels)
    cluster_counts = np.bincount(cluster_labels)
    
    stats = {
        'num_clusters': len(unique_labels),
        'cluster_sizes': cluster_counts,
        'min_size': np.min(cluster_counts),
        'max_size': np.max(cluster_counts),
    }
    
    return stats

def save_comprehensive_results(fig, demo_name, image_name):
    """Save comprehensive results with organized directory structure"""
    base_dir = get_base_dir()
    results_dir = os.path.join(base_dir, "results")
    
    ensure_results_dir()
    demo_dir = os.path.join(results_dir, demo_name)
    os.makedirs(demo_dir, exist_ok=True)
    
    output_path = os.path.join(demo_dir, f'{image_name}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved results to: {output_path}")
    return output_path
