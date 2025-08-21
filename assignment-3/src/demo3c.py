"""
Demo 3c: Run complete recursive n-cuts with T1=5, T2=0.20 on images d2a and d2b

This demonstration compares binary n-cuts split versus recursive adaptive segmentation,
showing how the algorithm determines optimal clustering based on thresholds.
"""
import numpy as np
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from n_cuts import n_cuts_recursive, calculate_n_cut_value, n_cuts
from demo_utils import (load_demo_images, visualize_segmentation, 
                       analyze_cluster_statistics, save_comprehensive_results)

def visualize_adaptive_segmentation(image, cluster_labels, title="Adaptive Segmentation"):
    """Helper function for adaptive segmentation (unknown number of clusters)"""
    M, N, C = image.shape
    segmented_image = np.zeros_like(image)
    
    # Reshape cluster labels back to image dimensions
    cluster_map = cluster_labels.reshape(M, N)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    
    # Assign average color to each segment
    for cluster_id in unique_clusters:
        mask = (cluster_map == cluster_id)
        if np.any(mask):
            avg_color = np.mean(image[mask], axis=0)
            segmented_image[mask] = avg_color
    
    return segmented_image, cluster_map, len(unique_clusters)

def print_ncut_values_for_segments(affinity_mat, cluster_labels, title="Segment N-cut Values"):
    """Print N-cut values for each segment in the clustering"""
    unique_clusters = np.unique(cluster_labels)
    print(f"\n{title}:")
    
    for cluster_id in unique_clusters:
        # Create binary mask for this cluster vs all others
        cluster_mask = (cluster_labels == cluster_id).astype(int)
        
        # Calculate N-cut value for this segment
        ncut_value = calculate_n_cut_value(affinity_mat, cluster_mask)
        cluster_size = np.sum(cluster_mask)
        
        print(f"  Segment {cluster_id}: N-cut = {ncut_value:.4f}, Size = {cluster_size}")

def main():
    print("=== Demo 3c: Binary N-Cuts vs Recursive N-Cuts ===")
    
    # Load images using utility function
    images = load_demo_images()
    if images is None:
        return
    
    # Parameters for recursive n-cuts
    T1 = 5    # Minimum cluster size threshold
    T2 = 0.8 # N-cut value threshold
    
    for img_name, image in images.items():
        print(f"\n=== Processing image {img_name} ===")
        print(f"Image shape: {image.shape}")
        
        # Convert image to graph
        print("Converting image to graph...")
        affinity_mat = image_to_graph(image)
        print(f"Generated affinity matrix with shape: {affinity_mat.shape}")
        
        # Perform binary n-cuts (k=2)
        print(f"\n--- Binary N-Cuts (k=2) ---")
        binary_labels = n_cuts(affinity_mat, k=2)
        binary_stats = analyze_cluster_statistics(binary_labels)
        
        print(f"Binary N-Cuts Results:")
        print(f"  Found {binary_stats['num_clusters']} clusters")
        print(f"  Cluster sizes: {binary_stats['cluster_sizes']}")
        
        # Print N-cut values for binary segments
        print_ncut_values_for_segments(affinity_mat, binary_labels, 
                                     "Binary N-Cuts Segment Analysis")
        
        # Visualize binary results
        binary_segmented, binary_map = visualize_segmentation(
            image, binary_labels, 2, 'Binary N-Cuts'
        )
        
        # Perform recursive n-cuts
        print(f"\n--- Recursive N-Cuts (T1={T1}, T2={T2}) ---")
        recursive_labels = n_cuts_recursive(affinity_mat, T1, T2)
        recursive_stats = analyze_cluster_statistics(recursive_labels)
        
        print(f"Recursive N-Cuts Results:")
        print(f"  Found {recursive_stats['num_clusters']} clusters adaptively")
        print(f"  Cluster sizes: {recursive_stats['cluster_sizes']}")
        
        # Print N-cut values for recursive segments
        print_ncut_values_for_segments(affinity_mat, recursive_labels, 
                                     "Recursive N-Cuts Segment Analysis")
        
        # Visualize recursive results
        recursive_segmented, recursive_map, num_recursive = visualize_adaptive_segmentation(
            image, recursive_labels, 'Recursive N-Cuts'
        )
        
        # Create comparison visualization (2x2 layout)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Row 1: Binary N-Cuts
        axes[0, 0].imshow(binary_segmented)
        axes[0, 0].set_title('Binary N-Cuts\n(k=2)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(binary_map, cmap='tab10', vmin=0, vmax=9)
        axes[0, 1].set_title('Binary Cluster Map')
        axes[0, 1].axis('off')
        
        # Row 2: Recursive N-Cuts
        axes[1, 0].imshow(recursive_segmented)
        axes[1, 0].set_title(f'Recursive N-Cuts\n({num_recursive} clusters)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(recursive_map, cmap='tab20')
        axes[1, 1].set_title('Recursive Cluster Map')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Binary N-Cuts vs Recursive N-Cuts - {img_name}', fontsize=16)
        plt.tight_layout()
        
        # Save results using utility function
        save_comprehensive_results(fig, "demo3c", img_name)
        plt.show()
                
        # Comparison analysis
        print(f"\nComparison Analysis:")
        print(f"  Binary N-Cuts: {binary_stats['num_clusters']} clusters")
        print(f"  Recursive N-Cuts: {recursive_stats['num_clusters']} clusters")
        print(f"  Adaptive advantage: {recursive_stats['num_clusters'] - binary_stats['num_clusters']} additional clusters found")
            
    print("\n=== Demo 3c completed successfully! ===")

if __name__ == "__main__":
    main()