"""
Demo 2: Apply complete pipeline (image-to-graph + spectral clustering) on images d2a and d2b for k=2,3,4

This demonstration shows the full workflow from raw image data to segmentation results,
helping understand how different k values affect segmentation quality.
"""
import numpy as np
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
from demo_utils import (load_demo_images, visualize_segmentation, 
                       analyze_cluster_statistics, save_comprehensive_results)

def plot_affinity_matrix(affinity_mat, img_name):
    """Plot the affinity matrix as a heatmap"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot full affinity matrix
    im1 = axes[0].imshow(affinity_mat, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Affinity Matrix - {img_name}')
    axes[0].set_xlabel('Pixel Index')
    axes[0].set_ylabel('Pixel Index')
    plt.colorbar(im1, ax=axes[0], label='Affinity Value')
    
    # Plot a zoomed-in section (first 200x200 if matrix is large enough)
    zoom_size = min(200, affinity_mat.shape[0])
    im2 = axes[1].imshow(affinity_mat[:zoom_size, :zoom_size], cmap='viridis', aspect='auto')
    axes[1].set_title(f'Affinity Matrix (First {zoom_size}x{zoom_size}) - {img_name}')
    axes[1].set_xlabel('Pixel Index')
    axes[1].set_ylabel('Pixel Index')
    plt.colorbar(im2, ax=axes[1], label='Affinity Value')
    
    plt.tight_layout()
    save_comprehensive_results(plt, "demo2_affinity", img_name)
    plt.show()

def main():
    print("=== Demo 2: Complete Pipeline (Image-to-Graph + Spectral Clustering) ===")
    
    # Load images using utility function
    images = load_demo_images()
    if images is None:
        return
    
    k_values = [2, 3, 4]
    
    for img_name, image in images.items():
        print(f"\n=== Processing image {img_name} ===")
        print(f"Image shape: {image.shape}")
        
        # Save the original input image
        print(f"Saving original image {img_name}...")
        fig_input = plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        save_comprehensive_results(plt, "demo2_input", img_name)
        plt.close(fig_input)
        
        # Convert image to graph
        print("Converting image to graph...")
        affinity_mat = image_to_graph(image)
        print(f"Generated affinity matrix with shape: {affinity_mat.shape}")
        print(f"Affinity matrix statistics:")
        print(f"  - Min value: {affinity_mat.min():.6f}")
        print(f"  - Max value: {affinity_mat.max():.6f}")
        print(f"  - Mean value: {affinity_mat.mean():.6f}")
        
        # Plot the affinity matrix
        plot_affinity_matrix(affinity_mat, img_name)
        
        # Create visualization for clustering results
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Show original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Store results for performance summary
        clustering_results = {}
        
        # Process each k value
        for i, k in enumerate(k_values):
            print(f"\n--- Applying spectral clustering with k={k} ---")
            
            # Perform spectral clustering
            cluster_labels = spectral_clustering(affinity_mat, k)
            
            # Analyze results using utility function
            stats = analyze_cluster_statistics(cluster_labels)
            clustering_results[f"k={k}"] = stats
            
            print(f"Cluster distribution: {stats['cluster_sizes']}")
            
            # Visualize results using utility function
            segmented_image, cluster_map = visualize_segmentation(image, cluster_labels, k, f'k={k}')
            
            # Show segmented image (average colors per segment)
            axes[0, i+1].imshow(segmented_image)
            axes[0, i+1].set_title(f'Segmented k={k}')
            axes[0, i+1].axis('off')
            
            # Show cluster map (different colors for each cluster)
            # Needed eg for first image that has only discrete colors 
            # and for k=4, two clusters are merged when using the average 
            discrete_cmap = plt.colormaps.get_cmap('tab20b').resampled(k)
            axes[1, i+1].imshow(cluster_map, cmap=discrete_cmap)
            axes[1, i+1].set_title(f'Cluster Map k={k}')
            axes[1, i+1].axis('off')
                    
        plt.suptitle(f'Spectral Clustering Results - {img_name}', fontsize=16)
        plt.tight_layout()
        
        # Save results using utility function
        save_comprehensive_results(plt, "demo2", img_name)
        plt.show()
            
    print("\n=== Demo 2 completed successfully! ===")

if __name__ == "__main__":
    main()