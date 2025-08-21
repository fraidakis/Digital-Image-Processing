"""
Demo 3b: Execute single-step recursive n-cuts (binary split) on images d2a and d2b

This demonstration isolates the effect of the generalized eigenvalue problem by
comparing binary splits and displaying n-cut values.
"""
import numpy as np
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from n_cuts import n_cuts, calculate_n_cut_value
from spectral_clustering import spectral_clustering
from demo_utils import (load_demo_images, visualize_binary_segmentation, 
                       analyze_cluster_statistics, save_comprehensive_results)

def main():
    print("=== Demo 3b: Binary N-Cuts Analysis ===")
    
    # Load images using utility function
    images = load_demo_images()
    if images is None:
        return
    
    for img_name, image in images.items():
        print(f"\n=== Processing image {img_name} ===")
        print(f"Image shape: {image.shape}")
        
        # Convert image to graph
        print("Converting image to graph...")
        affinity_mat = image_to_graph(image)
        print(f"Generated affinity matrix with shape: {affinity_mat.shape}")
        
        # Perform binary splits (k=2) with both methods
        print("\n--- Binary Clustering Comparison ---")
                
        # Spectral clustering binary split
        print("Applying Spectral Clustering (k=2)...")
        spectral_labels = spectral_clustering(affinity_mat, k=2)
        spectral_value = calculate_n_cut_value(affinity_mat, spectral_labels)
        spectral_stats = analyze_cluster_statistics(spectral_labels)

        # N-Cuts binary split
        print("Applying N-Cuts (k=2)...")
        ncuts_labels = n_cuts(affinity_mat, k=2)
        ncuts_value = calculate_n_cut_value(affinity_mat, ncuts_labels)
        ncuts_stats = analyze_cluster_statistics(ncuts_labels)
    
        # Display metrics
        print(f"\nResults for {img_name}:")
        print(f"N-Cuts:")
        print(f"  Cluster distribution: {ncuts_stats['cluster_sizes']}")
        print(f"  N-cut value: {ncuts_value:.6f}")
        
        print(f"Spectral:")
        print(f"  Cluster distribution: {spectral_stats['cluster_sizes']}")
        print(f"  N-cut value: {spectral_value:.6f}")
        
        print(f"\nN-cut value comparison:")
        if ncuts_value < spectral_value:
            print(f"  N-Cuts produces better cut (lower n-cut value)")
            print(f"  Improvement: {((spectral_value - ncuts_value) / spectral_value * 100):.2f}%")
        elif spectral_value < ncuts_value:
            print(f"  Spectral produces better cut (lower n-cut value)")
            print(f"  Improvement: {((ncuts_value - spectral_value) / ncuts_value * 100):.2f}%")
        else:
            print(f"  Both methods produce identical cuts")
        
        # Create detailed visualization
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        
        # Original image (top row, center)
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')
        
        # Spectral results (middle row)
        spectral_segmented, spectral_map = visualize_binary_segmentation(image, spectral_labels, 'Spectral')
        
        axes[1, 0].imshow(spectral_segmented)
        axes[1, 0].set_title(f'Spectral Binary Split\nN-cut: {spectral_value:.4f}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(spectral_map, cmap='RdYlBu')
        axes[1, 1].set_title('Spectral Cluster Map')
        axes[1, 1].axis('off')
        
        # N-cuts results (bottom row)
        ncuts_segmented, ncuts_map = visualize_binary_segmentation(image, ncuts_labels, 'N-Cuts')
        
        axes[2, 0].imshow(ncuts_segmented)
        axes[2, 0].set_title(f'N-Cuts Binary Split\nN-cut: {ncuts_value:.4f}')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(ncuts_map, cmap='RdYlBu')
        axes[2, 1].set_title('N-Cuts Cluster Map')
        axes[2, 1].axis('off')
        
        plt.suptitle(f'Binary Segmentation Analysis - {img_name}', fontsize=16)
        plt.tight_layout()
        
        # Save results using utility function
        save_comprehensive_results(fig, "demo3b", img_name)
        plt.show()
        
        # Summary table
        print(f"\nSummary Table for {img_name}:")
        print("Method    | N-cut Value | Cluster Sizes")
        print("-" * 50)
        print(f"N-Cuts    | {ncuts_value:11.6f} | {ncuts_stats['cluster_sizes']}")
        print(f"Spectral  | {spectral_value:11.6f} | {spectral_stats['cluster_sizes']}")
    
    print("\n=== Demo 3b completed successfully! ===")

if __name__ == "__main__":
    main()
