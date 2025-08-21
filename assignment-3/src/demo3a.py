"""
Demo 3a: Test non-recursive n-cuts on images d2a and d2b for k=2,3,4

This demonstration compares n-cuts with spectral clustering to understand the
differences between standard and normalized Laplacian approaches.
"""
import numpy as np
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from n_cuts import n_cuts
from spectral_clustering import spectral_clustering
from demo_utils import (load_demo_images, visualize_segmentation, 
                       analyze_cluster_statistics, save_comprehensive_results)

def main():
    print("=== Demo 3a: Non-Recursive N-Cuts vs Spectral Clustering ===")
    
    # Load images using utility function
    images = load_demo_images()
    if images is None:
        return
    
    k_values = [2, 3, 4]
    
    for img_name, image in images.items():
        print(f"\n=== Processing image {img_name} ===")
        print(f"Image shape: {image.shape}")
        
        # Convert image to graph
        print("Converting image to graph...")
        affinity_mat = image_to_graph(image)
        print(f"Generated affinity matrix with shape: {affinity_mat.shape}")
        
        # Create comprehensive comparison visualization (4 rows, 3 columns)
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        # Store results for detailed comparison
        comparison_results = {}
        
        # Show original image only in the first position
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Hide the other positions in the first row
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
        
        # Process each k value
        for i, k in enumerate(k_values):
            print(f"\n--- Comparing methods with k={k} ---")
                        
            # Spectral clustering
            print("  Applying Spectral Clustering...")
            spectral_labels = spectral_clustering(affinity_mat, k)
            spectral_stats = analyze_cluster_statistics(spectral_labels)
            comparison_results[f"Spectral k={k}"] = spectral_stats
            print(f"  Spectral cluster distribution: {spectral_stats['cluster_sizes']}")

            # N-Cuts clustering
            print("  Applying N-Cuts...")
            ncuts_labels = n_cuts(affinity_mat, k)
            ncuts_stats = analyze_cluster_statistics(ncuts_labels)
            comparison_results[f"N-Cuts k={k}"] = ncuts_stats
            print(f"  N-Cuts cluster distribution: {ncuts_stats['cluster_sizes']}")

            # Visualize Spectral results using utility function
            spectral_segmented, spectral_map = visualize_segmentation(image, spectral_labels, k, f'Spectral k={k}')
            axes[1, i].imshow(spectral_segmented)
            axes[1, i].set_title(f'Spectral k={k}')
            axes[1, i].axis('off')
            
            # Visualize N-Cuts results using utility function
            ncuts_segmented, ncuts_map = visualize_segmentation(image, ncuts_labels, k, f'N-Cuts k={k}')
            axes[2, i].imshow(ncuts_segmented)
            axes[2, i].set_title(f'N-Cuts k={k}')
            axes[2, i].axis('off')
            
            # Show cluster maps comparison
            discrete_cmap = plt.colormaps.get_cmap('tab20b').resampled(k)
            axes[3, i].imshow(ncuts_map, cmap=discrete_cmap)
            axes[3, i].set_title(f'N-Cuts Map k={k}')
            axes[3, i].axis('off')
                    
        plt.suptitle(f'N-Cuts vs Spectral Clustering - {img_name}', fontsize=16)
        plt.tight_layout()
        
        # Save results using utility function
        save_comprehensive_results(fig, "demo3a", img_name)
        plt.show()
                    
    print("\n=== Demo 3a completed successfully! ===")

if __name__ == "__main__":
    main()