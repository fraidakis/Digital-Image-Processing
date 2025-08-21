"""
Demo 1: Test spectral clustering on provided affinity matrix d1a for k=2,3,4

This demonstration validates the spectral clustering implementation on a controlled
dataset where the graph structure is pre-defined. It helps understand how the
algorithm performs on different cluster numbers.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

from spectral_clustering import spectral_clustering
from demo_utils import get_base_dir

def main():
    print("=== Demo 1: Spectral Clustering on Pre-built Affinity Matrix ===")

    base_dir = get_base_dir()

    # Create results directory
    results_dir = os.path.join(base_dir, "results", "demo1")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")
    
    data_file_path = os.path.join(base_dir, "docs", "dip_hw_3.mat")
        
    # Load the provided data
    print(f"Loading data from {data_file_path}...")
    try:
        data = loadmat(data_file_path)
        d1a = data["d1a"]  # Pre-built affinity matrix
        print(f"Successfully loaded affinity matrix with shape: {d1a.shape}")
    except FileNotFoundError:
        print(f"ERROR: {data_file_path} file not found!")
        print("Please ensure the data file is in the docs/ directory.")
        return
    except KeyError:
        print("ERROR: 'd1a' not found in the .mat file!")
        return
        
    # Test spectral clustering for k=2,3,4
    k_values = [2, 3, 4]
    results = {}
    
    print(f"\nTesting spectral clustering on {d1a.shape[0]} vertices...")
    
    # Create visualization with only the cluster labels row
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    for i, k in enumerate(k_values):
        print(f"\n--- Testing with k={k} clusters ---")
        
        # Perform spectral clustering
        cluster_labels = spectral_clustering(d1a, k)
        results[k] = cluster_labels
        
        # Analyze results
        unique_labels = np.unique(cluster_labels)
        cluster_counts = np.bincount(cluster_labels)
        
        print(f"Number of unique clusters found: {len(unique_labels)}")
        print(f"Cluster distribution: {cluster_counts}")
        print(f"Cluster sizes: {dict(zip(unique_labels, cluster_counts))}")
        
        # Only cluster labels as horizontal strip 
        im_strip = axes[i].imshow(cluster_labels[np.newaxis, :], aspect='auto', 
                                 cmap=plt.get_cmap('viridis', k))
        axes[i].set_title(f"d1a Labels (k={k})")
        axes[i].set_yticks([])
        axes[i].set_xlabel('Node Index')
        plt.colorbar(im_strip, ax=axes[i], shrink=0.8, ticks=range(k))
    
    plt.tight_layout()
    
    # Save the figure with high quality
    output_file = os.path.join(results_dir, 'demo1_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved as '{output_file}'")
    
    # Display the plot
    plt.show()

    # Summary analysis
    print("\n=== SUMMARY ANALYSIS ===")
    balance_ratios = []
    for k in k_values:
        cluster_counts = np.bincount(results[k])
        balance_ratio = np.min(cluster_counts) / np.max(cluster_counts)
        balance_ratios.append(balance_ratio)
        print(f"k={k}: Balance ratio = {balance_ratio:.3f} (1.0 = perfectly balanced)")
        print(f"      Cluster sizes: {cluster_counts.tolist()}")
    
    print(f"\nDemo 1 completed successfully!")
    print(f"All results saved in directory: {os.path.abspath(results_dir)}")

if __name__ == "__main__":
    main()