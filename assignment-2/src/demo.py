import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough
import cv2, os, time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_output_directory(dir_name='results'):
    """Create output directory for saved images."""
    output_path = os.path.join(BASE_DIR, dir_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def load_image(image_path='basketball_large.png'):
    """Load and preprocess image."""
    full_path = os.path.join(BASE_DIR, image_path)
    image = Image.open(full_path)
    gray = image.convert('L')  # Convert to grayscale
    gray = np.array(gray) / 255.0
    return gray

def set_plotting_style():
    """Set consistent and professional plotting style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Customize font sizes
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })


def process_sobel_edge(gray, output_dir, thresholds=None):
    """Process Sobel edge detection with varying thresholds."""
    print("\nRunning Sobel Edge Detection...\n")

    # Create output directory for Sobel images
    sobel_dir = os.path.join(output_dir, 'sobel')
    create_output_directory(sobel_dir)    
    
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.4, 4)  # Default threshold range

    edge_counts = []
    
    # Create a single figure for all thresholds
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7), constrained_layout=True) 
    axes = axes.flatten()
    
    for i, threshold in enumerate(thresholds):
        print(f'Processing threshold: {threshold:.2f}')

        sobel_edges = sobel_edge(gray, threshold)
        
        edge_counts.append(np.sum(sobel_edges))
        print(f'Edge pixel count: {edge_counts[-1]}\n')
        
        # Add to subplot with improved formatting
        im = axes[i].imshow(sobel_edges, cmap='gnuplot', interpolation='bilinear')
        axes[i].set_title(f'Threshold: {threshold:.2f}', fontsize=12, fontweight='bold')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Save individual images with enhanced styling
        plt.figure(figsize=(8, 6))
        plt.imshow(sobel_edges, cmap='gnuplot', interpolation='bilinear')
        if abs(threshold - 0.35) < 1e-2: 
            plt.title(f'Threshold: {threshold:.2f} (optimal)', fontsize=14, y=1.02, fontweight='bold')
        else:
            plt.title(f'Threshold: {threshold:.2f}', fontsize=14, y=1.02, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sobel_dir, f'sobel_thresh_{threshold:.2f}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Add colorbar to the combined figure
    fig.colorbar(im, ax=axes, location='right', shrink=0.85, label='Edge Intensity')
    
    fig.suptitle('Sobel Edge Detection at Various Thresholds', fontsize=16, fontweight='bold', y=1.04)
    plt.savefig(os.path.join(sobel_dir, 'all_sobel.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Threshold vs edge count plot with improved styling
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, edge_counts, 'o-', linewidth=2, color='#1f77b4', markersize=8)
    plt.xlabel('Threshold Value', fontsize=12)
    plt.ylabel('Edge Pixel Count', fontsize=12)
    plt.title('Edge Pixel Count vs. Threshold', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(sobel_dir, 'threshold_vs_edge_count.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Sobel Edge Detection completed.")
    print("\n--------------------------------------\n")

def process_log_edge(gray, output_dir, kernel_sizes=None):
    """Process LoG edge detection with varying mask sizes."""
    print("Running LoG Edge Detection...\n")
    
    # Create output directory for LoG images
    log_dir = os.path.join(output_dir, 'log')
    create_output_directory(log_dir)

    if kernel_sizes is None:
        kernel_sizes = [5, 7, 9, 10, 11, 13]  # Default mask sizes

    # Create a single figure for all mask sizes
    fig, axes = plt.subplots(3, 2, figsize=(13, 14.5), constrained_layout=True) 
    # Add more spacing between subplots (values are in inches)
    fig.set_constrained_layout_pads(hspace=0.07) 

    axes = axes.flatten()
    
    for i, kernel_size in enumerate(kernel_sizes):

        print(f'Processing LoG with mask size: {kernel_size}')

        log_edges = log_edge(gray, kernel_size)

        # Count edge pixels
        num_edge_pixels = np.sum(log_edges > 0)
        print(f'Edge pixel count: {num_edge_pixels}\n')
        
        # Add to subplot
        axes[i].imshow(log_edges, cmap='gnuplot')

        if kernel_size % 2 == 0:
            axes[i].set_title(f'Kernel Size: {kernel_size-1}', fontsize=12, fontweight='bold')
        else:
            axes[i].set_title(f'Kernel Size: {kernel_size}', fontsize=12, fontweight='bold')
        axes[i].set_xticks([]) 
        axes[i].set_yticks([])
        
        # Save individual images
        plt.figure(figsize=(8, 6))
        plt.imshow(log_edges, cmap='gnuplot')
        if kernel_size % 2 == 0:
            plt.title(f'Kernel Size: {kernel_size-1}', fontsize=14, fontweight='bold', y=1.02)
        elif kernel_size == 13:
            plt.title(f'Kernel Size: {kernel_size} (optimal)', fontsize=14, fontweight='bold', y=1.02)
        else:  
            plt.title(f'Kernel Size: {kernel_size}', fontsize=14, fontweight='bold', y=1.02)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'log_kernel_{kernel_size}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    fig.suptitle('LoG Edge Detection at Various Kernel Sizes', fontsize=16, fontweight='bold', y=1.03)
    plt.savefig(os.path.join(log_dir, 'all_log.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("LoG Edge Detection completed.")
    print("\n---------------------------------------------\n")

def process_hough_circles(gray, output_dir, max_edge_pixels=20000):
    """Process Hough Circle Detection using both Sobel and LoG edge detection."""
    print("Running Hough Circle Detection...\n")

    # Create output directory for Hough images
    hough_dir = os.path.join(output_dir, 'hough')
    os.makedirs(hough_dir, exist_ok=True)
    
    # Process with both Sobel and LoG edge detection
    for edge_method in ["Sobel", "LoG"]:
        print(f"\nRunning Hough Circle Detection with {edge_method} edge detection...")
        
        # Compute edges at full resolution based on the method
        if edge_method == "Sobel":
            edges = sobel_edge(gray, threshold=0.35)
            pixels = 4  # Hough circle detection parameter for Sobel
        else:  # LoG
            edges = log_edge(gray, kernel_size=13)
            pixels = 2  # Hough circle detection parameter for LoG
        
        # Count edge pixels
        num_edge_pixels = np.sum(edges > 0)
        print(f"Number of {edge_method} edge pixels: {num_edge_pixels}")
        
        # Determine if we need to downscale
        if num_edge_pixels > max_edge_pixels:
            # Calculate scaling factor to get approximately max_edge_pixels
            scale_factor = np.sqrt(max_edge_pixels / num_edge_pixels)
            print(f"Too many edge pixels, downscaling by factor: {scale_factor:.4f}")
            
            # Convert edges to float32 to prevent data loss during resize
            edges_float = edges.astype(np.float32)
            small_edges = cv2.resize(edges_float, (0, 0), fx=scale_factor, fy=scale_factor)
            
            # Get dimensions of downscaled image
            downscaled_h, downscaled_w = small_edges.shape
            print(f"Downscaled edge image size: {downscaled_w}x{downscaled_h}\n")
        else:
            print(f"Using full resolution edge image")
            small_edges = edges
        
        # Determine maximum radius based on the edge image dimensions
        R_max_base = int(0.5 * min(small_edges.shape))  # Maximum radius
        
        # Number of bins for a, b, r (based on the edge image size now)
        dim = np.array([
            int(small_edges.shape[1] / pixels), 
            int(small_edges.shape[0] / pixels), 
            int(R_max_base / pixels)
        ])
        
        R_min = 110
        R_max = 250
        nms_center_thresh_perc = 0.9
        nms_radius_thresh_perc = 0.2

        if edge_method == "Sobel":
            V_min = 230
        else:  # LoG
            V_min = 68

        centers, radii, votes = circ_hough(small_edges, R_max, dim, V_min, R_min, nms_center_thresh_perc, nms_radius_thresh_perc)
        
        print(f'\nNumber of circles detected with {edge_method}: {len(centers)}')
        
        votes_np = np.array(votes)
        # Define min and max vote values for normalization (color and thickness)
        min_vote_val = votes_np.min() if len(votes) > 0 else 0
        max_vote_val = votes_np.max() if len(votes) > 0 else 0

        # Create colormap for votes
        cmap = plt.cm.rainbow
        norm = plt.Normalize(min_vote_val, max_vote_val)

        # Create a single image with all circles
        display_image = cv2.cvtColor(((0 + small_edges) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Draw all circles with colormap
        for j, ((a, b), r_val, vote) in enumerate(zip(centers, radii, votes)):
            # Get color from colormap based on vote value
            color_rgb = cmap(norm(vote))[:3]  # Get RGB, exclude alpha
            # Convert to BGR for OpenCV (0-255 scale)
            color_bgr = (int(color_rgb[2]*255), int(color_rgb[1]*255), int(color_rgb[0]*255))
            
            # Determine thickness based on vote (1, 2, or 3)
            thickness = 2 # Default for lowest votes
            if max_vote_val > min_vote_val:
                # Normalize vote to 0-1 range for thickness determination
                normalized_vote_for_thickness = (vote - min_vote_val) / (max_vote_val - min_vote_val)
                if normalized_vote_for_thickness >= 2/3:
                    thickness = 4
                elif normalized_vote_for_thickness >= 1/3:
                    thickness = 3
            elif votes_np.size > 0: # Handles cases where all votes are the same
                thickness = 2 # Use a medium thickness if all votes are identical and exist

            cv2.circle(display_image, (int(a), int(b)), int(r_val), color_bgr, thickness)
        
        # Create and save the figure
        plt.figure(figsize=(10, 9))
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.title(f'{edge_method}: V_min: {V_min}, R = [{R_min}, {R_max}], NMSc: {nms_center_thresh_perc}, NMSr: {nms_radius_thresh_perc}', fontsize=12, fontweight='bold', y=1.02)
        plt.axis('off')
        
        # Add colorbar for vote strength
        if len(centers) > 0:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', shrink=0.8, pad=0.04) # Added shrink
            cbar.set_label('Vote Strength')
        
        # Save the figure
        save_path = os.path.join(hough_dir, f'{edge_method}_Rmin{R_min}_Rmax{R_max}_Vmin{V_min}_NMSc{nms_center_thresh_perc}_NMSr{nms_radius_thresh_perc}.png')        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved image: {save_path}")
        plt.show()
        plt.close()
    
    print("\nHough Circle Detection completed for both edge detection methods.")


def main():
    """Main function to run the image processing pipeline."""
    # Apply consistent plotting style
    set_plotting_style()
    
    # Create output directory
    output_dir = create_output_directory()

    # Load image
    gray = load_image(image_path='docs/input-image.png')

    process_sobel_edge(gray, output_dir)
   
    process_log_edge(gray, output_dir)
        
    process_hough_circles(gray, output_dir)


if __name__ == "__main__":
    main()


