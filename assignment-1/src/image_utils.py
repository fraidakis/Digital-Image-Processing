"""
Image Utilities Module
======================
This module provides functions for loading, processing, visualizing, and analyzing images,
with a focus on histograms and image enhancement operations.

The module includes functionality for:
- Image I/O operations (loading, saving)
- Visualization of images and histograms
- Metrics calculation for histogram operations
- Supporting histogram equalization and matching operations
"""


import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hist_utils import calculate_hist_of_img
from hist_modif import perform_hist_eq, perform_hist_matching
from metrics import evaluate_histogram_matching


# ======================= Global Configuration =======================
# Base directory for file operations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Available modes for histogram operations
MODES = ['greedy', 'non-greedy', 'post-disturbance']

# Color scheme for different modes in visualizations
MODE_COLORS = {
    'greedy': '#E41A1C',          # Red
    'non-greedy': '#498F28',      # Green
    'post-disturbance': '#109EEA' # Blue
}

# Set a professional style for all matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')



# ======================== Image I/O Functions =========================

def load_image(path):
    """
    Load an image and convert to grayscale with normalized values.
    
    Parameters:
        path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Normalized grayscale image with values in range [0,1]
        
    Notes:
        - Converts image to grayscale (L mode in PIL)
        - Normalizes pixel values to [0,1] range by dividing by 255
    """

    # Read the image into a PIL entity and keep only the Luminance component
    img = Image.open(path).convert('L')

    # Obtain the underlying numpy array and normalize to [0,1] range
    img_array = np.array(img).astype(float) / 255.0

    return img_array


def save_single_image(img, name):
    """
    Save a single image to the results directory.
    
    Parameters:
        img (numpy.ndarray): Image array to save
        name (str): Filename without extension
        
    Notes:
        - Creates a figure with the image
        - Image is saved to the 'results/images' directory
    """
    # Create a figure for the image
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    
    # Display the image without a colorbar for cleaner output
    display_image(img, '', ax, add_colorbar=False)
    plt.axis('off')
    
    # Save the figure
    save_figure(fig, name)


def save_figure(fig, name, is_histogram=False):
    """
    Save a figure with appropriate settings based on content type.
    
    Parameters:
        fig (matplotlib.figure.Figure): Figure to save
        name (str): Filename without extension
        is_histogram (bool): Flag to determine save location (histograms or images subfolder)
        
    Notes:
        - Creates appropriate subdirectories if they don't exist
        - Uses high DPI (300) for quality output
        - Automatically closes the figure after saving
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define subdirectory based on content type (images or histograms)
    subdir = 'histograms' if is_histogram else 'images'
    save_dir = os.path.join(results_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Full path for saving the figure
    save_path = os.path.join(save_dir, f"{name}.png")
    
    # Save with appropriate settings for quality output
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {save_path}")
    
    # Close the figure to free memory
    plt.close(fig)



# ========================= Visualization Functions ===========================

def create_styled_figure(title):
    """
    Create a figure with consistent styling for a professional appearance.
    
    Parameters:
        title (str): Main title for the figure
        
    Returns:
        matplotlib.figure.Figure: A styled figure object
        
    Notes:
        - Uses a light background for better readability
        - Applies consistent font styling across all figures
    """
    # Create a figure with appropriate size and background color
    fig = plt.figure(figsize=(15, 11), facecolor='#f9f9f9')
    
    # Add a styled title with specific font properties
    fig.suptitle(title, fontsize=18, y=0.98, fontweight='bold', fontfamily='Segoe UI', color='#2c3e50')
    
    # Set transparency level for the figure background
    fig.patch.set_alpha(0.9)
    
    return fig


def display_image(img, title, ax, add_colorbar=True):
    """
    Enhanced image display with frame, title styling, and optional colorbar.
    
    Parameters:
        img (numpy.ndarray): Image array to display
        title (str): Title for the image subplot
        ax (matplotlib.axes.Axes): Axes object to display the image on
        add_colorbar (bool): Whether to add a colorbar to the image
        
    Returns:
        matplotlib.image.AxesImage: The displayed image object
        
    Notes:
        - Always uses grayscale colormap with fixed range [0,1]
        - Adds subtle border for better visual separation
        - Includes optional colorbar for intensity reference
    """

    # Display the image with fixed intensity range for consistency
    im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    
    # Style the title with consistent font properties
    ax.set_title(title, fontsize=11, pad=10, fontfamily='Segoe UI', color='#2c3e50')
    
    # Remove default axis elements (ticks, labels)
    ax.axis('off')
    
    # Add a subtle border around the image for better visual separation
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#bdc3c7')
        spine.set_linewidth(0.5)
    
    # Add a colorbar only if requested
    if add_colorbar:
        # Create a divider for the existing axes to make space for the colorbar
        divider = make_axes_locatable(ax)
        
        # Append a new axes for the colorbar with specified size and padding
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        # Create the colorbar in the new axes
        cbar = plt.colorbar(im, cax=cax)
        
        # Style the colorbar ticks
        cbar.ax.tick_params(labelsize=8, colors='#7f8c8d')
        
        # Add a label to the colorbar
        cbar.set_label('Intensity', fontsize=8, fontfamily='Segoe UI', color='#34495e', labelpad=5)
    
    return im

def plot_histogram(hist, title, ax, color='#3182bd', show_ideal=False, ref_hist=None):
    """
    Plot histogram with enhanced styling and optional reference overlays.
    
    Parameters:
        hist (dict): Histogram data as {intensity: frequency}
        title (str): Title for the histogram subplot
        ax (matplotlib.axes.Axes): Axes object to plot on
        color (str): Color for the histogram bars
        show_ideal (bool): Whether to show the ideal uniform distribution
        ref_hist (dict): Optional reference histogram to overlay
        
    Returns:
        matplotlib.container.BarContainer: The bar plot object
        
    Notes:
        - X-axis shows normalized intensity values [0,1]
        - Includes optional reference lines for ideal uniform distribution
        - Can overlay a target histogram for comparison
        - Adds subtle styling for professional appearance
    """

    bars = ax.bar(hist.keys(), hist.values(), color=color, alpha=1.0, width=1/128)
    
    # Improved title and labels
    ax.set_title(title, fontsize=12, pad=12, fontweight='medium', 
                 fontfamily='Segoe UI', color='#2c3e50')
    ax.set_xlabel('Normalized Intensity', fontsize=10, fontfamily='Segoe UI', color='#34495e')
    ax.set_ylabel('Frequency', fontsize=10, fontfamily='Segoe UI', color='#34495e')
    
    # Subtle grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, color='#7f8c8d')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.tick_params(axis='both', which='major', labelsize=9, colors='#7f8c8d')

    # Ensure x-axis covers the full potential range slightly padded
    ax.set_xlim([0, 1])
    
    # Set x-axis ticks at 0.1 intervals
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    
    ax.autoscale(enable=True, axis='y', tight=False)
    ax.margins(y=0.05)
    
    # Add ideal equalization reference if requested
    if show_ideal:
        # Calculate the ideal uniform height
        total_frequency = sum(hist.values())
        num_bins = 256
        ideal_height = total_frequency / num_bins if num_bins > 0 else 0
        
        # Add a pale horizontal line showing the ideal uniform distribution
        ax.axhline(y=ideal_height, color='#FFFF00', linestyle='--', linewidth=2.0, alpha=0.8, label='Ideal uniform')
        
        # Create a subtle fill below the ideal line for emphasis
        x_range = np.array(list(hist.keys()))
        ax.fill_between(x_range, [ideal_height] * len(x_range), 0, color='#FFFF00', alpha=0.1)

    # Add reference histogram if provided
    if ref_hist is not None:
        # Scale the reference histogram to match the total frequency of the current histogram
        total_current = sum(hist.values())
        total_ref = sum(ref_hist.values())
        scale_factor = total_current / total_ref if total_ref > 0 else 1
        
        # Calculate normalized counts
        ref_x = list(ref_hist.keys())
        ref_y = [ref_hist[x] * scale_factor for x in ref_x]
        
        # Plot the reference histogram as a smooth line with gradient effect
        ax.plot(ref_x, ref_y, color='#7A26BA', linewidth=1.8, alpha=0.9, label='Target histogram', zorder=10, linestyle='-')

        # Add subtle area fill below the reference line for better visibility
        ax.fill_between(ref_x, ref_y, 0, color='#7A26BA', alpha=0.1)    
    
    # Add a proper legend with consistent styling for both reference elements
    if show_ideal or ref_hist is not None:
        legend = ax.legend(loc='upper right', frameon=True, fontsize=9, 
                     framealpha=0.7, edgecolor='#d5d8dc')
        legend.get_frame().set_facecolor('#f8f9fa')
        
        # Style legend text
        for text in legend.get_texts():
            text.set_fontfamily('Segoe UI')
            text.set_color('#2c3e50')        
    return bars


def save_single_histogram(hist, name, color='#3182bd'):
    """
    Save a single histogram plot to the results directory.
    
    Parameters:
        hist (dict): Histogram data as {intensity: frequency}
        name (str): Filename without extension
        color (str): Color for the histogram bars
        
    Notes:
        - Creates a figure with the histogram
        - Image is saved to the 'results/histograms' directory
    """
    # Create a figure for the histogram
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    
    # Plot the histogram with specified color
    plot_histogram(hist, '', ax, color=color)
    
    # Save as a histogram type figure
    save_figure(fig, name, is_histogram=True)



# ======================= Metrics Calculation ========================

def calculate_and_display_metrics(input_img, ref_img, results_type):
    """
    Calculate metrics for all histogram operation modes and display them.
    
    Parameters:
        input_img (numpy.ndarray): Input image array
        ref_img (numpy.ndarray): Reference image array
        results_type (str): Type of operation ('equalization' or 'matching')
        
    Returns:
        pandas.DataFrame: DataFrame containing metrics for all modes
        
    Notes:
        - For equalization, compares against an ideal uniform histogram
        - For matching, compares against the reference image's histogram
        - Metrics are calculated for all three modes (greedy, non-greedy, post-disturbance)
        - Results are saved to CSV and displayed in console
    """
    
    print(f"\n----- {results_type.capitalize()} Metrics -----")
    
    # Set up data storage for metrics from each mode
    metrics_data = {mode: {} for mode in MODES}
    
    # Handle histogram equalization metrics
    if results_type == 'equalization':
        # Create uniform reference histogram (ideal equalization target)
        Lg = 256  # Number of gray levels
        uniform_hist = {i/(Lg-1): 1/Lg for i in range(Lg)}
        
        # Calculate metrics for each mode
        for mode in MODES:
            # Perform histogram equalization using the current mode
            equalized = perform_hist_eq(input_img, mode)
            
            # Calculate metrics comparing to ideal uniform histogram
            metrics = evaluate_histogram_matching(equalized, uniform_hist)
            metrics_data[mode] = metrics
    
    # Handle histogram matching metrics        
    elif results_type == 'matching':
        # Calculate reference histogram from the reference image
        ref_hist = calculate_hist_of_img(ref_img, return_normalized=True)
        
        # Calculate metrics for each mode
        for mode in MODES:
            # Perform histogram matching using the current mode
            matched = perform_hist_matching(input_img, ref_img, mode)
            
            # Calculate metrics comparing to reference histogram
            metrics = evaluate_histogram_matching(matched, ref_hist)
            metrics_data[mode] = metrics
    
    # Convert metrics to DataFrame for easier display and analysis
    df = pd.DataFrame(metrics_data).T
    print(df)
    
    return df