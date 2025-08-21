import os
import numpy as np
from image_utils import (
    load_image, 
    save_single_image, 
    save_figure,
    create_styled_figure,
    display_image,
    plot_histogram,
    save_single_histogram,
    calculate_and_display_metrics,
    BASE_DIR,
    MODES,
    MODE_COLORS
)
import matplotlib.pyplot as plt

from hist_utils import calculate_hist_of_img
from hist_modif import perform_hist_eq, perform_hist_matching


###########################  Processing Functions  ############################

def create_equalization_display(input_img):
    """
    Create and display a comparison of histogram equalization methods.
    
    This function applies histogram equalization to the input image and generates a figure with:
    - Original image and its histogram
    - Equalized images in each mode
    - Corresponding histograms with ideal uniform distribution reference
    
    Parameters:
        input_img (numpy.ndarray): The input image to be equalized
        
    Returns:
        matplotlib.figure.Figure: The complete comparison figure
    """

    # Create a styled figure for the visualization
    fig_eq = create_styled_figure('Histogram Equalization Comparison')

    # First row: Display original input image and its histogram
    ax1 = plt.subplot(4, 2, 1)
    display_image(input_img, 'Input Image', ax1)

    ax2 = plt.subplot(4, 2, 2)
    input_hist = calculate_hist_of_img(input_img, False)
    plot_histogram(input_hist, 'Histogram of Input Image', ax2, color='#636363')

    # Each mode in its own row 
    for i, mode in enumerate(MODES):
        equalized = perform_hist_eq(input_img, mode)
        # Save the equalized image 
        save_single_image(equalized, f'equalized_{mode}')

        # Display equalized image in the figure
        ax_img = plt.subplot(4, 2, 2*i+3)
        display_image(equalized, f'Equalized ({mode})', ax_img)

        # Calculate and display the histogram of equalized image
        # The ideal histogram would be uniform (shown as reference)
        ax_hist = plt.subplot(4, 2, 2*i+4)
        hist = calculate_hist_of_img(equalized, False)
        save_single_histogram(hist, f'equalized_histogram_{mode}', MODE_COLORS[mode])
        plot_histogram(hist, f'Histogram ({mode} Equalization)', ax_hist, color=MODE_COLORS[mode], show_ideal=True)

    # Adjust layout for better readability
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])
    plt.subplots_adjust(hspace=0.45, wspace=0.25)
    
    return fig_eq


def create_matching_display(input_img, ref_img):
    """
    Create and display a comparison of histogram matching methods.
    
    This function transforms the input image to match the histogram distribution of the reference image and generates a figure with:
    - Reference image and its histogram
    - Matched images in each color mode
    - Corresponding histograms with reference histogram overlay
    
    Parameters:
        input_img (numpy.ndarray): The input image to be transformed
        ref_img (numpy.ndarray): The reference image whose histogram will be matched
        
    Returns:
        matplotlib.figure.Figure: The complete comparison figure
    """

    # Create a styled figure for the visualization
    fig_match = create_styled_figure('Histogram Matching Comparison')

    # Calculate the histogram of the reference image for comparison and matching
    ref_hist = calculate_hist_of_img(ref_img, False)

    # First row: Display reference image and its histogram
    ax1 = plt.subplot(4, 2, 1)
    display_image(ref_img, 'Reference Image', ax1)

    ax2 = plt.subplot(4, 2, 2)
    plot_histogram(ref_hist, 'Histogram of Reference Image', ax2, color='#636363')

    # Each mode in its own row
    for i, mode in enumerate(MODES):

        # Apply histogram matching to make input histogram match reference
        matched = perform_hist_matching(input_img, ref_img, mode)
        # Save the matched image
        save_single_image(matched, f'matched_{mode}')

        # Display matched image in the figure
        ax_img = plt.subplot(4, 2, 2*i+3)
        display_image(matched, f'Matched ({mode})', ax_img)

        # Calculate and display the histogram of matched image
        # The reference histogram is shown as an overlay for comparison
        ax_hist = plt.subplot(4, 2, 2*i+4)
        hist = calculate_hist_of_img(matched, False)
        save_single_histogram(hist, f'matched_histogram_{mode}', MODE_COLORS[mode])
        plot_histogram(hist, f'Histogram ({mode} Matching)', ax_hist, color=MODE_COLORS[mode], ref_hist=ref_hist)
    
    # Adjust layout for better readability
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])
    plt.subplots_adjust(hspace=0.45, wspace=0.25)
    
    return fig_match



###############################  Main Function  ###############################

def main():
    """
    Main function to run the histogram processing demonstration.
    
    This function performs the following steps:
    1. Loads the input and reference images from the specified paths
    2. Creates and displays the histogram equalization comparison
    3. Creates and displays the histogram matching comparison
    4. Calculates and displays performance metrics for both techniques
    """

    # Load images from the specified paths
    input_img_path = os.path.join(BASE_DIR, 'docs', 'input-images', 'ref_img.jpg')
    ref_img_path = os.path.join(BASE_DIR, 'docs', 'input-images', 'input_img.jpg')
    input_img = load_image(input_img_path)
    ref_img = load_image(ref_img_path)
    
    # Generate and display the histogram equalization results
    fig_eq = create_equalization_display(input_img)
    plt.show(block=True)  # Show and wait for it to be closed
    save_figure(fig_eq, 'histogram_equalization_comparison')
    
    # Generate and display the histogram matching results
    fig_match = create_matching_display(input_img, ref_img)
    plt.show()  # Show the matching figure
    save_figure(fig_match, 'histogram_matching_comparison')
    
    # Evaluate the performance of both techniques using quantitative metrics
    calculate_and_display_metrics(input_img, ref_img, 'equalization')
    calculate_and_display_metrics(input_img, ref_img, 'matching')

# Entry point of the script - executes main() when run directly
if __name__ == "__main__":
    main()
