import numpy as np
from result_refinement import non_maximum_suppression


def circ_hough(
    in_img_array: np.ndarray,
    R_max: float,
    dim: np.ndarray,
    V_min: int,
    R_min: float = 1,
    nms_center_thresh_perc: float = 1,
    nms_radius_thresh_perc: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detects circles in a binary edge map using the Circular Hough Transform.

    The Hough Transform for circles works by transforming edge points (x, y)
    into a parameter space (a, b, r) representing circle centers (a, b) and radii (r).
    Each edge point votes for all possible circles that could pass through it.
    The accumulator array H(a, b, r) stores these votes. Peaks in the accumulator
    correspond to detected circles.
    
    Parameters:
    -----------
    in_img_array : np.ndarray
        Input 2D binary edge map (0s and 1s).
    
    R_max : float
        Maximum radius of circles to detect. Defines the upper bound for the
        radius dimension of the accumulator.

    dim : np.ndarray
        Dimensions of the accumulator array [num_a_bins, num_b_bins, num_r_bins].
        - num_a_bins: Number of bins for x-coordinates of circle centers.
        - num_b_bins: Number of bins for y-coordinates of circle centers.
        - num_r_bins: Number of bins for radii.
    
    V_min : int
        Minimum number of votes in the accumulator for a (center, radius)
        combination to be considered a detected circle.

    R_min: float, optional (default=1)
        Minimum radius of circles to detect.

    nms_center_thresh_perc: float, optional (default=1)
        Distance threshold for non-maximum suppression, as a percentage of maximum radius.
        Controls how close detected circle centers can be.

    nms_radius_thresh_perc: float, optional (default=0.2)
        Radius difference threshold for non-maximum suppression, as a percentage.
        Controls how similar radii can be for nearby circles.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray]:
        - final_centers : np.ndarray
            Array of [x, y] coordinates for the centers of detected circles.
            Empty if no circles are found.
        - final_radii : np.ndarray
            Array of radii for the detected circles. Empty if no circles are found.
        - final_votes : np.ndarray
            Array of votes for the detected circles. Empty if no circles are found.
    """
    
    H, W = in_img_array.shape  # Image height and width

    # Define the centers of the bins for each dimension of the accumulator
    # These represent the discrete values of a, b, and r that will be voted for.
    a_bin_centers = np.linspace(0, W - 1, dim[0])  # Centers for x-coordinates of circle centers
    b_bin_centers = np.linspace(0, H - 1, dim[1])  # Centers for y-coordinates of circle centers
    r_bin_values = np.linspace(R_min, R_max, dim[2])  # Discrete radius values

    # Initialize the accumulator array with zeros
    accumulator = np.zeros(dim, dtype=int)

    # Find the coordinates of all edge pixels (pixels with value 1)
    y_edges, x_edges = np.nonzero(in_img_array)

    num_edge_pixels = len(y_edges)
    if num_edge_pixels == 0:
        print("No edge pixels found in the input image.")
        return np.array([]), np.array([]), np.array([])

    # Pre-compute step sizes for converting continuous coordinates to bin indices.
    # This avoids redundant calculations inside the loop.

    # Step size for a-dimension (x-coordinate of center)
    if dim[0] > 1:
        # Avoid division by zero if W=1 (single column image) or dim[0]-1 is zero
        a_step = (a_bin_centers[-1] - a_bin_centers[0]) / (dim[0] - 1) if (dim[0] - 1) > 0 else 1
        if a_step == 0:  # Handles case where W=1, making all a_bin_centers potentially the same
            a_step = 1 
    else:  # Only one bin for a-dimension
        a_step = 1

    # Step size for b-dimension (y-coordinate of center)
    if dim[1] > 1:
        # Avoid division by zero if H=1 (single row image) or dim[1]-1 is zero
        b_step = (b_bin_centers[-1] - b_bin_centers[0]) / (dim[1] - 1) if (dim[1] - 1) > 0 else 1
        if b_step == 0:  # Handles case where H=1
            b_step = 1
    else:  # Only one bin for b-dimension
        b_step = 1
    
    
    # Process each potential radius value
    for r_idx, r_val in enumerate(r_bin_values):
        # Dynamically adjust number of angles based on radius for efficiency
        # More points needed for larger circles to maintain accuracy
        num_angles = max(16, min(60, int(np.pi * r_val)))
        theta = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Calculate potential circle centers for all edge pixels and all angles
        # For each edge pixel, we compute potential circle centers in all directions
        a_coords_all = x_edges[:, np.newaxis] - r_val * cos_theta  # potential x-coordinates
        b_coords_all = y_edges[:, np.newaxis] - r_val * sin_theta  # potential y-coordinates
        
        # Convert real coordinates to accumulator bin indices
        if dim[0] > 1:
            a_indices = ((a_coords_all - a_bin_centers[0]) / a_step).astype(int)
        else:
            a_indices = np.zeros_like(a_coords_all, dtype=int)
            
        if dim[1] > 1:
            b_indices = ((b_coords_all - b_bin_centers[0]) / b_step).astype(int)
        else:
            b_indices = np.zeros_like(b_coords_all, dtype=int)
            
        # Filter out indices outside the accumulator bounds
        valid_mask = (a_indices >= 0) & (a_indices < dim[0]) & \
                     (b_indices >= 0) & (b_indices < dim[1])
        
        # Extract valid indices
        valid_a_idx = a_indices[valid_mask]
        valid_b_idx = b_indices[valid_mask]
        
        # Increment accumulator votes for the current radius
        if valid_a_idx.size > 0:
            np.add.at(accumulator[:, :, r_idx], (valid_a_idx, valid_b_idx), 1)
        
        # Display progress
        if (r_idx + 1) % (max(1, len(r_bin_values) // 10)) == 0 or r_idx == len(r_bin_values) - 1:
             print(f"Processed radius {r_idx+1}/{len(r_bin_values)} (r={r_val:.2f})")
    
    # Find circle candidates that meet minimum vote threshold
    circle_indices = np.argwhere(accumulator >= V_min)  # [a_idx, b_idx, r_idx]
    
    # Handle case with no circle candidates
    if len(circle_indices) == 0:
        print("No circles detected before NMS.")
        return np.array([]), np.array([]), np.array([])
    
    # Convert bin indices back to image coordinates and radii
    detected_centers_a = a_bin_centers[circle_indices[:, 0]]  # x-coordinates
    detected_centers_b = b_bin_centers[circle_indices[:, 1]]  # y-coordinates
    detected_radii = r_bin_values[circle_indices[:, 2]]  # radii
    
    # Prepare arrays for all candidates
    all_centers = np.column_stack([detected_centers_a, detected_centers_b])
    all_radii = detected_radii
    all_votes = accumulator[circle_indices[:, 0], circle_indices[:, 1], circle_indices[:, 2]]
    
    # Sort candidates by vote count (descending)
    sorted_vote_indices = np.argsort(all_votes)[::-1]
    candidate_centers = all_centers[sorted_vote_indices]
    candidate_radii = all_radii[sorted_vote_indices]
    candidate_votes = all_votes[sorted_vote_indices]

    print(f"\nNumber of candidate circles before NMS: {len(candidate_centers)}")

    # Apply Non-Maximum Suppression to remove duplicate/overlapping circles
    final_centers, final_radii, final_votes = non_maximum_suppression(
        candidate_centers,
        candidate_radii,
        candidate_votes,
        nms_radius_thresh_perc,
        nms_center_thresh = nms_center_thresh_perc * max(1.0, R_max),
    )

    # Display detected circles after NMS
    print("\nDetected circles after NMS:")
    if len(final_centers) > 0:
        for i in range(len(final_centers)):
            center = final_centers[i]
            radius = final_radii[i]
            votes = final_votes[i]
            print(f"Circle #{i+1}: Center: [{center[0]:.2f}, {center[1]:.2f}], "
                  f"Radius: {radius:.2f}, Votes: {votes}")
    else:
        print("No circles detected after NMS.")
    
    return final_centers, final_radii, final_votes