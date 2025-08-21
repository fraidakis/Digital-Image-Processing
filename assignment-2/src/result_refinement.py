import numpy as np
from scipy import ndimage


def remove_small_components(edge_map, min_size=50):
    """
    Clean edge map by removing small connected components.
    
    Parameters:
    -----------
    edge_map : ndarray
        Binary edge map.
    min_size : int
        Minimum size of connected components to keep.
        
    Returns:
    --------
    ndarray
        Cleaned edge map.
    """
    # Label connected components
    labeled, num_features = ndimage.label(edge_map)
    
    # Calculate size of each component
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background
    
    # Remove small components
    mask = component_sizes >= min_size
    cleaned_edges = mask[labeled]
    
    return cleaned_edges.astype(np.uint8)



def non_maximum_suppression(
    candidate_centers: np.ndarray,
    candidate_radii: np.ndarray,
    candidate_votes: np.ndarray,
    nms_radius_thresh_perc: float,
    nms_center_thresh: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Non-Maximum Suppression on candidate circles.

    Args:
        candidate_centers: Array of candidate circle centers.
        candidate_radii: Array of candidate circle radii.
        candidate_votes: Array of votes for candidate circles (sorted descending).
        nms_radius_thresh_perc: Maximum relative difference (percentage) between radii for NMS.
        nms_center_thresh: Maximum distance (pixels) between centers for NMS.

    Returns:
        A tuple containing:
            - final_centers: Centers of circles after NMS.
            - final_radii: Radii of circles after NMS.
            - final_votes: Votes of circles after NMS.
    """
    final_centers_list = []
    final_radii_list = []
    final_votes_list = []
    
    discard_indices = [False] * len(candidate_centers)

    for i in range(len(candidate_centers)):
        if discard_indices[i]:
            continue

        # Add current circle to final list (it's the best so far among non-discarded)
        final_centers_list.append(candidate_centers[i])
        final_radii_list.append(candidate_radii[i])
        final_votes_list.append(candidate_votes[i])

        # Compare with subsequent circles
        for j in range(i + 1, len(candidate_centers)):
            if discard_indices[j]:
                continue

            center1 = candidate_centers[i]
            radius1 = candidate_radii[i]
            center2 = candidate_centers[j]
            radius2 = candidate_radii[j]

            dist_centers = np.linalg.norm(center1 - center2)
            
            if radius1 == 0: # Should not happen given r_min=10
                radius_diff_perc = float('inf') if radius1 != radius2 else 0
            else:
                radius_diff_perc = abs(radius1 - radius2) / radius1
            
            centers_overlap = dist_centers < nms_center_thresh
            radii_overlap = radius_diff_perc < nms_radius_thresh_perc
            
            if centers_overlap and radii_overlap:
                discard_indices[j] = True # Suppress circle j

    final_centers = np.array(final_centers_list)
    final_radii = np.array(final_radii_list)
    final_votes_arr = np.array(final_votes_list)
    
    return final_centers, final_radii, final_votes_arr
