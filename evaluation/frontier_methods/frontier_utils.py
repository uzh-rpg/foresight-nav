import numpy as np
import cv2

def detect_frontiers(occupancy_map):
    """
    Detects frontiers in the occupancy map in a vectorized manner.
    
    Parameters:
    occupancy_map (np.ndarray): 2D array where each cell is either 0 (free), 0.5 (unobserved), or 1 (occupied)
    
    Returns:
        f_idx: List of frontier cell coordinates (x, y)
        frontier_img: Binary image with frontiers marked as 1
        labels: Connected components of frontiers
    """
    occupancy_map = cv2.dilate(occupancy_map, np.ones((5, 5), np.uint8), iterations=1)


    frontiers = np.zeros_like(occupancy_map)
    frontiers[occupancy_map == 0.5] = 1
    frontiers = cv2.dilate(frontiers, np.ones((3, 3), np.uint8), iterations=1)
    frontiers[occupancy_map == 1] = 0
    frontiers[occupancy_map == 0.5] = 0

    f_idx = frontiers==1

    # TODO: fix handling of no frontiers detected
    if len(f_idx) == 0:
        pass

    # get list of frontier zones (get connected components)
    frontier_img = np.zeros_like(occupancy_map)
    frontier_img[f_idx] = 1
    _, labels = cv2.connectedComponents(frontier_img.astype(np.uint8))

    return f_idx, frontier_img, labels


def visualize_frontiers(occupancy_map, frontiers):
    """
    Visualizes frontiers on the occupancy map.
    
    Parameters:
    occupancy_map (np.ndarray): 2D array where each cell is either 0 (free), 0.5 (unobserved), or 1 (occupied)
    frontiers (np.ndarray): Binary image with frontiers marked as 1
    
    Returns:
    np.ndarray: Image with frontiers overlaid
    """
    img = cv2.cvtColor((occupancy_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img[frontiers == 1] = [0, 0, 255]
    return img