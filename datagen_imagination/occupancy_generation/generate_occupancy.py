import argparse
from tqdm import tqdm
import os
import pickle
import numpy as np
import cv2
from scipy.ndimage import binary_erosion
from skimage.morphology import opening

from datagen_imagination.occupancy_generation.common_utils import read_scene_pc
from datagen_imagination.occupancy_generation.stru3d_utils import (
    normalize_annotations,
    parse_floor_plan_polys,
    generate_coco_dict,
    invalid_scenes_ids
)

# set random seed for reproducibility
np.random.seed(0)


def config():
    a = argparse.ArgumentParser(description='Generate occupancy from point cloud for Structured3D')
    a.add_argument('--data_dir', type=str, help='path to data directory', required=True)
    a.add_argument('--num_scenes', default=None, type=int, help='Number of scenes to process')
    a.add_argument('--width', default=224, type=int, help='Width of the occupancy map')
    a.add_argument('--height', default=224, type=int, help='Height of the occupancy map')
    args = a.parse_args()
    return args

def remove_isolated_pixels(binary_map, kernel_size=4):
    """
    Removes isolated pixels using morphological opening.

    Args:
        binary_map: A numpy array representing the binary occupancy map.
        kernel_size: Size of the structuring element for opening operation.

    Returns:
        A new numpy array with isolated pixels removed.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_map = opening(binary_map, kernel)
    return opened_map

def close_small_holes(binary_map, kernel_size):
  """
  Closes small holes in the binary map using morphological closing.

  Args:
      binary_map: A numpy array representing the binary occupancy map.
      kernel_size: Size of the structuring element for closing operation.

  Returns:
      A new numpy array with small holes in walls closed.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  closed_map = cv2.morphologyEx(binary_map.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
  return closed_map.astype(np.uint8)

def remove_small_artifacts(binary_map, threshold):
    """
    Removes artifacts smaller than a threshold size from a binary map.

    Args:
        binary_map: A numpy array representing the binary occupancy map.
        threshold: Minimum size (area) of an object to be considered valid.

    Returns:
        A new numpy array with artifacts removed.
    """

    # Get statistics for each component
    output = cv2.connectedComponentsWithStats(binary_map.astype(np.uint8), connectivity=8)
    areas = output[2][:, cv2.CC_STAT_AREA] # Get the areas of all connected components
    labels = output[1] # Get the labels of all connected components

    # Create a mask to keep valid objects
    mask = np.ones(binary_map.shape, np.uint8)
    for i in range(1, len(areas)):
        if areas[i] < threshold:
            mask[labels == i] = 0

    # Apply mask to remove artifacts
    filtered_map = cv2.bitwise_and(binary_map.astype(np.uint8), mask)
    return filtered_map.astype(np.uint8)

def decrease_resolution(occupancy_map, scale_factor):
    """
    Downscale the occupancy map by a given scale factor.
    """
    # Original dimensions of the occupancy map
    original_height, original_width = occupancy_map.shape
    
    # Calculate new dimensions after scaling
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Initialize the downscaled map with zeros
    downsampled_map = np.zeros((new_height, new_width), dtype=np.uint8)
    
    # Calculate block sizes for downsampling
    block_height = original_height // new_height
    block_width = original_width // new_width
    
    # Iterate over each block in the downscaled map
    for i in range(new_height):
        for j in range(new_width):
            # Define the region in the original map corresponding to this block
            start_row = i * block_height
            end_row = (i + 1) * block_height
            start_col = j * block_width
            end_col = (j + 1) * block_width
            
            # Get the corresponding region from the original occupancy map
            block_region = occupancy_map[start_row:end_row, start_col:end_col]
            
            # Determine if any cell in the block region is occupied (value of 1)
            if np.any(block_region == 1):
                downsampled_map[i, j] = 1  # Set the value to 1 (occupied)
            else:
                downsampled_map[i, j] = 0  # Set the value to 0 (unoccupied)
    
    # Create a new 256x256 array and paste the downscaled map in the center
    padded_map = np.zeros((256, 256), dtype=np.uint8)
    pad_top = (256 - new_height) // 2
    pad_left = (256 - new_width) // 2
    padded_map[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = downsampled_map
    
    return padded_map

def decrease_resolution_cv2(occupancy_map, scale_factor):
    """
    Downscale the occupancy map by a given scale factor using OpenCV for faster processing.
    """
    # Calculate new dimensions after scaling
    new_width = int(occupancy_map.shape[1] * scale_factor)
    new_height = int(occupancy_map.shape[0] * scale_factor)
    
    # Resize the occupancy map using OpenCV with INTER_AREA interpolation
    downsampled_map = cv2.resize(occupancy_map, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a new 256x256 array and paste the downscaled map in the center
    padded_map = np.zeros((256, 256), dtype=np.uint8)
    pad_top = (256 - new_height) // 2
    pad_left = (256 - new_width) // 2
    padded_map[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = downsampled_map
    
    return padded_map

def ply_to_occ(point_cloud, width=256, height=256):
    ps = point_cloud

    image_res = np.array((width, height))

    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)

    max_m_min = max_coords - min_coords

    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    # enforce square aspect ratio
    x_range = max_coords[0] - min_coords[0]
    y_range = max_coords[1] - min_coords[1]
    if x_range > y_range:
        min_coords[1] -= (x_range - y_range) / 2
        max_coords[1] += (x_range - y_range) / 2
    else:
        min_coords[0] -= (y_range - x_range) / 2
        max_coords[0] += (y_range - x_range) / 2

    # map resolution in meters per pixel
    map_res = (0.001*(max_coords[:2] - min_coords[:2]))/image_res
    assert map_res[0] == map_res[1]

    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res
    normalization_dict["map_res"] = map_res[0]

    # point cloud values are in milimeters
    # we require points on the walls, so cut off floor and ceiling
    up_limit = 0.7 * np.max(ps[:, 2])
    down_limit = 0.4 * np.max(ps[:, 2])
    ps = ps[ps[:, 2] < up_limit]
    ps = ps[ps[:, 2] > down_limit]

    # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
    coordinates = \
        np.round(
            (ps[:, :2] - min_coords[None, :2]) / (max_coords[None,:2] - min_coords[None, :2]) * image_res[None])
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                image_res - 1)

    occ = np.zeros((height, width), dtype=np.float32)

    # count the number of points in each pixel (density map)
    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    unique_coordinates = unique_coordinates.astype(np.int32)
    occ[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts

    # convert to binary occupancy map - threshold at 50% of the maximum value
    occ = occ > 0.05 * np.max(occ)
    occ = occ.astype(np.float32)

    # Remove small artifacts and close small holes in the occupancy map
    occ = remove_small_artifacts(occ, threshold=4)
    occ = close_small_holes(occ, kernel_size=4)

    # add a buffer of 1 pixel around the walls (agent radius)
    occ = 1-binary_erosion(1-occ, iterations=1, structure=np.ones((2, 2)), border_value=1)
 
    return occ, normalization_dict

def generate_gt_occ(polygons, normalization_dict):
    """
    Generate a ground truth occupancy map from the given polygons.

    Args:
        polygons: A list of polygons representing the walls.
        normalization_dict: A dictionary containing normalization parameters.

    Returns:
        A numpy array representing the ground truth occupancy map.
    """
    image_res = normalization_dict["image_res"]

    room_centroids = []
    room_rand_pts = []

    # create a blank occupancy map and door occupancy map (for doors only)
    occ = np.zeros(image_res)
    door_occ = np.zeros(image_res)

    for poly in polygons:

        polygons = poly['segmentation'][0]
        poly_line = np.array(polygons).reshape(-1, 2)
        
        if poly['category_id'] == 'door':
            cv2.fillPoly(door_occ, [poly_line.astype(np.int32)], 1)
            cv2.fillPoly(occ, [poly_line.astype(np.int32)], 1)
            continue
        elif poly['category_id'] == 'window':
            cv2.fillPoly(occ, [poly_line.astype(np.int32)], 1)
            continue
        
        # draw the polygon borders on the occupancy map and fill the inside
        cv2.fillPoly(occ, [poly_line.astype(np.int32)], 1)
        
        center = np.mean(poly_line, axis=0)
        # convert the center to occupancy map coordinates
        center = np.round(center)[::-1].astype(np.int32)
        room_centroids.append(center)

        # sample random point inside each room
        occ_cop = np.zeros_like(occ)
        cv2.fillPoly(occ_cop, [poly_line.astype(np.int32)], 1)
        # cv2.polylines(occ_cop, [poly_line.astype(np.int32)], isClosed=True, color=0, thickness=2)
        
        # get the indices of the non-zero elements
        indices = np.argwhere(occ_cop == 1)
        random_point = indices[np.random.choice(indices.shape[0])]
        room_rand_pts.append(random_point)
            
    # erode the doors by 2 pixels
    door_occ = binary_erosion(door_occ, iterations=1, structure=np.ones((3, 3)), border_value=0)

    return occ, room_centroids, room_rand_pts, door_occ

def process_scene(scene_path, width, height):
    
    ply_path = os.path.join(scene_path, 'point_cloud.ply')

    points = read_scene_pc(ply_path)
    xyz = points[:, :3]

    # project point cloud to occupancy map
    occ, normalization_dict = ply_to_occ(xyz, width, height)     

    # save occupancy map as png
    save_path = os.path.join(scene_path, 'occupancy_map.png')
    occ_uint8 = (occ * 255).astype(np.uint8)
    cv2.imwrite(save_path, occ_uint8)  
    
    # generate annotations
    normalized_annos = normalize_annotations(scene_path, normalization_dict)
    polys = parse_floor_plan_polys(normalized_annos)
    polygons_list = generate_coco_dict(normalized_annos, polys, curr_instance_id=0, curr_img_id=0, ignore_types=['outwall'])
    gt_occ, room_centroids, room_rand_pts, door_occ = generate_gt_occ(polygons_list, normalization_dict)

    normalization_dict['room_centroids'] = room_centroids
    normalization_dict['room_random_points'] = room_rand_pts

    # save normalization_dict as pkl
    mp_data_path = os.path.join(scene_path, 'map_data.pkl')
    with open(mp_data_path, 'wb') as f:
        pickle.dump(normalization_dict, f)

    #save ground truth occupancy map as png
    save_path = os.path.join(scene_path, 'gt_occupancy_map.png')
    gt_occ_uint8 = (gt_occ * 255).astype(np.uint8)
    cv2.imwrite(save_path, gt_occ_uint8)

    """
    occ = occ - door_occ
    save_path = os.path.join(scene_path, 'occupancy_map_wdoors.png')
    occ_uint8 = (occ * 255).astype(np.uint8)
    cv2.imwrite(save_path, occ_uint8)

    # save door occupancy map as png
    save_path = os.path.join(scene_path, 'door_occupancy_map.png')
    door_occ_uint8 = (door_occ * 255).astype(np.uint8)
    cv2.imwrite(save_path, door_occ_uint8)
    """

def main(args):
    data_dir = args.data_dir
    scene_list_path = os.path.join(data_dir, 'scene_list_occ.txt')
    with open(scene_list_path, 'r') as f:
        scenes = f.readlines()
        scenes = [scene.strip() for scene in scenes]

    if args.num_scenes:
        scenes = scenes[:args.num_scenes]

    for scene in tqdm(scenes):
        scene_path = os.path.join(data_dir, scene)
        scene_id = scene.split('_')[-1]
        if int(scene_id) in invalid_scenes_ids:
            print('skip {}'.format(scene))
            continue
        process_scene(scene_path, args.width, args.height)

    print("Done!")

if __name__ == "__main__":

    main(config())