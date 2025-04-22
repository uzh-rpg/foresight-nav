"""
Prepare the GeoSem Maps as a single npy file for each scene for faster loading during training.

The final processed training file is a numpy array of shape (H, W, D+4), a concatenation of:
- CLIP feature map (H, W, D)
- Visibility Map (H, W, 1)
- Simulated Occupancy Map (H, W, 1)
- Ground Truth Occupancy Map (H, W, 1)
- Interior Mask (H, W, 1)

The CLIP feature Map is normalized along the channel dimension to normalize the CLIP features.
The visibility map is a binary mask indicating the pixels having semantic information (Not Used).
The simulated occupancy map is a binary mask indicating the observed occupancy of the scene.
The ground truth occupancy map is a binary mask indicating the complete occupancy of the scene.
The interior mask is a binary mask indicating the indoor area in the scene.

Example Usage:
    python -m datagen_imagination.prepare_geosem_maps \
    --data_dir=/scratch/hshah/ForeSightDataset/Structured3D/ \
    --output_dir=/scratch/hshah/ForeSightDataset/training
"""

import shutil
import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(0)


class GeoSemMapData():
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir

        scene_list_path_train = os.path.join(data_dir, "scene_list_train_geosem_map.txt")
        scene_list_path_val = os.path.join(data_dir, "scene_list_val_geosem_map.txt")

        with open(scene_list_path_train, "r") as f:
            self.train_scenes = f.read().splitlines()
        with open(scene_list_path_val, "r") as f:
            self.val_scenes = f.read().splitlines()

        print(f"Found {len(self.train_scenes)} training scenes.")
        print(f"Found {len(self.val_scenes)} validation scenes.")

        # Setup output directories
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

        # move train and val lists to output directory
        shutil.copy(scene_list_path_train, os.path.join(output_dir, "scene_list_train_geosem_map.txt"))
        shutil.copy(scene_list_path_val, os.path.join(output_dir, "scene_list_val_geosem_map.txt"))
        
    def get_image_paths(self, scene_list):
        image_paths = []
        for scene in scene_list:
            sim_map_dir = os.path.join(self.data_dir, scene, "simulated_maps")
            if not os.path.exists(sim_map_dir):
                self.num_scenes -= 1
                continue

            geosem_map_dir = os.path.join(self.data_dir, scene, "GeoSemMap")
            if not os.path.exists(geosem_map_dir):
                self.num_scenes -= 1
                continue

            img_paths = glob.glob(os.path.join(sim_map_dir, "[0-9]*.png"))
            image_paths.extend(img_paths)
        
        return image_paths
    
    def load_npy(self, path):
        with open(path, 'rb') as f:
            mat = np.load(f)
        return mat

    def save_processed_map(self, scene, phase):

        sim_map_dir = os.path.join(self.data_dir, scene, "simulated_maps")
        geosem_map_dir = os.path.join(self.data_dir, scene, "GeoSemMap")

        if (not os.path.exists(geosem_map_dir)): 
            print(f"GeoSemMaps Directory not found for {scene}, skipping.")
            return
        
        if (not os.path.exists(sim_map_dir)):
            print(f"Simulated Maps Directory not found for {scene}, skipping.")
            return

        os.makedirs(os.path.join(self.output_dir, phase, scene), exist_ok=True)

        gt_occ_path = os.path.join(self.data_dir, scene, "occupancy_map.png")
        int_mask_path = os.path.join(self.data_dir, scene, "gt_occupancy_map.png")

        # GeoSem Map paths
        clipfeat_map_path = os.path.join(self.data_dir, scene, "GeoSemMap", "clipfeat_map.npy")
        vis_obs_path = os.path.join(self.data_dir, scene, "GeoSemMap", "vis_obs.npy")

        # Read the ground truth occupancy map
        gt_occ = cv2.imread(gt_occ_path, cv2.IMREAD_GRAYSCALE)
        gt_occ = gt_occ.astype(np.float32) / 255.0

        # Read the interior mask (binary mask, 1-indoor, 0-outdoor)
        int_mask = cv2.imread(int_mask_path, cv2.IMREAD_GRAYSCALE)
        int_mask = int_mask.astype(np.float32) / 255.0

        # Read the GeoSem Map
        clipfeat_map = self.load_npy(clipfeat_map_path)
        vis_obs = self.load_npy(vis_obs_path)
        vis_obs = (vis_obs > 0).astype(np.float32)

        # Normalize the clipfeat_map (unit norm along the channel dimension)
        # Normalize only the channels with non-zero values
        norm = np.linalg.norm(clipfeat_map, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        clipfeat_map = clipfeat_map / norm

        # Read the simulated occupancy maps
        sim_occ_paths = glob.glob(os.path.join(sim_map_dir, "[0-9]*.png"))
        print(f"Found {len(sim_occ_paths)} simulated maps in {scene}")

        for i in range(len(sim_occ_paths)):
            sim_occ_path = sim_occ_paths[i]
         
            sim_occ = cv2.imread(sim_occ_path, cv2.IMREAD_GRAYSCALE)
            uk_mask = sim_occ == 127    # unknown mask
            sim_occ = sim_occ.astype(np.float32) / 255.0
            sim_occ[uk_mask] = 0.5

            # Concatenate to form the final processed map
            processed_map = np.concatenate(
                [
                    clipfeat_map,
                    vis_obs[..., None],
                    sim_occ[..., None],
                    gt_occ[..., None],
                    int_mask[..., None]
                ],
                axis=-1
            )
            
            idx = sim_occ_path.split("/")[-1].split(".")[0]

            # Save the processed map
            output_path = os.path.join(self.output_dir, phase, scene, f"{idx}.npy")
            np.save(output_path, processed_map)
    
    def prepare_geosem_maps(self):
        for scene in tqdm(self.train_scenes):
            self.save_processed_map(scene, "train")
        for scene in tqdm(self.val_scenes):
            self.save_processed_map(scene, "val")

if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument("--data_dir", type=str)
    config.add_argument("--output_dir", type=str)
    args = config.parse_args()

    geosem_map_data = GeoSemMapData(args.data_dir, args.output_dir)
    geosem_map_data.prepare_geosem_maps()

    print("Done!")
