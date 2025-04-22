import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import cv2
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)


class OccupancyDataset():
    def __init__(self, data_dir, train_size, val_size):
        self.data_dir = data_dir

        scene_list_path_train = os.path.join(data_dir, "scene_list_train_occ.txt")
        scene_list_path_val = os.path.join(data_dir, "scene_list_val_occ.txt")

        with open(scene_list_path_train, "r") as f:
            self.train_scenes = f.read().splitlines()
        with open(scene_list_path_val, "r") as f:
            self.val_scenes = f.read().splitlines()

        self.num_scenes = len(self.train_scenes) + len(self.val_scenes)
        self.image_paths_train = self.get_image_paths(self.train_scenes)
        self.image_paths_val = self.get_image_paths(self.val_scenes)

        print(f"Found {self.num_scenes} total scenes.")
        print(f"Found {len(self.train_scenes)} training scenes.")
        print(f"Found {len(self.val_scenes)} validation scenes.")
        print(f"Found {len(self.image_paths_train)} training images.")
        print(f"Found {len(self.image_paths_val)} validation images.")

        if train_size is None:
            train_size = len(self.image_paths_train)
        if val_size is None:
            val_size = len(self.image_paths_val)
        
        if train_size > len(self.image_paths_train):
            raise ValueError(
                f"train_size {train_size} is greater than the number of training images {len(self.image_paths_train)}")
        if val_size > len(self.image_paths_val):
            raise ValueError(
                f"val_size {val_size} is greater than the number of validation images {len(self.image_paths_val)}")

        self.train_size = train_size
        self.val_size = val_size

        train_images = self.image_paths_train[:train_size]
        val_images = self.image_paths_val[:val_size]

        self.images = {
            "train": train_images,
            "val": val_images
        }

        print(f"Using {train_size} training images.")
        print(f"Using {val_size} validation images.")
    
    def get_image_paths(self, scene_list):
        image_paths = []
        for scene in scene_list:
            sim_map_dir = os.path.join(self.data_dir, scene, "simulated_maps")
            if not os.path.exists(sim_map_dir):
                self.num_scenes -= 1
                continue

            img_paths = glob.glob(os.path.join(sim_map_dir, "[0-9]*.png"))
            image_paths.extend(img_paths)
        
        return image_paths
    
    def get_dataset(self, split):
        return _Dataset(self.data_dir, self.images[split])


class _Dataset(Dataset):
    def __init__(self, data_dir, image_paths):
        self.data_dir = data_dir
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)
    
    def scale_transform(self, occ_map, scale, pad_value=0.0):
        target_h, target_w = occ_map.shape

        # Calculate new dimensions after scaling
        new_width = int(occ_map.shape[1] * scale)
        new_height = int(occ_map.shape[0] * scale)
        
        # Resize the occupancy map using OpenCV with INTER_AREA interpolation
        downsampled_map = cv2.resize(occ_map, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        if scale < 1:
            # Create a new array and paste the downscaled map in the center
            if pad_value == 0.0:
                padded_map = np.zeros((target_h, target_w), dtype=np.float32)
            else:
                padded_map = np.ones((target_h, target_w), dtype=np.float32) * pad_value
            pad_top = (target_h - new_height) // 2
            pad_left = (target_w - new_width) // 2
            padded_map[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = downsampled_map

        else:
            # Crop the map to the target dimensions
            crop_top = (new_height - target_h) // 2
            crop_left = (new_width - target_w) // 2
            padded_map = downsampled_map[crop_top:crop_top + target_h, crop_left:crop_left + target_w]
        
        return padded_map
    
    def rotate_transform(self, occ_map, angle, pad_value=0.0):
        # Rotate the occupancy map using OpenCV
        height, width = occ_map.shape
        center = (width // 2, height // 2)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_map = cv2.warpAffine(
            occ_map, rot, (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=pad_value
        )

        return rotated_map
    
    def transform(self, sim_occ, gt_occ):
        """
        Apply rotation and scaling transformations to the occupancy map randomly.
        """

        # Apply rotation and scaling with a probability of 50%
        if np.random.rand() < 0.9:

            # Randomly select a scale factor and angle
            scale = np.random.uniform(0.5, 1.5)
            angle = np.random.uniform(-45, 45)

            sim_scaled = self.scale_transform(sim_occ, scale, pad_value=0.5)
            sim_occ = self.rotate_transform(sim_scaled, angle, pad_value=0.5)

            gt_scaled = self.scale_transform(gt_occ, scale)
            gt_occ = self.rotate_transform(gt_scaled, angle)

        # Convert to tensor
        sim_occ = torch.from_numpy(sim_occ)
        gt_occ = torch.from_numpy(gt_occ)

        # Convert to 3 channel tensor (pad same value in all channels)
        sim_occ = sim_occ.unsqueeze(0).repeat(3, 1, 1)
        gt_occ = gt_occ.unsqueeze(0).repeat(3, 1, 1)

        return sim_occ, gt_occ

    def __getitem__(self, idx):
        sim_occ_path = self.image_paths[idx]
        
        scene_path = sim_occ_path.split("/")[-3]
        gt_occ_path = os.path.join(self.data_dir, scene_path, "occupancy_map.png")

        # Read the simulated occupancy map
        sim_occ = cv2.imread(sim_occ_path, cv2.IMREAD_GRAYSCALE)
        sim_occ = sim_occ.astype(np.float32) / 255.0

        # Read the ground truth occupancy map
        gt_occ = cv2.imread(gt_occ_path, cv2.IMREAD_GRAYSCALE)
        gt_occ = gt_occ.astype(np.float32) / 255.0
        
        # Apply random transformations to the sim and gt occupancy maps
        sim_occ, gt_occ = self.transform(sim_occ, gt_occ)

        return {
            "sim_occ": sim_occ,
            "gt_occ": gt_occ
        }

if __name__ == "__main__":
    data_dir = "/scratch/hshah/ForeSightDataset/Structured3D/"

    train_size = 100
    val_size = 20

    dataset = OccupancyDataset(data_dir, train_size, val_size)
    train_dataset = dataset.get_dataset("train")
    val_dataset = dataset.get_dataset("val")
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Create tmp directory and save the transformed images for debugging
    if os.path.exists("tmp"):
        os.system("rm -r tmp")
    os.makedirs("tmp")

    for i, data in enumerate(train_loader):
        sim_occ = data["sim_occ"]
        gt_occ = data["gt_occ"]

        # store the entire batch
        for j in range(sim_occ.size(0)):
            sim_occ_img = sim_occ[j].permute(1, 2, 0).numpy()
            gt_occ_img = gt_occ[j].permute(1, 2, 0).numpy()

            sim_occ_img = (sim_occ_img * 255).astype(np.uint8)
            gt_occ_img = (gt_occ_img * 255).astype(np.uint8)

            sim_occ_img = Image.fromarray(sim_occ_img)
            gt_occ_img = Image.fromarray(gt_occ_img)

            sim_occ_img.save(f"tmp/sim_occ_{i}_{j}.png")
            gt_occ_img.save(f"tmp/gt_occ_{i}_{j}.png")

        if i == 0:
            break