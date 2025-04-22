import os
from PIL import Image
import glob
import cv2
import numpy as np
import argparse
from omegaconf import OmegaConf
from typing import Optional, List

import torch
from torch.utils.data import Dataset, DataLoader

from train_utils.log_geosem import visualize_topdown_semantic
from train_utils.imagine_loss import ClipLossType

# Set random seed for reproducibility
np.random.seed(0)


class GeoSemMapDataset():
    def __init__(self,
        data_dir: str,
        conf: OmegaConf,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.conf = conf

        self.num_train_scenes = conf.num_train_scenes
        self.num_val_scenes = conf.num_val_scenes

        scene_list_path_train = os.path.join(data_dir, "scene_list_train_geosem_map.txt")
        scene_list_path_val = os.path.join(data_dir, "scene_list_val_geosem_map.txt")

        with open(scene_list_path_train, "r") as f:
            self.train_scenes = f.read().splitlines()
        with open(scene_list_path_val, "r") as f:
            self.val_scenes = f.read().splitlines()

        if self.num_train_scenes is not None:
            self.train_scenes = np.random.choice(self.train_scenes, self.num_train_scenes, replace=False)
        if self.num_val_scenes is not None:
            self.val_scenes = np.random.choice(self.val_scenes, self.num_val_scenes, replace=False)

        self.num_scenes = len(self.train_scenes) + len(self.val_scenes)
        self.image_paths_train = self.get_image_paths(self.train_scenes, "train")
        self.image_paths_val = self.get_image_paths(self.val_scenes, "val")

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

        if self.conf.skip_transform:
            print("Skipping Rotation and Scaling Transformations.")

        print(f"Using {train_size} training images.")
        print(f"Using {val_size} validation images.")
    
    def get_image_paths(self, scene_list, phase):
        image_paths = []
        for scene in scene_list:
            load_dir = os.path.join(self.data_dir, phase, scene)
            if not os.path.exists(load_dir):
                self.num_scenes -= 1
                continue

            img_paths = glob.glob(os.path.join(load_dir, "[0-9]*.npy"))
            image_paths.extend(img_paths)
        
        return image_paths
    
    def get_dataset(self, split):
        return _Dataset(self.data_dir, self.images[split], self.conf)


class _Dataset(Dataset):
    def __init__(self, data_dir: str, image_paths: List[str], conf: OmegaConf):
        self.data_dir = data_dir
        self.image_paths = image_paths
        self.conf = conf
        self.use_category_wts = False

        # Transform category frequencies to weights
        if ClipLossType(conf.clip_loss_type) == ClipLossType.CATEGORY_COSINE_SIM:
            self.use_category_wts = True
            category_freq_path = conf.category_freq_path
            self.category_freq = self.load_npy(category_freq_path)
            self.category_freq[self.category_freq == 0] = 1
            self.category_wts = self.category_freq.sum() / (self.category_freq * len(self.category_freq))

        if self.conf.remove_floor or self.use_category_wts:
            text_feat_path = self.conf.text_feat_path
            self.text_feats = self.load_npy(text_feat_path)


    def __len__(self):
        return len(self.image_paths)
    
    def scale_transform(self, cur_mat: np.ndarray, scale: float, pad_value: float=0.0):
        target_h, target_w = cur_mat.shape[:2]
        target_shape = tuple(cur_mat.shape)

        # Calculate new dimensions after scaling
        new_width = int(cur_mat.shape[1] * scale)
        new_height = int(cur_mat.shape[0] * scale)
        
        # Resize the occupancy map using OpenCV with INTER_NEAREST interpolation to avoid interpolation artifacts and float values in binary maps
        downsampled_map = cv2.resize(cur_mat, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        if scale < 1:
            # Create a new array and paste the downscaled map in the center
            if pad_value == 0.0:
                padded_map = np.zeros(target_shape, dtype=np.float32)
            else:
                padded_map = np.ones(target_shape, dtype=np.float32) * pad_value
            pad_top = (target_h - new_height) // 2
            pad_left = (target_w - new_width) // 2
            padded_map[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = downsampled_map

        else:
            # Crop the map to the target dimensions
            crop_top = (new_height - target_h) // 2
            crop_left = (new_width - target_w) // 2
            padded_map = downsampled_map[crop_top:crop_top + target_h, crop_left:crop_left + target_w]
        
        return padded_map
    
    def rotate_transform(self, cur_mat: np.ndarray, angle: float, pad_value: float=0.0):
        # Rotate the representation using OpenCV
        height, width = cur_mat.shape[:2]
        center = (width // 2, height // 2)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_map = cv2.warpAffine(
            cur_mat, rot, (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=pad_value
        )

        return rotated_map
    
    def transform(self,
        sim_occ: np.ndarray,
        gt_occ: np.ndarray,
        clipfeat_map: np.ndarray,
        vis_obs: np.ndarray,
        int_mask: np.ndarray
    ):
        """
        Apply rotation and scaling transformations to the occupancy map and clipfeat_map randomly.
        Return the data_dict containing the transformed maps, clipfeat_map and vis_obs, and the sim_clipfeat_map and sim_vis_obs.
        """

        # Apply rotation and scaling with a probability of 50%
        if (not self.conf.skip_transform) and (np.random.rand() < 0.9):

            # Randomly select a scale factor and angle
            scale = np.random.uniform(0.5, 1.5)
            angle = np.random.uniform(-45, 45)

            sim_scaled = self.scale_transform(sim_occ, scale, pad_value=0.5)
            sim_occ = self.rotate_transform(sim_scaled, angle, pad_value=0.5)

            gt_scaled = self.scale_transform(gt_occ, scale)
            gt_occ = self.rotate_transform(gt_scaled, angle)

            int_mask_scaled = self.scale_transform(int_mask, scale)
            int_mask = self.rotate_transform(int_mask_scaled, angle)

            vis_obs_scaled = self.scale_transform(vis_obs, scale)
            vis_obs = self.rotate_transform(vis_obs_scaled, angle)

            clipfeat_map_scaled = self.scale_transform(clipfeat_map, scale)
            clipfeat_map = self.rotate_transform(clipfeat_map_scaled, angle)
        
        # remove floor from the clipfeat_map
        if self.conf.remove_floor:
            grid_size = clipfeat_map.shape[0]
    
            map_feats = clipfeat_map.reshape((-1, clipfeat_map.shape[-1]))
            scores_list = map_feats @ self.text_feats.T

            predicts = np.argmax(scores_list, axis=1)
            predicts = predicts.reshape((grid_size, grid_size))
            floor_mask = predicts == 2

            clipfeat_map[floor_mask] = 0.0

        # Prepare pixel-wise category weights
        if self.use_category_wts:
            grid_size = clipfeat_map.shape[0]
    
            map_feats = clipfeat_map.reshape((-1, clipfeat_map.shape[-1]))
            scores_list = map_feats @ self.text_feats.T

            predicts = np.argmax(scores_list, axis=1)
            predicts = predicts.reshape((grid_size, grid_size))

            category_wts = self.category_wts[predicts]
            category_wts = torch.from_numpy(category_wts).unsqueeze(0)

        sim_clipfeat_map = clipfeat_map.copy()
        sim_vis_obs = vis_obs.copy()
        sim_clipfeat_map[sim_occ == 0.5] = 0.0
        sim_vis_obs[sim_occ == 0.5] = 0.5

        # Convert to tensor
        sim_occ = torch.from_numpy(sim_occ)
        gt_occ = torch.from_numpy(gt_occ)
        int_mask = torch.from_numpy(int_mask)
        clipfeat_map = torch.from_numpy(clipfeat_map)
        vis_obs = torch.from_numpy(vis_obs)
        sim_clipfeat_map = torch.from_numpy(sim_clipfeat_map)
        sim_vis_obs = torch.from_numpy(sim_vis_obs)

        # Convert to 3 channel tensor 
        sim_occ = sim_occ.unsqueeze(0)
        gt_occ = gt_occ.unsqueeze(0)
        int_mask = int_mask.unsqueeze(0)
        vis_obs = vis_obs.unsqueeze(0)
        clipfeat_map = clipfeat_map.permute(2, 0, 1)    # C, H, W
        sim_clipfeat_map = sim_clipfeat_map.permute(2, 0, 1)    # C, H, W
        sim_vis_obs = sim_vis_obs.unsqueeze(0)

        data_dict = {
            "sim_occ": sim_occ,
            "gt_occ": gt_occ,
            "int_mask": int_mask,
            "gt_clipfeat_map": clipfeat_map,
            "gt_vis_obs": vis_obs,
            "sim_clipfeat_map": sim_clipfeat_map,
            "sim_vis_obs": sim_vis_obs
        }

        if self.use_category_wts:
            data_dict["category_wts"] = category_wts

        return data_dict

    def load_npy(self, path):
        with open(path, 'rb') as f:
            mat = np.load(f)
        return mat

    def __getitem__(self, idx):
        load_path = self.image_paths[idx]
        
        processed_map = self.load_npy(load_path)

        clipfeat_map = processed_map[:, :, :512]
        vis_obs = processed_map[:, :, 512]
        sim_occ = processed_map[:, :, 513]
        gt_occ = processed_map[:, :, 514]
        int_mask = processed_map[:, :, 515]

        # Apply random transformations to the clipfeat_map and occupancy maps
        data_dict = self.transform(sim_occ, gt_occ, clipfeat_map, vis_obs, int_mask)

        return data_dict

if __name__ == "__main__":
    data_dir = "/scratch/hshah/ForeSightDataset/imagination_training_data"
    conf_path = "configs/geosem_loader_conf.yaml"

    conf = OmegaConf.load(conf_path)
    dataset = GeoSemMapDataset(data_dir, conf, train_size=100, val_size=10)
    train_dataset = dataset.get_dataset("train")
    val_dataset = dataset.get_dataset("val")
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    # Create tmp directory and save the transformed images for debugging
    if os.path.exists("tmp"):
        os.system("rm -r tmp")
    os.makedirs("tmp")

    # Prepare text feats
    text_feat_path = "/scratch/hshah/ForeSightDataset/training_utils/mp3d_text_feats.npy"
    with open(text_feat_path, 'rb') as f:
        text_feats = np.load(f)

    for i, data in enumerate(train_loader):
        sim_occ = data["sim_occ"]
        gt_occ = data["gt_occ"]
        int_mask = data["int_mask"]
        clipfeat_map = data["gt_clipfeat_map"]
        vis_obs = data["gt_vis_obs"]
        sim_clipfeat_map = data["sim_clipfeat_map"]
        sim_vis_obs = data["sim_vis_obs"]

        os.makedirs("tmp/{}".format(i))

        # store the entire batch
        for j in range(sim_occ.size(0)):
            sim_occ_img = sim_occ[j].permute(1, 2, 0).numpy().squeeze(-1)
            gt_occ_img = gt_occ[j].permute(1, 2, 0).numpy().squeeze(-1)
            int_mask_img = int_mask[j].permute(1, 2, 0).numpy().squeeze(-1)
            clipfeat_map_img = clipfeat_map[j].permute(1, 2, 0).numpy()
            vis_obs_img = vis_obs[j].permute(1, 2, 0).numpy().squeeze(-1)
            sim_clipfeat_map_img = sim_clipfeat_map[j].permute(1, 2, 0).numpy()
            sim_vis_obs_img = sim_vis_obs[j].permute(1, 2, 0).numpy().squeeze(-1)

            print(f"Unique values in sim_occ: {np.unique(sim_occ_img)}")
            print(f"Unique values in gt_occ: {np.unique(gt_occ_img)}")
            print(f"Unique values in int_mask: {np.unique(int_mask_img)}")
            print(f"Unique values in vis_obs: {np.unique(vis_obs_img)}")
            print(f"Unique values in sim_vis_obs: {np.unique(sim_vis_obs_img)}")

            sim_occ_img = (sim_occ_img * 255).astype(np.uint8)
            gt_occ_img = (gt_occ_img * 255).astype(np.uint8)
            int_mask_img = (int_mask_img * 255).astype(np.uint8)
            vis_obs_img = (vis_obs_img * 255).astype(np.uint8)
            sim_vis_obs_img = (sim_vis_obs_img * 255).astype(np.uint8)

            sim_occ_img = Image.fromarray(sim_occ_img)
            gt_occ_img = Image.fromarray(gt_occ_img)
            int_mask_img = Image.fromarray(int_mask_img)
            vis_obs_img = Image.fromarray(vis_obs_img)
            sim_vis_obs_img = Image.fromarray(sim_vis_obs_img)

            sim_occ_img.save(f"tmp/{i}/sim_occ_{j}.png")
            gt_occ_img.save(f"tmp/{i}/gt_occ_{j}.png")
            int_mask_img.save(f"tmp/{i}/int_mask_{j}.png")
            vis_obs_img.save(f"tmp/{i}/vis_obs_{j}.png")
            sim_vis_obs_img.save(f"tmp/{i}/sim_vis_obs_{j}.png")

            # Visualize the clipfeat_map
            clipfeat_map_img = Image.fromarray(
                visualize_topdown_semantic(clipfeat_map_img, text_feats, vis_obs_img, pred_mask=None))
            clipfeat_map_img.save(f"tmp/{i}/clipfeat_map_{j}.png")

            # Visualize the sim_clipfeat_map
            sim_clipfeat_map_img = Image.fromarray(
                visualize_topdown_semantic(sim_clipfeat_map_img, text_feats, sim_vis_obs_img, pred_mask=sim_occ_img))
            sim_clipfeat_map_img.save(f"tmp/{i}/sim_clipfeat_map_{j}.png")

        if i == 0:
            break