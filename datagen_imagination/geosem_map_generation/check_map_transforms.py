"""
Check rotation and scaling transforms on GeoSem Maps.

Example Usage:
python -m datagen_imagination.geosem_map_generation.check_map_transforms \
    --root_path=/scratch/hshah/ForeSightDataset/Structured3D/scene_00000 \
    --text_feat_path=/scratch/hshah/ForeSightDataset/training_utils/mp3d_text_feats.npy \
    --rotation=30 \
    --scale=0.5
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import cv2

from datagen_imagination.geosem_map_generation.utils.mp3dcat import mp3dcat
from datagen_imagination.geosem_map_generation.utils.viz_utils import get_new_pallete, get_new_mask_pallete

class GeoSemMap_Check:
    def __init__(self, root_path, text_feat_path, vis_path=None):
        
        # load CLIP-Embedded Floorplan and other data
        self.root_path = root_path
        print(f'Loading CLIP-Embedded Floorplan data from {self.root_path}')
        self.clipfeat_map = self.load_npy(os.path.join(self.root_path, 'clipfeat_map.npy'))
        self.obstacles = self.load_npy(os.path.join(self.root_path, 'obstacles.npy'))
        self.c_top_down = self.load_npy(os.path.join(self.root_path, 'color_top_down.npy'))
        self.vis_obs = self.load_npy(os.path.join(self.root_path, 'vis_obs.npy'))
        self.weight = self.load_npy(os.path.join(self.root_path, 'weight.npy'))
        self.grid_size = self.clipfeat_map.shape[0]

        # load text features for Matterport3D categories
        self.text_feat_path = text_feat_path
        print(f'Loading text features from {self.text_feat_path}')
        self.text_feats = self.load_npy(self.text_feat_path)

        # visualization path
        if vis_path is None:
            self.vis_path = os.path.join(self.root_path, 'check_transforms')
        else:
            self.vis_path = vis_path

        # create visualization directory
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)

    def load_npy(self, path):
        with open(path, 'rb') as f:
            mat = np.load(f)
        return mat
    
    def rotate_transform(self, cur_mat, angle, pad_value=0.0):
        """
        Rotate the GeoSem Map data by a given angle using OpenCV.
        """
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
    
    def scale_transform(self, cur_mat, scale, pad_value=0.0):
        target_h, target_w, dim = cur_mat.shape

        # Calculate new dimensions after scaling
        new_width = int(cur_mat.shape[1] * scale)
        new_height = int(cur_mat.shape[0] * scale)
        
        # Resize the occupancy map using OpenCV with INTER_AREA interpolation
        downsampled_map = cv2.resize(cur_mat, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        if downsampled_map.ndim == 2:
            downsampled_map = np.expand_dims(downsampled_map, axis=-1)
        
        if scale < 1:
            # Create a new array and paste the downscaled map in the center
            if pad_value == 0.0:
                padded_map = np.zeros((target_h, target_w, dim), dtype=np.float32)
            else:
                padded_map = np.ones((target_h, target_w, dim), dtype=np.float32) * pad_value
            pad_top = (target_h - new_height) // 2
            pad_left = (target_w - new_width) // 2
            padded_map[pad_top:pad_top + new_height, pad_left:pad_left + new_width, :] = downsampled_map

        else:
            # Crop the map to the target dimensions
            crop_top = (new_height - target_h) // 2
            crop_left = (new_width - target_w) // 2
            padded_map = downsampled_map[crop_top:crop_top + target_h, crop_left:crop_left + target_w, :]
        
        return padded_map
    
    def visualize_topdown_semantic(self, clipfeat_map, vis_obs, transform_type):
        obstacles = np.logical_not(vis_obs).astype(np.uint8)
        no_map_mask = obstacles > 0

        lang = mp3dcat 
        
        text_feats = self.text_feats

        grid = clipfeat_map
        map_feats = grid.reshape((-1, grid.shape[-1]))
        scores_list = map_feats @ text_feats.T

        predicts = np.argmax(scores_list, axis=1)
        predicts = predicts.reshape((self.grid_size, self.grid_size))
        floor_mask = predicts == 2

        new_pallete = get_new_pallete(len(lang))
        mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang)
        seg = mask.convert("RGBA")
        seg = np.array(seg)
        seg[no_map_mask] = [225, 225, 225, 255]
        seg[floor_mask] = [225, 225, 225, 255]
        seg = Image.fromarray(seg)
        plt.figure(figsize=(10, 6), dpi=120)
        plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
        plt.axis('off')
        plt.title("Top Down Semantic Segmentation")
        plt.imshow(seg)
        plt.savefig(os.path.join(self.vis_path, f'TopDown_Semantic_{transform_type}.png'))
        plt.close()

def config():
    parser = argparse.ArgumentParser(description='Visualize transformed CLIP-Embedded Floorplan')
    parser.add_argument('--root_path', type=str, required=True, help='Path to the CLIP-Embedded Floorplan')
    parser.add_argument('--text_feat_path', type=str, required=True, help='Path to the text features')
    parser.add_argument('--vis_path', type=str, default=None, help='Path to save the visualization')
    parser.add_argument('--rotation', type=int, default=0, help='Rotation angle')
    parser.add_argument('--scale', type=float, default=1.0, help='Scaling factor')
    args = parser.parse_args()
    return args

def main():
    args = config()
    chk = GeoSemMap_Check(args.root_path, args.text_feat_path, args.vis_path)
    chk.visualize_topdown_semantic(chk.clipfeat_map, chk.vis_obs, 'original')

    # check normalization of clipfeat_map and range of values
    print(f'clipfeat_map min: {chk.clipfeat_map.min()}, max: {chk.clipfeat_map.max()}')
    print(f'Is clipfeat_map normalized? {np.allclose(np.linalg.norm(chk.clipfeat_map, axis=-1), 1)}')
    print(f'Normalized clipfeat_map min: {np.linalg.norm(chk.clipfeat_map, axis=-1).min()}, max: {np.linalg.norm(chk.clipfeat_map, axis=-1).max()}')

    # rotation
    rotated_map = chk.rotate_transform(chk.clipfeat_map, args.rotation)
    rotated_vis_obs = chk.rotate_transform(chk.vis_obs, args.rotation)
    chk.visualize_topdown_semantic(rotated_map, rotated_vis_obs, f'rotated_{args.rotation}')
    cv2.imwrite(os.path.join(chk.vis_path, f'occ_rotated_{args.rotation}.png'), rotated_vis_obs*255)

    # scaling
    scaled_map = chk.scale_transform(chk.clipfeat_map, args.scale)
    scaled_vis_obs = chk.scale_transform(np.expand_dims(chk.vis_obs, axis=-1), args.scale)[:, :, 0]
    chk.visualize_topdown_semantic(scaled_map, scaled_vis_obs, f'scaled_{args.scale}')
    cv2.imwrite(os.path.join(chk.vis_path, f'occ_scaled_{args.scale}.png'), scaled_vis_obs*255)

if __name__ == "__main__":
    main()
