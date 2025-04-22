"""
Get the category weights for the Imagine loss for the matterport3d categories.
Plot the histogram of distribution of the categories over the training scenes.

Example Usage:
    python -m datagen_imagination.get_category_freq \
        --data_dir=/scratch/hshah/ForeSightDataset/Structured3D/ \
        --output_dir=/scratch/hshah/ForeSightDataset/training_utils \
        --text_feat_path=/scratch/hshah/ForeSightDataset/training_utils/mp3d_text_feats.npy
"""

import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datagen_imagination.geosem_map_generation.utils.mp3dcat import mp3dcat

# Set random seed for reproducibility
np.random.seed(0)


class GeoSemMapData():
    def __init__(self, data_dir, output_dir, text_feats):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.text_feats = text_feats

        # Load the scene list
        scene_list_path_train = os.path.join(data_dir, "scene_list_train_geosem_map.txt")
        with open(scene_list_path_train, "r") as f:
            self.train_scenes = f.read().splitlines()
        print(f"Found {len(self.train_scenes)} training scenes.")

        # Setup output directories
        os.makedirs(output_dir, exist_ok=True)

        # Setup the category distribution
        self.cat_freq = {cat: 0 for cat in mp3dcat}
        
    def load_npy(self, path):
        with open(path, 'rb') as f:
            mat = np.load(f)
        return mat

    def get_category_frequency(self, scene):

        # Read the CLIP feature map 
        clipfeat_map_path = os.path.join(self.data_dir, scene, "GeoSemMap", "clipfeat_map.npy")
        clipfeat_map = self.load_npy(clipfeat_map_path)

        # Normalize the clipfeat_map (unit norm along the channel dimension)
        # Normalize only the channels with non-zero values
        norm = np.linalg.norm(clipfeat_map, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        clipfeat_map = clipfeat_map / norm

        # Get pixel-wise category for the clipfeat_map
        grid_size = clipfeat_map.shape[0]
    
        map_feats = clipfeat_map.reshape((-1, clipfeat_map.shape[-1]))
        scores_list = map_feats @ self.text_feats.T

        predicts = np.argmax(scores_list, axis=1)
        predicts = predicts.reshape((grid_size, grid_size))

        # Get the category frequency
        for cat in mp3dcat:
            self.cat_freq[cat] += np.sum(predicts == mp3dcat.index(cat))
    
    def process_geosem_maps(self):
        for scene in tqdm(self.train_scenes):
            self.get_category_frequency(scene)

    def plot_category_distribution(self):
        plt.figure(figsize=(10, 6))

        # Set limits for the y-axis
        plt.ylim(0, np.mean(list(self.cat_freq.values())))

        plt.bar(self.cat_freq.keys(), self.cat_freq.values())
        plt.xticks(rotation=90)
        plt.ylabel("Frequency")
        plt.title("Category Distribution in GeoSemMaps")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "category_distribution.png"))

if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument("--data_dir", type=str)
    config.add_argument("--output_dir", type=str)
    config.add_argument("--text_feat_path", type=str)
    args = config.parse_args()

    # Load the CLIP features for the matterport3d categories (text)
    with open(args.text_feat_path, "rb") as f:
        text_feats = np.load(f)

    geosem_map_data = GeoSemMapData(args.data_dir, args.output_dir, text_feats)
    geosem_map_data.process_geosem_maps()
    geosem_map_data.plot_category_distribution()

    # Save the category frequency
    freq = []
    for cat in mp3dcat:
        freq.append(geosem_map_data.cat_freq[cat])
    freq = np.array(freq)
    with open(os.path.join(args.output_dir, "category_frequency.npy"), "wb") as f:
        np.save(f, freq)

    print("Done!")
