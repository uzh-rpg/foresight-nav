"""
Save text features for Matterport3D categories in a numpy file.
Example Usage:
    python -m datagen_imagination.geosem_map_generation.export_mp3d_text_feats \
    --save_dir=/scratch/hshah/ForeSightDataset/training_utils
"""
import os
import argparse
import numpy as np
import clip
import torch

from datagen_imagination.geosem_map_generation.utils.clip_utils import get_text_feats
from datagen_imagination.geosem_map_generation.utils.mp3dcat import mp3dcat


def config():
    parser = argparse.ArgumentParser(description='Export CLIP text features for Matterport3D Categories')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory for saving the features')
    
    args = parser.parse_args()
    return args

def main():
    args = config()

    # load CLIP model for text embeddings of language queries
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = 512
    clip_model, preprocess = clip.load(clip_version)
    clip_model.to(device).eval()

    # save text features for Matterport3D categories
    text_feats = get_text_feats(
        mp3dcat,
        clip_model,
        clip_feat_dim,
    )

    # create directory if does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save text features
    save_path = os.path.join(args.save_dir, "mp3d_text_feats.npy")
    np.save(save_path, text_feats)

    print(f"Text features saved to {save_path}")

if __name__ == "__main__":
    main()
