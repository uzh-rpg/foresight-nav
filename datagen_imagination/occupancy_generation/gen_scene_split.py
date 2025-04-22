import argparse
import os
import random

# set random seed for reproducibility
random.seed(0)

def config():
    a = argparse.ArgumentParser(description='Generate point cloud for Structured3D')
    a.add_argument('--data_root', type=str, required=True,
                     help='path to raw Structured3D folder')
    a.add_argument('--num_train', type=int, default=None,
                     help='Number of scenes to process for training')
    a.add_argument('--num_val', type=int, default=None,
                     help='Number of scenes to process for validation')
    args = a.parse_args()
    return args

def main(args):
    data_root = args.data_root
    scene_list = os.path.join(data_root, 'scene_list_occ.txt')

    train_list_path = os.path.join(data_root, 'scene_list_train_occ.txt')
    val_list_path = os.path.join(data_root, 'scene_list_val_occ.txt')

    with open(scene_list, 'r') as f:
        scenes = f.readlines()
    scenes = [scene.strip() for scene in scenes]

    num_train = args.num_train
    num_val = args.num_val

    """
    # if num_train or num_val is not specified, use all scenes
    # split at 80% for training and 20% for validation
    if num_train is None or num_val is None:
        num_train = int(len(scenes) * 0.8)
        num_val = len(scenes) - num_train
    
    # randomly shuffle scenes
    random.shuffle(scenes)
    """

    # if num_train or num_val is not specified, use all scenes
    # split at the given scene numbers
    # Reference given in Structured3D README: https://github.com/bertjiazheng/Structured3D?tab=readme-ov-file#data
    if num_train is None:
        num_train = 2999
    if num_val is None:
        # ideally should be 3299, 3299 - 3499 is test set
        num_val = 3499  

    train_idx = scenes.index(f"scene_{num_train:05d}")
    val_idx = scenes.index(f"scene_{num_val:05d}")
    print(f"Train scene index: {train_idx}")
    print(f"Val scene index: {val_idx}")

    train_scenes = scenes[:train_idx+1]
    val_scenes = scenes[train_idx+1:val_idx+1]

    with open(train_list_path, 'w') as f:
        for scene in train_scenes:
            f.write(scene + '\n')
    
    with open(val_list_path, 'w') as f:
        for scene in val_scenes:
            f.write(scene + '\n')
    
    print(f"Train scenes: {len(train_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")
    
if __name__ == "__main__":
    main(config())