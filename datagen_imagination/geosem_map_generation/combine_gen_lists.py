"""
Combine the generated shards into a single list of valid scenes of Structured3D, and split into train, and val sets.
Usage:
    python -m datagen_imagination.geosem_map_generation.combine_gen_lists \
        --shard_dir='/home/hshah/scratch/ForeSightDataset/Structured3D' \
        --num_shards=4 \
        --output_dir='/home/hshah/scratch/ForeSightDataset/Structured3D'
"""

import os
import argparse

def config():
    parser = argparse.ArgumentParser(description='Combine the generated shards into a single list of valid scenes of Structured3D, and split into train, and val sets.')
    parser.add_argument('--shard_dir', type=str, required=True, help='Path to the directory containing the shards')
    parser.add_argument('--num_shards', type=int, required=True, help='Number of shards')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')

    args = parser.parse_args()
    return args

def main():
    args = config()
    shard_dir = args.shard_dir
    num_shards = args.num_shards
    output_dir = args.output_dir

    if num_shards > 0:
        file_paths = [os.path.join(shard_dir, f'scene_list_geosem_map_shard{i}.txt') for i in range(num_shards)]
        scene_list = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                cur_list = f.readlines()
                cur_list = [scene.strip() for scene in cur_list]
                scene_list += cur_list
    else:
        file_path = os.path.join(shard_dir, f'scene_list_geosem_map.txt')
        with open(file_path, 'r') as f:
            saved_scenes = f.readlines()
        scene_list = [scene.strip() for scene in saved_scenes]

    scene_list = list(set(scene_list))
    scene_list.sort()

    # if num_train or num_val is not specified, use all scenes
    # split at the given scene numbers
    # Reference given in Structured3D README: https://github.com/bertjiazheng/Structured3D?tab=readme-ov-file#data
    num_train = 2999
    # ideally should be 3299, 3299 - 3499 is test set
    num_val = 3499  

    train_idx = scene_list.index(f"scene_{num_train:05d}")
    val_idx = scene_list.index(f"scene_{num_val:05d}")
    print(f"Train scene index: {train_idx}")
    print(f"Val scene index: {val_idx}")

    train_scenes = scene_list[:train_idx+1]
    val_scenes = scene_list[train_idx+1:val_idx+1]

    scene_list_path = os.path.join(output_dir, 'scene_list_geosem_map.txt')
    train_list_path = os.path.join(output_dir, 'scene_list_train_geosem_map.txt')
    val_list_path = os.path.join(output_dir, 'scene_list_val_geosem_map.txt')

    with open(scene_list_path, 'w') as f:
        for scene in scene_list:
            f.write(scene + '\n')

    with open(train_list_path, 'w') as f:
        for scene in train_scenes:
            f.write(scene + '\n')
    
    with open(val_list_path, 'w') as f:
        for scene in val_scenes:
            f.write(scene + '\n')
    
    print(f"Train scenes: {len(train_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")

if __name__ == '__main__':
    main()