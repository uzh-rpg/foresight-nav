import argparse
import os
from tqdm import tqdm

from datagen_imagination.occupancy_generation.PointCloudReaderPanorama import PointCloudReaderPanorama


def config():
    a = argparse.ArgumentParser(description='Generate point cloud for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--num_scenes', default=None, type=int, help='Number of scenes to process')
    args = a.parse_args()
    return args

def main(args):
    print("Creating point cloud from perspective views...")
    data_root = args.data_root

    valid_scene_path = os.path.join(data_root, 'scene_list_occ.txt')
    if os.path.exists(valid_scene_path):
        # get latest scene and continue from there
        with open(valid_scene_path, 'r') as f:
            scenes = f.readlines()
        scenes = [scene.strip() for scene in scenes]
        last_scene = scenes[-1]
    
    scenes = os.listdir(os.path.join(data_root))
    scenes.sort()
    if args.num_scenes:
        scenes = scenes[:args.num_scenes]
    if os.path.exists(valid_scene_path):
        scenes = scenes[scenes.index(last_scene)+1:]
    print(f"Total scenes: {len(scenes)}")
    
    for scene in tqdm(scenes):
        print(f"Processing {scene}...")
        scene_path = os.path.join(data_root, scene)

        try:
            reader = PointCloudReaderPanorama(
                scene_path,
                resolution='empty',
                random_level=0,
                generate_color=False,
                generate_normal=False)
        except:
            continue

        save_path = os.path.join(data_root, scene, 'point_cloud.ply')
        reader.export_ply(save_path)

        with open(valid_scene_path, 'a') as f:
            f.write(scene + '\n')
        # reader.visualize()

if __name__ == "__main__":
    main(config())
