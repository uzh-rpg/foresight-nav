"""
Generate GeoSem Maps for the  Structured3D dataset
from Panoramic RGB, Depth and Pose data. (Vectorized Implementation)

Example Usage:
python -m datagen_imagination.geosem_map_generation.gen_geosem_map \
    --config='configs/geosem_map_gen.yaml'
"""

import os
import cv2
import argparse
import json
import yaml
import time
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_erosion


from datagen_imagination.geosem_map_generation.utils.viz_utils import (
    get_new_pallete, get_new_mask_pallete)
from datagen_imagination.geosem_map_generation.utils.clip_utils import get_text_feats
from datagen_imagination.geosem_map_generation.utils.mp3dcat import mp3dcat
from datagen_imagination.geosem_map_generation.utils.panaroma_to_perspective import e2p
from datagen_imagination.geosem_map_generation.utils.lseg_inference import LSegEncoder


def precompute_offsets():
    pre_h = 512
    pre_w = 1024

    alpha = np.zeros((pre_h, pre_w))
    beta = np.zeros((pre_h, pre_w))

    x_tick = 180.0/pre_h
    y_tick = 360.0/pre_w

    for x in range(0, pre_h):
        for y in range(0, pre_w):
            # need 90 - -09
            alpha_c = 90 - (x * x_tick)
            beta_c = y * y_tick -180

            alpha[x,y] = np.deg2rad(alpha_c)
            beta[x,y] = np.deg2rad(beta_c)

    alpha = alpha
    beta = beta 

    return {
        'pre_h': pre_h,
        'pre_w': pre_w,
        'alpha': alpha,
        'beta': beta
    }

class GeoSemMap_Generator():

    def __init__(
            self,
            scene_path,
            clip_encoder: LSegEncoder,
            precomputed_offsets,
            save_root=None,
            grid_size=1000,
            room_type='full',
            fov=90,
            overlap=0.5,
            mode='bilinear',
            out_hw=(480, 640)
        ):
        # general parameters
        self.scene_path = scene_path
        self.clip_encoder = clip_encoder
        self.grid_size = grid_size
        self.room_type = room_type
        
        # perspective image parameters
        self.fov = fov
        self.overlap = overlap
        self.mode = mode
        self.out_hw = tuple(out_hw)

        # paths to RGB, Depth and Camera data
        sections = [p for p in os.listdir(os.path.join(scene_path, "2D_rendering"))]
        self.annotation_path = os.path.join(scene_path, "annotation_3d.json")
        self.depth_paths = [os.path.join(*[scene_path, "2D_rendering", p, "panorama", self.room_type, "depth.png"]) for p in sections]
        self.rgb_paths = [os.path.join(*[scene_path, "2D_rendering", p, "panorama", self.room_type, "rgb_coldlight.png"]) for p in sections]
        self.camera_paths = [os.path.join(*[scene_path, "2D_rendering", p, "panorama", "camera_xyz.txt"]) for p in sections]
        self.camera_centers = self.read_camera_center()

        # Precomputed offsets and min-max point cloud coordinates from annotations
        self.get_min_max_pc()
        self.pre_h = precomputed_offsets['pre_h']
        self.pre_w = precomputed_offsets['pre_w']
        self.alpha = precomputed_offsets['alpha']
        self.beta = precomputed_offsets['beta']

        # CLIP-Floorplan parameters
        clip_feat_dim = self.clip_encoder.clip_feat_dim
        self.clipfeat_map = np.zeros((self.grid_size, self.grid_size, clip_feat_dim), dtype=np.float32)
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.weight = np.zeros((self.grid_size, self.grid_size), dtype=float)
        self.color_top_down = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.vis_obs = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # MP3D Categories Text Features for visualization
        self.mp3d_text_feats = get_text_feats(
            mp3dcat,
            self.clip_encoder.clip_model,
            self.clip_encoder.clip_feat_dim,
        )

        # save path
        if save_root is not None:
            self.save_path = os.path.join(save_root, scene_path.split("/")[-1])
        else:
            self.save_path = os.path.join(scene_path, "GeoSemMap")
        self.vis_path = os.path.join(self.save_path, "vis")

        if os.path.exists(self.save_path):
            os.system(f'rm -r {self.save_path}')
        os.makedirs(self.save_path)
        os.makedirs(self.vis_path)

    def read_camera_center(self):
        camera_centers = []
        for i in range(len(self.camera_paths)):
            with open(self.camera_paths[i], 'r') as f:
                line = f.readline()
            center = list(map(float, line.strip().split(" ")))
            camera_centers.append(np.asarray([center[0], center[1], center[2]]))
        return camera_centers
    
    def get_min_max_pc(self):
        """
        Get the minimum and maximum coordinates of the point cloud from annotations
        """
        with open(self.annotation_path, 'r') as f:
            annos = json.load(f)
        
        min_coords = np.array([np.inf, np.inf, np.inf])
        max_coords = np.array([-np.inf, -np.inf, -np.inf])

        for line in annos['lines']:
            point = np.array(line['point'])
            min_coords = np.minimum(min_coords, point)
            max_coords = np.maximum(max_coords, point)
        
        for junction in annos['junctions']:
            point = np.array(junction['coordinate'])
            min_coords = np.minimum(min_coords, point)
            max_coords = np.maximum(max_coords, point)
        
        self.min_pc = min_coords
        self.max_pc = max_coords
    
    def get_perspective_images(self, pano_img_path, base_idx_img, fov=90, overlap=0.5, mode='bilinear', out_hw=(480, 640)):
        imgs = []
        idx_imgs = []

        num_x = int(360/(fov*(1-overlap)))
        num_y = int(180/(fov*(1-overlap)))

        viewdir_x = np.linspace(-180, 180, num_x, endpoint=False)
        viewdir_y = np.linspace(-105 + fov//2, 105 - fov//2, num_y+1, endpoint=True)

        img = cv2.imread(pano_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for x in viewdir_x:
            for y in viewdir_y:
                persp_img = e2p(
                    e_img=img,
                    fov_deg=fov,
                    u_deg=x,
                    v_deg=y,
                    out_hw=out_hw,
                    mode=mode
                )

                idx_img = e2p(
                    e_img=base_idx_img,
                    fov_deg=fov,
                    u_deg=x,
                    v_deg=y,
                    out_hw=out_hw,
                    mode='nearest'
                )

                imgs.append(persp_img)
                idx_imgs.append(idx_img)

        return imgs, idx_imgs
    
    def viz_perspective_images(self):
        persp_path = os.path.join(self.save_path, 'perspective_images')
        os.makedirs(persp_path, exist_ok=True)
        for i in range(len(self.rgb_paths)):
            persp_imgs, _ = self.get_perspective_images(self.rgb_paths[i])
            for j in range(len(persp_imgs)):
                cv2.imwrite(os.path.join(persp_path, f'img_{i}_{j}.png'), persp_imgs[j])
    
    def get_local_pc(self, depth_img, cam_center, base_idx):

        coords = []
        pt_idx = []
        local_pc = np.zeros((0, 4))

        if depth_img.shape[0] != self.pre_h or depth_img.shape[1] != self.pre_w:
            raise ValueError("Depth image shape does not match precomputed offsets")
        
        # use precomputed offsets
        else:
            z_offset = depth_img * np.sin(self.alpha)
            xy_offset = depth_img * np.cos(self.alpha)
            x_offset = xy_offset * np.sin(self.beta)
            y_offset = xy_offset * np.cos(self.beta)
            pt_idx = np.arange(depth_img.shape[0]*depth_img.shape[1]).reshape(depth_img.shape[0], depth_img.shape[1])
            pt_idx += base_idx
            point = np.stack([x_offset, y_offset, z_offset], axis=-1)

            mask = depth_img > 500.
            point = point[mask]
            pt_idx = pt_idx[mask]

            coords = point + cam_center
        
        coords[:,:2] = np.round(coords[:,:2] / 10) * 10.
        coords[:,2] = np.round(coords[:,2] / 100) * 100.
        unique_coords, unique_ind = np.unique(coords, return_index=True, axis=0)

        coords = coords[unique_ind]
        pt_idx = pt_idx[unique_ind]        
        coords_idx = np.concatenate([coords, pt_idx.reshape(-1, 1)], axis=1)

        # local_pc is now a 2D array with columns [x, y, z, idx]
        local_pc = np.concatenate([local_pc, coords_idx], axis=0)

        return local_pc
    
    def get_full_pc(self):
        pc = np.zeros((0, 4))
        idx_imgs = []
        base_idx = 0

        for i in range(len(self.depth_paths)):
            depth = cv2.imread(self.depth_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            cam_center = self.camera_centers[i]

            local_pc = self.get_local_pc(depth, cam_center, base_idx)
            pc = np.concatenate([pc, local_pc], axis=0)
            
            idx_img = np.arange(depth.shape[0]*depth.shape[1]).reshape(depth.shape[0], depth.shape[1]) + base_idx
            idx_imgs.append(idx_img)
            base_idx += depth.shape[0]*depth.shape[1]
        
        # remove points outside min and max pc bounds (x and y)
        pc = pc[pc[:, 0] >= self.min_pc[0]]
        pc = pc[pc[:, 0] <= self.max_pc[0]]
        pc = pc[pc[:, 1] >= self.min_pc[1]]
        pc = pc[pc[:, 1] <= self.max_pc[1]]
        
        return pc, idx_imgs
    
    def get_map_resolution(self, pc):
        max_coords = np.max(pc, axis=0)
        min_coords = np.min(pc, axis=0)

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

        # get map resolution in meters from grid size
        map_res = (0.001*(max_coords[:2] - min_coords[:2]))/self.grid_size

        assert map_res[0] == map_res[1]

        return map_res[0], min_coords, max_coords

    def occ_from_pc(self, pc, up_limit, down_limit, min_coords, max_coords, map_res):
        # we require points on the walls, so cut off floor and ceiling
        pc = pc[pc[:, 2] < up_limit]
        pc = pc[pc[:, 2] > down_limit]

        image_res = np.array([self.grid_size, self.grid_size])
        coordinates = \
            np.round(
                (pc[:, :2] - min_coords[None, :2]) / (map_res * 1000))
        coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                    image_res - 1)

        occ = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # count the number of points in each pixel (density map)
        unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
        unique_coordinates = unique_coordinates.astype(np.int32)
        occ[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts

        density_map = occ.copy()

        # convert to binary occupancy map - threshold at 5% of the maximum value
        occ = occ > 0.05 * np.max(occ)
        occ = occ.astype(np.float32)

        # Remove small artifacts and close small holes in the occupancy map
        occ = self.remove_small_artifacts(occ, threshold=4)
        occ = self.close_small_holes(occ, kernel_size=4)

        # add a buffer of 1 pixel around the walls (agent radius)
        occ = 1-binary_erosion(1-occ, iterations=1, structure=np.ones((2, 2)), border_value=1)
    
        return occ, density_map

    
    def get_geosem_map(self, seg_vis=False):

        seg_imgs = []

        # pc is a 2D array with columns [x, y, z, idx]
        pc, base_idx_imgs = self.get_full_pc()
        points = deepcopy(pc)[:,:3]

        # get map resolution from grid size
        map_res, min_coords, max_coords = self.get_map_resolution(pc)

        # get floor and ceiling bounds for GeoSem Map
        up_limit = 0.7 * np.max(pc[:,2])
        down_limit = 0.4 * np.max(pc[:,2])

        # get obstacles from pc
        self.obstacles, self.density_map = self.occ_from_pc(
            points, up_limit, down_limit, min_coords, max_coords, map_res)

        # precompute grid_x and grid_y for each point
        grid_x = np.round((pc[:,0] - min_coords[0]) / (map_res*1000)).astype(int)
        grid_y = np.round((pc[:,1] - min_coords[1]) / (map_res*1000)).astype(int)

        if grid_x.max() >= self.grid_size or grid_y.max() >= self.grid_size:
            raise ValueError("Grid location out of bounds")
        if grid_x.min() < 0 or grid_y.min() < 0:
            raise ValueError("Grid location out of bounds")
        
        # concatenate grid_x and grid_y to pc
        # pc is now a 2D array with columns [x, y, z, idx, grid_x, grid_y]
        pc = np.concatenate([pc, grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], axis=1)

        # sort pc by idx
        sorted_indices = np.argsort(pc[:, -3])
        pc = pc[sorted_indices]

        # get down limit for obstacles (approx values of z-coordinate for floor points with 0.2m buffer)
        down_limit = np.min(pc[:,2]) + (0.2*1000)
        
        # initialize color_top_down_ht
        self.color_top_down_ht = np.ones((self.grid_size, self.grid_size)) * np.min(pc[:,2]) - 1000

        for i in range(len(self.rgb_paths)):
            # print(f"Generating Perspective Images for room {i}")
            persp_imgs, idx_imgs = self.get_perspective_images(
                self.rgb_paths[i],
                base_idx_imgs[i],
                fov=self.fov,
                overlap=self.overlap,
                mode=self.mode,
                out_hw=self.out_hw
            )

            # for j in tqdm(range(len(persp_imgs))):
            for j in range(len(persp_imgs)):
                cur_rgb = persp_imgs[j]

                # get pixel-wise clip features
                pix_feats, seg_img = self.clip_encoder.get_lseg_feat(cur_rgb, vis=seg_vis)
                if seg_vis:
                    seg_imgs.append(seg_img)

                # vectorized implementation
                feats = pix_feats[0].transpose(1, 2, 0)
                feats = feats.reshape(-1, feats.shape[-1])
                idx = idx_imgs[j].reshape(-1)
                cur_rgb = cur_rgb.reshape(-1, 3)

                # get all points corresponding to the current perspective image
                indices = np.searchsorted(pc[:, -3], idx)
                indices[indices == len(pc)] -= 1
                valid_indices = pc[indices, -3] == idx
                indices = indices[valid_indices]
                idx = idx[valid_indices]
                cur_rgb = cur_rgb[valid_indices]
                feats = feats[valid_indices]

                pc_cur = pc[indices]

                # associate each point with rgb
                # pc_cur is now a 2D array with columns [x, y, z, idx, grid_x, grid_y, r, g, b]
                pc_cur = np.concatenate([pc_cur, cur_rgb], axis=1)

                # get mask for ceiling points
                ceiling_mask = pc_cur[:, 2] >= up_limit
                pc_cur = pc_cur[~ceiling_mask]
                feats = feats[~ceiling_mask]

                # update color_top_down_ht and color_top_down
                cur_gridx = pc_cur[:, -5].astype(int)
                cur_gridy = pc_cur[:, -4].astype(int)
                cur_rgb = pc_cur[:, -3:]

                cur_top_down_ht = self.color_top_down_ht[cur_gridy, cur_gridx]
                update_mask = pc_cur[:, 2] > cur_top_down_ht
                self.color_top_down[cur_gridy[update_mask], cur_gridx[update_mask]] = cur_rgb[update_mask]

                # update clipfeat_map
                self.clipfeat_map[cur_gridy, cur_gridx] = ((self.clipfeat_map[cur_gridy, cur_gridx] * self.weight[cur_gridy, cur_gridx].reshape(-1, 1)) + feats) / (self.weight[cur_gridy, cur_gridx].reshape(-1, 1) + 1)
                self.weight[cur_gridy, cur_gridx] += 1

                # """
                # update obstacles
                obstacle_mask = pc_cur[:, 2] > down_limit
                self.vis_obs[cur_gridy[obstacle_mask], cur_gridx[obstacle_mask]] += 1
                # """

        if seg_vis:
            return seg_imgs
    
    def remove_small_artifacts(self, binary_map, threshold):
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
    
    def close_small_holes(self, binary_map, kernel_size):
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
        
    def vis_obstacle_map(self):
        obs_map = (self.density_map/np.max(self.density_map) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.vis_path, 'density_map.png'), obs_map)

        # threshold map to save a binary image
        occ = (self.obstacles * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.vis_path, 'obstacle_map.png'), occ)

    def vis_top_down_map(self):
        cv2.imwrite(os.path.join(self.vis_path, 'top_down_map.png'), self.color_top_down[:,:,::-1])

    def vis_clipfeat_map(self, lang = None):
        obstacles = np.logical_not(self.vis_obs).astype(np.uint8)
        no_map_mask = obstacles > 0

        if lang is None:
            lang = mp3dcat 
            text_feats = self.mp3d_text_feats
        else:
            lang = lang.split(",")
            text_feats = get_text_feats(
                lang,
                self.clip_encoder.clip_model,
                self.clip_encoder.clip_feat_dim,
            )

        grid = self.clipfeat_map
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
        plt.title("GeoSem Map")
        plt.imshow(seg)
        plt.savefig(os.path.join(self.vis_path, 'clipfeat_map.png'))
        plt.close()

    def save_geosem_map(self):
        # """
        np.save(os.path.join(self.save_path, 'clipfeat_map.npy'), self.clipfeat_map)
        np.save(os.path.join(self.save_path, 'obstacles.npy'), self.obstacles)
        np.save(os.path.join(self.save_path, 'color_top_down.npy'), self.color_top_down)
        np.save(os.path.join(self.save_path, 'weight.npy'), self.weight)
        np.save(os.path.join(self.save_path, 'vis_obs.npy'), self.vis_obs)
        # """

        """
        print(f"Size of clipfeat_map in bytes: {self.clipfeat_map.nbytes}")
        print(f"Size of obstacles in bytes: {self.obstacles.nbytes}")
        print(f"Size of color_top_down in bytes: {self.color_top_down.nbytes}")
        print(f"Size of weight in bytes: {self.weight.nbytes}")
        """

        self.vis_obstacle_map()
        self.vis_top_down_map()
        self.vis_clipfeat_map()

def config():
    a = argparse.ArgumentParser(description='Generate GeoSem Maps from Structured3D')
    a.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    args = a.parse_args()
    
    # open config (yaml) file
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    return cfg

def main(cfg):
    print("Creating GeoSem Maps from panorama...")

    # get parameters from cfg
    lseg_params = cfg['LSEG']
    clipfp_params = cfg['GeoSemMap']
    data_root = cfg['data_root']

    # scene parameters
    num_scenes = cfg['num_scenes']
    start_scene = cfg['start_scene']
    shard = cfg['shard']
    
    scenes = os.listdir(os.path.join(data_root))
    # remove non-directory files
    scenes = [scene for scene in scenes if os.path.isdir(os.path.join(data_root, scene)) and 'scene' in scene]
    scenes.sort()

    if start_scene is not None:
        start_scene = f'scene_{start_scene:05d}'
        scenes = scenes[scenes.index(start_scene):]
    if num_scenes:
        scenes = scenes[:num_scenes]

    # remove existing save_root
    if clipfp_params['save_root'] is not None:
        save_root = clipfp_params['save_root']
        if os.path.exists(save_root):
            os.system(f'rm -r {save_root}')
        os.makedirs(save_root)

    # initialize LSeg model
    clip_encoder = LSegEncoder(
        lseg_params['ckpt_path'],
        lseg_params['clip_version'],
        lseg_params['crop_size'],
        lseg_params['base_size']
    )

    # precompute offsets
    precomputed_offsets = precompute_offsets()

    # valid scenes - save in text file
    if clipfp_params['save_root'] is not None:
        valid_scene_path = os.path.join(clipfp_params['save_root'], 'scene_list_geosem_map.txt')
    else: 
        valid_scene_path = os.path.join(data_root, 'scene_list_geosem_map.txt')

    if shard is not None:
        valid_scene_path = valid_scene_path.replace('.txt', f'_shard{shard}.txt')

    if os.path.exists(valid_scene_path):
        # get latest scene and continue from there
        with open(valid_scene_path, 'r') as f:
            saved_scenes = f.readlines()
        saved_scenes = [scene.strip() for scene in saved_scenes]
        last_scene = saved_scenes[-1]
        scenes = scenes[scenes.index(last_scene)+1:]
        print(f"Starting generation from {scenes[0]}")
    print(f"Total scenes: {len(scenes)}")

    start_full = time.time()
    # for scene in tqdm(scenes):
    for scene in scenes:
        print(f"Processing {scene}...")
        scene_path = os.path.join(data_root, scene)
        try:
            gen = GeoSemMap_Generator(
                scene_path,
                clip_encoder,
                precomputed_offsets,
                clipfp_params['save_root'],
                clipfp_params['grid_size'],
                clipfp_params['room_type'],
                clipfp_params['fov'],
                clipfp_params['overlap'],
                clipfp_params['mode'],
                clipfp_params['out_hw']
            )

            # gen.viz_perspective_images()
            start = time.time()
            gen.get_geosem_map()
            gen.save_geosem_map()
            print(f"Time taken for scene {scene}: {time.time() - start}")
        
        except Exception as e:
            print(f"Error processing {scene}: {e}")
            continue
        
        with open(valid_scene_path, 'a') as f:
            f.write(scene + '\n')

    print(f"Time taken for all scenes: {time.time() - start_full}")

if __name__ == "__main__":
    cfg = config()
    main(cfg)