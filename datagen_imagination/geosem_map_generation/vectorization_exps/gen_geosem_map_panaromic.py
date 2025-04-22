"""
Code to generate GeoSemMaps from Panoramic RGB, Depth and Pose data. (Non - Vecotrized)
For loop based implementation - iterates over each pixel in all
the perspective images and associates it with a 3D point in the point cloud.
"""

import os
import cv2
import math
import numpy as np
from tqdm import tqdm
import argparse
# from py360convert import e2p
from datagen_imagination.geosem_map_generation.utils.panaroma_to_perspective import e2p

import torch
import torchvision.transforms as transforms
import clip
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

from PIL import Image
from datagen_imagination.geosem_map_generation.utils.viz_utils import get_new_pallete, get_new_mask_pallete
from utils.clip_utils import get_text_feats
from utils.mp3dcat import mp3dcat
import clip


from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image

class LSegEncoder():
    def __init__(self, clip_version="ViT-B/32"):
        self.crop_size = 480 # 480
        self.base_size = 640 # 520
        lang = "door,chair,ground,ceiling,other"
        self.labels = lang.split(",")
        self.clip_version = clip_version
        

        # loading models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        self.device = device

        # initialize CLIP model
        self.init_clip_model()
        
        model = LSegEncNet(
            self.labels,
            arch_option=0,
            block_depth=0,
            activation='lrelu',
            crop_size=self.crop_size
        )
        model_state_dict = model.state_dict()
        pretrained_state_dict = torch.load("/scratch/hashah/vlmaps/demo_e200.ckpt")
        pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

        model.eval()
        self.model = model.cuda()

        self.norm_mean= [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]
        self.padding = [0.0] * 3
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    
    def get_lseg_feat(
            self,
            image: np.array,
            vis=False
        ):
        
        vis_image = image.copy()
        image = self.transform(image).unsqueeze(0).cuda()
        img = image[0].permute(1,2,0)
        img = img * 0.5 + 0.5
        
        batch, _, h, w = image.size()
        stride_rate = 2.0/3.0
        stride = int(self.crop_size * stride_rate)

        long_size = self.base_size
        if h > w:
            height = long_size
            width = int(1.0 * w * long_size / h + 0.5)
            short_size = width
        else:
            width = long_size
            height = int(1.0 * h * long_size / w + 0.5)
            short_size = height


        cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

        if long_size <= self.crop_size:
            pad_img = pad_image(cur_img, self.norm_mean,
                                self.norm_std, self.crop_size)
            print(pad_img.shape)
            with torch.no_grad():
                outputs, logits = self.model(pad_img, self.labels)
            outputs = crop_image(outputs, 0, height, 0, width)
        else:
            if short_size < self.crop_size:
                # pad if needed
                pad_img = pad_image(cur_img, self.norm_mean,
                                    self.norm_std, self.crop_size)
            else:
                pad_img = cur_img
            _,_,ph,pw = pad_img.shape #.size()
            assert(ph >= height and pw >= width)
            h_grids = int(math.ceil(1.0 * (ph-self.crop_size)/stride)) + 1
            w_grids = int(math.ceil(1.0 * (pw-self.crop_size)/stride)) + 1
            with torch.cuda.device_of(image):
                with torch.no_grad():
                    outputs = image.new().resize_(batch, self.model.out_c,ph,pw).zero_().cuda()
                    logits_outputs = image.new().resize_(batch, len(self.labels),ph,pw).zero_().cuda()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
            # grid evaluation
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + self.crop_size, ph)
                    w1 = min(w0 + self.crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # pad if needed
                    pad_crop_img = pad_image(crop_img, self.norm_mean,
                                                self.norm_std, self.crop_size)
                    with torch.no_grad():
                        output, logits = self.model(pad_crop_img, self.labels)
                    cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                    cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                    outputs[:,:,h0:h1,w0:w1] += cropped
                    logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                    count_norm[:,:,h0:h1,w0:w1] += 1
            assert((count_norm==0).sum()==0)
            outputs = outputs / count_norm
            logits_outputs = logits_outputs / count_norm
            outputs = outputs[:,:,:height,:width]
            logits_outputs = logits_outputs[:,:,:height,:width]
        outputs = outputs.cpu()
        outputs = outputs.numpy() # B, D, H, W
        predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
        pred = predicts[0]

        vis_fig = None
        if vis:
            new_palette = get_new_pallete(len(self.labels))
            mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=self.labels)
            seg = mask.convert("RGBA")
            
            # show image and segmentation side by side
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(vis_image)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(seg)
            plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 8})
            plt.axis("off")
            plt.tight_layout()

            vis_fig = fig.canvas.draw()
            fig_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            fig_data = fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            vis_fig = fig_data
            plt.close(fig)

        return outputs, vis_fig
    
    def init_clip_model(self):
        clip_models = {
            'RN50': 1024,
            'RN101': 512,
            'RN50x4': 640,
            'RN50x16': 768,
            'RN50x64': 1024,
            'ViT-B/32': 512,
            'ViT-B/16': 512,
            'ViT-L/14': 768
        }
        self.clip_feat_dim = clip_models[self.clip_version]
        clip_model, preprocess = clip.load(self.clip_version)
        clip_model.to(self.device).eval()
        self.clip_model = clip_model

class GeoSemMapGenerator():

    def __init__(
            self,
            scene_path,
            clip_encoder: LSegEncoder,
            save_root,
            grid_size=1000,
            room_type='full',
        ):
        self.scene_path = scene_path
        self.clip_encoder = clip_encoder
        self.save_path = os.path.join(save_root, scene_path.split("/")[-1])
        self.grid_size = grid_size
        self.room_type = room_type

        sections = [p for p in os.listdir(os.path.join(scene_path, "2D_rendering"))]
        self.depth_paths = [os.path.join(*[scene_path, "2D_rendering", p, "panorama", self.room_type, "depth.png"]) for p in sections]
        self.rgb_paths = [os.path.join(*[scene_path, "2D_rendering", p, "panorama", self.room_type, "rgb_coldlight.png"]) for p in sections]
        self.camera_paths = [os.path.join(*[scene_path, "2D_rendering", p, "panorama", "camera_xyz.txt"]) for p in sections]
        self.camera_centers = self.read_camera_center()
        self.precompute_offsets()

        clip_feat_dim = self.clip_encoder.clip_feat_dim
        self.clipfeat_map = np.zeros((self.grid_size, self.grid_size, clip_feat_dim), dtype=np.float32)
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.weight = np.zeros((self.grid_size, self.grid_size), dtype=float)
        self.color_top_down = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        if os.path.exists(self.save_path):
            os.system(f'rm -r {self.save_path}')
        os.makedirs(self.save_path)

    def read_camera_center(self):
        camera_centers = []
        for i in range(len(self.camera_paths)):
            with open(self.camera_paths[i], 'r') as f:
                line = f.readline()
            center = list(map(float, line.strip().split(" ")))
            camera_centers.append(np.asarray([center[0], center[1], center[2]]))
        return camera_centers
    
    def precompute_offsets(self):
        self.pre_h = 512
        self.pre_w = 1024

        alpha = np.zeros((self.pre_h, self.pre_w))
        beta = np.zeros((self.pre_h, self.pre_w))

        x_tick = 180.0/self.pre_h
        y_tick = 360.0/self.pre_w

        for x in range(0, self.pre_h):
            for y in range(0, self.pre_w):
                # need 90 - -09
                alpha_c = 90 - (x * x_tick)
                beta_c = y * y_tick -180

                alpha[x,y] = np.deg2rad(alpha_c)
                beta[x,y] = np.deg2rad(beta_c)

        self.alpha = alpha
        self.beta = beta 
        
    
    def get_perspective_images(self, pano_img_path, base_idx_img, fov=90, overlap=0.5, mode='bilinear', out_hw=(480, 640)):
        imgs = []
        idx_imgs = []

        num_x = int(360/(fov*(1-overlap)))
        num_y = int(180/(fov*(1-overlap)))

        viewdir_x = np.linspace(-180, 180, num_x, endpoint=False)
        viewdir_y = np.linspace(-90, 90, num_y, endpoint=False)

        img = cv2.imread(pano_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        """
        # create an index image to map the perspective images back to the panorama space
        h, w = img.shape[:2]
        base_idx_img = np.arange(h*w).reshape(h, w)
        """
        
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

        points = {}
        coords = []
        pt_idx = []
        local_pc = np.zeros((0, 3))

        if depth_img.shape[0] != self.pre_h or depth_img.shape[1] != self.pre_w:
            raise ValueError("Depth image shape does not match precomputed offsets")
        # if True:
            x_tick = 180.0/depth_img.shape[0]
            y_tick = 360.0/depth_img.shape[1]

            for x in range(0, depth_img.shape[0]):
                for y in range(0, depth_img.shape[1]):
                    # need 90 - -09
                    alpha = 90 - (x * x_tick)
                    beta = y * y_tick -180

                    depth = depth_img[x,y]

                    if depth > 500.:
                        z_offset = depth*np.sin(np.deg2rad(alpha))
                        xy_offset = depth*np.cos(np.deg2rad(alpha))
                        x_offset = xy_offset * np.sin(np.deg2rad(beta))
                        y_offset = xy_offset * np.cos(np.deg2rad(beta))
                        point = np.asarray([x_offset, y_offset, z_offset])
                        coords.append(point + cam_center)
                        pt_idx.append(x*depth_img.shape[1] + y + base_idx)

            coords = np.asarray(coords)
            pt_idx = np.asarray(pt_idx)
        
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

        for idx in range(len(coords)):
            points[pt_idx[idx]] = coords[idx]
        
        local_pc = np.concatenate([local_pc, coords], axis=0)

        return points, local_pc
    
    def get_full_pc(self):
        pc = np.zeros((0, 3))
        points = {}
        idx_imgs = []
        base_idx = 0

        for i in range(len(self.depth_paths)):
            depth = cv2.imread(self.depth_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            cam_center = self.camera_centers[i]

            local_points, local_pc = self.get_local_pc(depth, cam_center, base_idx)
            points.update(local_points)
            pc = np.concatenate([pc, local_pc], axis=0)
            
            idx_img = np.arange(depth.shape[0]*depth.shape[1]).reshape(depth.shape[0], depth.shape[1]) + base_idx
            idx_imgs.append(idx_img)
            base_idx += depth.shape[0]*depth.shape[1]
        
        return points, pc, idx_imgs
    
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

    
    def get_geosem_map(self, seg_vis=False):

        seg_imgs = []

        points, pc, base_idx_imgs = self.get_full_pc()

        # get map resolution from grid size
        map_res, min_coords, max_coords = self.get_map_resolution(pc)
        
        # get floor and ceiling bounds for GeoSem Map
        up_limit = 0.7 * np.max(pc[:,2])
        down_limit = 0.4 * np.max(pc[:,2])

        # initialize color_top_down_ht
        self.color_top_down_ht = np.ones((self.grid_size, self.grid_size)) * np.min(pc[:,2]) - 1000

        for i in range(len(self.rgb_paths)):
            print(f"Generating Perspective Images for room {i}")
            persp_imgs, idx_imgs = self.get_perspective_images(self.rgb_paths[i], base_idx_imgs[i])

            for j in tqdm(range(len(persp_imgs))):
                cur_rgb = persp_imgs[j]

                # get pixel-wise clip features
                pix_feats, seg_img = self.clip_encoder.get_lseg_feat(cur_rgb, vis=seg_vis)
                if seg_vis:
                    seg_imgs.append(seg_img)

                # associate each pixel with a 3D point
                for l in range(cur_rgb.shape[0]):
                    for m in range(cur_rgb.shape[1]):
                        idx = idx_imgs[j][l,m]
                        if idx not in points:
                            continue

                        x, y, z = points[idx]

                        # get grid location from 3D point using map resolution and min_coords
                        grid_x = int((x - min_coords[0]) / (map_res*1000))
                        grid_y = int((y - min_coords[1]) / (map_res*1000))

                        if grid_x < 0 or grid_x >= self.grid_size or grid_y < 0 or grid_y >= self.grid_size:
                            raise ValueError("Grid location out of bounds")
                        
                        # if point is from ceiling, continue
                        if z >= up_limit:
                            continue

                        # update color_top_down and color_top_down_ht
                        if z > self.color_top_down_ht[grid_y, grid_x]:
                            self.color_top_down[grid_y, grid_x] = cur_rgb[l,m]
                            self.color_top_down_ht[grid_y, grid_x] = z
                        
                        # update GeoSem Map
                        feat = pix_feats[0,:,l,m]
                        self.clipfeat_map[grid_y, grid_x] = ((self.clipfeat_map[grid_y, grid_x] * self.weight[grid_y, grid_x]) + feat) / (self.weight[grid_y, grid_x] + 1)
                        self.weight[grid_y, grid_x] += 1

                        # update obstacles
                        if z <= down_limit:
                            continue
                        self.obstacles[grid_y, grid_x] += 1

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
        obs_map = (self.obstacles/np.max(self.obstacles) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.save_path, 'density_map.png'), obs_map)

        # threshold map to save a binary image
        occ = (self.obstacles > 0.05 * np.max(self.obstacles)).astype(np.float32)
        
        # Remove small artifacts and close small holes in the occupancy map
        occ = self.remove_small_artifacts(occ, threshold=4)
        occ = self.close_small_holes(occ, kernel_size=4)

        # add a buffer of 1 pixel around the walls (agent radius)
        occ = 1-binary_erosion(1-occ, iterations=1, structure=np.ones((2, 2)), border_value=1)
        occ = (occ * 255).astype(np.uint8)

        # save the binary occupancy map
        cv2.imwrite(os.path.join(self.save_path, 'obstacle_map.png'), occ)

    def vis_top_down_map(self):
        cv2.imwrite(os.path.join(self.save_path, 'top_down_map.png'), self.color_top_down[:,:,::-1])

    def vis_clipfeat_map(self, lang = None):
        obstacles = np.logical_not(self.obstacles).astype(np.uint8)
        no_map_mask = obstacles > 0

        if lang is None:
            lang = mp3dcat 
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
        plt.title("GeoSem Maps")
        plt.imshow(seg)
        plt.savefig(os.path.join(self.save_path, 'clipfeat_map.png'))

    def save_geosem_map(self):
        """
        np.save(os.path.join(self.save_path, 'clipfeat_map.npy'), self.clipfeat_map)
        np.save(os.path.join(self.save_path, 'obstacles.npy'), self.obstacles)
        np.save(os.path.join(self.save_path, 'color_top_down.npy'), self.color_top_down)
        np.save(os.path.join(self.save_path, 'weight.npy'), self.weight)
        """
        print(f"Size of clipfeat_map in bytes: {self.clipfeat_map.nbytes}")
        print(f"Size of obstacles in bytes: {self.obstacles.nbytes}")
        print(f"Size of color_top_down in bytes: {self.color_top_down.nbytes}")
        print(f"Size of weight in bytes: {self.weight.nbytes}")
        self.vis_obstacle_map()
        self.vis_top_down_map()
        self.vis_clipfeat_map()

def config():
    a = argparse.ArgumentParser(description='Generate GeoSem Maps from Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--num_scenes', default=None, type=int, help='Number of scenes to process')
    a.add_argument('--save_root', default='Structured3D_GeoSem_Maps', type=str, help='path to save GeoSem Maps')
    a.add_argument('--grid_size', default=1000, type=int, help='Size of grid for GeoSem Map')
    a.add_argument('--room_type', default='full', type=str, help='Type of room to process')
    args = a.parse_args()
    return args

def main(args):
    print("Creating GeoSem Maps from panorama...")
    data_root = args.data_root
    
    scenes = os.listdir(os.path.join(data_root))
    scenes.sort()
    if args.num_scenes:
        scenes = scenes[:args.num_scenes]
    print(f"Total scenes: {len(scenes)}")

    if os.path.exists(args.save_root):
        os.system(f'rm -r {args.save_root}')
    os.makedirs(args.save_root)

    # initialize LSeg model
    clip_encoder = LSegEncoder()
    
    for scene in tqdm(scenes):
        print(f"Processing {scene}...")
        scene_path = os.path.join(data_root, scene)
        gen = GeoSemMapGenerator(scene_path, clip_encoder, args.save_root, args.grid_size, args.room_type)

        # imgs = gen.viz_perspective_images()
        gen.get_geosem_map()
        gen.save_geosem_map()

        break

if __name__ == "__main__":
    main(config())