"""
Code to generate GeoSemMaps from Panoramic RGB, Depth and Pose data (Vectorized version - 2)
Different from final implementation - Indexing is done using unique index images
for each panoramic image depending on the point cloud data (depth panorama).

Masking in Point Cloud due to:
    1. invalid depth values (< 500mm)
    2. Different points mapped to same point by reducing depth precision
    3. Points outside the room bounds using the annotations in Structured3D

After Point Cloud masking, we compute unique indices for each point, which form a 
unique index image for each panorama. This index image is used to map the perspective
images back to the panorama space.
"""

import os
import cv2
import math
import numpy as np
from tqdm import tqdm
import argparse

from datagen_imagination.geosem_map_generation.utils.panaroma_to_perspective import e2p

import torch
import torchvision.transforms as transforms
import clip
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from copy import deepcopy
import json
import yaml
import time

from PIL import Image
from datagen_imagination.geosem_map_generation.utils.viz_utils import get_new_pallete, get_new_mask_pallete
from utils.clip_utils import get_text_feats
from utils.mp3dcat import mp3dcat
import clip


from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image

class LSegEncoder():
    def __init__(
            self,
            ckpt_path,
            clip_version="ViT-B/32",
            crop_size=480,
            base_size=640,
        ):

        self.crop_size = crop_size # 480
        self.base_size = base_size # 520
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
        pretrained_state_dict = torch.load(ckpt_path, map_location=device)
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

class GeoSemMapGenerator():

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

        coords = []
        local_pc = np.zeros((0, 3))
        
        h, w = depth_img.shape[:2]
        pt_idx = np.arange(h*w).reshape(h, w)

        if depth_img.shape[0] != self.pre_h or depth_img.shape[1] != self.pre_w:
            raise ValueError("Depth image shape does not match precomputed offsets")
        
        # use precomputed offsets
        z_offset = depth_img * np.sin(self.alpha)
        xy_offset = depth_img * np.cos(self.alpha)
        x_offset = xy_offset * np.sin(self.beta)
        y_offset = xy_offset * np.cos(self.beta)
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

        # mask using min-max pc
        mask = coords[:,0] >= self.min_pc[0]
        mask = np.logical_and(mask, coords[:,0] <= self.max_pc[0])
        mask = np.logical_and(mask, coords[:,1] >= self.min_pc[1])
        mask = np.logical_and(mask, coords[:,1] <= self.max_pc[1])
        coords = coords[mask]
        pt_idx = pt_idx[mask]

        idx_img = np.ones_like(depth_img) * -1
        idx_img[pt_idx // w, pt_idx % w] = base_idx + np.arange(len(coords))

        # local_pc is now a 2D array with columns [x, y, z]
        local_pc = np.concatenate([local_pc, coords], axis=0)

        return local_pc, idx_img
    
    def get_full_pc(self):
        pc = np.zeros((0, 3))
        idx_imgs = []
        base_idx = 0

        for i in range(len(self.depth_paths)):
            depth = cv2.imread(self.depth_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            cam_center = self.camera_centers[i]

            local_pc, idx_img = self.get_local_pc(depth, cam_center, base_idx)
            pc = np.concatenate([pc, local_pc], axis=0)
            
            idx_imgs.append(idx_img)
            base_idx += len(local_pc)
        
        """
        assert (pc[:, -1] == np.arange(pc.shape[0])).all()
        assert (pc[:, -1] == np.arange(len(self.depth_paths)*depth.shape[0]*depth.shape[1])).all()
        """

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

        # pc is a 2D array with columns [x, y, z]
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
        # pc is now a 2D array with columns [x, y, z, grid_x, grid_y]
        pc = np.concatenate([pc, grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], axis=1)

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
                indices = idx_imgs[j].reshape(-1)
                cur_rgb = cur_rgb.reshape(-1, 3)

                # get valid indices (remove -1 from idx_img)
                valid_mask = indices != -1
                indices = indices[valid_mask]
                feats = feats[valid_mask]
                cur_rgb = cur_rgb[valid_mask]

                # get all points corresponding to the current perspective image
                pc_cur = pc[indices]

                # associate each point with rgb
                # pc_cur is now a 2D array with columns [x, y, z, grid_x, grid_y, r, g, b]
                pc_cur = np.concatenate([pc_cur, cur_rgb], axis=1)

                # remove points with idx = 0 (depth < 500mm)
                depth_mask = pc_cur[:, 3] > 0
                pc_cur = pc_cur[depth_mask]
                feats = feats[depth_mask]

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

                # update GeoSem Map
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
        plt.title("GeoSemMap")
        plt.imshow(seg)
        plt.savefig(os.path.join(self.vis_path, 'geosem_map.png'))

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
    a.add_argument('-c', '--config', default='config.yaml', type=str, help='path to config file')
    args = a.parse_args()
    
    # open config (yaml) file
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    return cfg

def main(cfg):
    print("Creating GeoSemMaps from panorama...")

    # get parameters from cfg
    lseg_params = cfg['LSEG']
    clipfp_params = cfg['GeoSemMap']
    data_root = cfg['data_root']
    num_scenes = cfg['num_scenes']
    
    scenes = os.listdir(os.path.join(data_root))
    scenes.sort()
    if num_scenes:
        scenes = scenes[:num_scenes]
    print(f"Total scenes: {len(scenes)}")

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
    if os.path.exists(valid_scene_path):
        os.system(f'rm {valid_scene_path}')

    start_full = time.time()
    # for scene in tqdm(scenes):
    for scene in scenes:
        print(f"Processing {scene}...")
        scene_path = os.path.join(data_root, scene)
        try:
            gen = GeoSemMapGenerator(
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
        
        with open(valid_scene_path, 'a') as f:
            f.write(scene + '\n')

    print(f"Time taken for all scenes: {time.time() - start_full}")

if __name__ == "__main__":
    cfg = config()
    main(cfg)