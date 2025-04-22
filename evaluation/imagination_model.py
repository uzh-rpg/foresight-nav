"""
Imagination model to predict the CLIP floorplan using a given model.
"""

import torch
import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf

from models.imagine_unet import UNet2D_GeoSem_Pred
from models.imagine_mae import model_factory as mae_model_factory
from evaluation.exploration_model import ExplorationModel
from evaluation.clip_model import CLIP_Model

class ImaginationModel(ExplorationModel):
    """
    Imagination model to predict the CLIP floorplan using a given model.
    """
    def __init__(self, clip_model: CLIP_Model, conf: OmegaConf):
        self.clip_model = clip_model

        self.device = conf.device
        model_type = conf.model_type
        model_conf = conf.model_conf
        model_ckpt = conf.model_ckpt

        if model_type == 'unet':
            self.model = UNet2D_GeoSem_Pred(**OmegaConf.to_container(model_conf))
        else:
            self.model = mae_model_factory[model_type](**OmegaConf.to_container(model_conf))

        ckpt = torch.load(model_ckpt, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        print(f"Successfully loaded model from {model_ckpt}!")

        self.occ_thresh = conf.occ_thresh
        self.int_thresh = conf.int_thresh
        self.use_pred_occ = conf.use_pred_occ
        self.use_int_mask = conf.use_int_mask

        self.model.to(self.device)
        self.model.eval()

    def get_model(self):
        return self.model
    
    def reset_eval(self, goal_category: str, clipfeat_map: np.ndarray):
        # get goal category clip features
        self.goal_category = goal_category
        self.goal_clip_feat = self.clip_model.get_text_feats([self.goal_category])

        self.gt_clipfeat_map = clipfeat_map
    
    def get_goal(self, input):
        """
        Predict the CLIP floorplan using the given model.
        Use the similarity score with the predicted CLIP floorplan to select the goal.
        Return the predicted occupancy as well for replanning.
        """
        x, y = input["x"], input["y"]
        pred_map = torch.from_numpy(input["pred_map"]).unsqueeze(0)      # C, H, W

        clipfeat_map = deepcopy(self.gt_clipfeat_map)
        clipfeat_map[pred_map.squeeze(0) == 0.5] = 0

        norm = np.linalg.norm(clipfeat_map, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        clipfeat_map = clipfeat_map / norm
        clipfeat_map = torch.from_numpy(clipfeat_map).permute(2, 0, 1)    # C, H, W

        cur_geosem_map = torch.cat([clipfeat_map, pred_map], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_geosem_map = self.model(
                input = cur_geosem_map,
                inference = True
            )

        pred_geosem_map = pred_geosem_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred_clipfeat_map = pred_geosem_map[:, :, :512]
        pred_occ = pred_geosem_map[:, :, 512] > self.occ_thresh
        int_mask = pred_geosem_map[:, :, 513] > self.int_thresh

        # mark exterior locations in CLIP floorplan as zeros
        if self.use_int_mask:
            pred_clipfeat_map[~int_mask] = 0
            pred_clipfeat_map[pred_map.squeeze(0) != 0.5] = 0

        # Get goal
        map_feats = pred_clipfeat_map.reshape(-1, pred_clipfeat_map.shape[-1])
        goal_sim = map_feats @ self.goal_clip_feat.T
        goal_sim = goal_sim.reshape(pred_clipfeat_map.shape[:2])

        # Choose the goal with highest similarity in goal_sim
        goal_idx = np.argmax(goal_sim)
        goal_x, goal_y = np.unravel_index(goal_idx, goal_sim.shape)

        output = {
            "goal_x": goal_x,
            "goal_y": goal_y,
            "roll_back": False,
        }

        if self.use_pred_occ:
            output["pred_occ"] = pred_occ

        return output

    