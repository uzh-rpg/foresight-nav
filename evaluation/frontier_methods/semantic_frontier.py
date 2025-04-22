"""
Exploration using a Semantic map, selecting the frontier with the highest semantic score.
"""

import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf

from evaluation.exploration_model import ExplorationModel
from evaluation.frontier_methods.frontier_utils import detect_frontiers
from evaluation.clip_model import CLIP_Model
from evaluation.categories import coco_categories


class SemanticFrontierModel(ExplorationModel):
    """
    Exploration using a Semantic map, selecting the frontier with the highest semantic score.
    """
    def __init__(self, clip_model: CLIP_Model, conf: OmegaConf):
        self.clip_model = clip_model

    def reset_eval(self, goal_category: str, clipfeat_map: np.ndarray):
        self.goal_category = goal_category
        self.goal_cat_idx = coco_categories.index(goal_category)
        self.clipfeat_map = clipfeat_map

        # generate semantic map by using clipfeat_map and coco_categories
        map_feats = self.clipfeat_map.reshape(-1, self.clipfeat_map.shape[-1])
        self.semantic_map = map_feats @ self.clip_model.coco_feat_other.T
        self.semantic_map = np.argmax(self.semantic_map, axis=1) 
        self.semantic_map = self.semantic_map.reshape(self.clipfeat_map.shape[:2]) 

    def get_goal(self, input):
        """
        Get goal x,y from current semantic_map and c2c_matrix.
        Get frontiers from pred_map, and choose the one with highest similarity to goal.
        """
        x, y = input["x"], input["y"]
        pred_map = input["pred_map"]

        # get frontiers 
        cur_pred = deepcopy(pred_map)
        frontiers, frontier_img, labels = detect_frontiers(cur_pred)

        if len(np.where(frontiers)[0]) == 0:
            return {
                "abort": True
            }

        # select current goal as frontier with highest similarity with goal in c2c_matrix
        semantic_pred = self.semantic_map[frontiers]
        
        # get goal similarity with frontiers
        goal_idx = np.argmax(self.clip_model.c2c_matrix_other[self.goal_cat_idx, semantic_pred])
        goal_x, goal_y = np.where(frontiers)[0][goal_idx], np.where(frontiers)[1][goal_idx]

        roll_back = False
        if goal_x == x and goal_y == y:
            print("[WARNING] Goal is same as current position in SemanticFrontierModel. Choosing a random frontier goal.")

            # mark current frontier contour as obstacle
            cur_contour = labels == labels[goal_x, goal_y]
            input["pred_map"][cur_contour] = 1
            frontier_img[cur_contour] = 0
            frontiers = frontier_img == 1
            
            # TODO: fix in main
            # self.planner.update_weights()

            # move agent to free cell 
            # x, y = self.path_history[-10]

            if len(np.where(frontiers)[0]) == 0:
                return {
                    "abort": True
                }

            # return random frontier point
            idx = np.random.choice(range(len(np.where(frontiers)[0])))
            goal_x, goal_y = np.where(frontiers)[0][idx], np.where(frontiers)[1][idx]
            roll_back = True

        output = {
            "goal_x": goal_x,
            "goal_y": goal_y,
            "roll_back": roll_back,
        }
        return output