"""
Exploration strategy that selects a random frontier.
"""

import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf

from evaluation.exploration_model import ExplorationModel
from evaluation.frontier_methods.frontier_utils import detect_frontiers

class RandomFrontierModel(ExplorationModel):
    """
    Exploration strategy that selects a random frontier.
    """
    def __init__(self, conf: OmegaConf):
        self.seed = conf.seed

        # set seed for reproducibility
        self.rng = np.random.default_rng(self.seed)

    def get_goal(self, input):
        pred_map = input["pred_map"]

        # get frontiers 
        cur_pred = deepcopy(pred_map)
        _, _, labels = detect_frontiers(cur_pred)

        if len(labels) == 0:
            return {
                "abort": True
            }

        random_contour = self.rng.choice(np.unique(labels))
        goal_x = self.rng.choice(np.where(labels == random_contour)[0])
        goal_y = self.rng.choice(np.where(labels == random_contour)[1])

        roll_back = False

        output = {
            "goal_x": goal_x,
            "goal_y": goal_y,
            "roll_back": roll_back,
        }
        return output