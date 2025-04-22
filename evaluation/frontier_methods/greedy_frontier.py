"""
Exploration strategy that chooses the closest frontier to the current position.
"""

import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf

from evaluation.exploration_model import ExplorationModel
from evaluation.frontier_methods.frontier_utils import detect_frontiers


class GreedyFrontierModel(ExplorationModel):
    """
    Exploration strategy that chooses the closest frontier to the current position.
    """

    def get_goal(self, input):
        """
        Choose the closest frontier as the goal.
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

        # select current goal as frontier which is closest to current position
        dist = np.linalg.norm(np.array(np.where(frontiers)).T - np.array([x, y]), axis=1)
        goal_idx = np.argmin(dist)
        goal_x, goal_y = np.where(frontiers)[0][goal_idx], np.where(frontiers)[1][goal_idx]

        roll_back = False
        if goal_x == x and goal_y == y:
            print("[WARNING] Goal is same as current position in GreedyFrontierModel. Choosing a random frontier goal.")
            
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

            idx = np.random.choice(range(len(np.where(frontiers)[0])))
            goal_x, goal_y = np.where(frontiers)[0][idx], np.where(frontiers)[1][idx]
            roll_back = True

        output = {
            "goal_x": goal_x,
            "goal_y": goal_y,
            "roll_back": roll_back,
        }
        return output