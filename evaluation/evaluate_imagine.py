"""
Evalution script for different exploration algorithms deployed on an agent in
the Structured-3D validation scenes.
"""

import numpy as np
import random
import argparse
import cv2
import os
import pickle
import warnings
from copy import deepcopy
from tqdm import tqdm
import enum
from pprint import pprint
import pandas as pd
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import seaborn as sns
from omegaconf import OmegaConf

import wandb
import json

from datagen_imagination.occupancy_generation.stru3d_utils import invalid_scenes_ids
from evaluation.categories import objectnav_categories
from evaluation.astar_planner import AstarPlanner
from evaluation.robot import Robot
from evaluation.clip_model import CLIP_Model
from evaluation.gt_objgoal_setter import GoalSetter
from evaluation.imagination_model import ImaginationModel
from evaluation.frontier_methods.random_frontier import RandomFrontierModel
from evaluation.frontier_methods.semantic_frontier import SemanticFrontierModel
from evaluation.frontier_methods.clip_frontier import ClipFrontierModel
from evaluation.frontier_methods.greedy_frontier import GreedyFrontierModel

# set random seed
random.seed(0)
np.random.seed(0)


class ExplorationMethod(enum.Enum):

    # Imagination based methods
    IMAGINE = 'imagine'

    # Frontier based methods
    FRONTIER_CLIP = 'frontier_clip'
    SEMANTIC = 'semantic'
    RANDOM = 'random'
    GREEDY = 'greedy'


# Mapping from method type to method class, requires_clip_model
method_mapping = {
    ExplorationMethod.IMAGINE.value: (ImaginationModel, True),
    ExplorationMethod.FRONTIER_CLIP.value: (ClipFrontierModel, True),
    ExplorationMethod.SEMANTIC.value: (SemanticFrontierModel, True),
    ExplorationMethod.RANDOM.value: (RandomFrontierModel, False),
    ExplorationMethod.GREEDY.value: (GreedyFrontierModel, False),
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Evaluator:
    """
    Initialize all methods to be benchmarked, keep track of metrics, log the path trajectories for every method.
    """

    metric_str = ['SPL', 'D2Goal', 'SuccessRate', 'Time']

    def __init__(self, conf: OmegaConf):

        # Initialize scene list
        self.scene_list = []
        self.data_dir = conf.data_dir
        scene_list_path_val = os.path.join(self.data_dir, "scene_list_val_geosem_map.txt")
        with open(scene_list_path_val, "r") as f:
            self.scene_list = f.read().splitlines()

        self.scene_list = [scene for scene in self.scene_list if int(scene.split('_')[-1]) not in invalid_scenes_ids]
        self.scene_list = self.scene_list[:conf.num_scenes] if conf.num_scenes > 0 else self.scene_list
        self.scene_list = sorted(self.scene_list)

        # Initialize CLIP model
        self.clip_model = CLIP_Model()

        # Initialize metrics[SPL, D2Goal, SuccessRate, Time] for all scenes and methods, and goal object category 
        goal_cats = objectnav_categories

        self.metrics = {
            method: {
                scene: {
                    cat: {metric: [] for metric in self.metric_str} for cat in goal_cats
                } for scene in self.scene_list
            } for method in conf.methods.keys()
        }

        self.traj = {
            method: {
                scene: {
                    cat: [] for cat in goal_cats
                } for scene in self.scene_list
            } for method in conf.methods.keys()
        }

        # Initialize results directory
        self.results_dir = conf.results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        result_fname = f"{conf.exp_name}_results.json"
        self.result_path = os.path.join(self.results_dir, result_fname)
        if conf.resume and os.path.exists(self.result_path):
            with open(self.result_path, 'r') as f:
                self.metrics = json.load(f)
        else:
            with open(self.result_path, 'w') as f:
                json.dump(self.metrics, f)
            
        traj_fname = f"{conf.exp_name}_traj.json"
        self.traj_path = os.path.join(self.results_dir, traj_fname)
        if conf.resume and os.path.exists(self.traj_path):
            with open(self.traj_path, 'r') as f:
                self.traj = json.load(f)
            # resume evaluation from the last scene
            # last evaluated scene is the scene with a non-empty traj list for at least one method and category
            last_eval_scene = None
            for scene in reversed(self.scene_list):
                for method in self.metrics.keys():
                    for cat in goal_cats:
                        if len(self.traj[method][scene][cat]) > 0:
                            last_eval_scene = scene
                            break
                    if last_eval_scene is not None:
                        break
                if last_eval_scene is not None:
                    break
            self.scene_list = self.scene_list[self.scene_list.index(last_eval_scene):]
            print(f"[LOG] Resuming evaluation from scene {last_eval_scene}")

        else:
            with open(self.traj_path, 'w') as f:
                json.dump(self.traj, f)

        # Initialize json file for saving start and goal 
        self.positions = {
            scene: {
                cat: {
                    'start': [],
                    'theta': [],
                    'goal': []
                } for cat in goal_cats
            } for scene in self.scene_list
        }
        positions_fname = f"positions.json"
        self.positions_path = os.path.join(self.results_dir, positions_fname)
        if not os.path.exists(self.positions_path):
            with open(self.positions_path, 'w') as f:
                json.dump(self.positions, f)
        else:
            with open(self.positions_path, 'r') as f:
                self.positions = json.load(f)

        # Initialize wandb logging
        self.log = conf.log.do
        log_dir = os.path.join(conf.log.log_dir, conf.exp_name)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        if self.log:
            run = wandb.init(
                project = "Imagination-CLIPMap",
                entity="rpg-objectnav",
                name = conf.exp_name,
                config = conf,
                dir=log_dir,
                id=conf.log.run_id,
                resume='allow',
            )

        # Initialize all methods
        self.methods = {}
        for method, method_conf in conf.methods.items():
            method_type = method
            method_class, requires_clip = method_mapping[method_type]

            if requires_clip:
                self.methods[method] = method_class(self.clip_model, method_conf)
            else:
                self.methods[method] = method_class(method_conf)

        # Initialize random seed
        self.rng = np.random.default_rng(conf.seed)

        # Initialize goal setter
        self.goal_setter = GoalSetter(conf.goal_setter)

        self.conf = conf

    def set_start_goal(self, occ_map: np.ndarray, gt_map: np.ndarray, clipfeat_map: np.ndarray, cat: str, ctop_down: np.ndarray = None):
        """
        Set the start and goal positions for a given scene and category.
        """
        # Get the starting position and goal position for the scene and category
        goal_x, goal_y = self.goal_setter.get_gt_goal(
            goal_category=cat,
            occ_map=occ_map,
            clipfeat_map=clipfeat_map,
            clip_model=self.clip_model,
            ctop_down=ctop_down
        )

        if (goal_x is None) or (goal_y is None):
            return (0,0), 0, (goal_x, goal_y)
        
        # Set current position so that path to gt_goal can be planned
        attempts = 0
        x, y = None, None
        gt_planner = AstarPlanner(occ_map)
        while (x is None or y is None) and attempts < 10:
            x, y = Robot.set_random_free(
                gt_map = gt_map,
                occ_map = occ_map,
                rng = self.rng
            )
            gt_path = gt_planner.plan_path((x, y), (goal_x, goal_y))

            if gt_path is None:
                x, y = None, None
                continue

            attempts += 1
        
        start_x, start_y = x, y
        start_theta = self.rng.uniform(0, 2*np.pi)

        return (start_x, start_y), start_theta, (goal_x, goal_y)

    def get_robot_start_config(self, scene: str, occ_map: np.ndarray, gt_map: np.ndarray, clipfeat_map: np.ndarray, cat: str, ctop_down: np.ndarray = None):
        """
        Get the start and goal positions for a given scene and category.
        """
        if len(self.positions[scene][cat]['start']) == 0:
            start_pos, start_theta, goal_pos = self.set_start_goal(
                occ_map=occ_map,
                gt_map=gt_map,
                clipfeat_map=clipfeat_map,
                cat=cat,
                ctop_down=ctop_down
            )
            self.positions[scene][cat]['start'].append(start_pos)
            self.positions[scene][cat]['theta'].append(start_theta)
            self.positions[scene][cat]['goal'].append(goal_pos)

            with open(self.positions_path, 'w') as f:
                json.dump(self.positions, f, cls=NpEncoder)
        
        else:
            start_pos = self.positions[scene][cat]['start'][0]
            start_theta = self.positions[scene][cat]['theta'][0]
            goal_pos = self.positions[scene][cat]['goal'][0]

        return start_pos, start_theta, goal_pos

    def evaluate_scene(self, scene: str):
        """
        Evaluate all methods for a given scene.
        """
        scene_path = os.path.join(self.data_dir, scene)
        occ_path = os.path.join(scene_path, 'occupancy_map.png')
        gt_occ_path = os.path.join(scene_path, 'gt_occupancy_map.png')
        map_data_path = os.path.join(scene_path, 'map_data.pkl')
        clipfeat_map_path = os.path.join(scene_path, 'GeoSemMap', 'clipfeat_map.npy')
        ctopd_path = os.path.join(scene_path, 'GeoSemMap', 'color_top_down.npy')

        with open(clipfeat_map_path, 'rb') as f:
            clipfeat_map = np.load(f)
        
        with open(ctopd_path, 'rb') as f:
            c_top_down = np.load(f)

        occ_map = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE)
        occ_map = (occ_map // 255).astype(np.uint8)
        occ_map = cv2.dilate(occ_map, kernel=np.ones((3, 3), np.uint8), iterations=1)

        gt_map = cv2.imread(gt_occ_path, cv2.IMREAD_GRAYSCALE)
        gt_map = (gt_map // 255).astype(np.uint8)

        # Load map data
        with open(map_data_path, 'rb') as f:
            map_data = pickle.load(f)

        # Iterate over all goal categories and methods
        viz_imgs = []
        for cat in objectnav_categories:
            
            # Get the starting position and goal position for the scene and category
            start_pos, start_theta, goal_pos = self.get_robot_start_config(
                scene=scene,
                occ_map=occ_map,
                gt_map=gt_map,
                clipfeat_map=clipfeat_map,
                cat=cat,
                ctop_down=c_top_down if self.log else None
            )

            if start_pos[0] is None or start_pos[1] is None:
                print(f"Could not find a valid starting position for scene {scene} and category {cat}. Skipping...")
                viz_imgs.append(wandb.Image(np.zeros((100, 100, 3), dtype=np.uint8)))
                continue
            if goal_pos[0] is None or goal_pos[1] is None:
                print(f"Could not find a valid goal position for scene {scene} and category {cat}. Maybe {cat} does not exist in the scene. Skipping...")
                viz_imgs.append(wandb.Image(np.zeros((100, 100, 3), dtype=np.uint8)))
                continue

            for method, method_func in self.methods.items():

                # Reset the method object
                method_func.reset_eval(goal_category=cat, clipfeat_map=clipfeat_map)

                if self.conf.robot_conf.debug:
                    debug_dir = os.path.join(self.results_dir, 'debug', scene, cat, method)
                else:
                    debug_dir = None

                # Initialize the robot
                robot = Robot(
                    exp_method=method_func,
                    start_pos=start_pos,
                    start_theta=start_theta,
                    goal_pos=goal_pos,
                    map_data=map_data,
                    occ_map=occ_map,
                    gt_map=gt_map,
                    debug_dir=debug_dir,
                    conf=self.conf.robot_conf,
                )

                # Run the simulation
                run_metrics = robot.run_simulation()

                print(f"[LOG] Scene: {scene}, Category: {cat}, Method: {method}, Metrics: {run_metrics}")

                # Log the path trajectory
                self.traj[method][scene][cat] = robot.path_history

                # Log the metrics
                for metric_name in self.metric_str:
                    self.metrics[method][scene][cat][metric_name].append(run_metrics[metric_name])

                # Save the results
                with open(self.result_path, 'w') as f:
                    json.dump(self.metrics, f, cls=NpEncoder)

                # Save the trajectory
                with open(self.traj_path, 'w') as f:
                    json.dump(self.traj, f, cls=NpEncoder)

            # Visualize trajectories on color-top down map of all methods
            if self.log:
                # plt.figure(figsize=(c_top_down.shape[1]//10, c_top_down.shape[0]//10))
                fig = plt.figure(figsize=(12,6))
                plt.imshow(c_top_down)
                plt.plot(start_pos[1], start_pos[0], 'ro', label='Start')
                plt.plot(goal_pos[1], goal_pos[0], 'go', label='Goal')

                for method in self.methods.keys():
                    traj = self.traj[method][scene][cat]
                    if len(traj) > 0:
                        traj = np.array(traj)
                        plt.plot(traj[:, 1], traj[:, 0], label=method)

                plt.legend()
                plt.axis('off')

                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                viz_imgs.append(wandb.Image(img))

                plt.close()


        # Log the metrics to wandb
        if self.log:
            overall_metrics, cat_metrics = self.get_mean_metrics(self.metrics)
            
            o_tab = pd.DataFrame(overall_metrics)
            c_tab = pd.DataFrame(cat_metrics)

            wandb.log({
                "Overall Metrics": wandb.Table(dataframe=o_tab),
                "Category Metrics": wandb.Table(dataframe=c_tab)
            }, commit=False)

            # Log the visualization images
            img_table = wandb.Table(data=[viz_imgs], columns=objectnav_categories)
            wandb.log({"Trajectories": img_table}, commit=True)
            
            """
            log_dict = {}
            for metric_name in self.metric_str:
                for method, method_metrics in overall_metrics.items():
                    log_dict[f"{metric_name}/{method}"] = method_metrics[metric_name]

            wandb.log(log_dict)

            log_dict = {}
            for metric_name in self.metric_str:
                for method, method_cat_metrics in cat_metrics.items():
                    for cat, cat_metrics in method_cat_metrics.items():
                        log_dict[f"{metric_name}/{cat}/{method}"] = cat_metrics[metric_name]

            wandb.log(log_dict)
            """
    def evaluate_all(self):
        """
        Evaluate all methods for all scenes.
        """
        for scene in self.scene_list:
            self.evaluate_scene(scene)
            print(f"Scene {scene} evaluation complete!")

        wandb.finish()

    @classmethod
    def get_mean_metrics(cls, metrics):
        """
        For each method, get the mean of all metrics across all scenes and categories.
        """
        overall_metrics = {
            method: {
                metric: [] for metric in cls.metric_str
            } for method in metrics.keys()
        }
        cat_metrics = {
            method: {
                cat: {
                    metric: [] for metric in cls.metric_str
                } for cat in objectnav_categories
            } for method in metrics.keys()
        }

        for method, method_metrics in metrics.items():
            for metric in cls.metric_str:
                for scene_metrics in method_metrics.values():
                    for cat, pcat_metrics in scene_metrics.items():
                        overall_metrics[method][metric].extend(pcat_metrics[metric])
                        cat_metrics[method][cat][metric].extend(pcat_metrics[metric])

        for method, method_metrics in overall_metrics.items():
            for metric, metric_vals in method_metrics.items():
                overall_metrics[method][metric] = np.mean(metric_vals)

        for method, method_cat_metrics in cat_metrics.items():
            for cat, pcat_metrics in method_cat_metrics.items():
                for metric, metric_vals in pcat_metrics.items():
                    cat_metrics[method][cat][metric] = np.mean(metric_vals)

        return overall_metrics, cat_metrics


def config():
    args = argparse.ArgumentParser(description='Evalution Object-Nav using Imagination')

    # Data and Scene Parameters
    args.add_argument('--conf', type=str, default='configs/evaluation_conf.yaml', help='Path to config file')
    args = args.parse_args()

    return args

def main():
    args = config()

    conf = OmegaConf.load(args.conf)
    evaluator = Evaluator(conf)
    evaluator.evaluate_all()

    # Save the results
    with open(evaluator.result_path, 'w') as f:
        json.dump(evaluator.metrics, f)

    # Save the trajectory
    with open(evaluator.traj_path, 'w') as f:
        json.dump(evaluator.traj, f)

    # Print the results
    overall_metrics, cat_metrics = Evaluator.get_mean_metrics(evaluator.metrics)
    pprint(f"Overall Metrics: ")
    pprint(pd.DataFrame(overall_metrics))

    pprint(f"Category Metrics: ")
    pprint(pd.DataFrame(cat_metrics))

    print("Done!")

if __name__ == "__main__":
    main()
    