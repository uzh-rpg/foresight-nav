"""
Robot class for evaluating exploration methods on Structured3D dataset.
"""

import os
import numpy as np
import cv2
import wandb
from omegaconf import OmegaConf

from evaluation.exploration_model import ExplorationModel
from evaluation.astar_planner import AstarPlanner


class Robot:
    def __init__(
            self,
            exp_method: ExplorationModel,
            start_pos: tuple,
            start_theta: float,
            goal_pos: tuple,
            map_data: dict,
            occ_map: np.ndarray,
            gt_map: np.ndarray,
            debug_dir: str,
            conf: OmegaConf,
        ):

        self.exp_method = exp_method
        
        self.occ_map = occ_map
        self.gt_map = gt_map
        self.map_data = map_data

        # Predicted map - initialize with unknown cells
        self.pred_map = np.ones_like(self.occ_map).astype(np.float32) * 0.5

        # Initialize sensor parameters
        self.field_of_view = np.radians(conf.field_of_view)
        self.turn_angle = np.radians(conf.turn_angle)
        self.max_steps = conf.max_steps

        # Turning parameters
        self.max_rotation = 2*np.pi / self.turn_angle
        self.cur_rotation = 0

        # Get sensor range in pixels
        self.sensor_range = conf.sensor_range
        map_res = self.map_data['map_res']
        self.sensor_range = int(self.sensor_range / map_res)

        # Check save path
        if conf.debug:
            self.debug_dir = debug_dir
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
        self.debug = conf.debug
        self.debug_freq = conf.debug_freq
        self.log = conf.log

        if self.log:
            self.log_prefix = debug_dir.split('/')[-3:].join('/')

        # Initialize A* planner
        self.planner = AstarPlanner(self.pred_map)
        self.gt_planner = AstarPlanner(self.occ_map)

        # Initialize robot orientation
        self.theta = start_theta # Angle with respect to x-axis (image width axis)

        # Initialize goal position
        self.gt_goal_x, self.gt_goal_y = goal_pos
        self.goal_x, self.goal_y = None, None
        self.x, self.y = None, None
        self.gt_goal_found = False

        self.x, self.y = start_pos
        self.gt_path = self.gt_planner.plan_path((self.x, self.y), (self.gt_goal_x, self.gt_goal_y))
        self.start_x, self.start_y = self.x, self.y

        self.replan_freq = conf.replan_freq

        # metrics
        self.best_pathlen = len(self.gt_path)
        self.pathlen = 0
        self.path_history = []

        # Initialize the agent
        self.agent_initialization()

    def agent_initialization(self):
        """
        Rotate in place for 360 degrees to get initial sensor data.
        """
        self.cur_rotation = 0
        while self.cur_rotation < self.max_rotation:
            self.get_sensor_data()
            self.cur_rotation += 1
            self.theta -= self.turn_angle
            self.theta = self.theta % (2 * np.pi)

    @classmethod
    def set_random_free(cls, gt_map, occ_map, rng):
        """Sets random position using gt_map and occupancy_map."""

        valid_indices = np.where(np.logical_and(gt_map == 1, occ_map == 0))
        if len(valid_indices[0]) == 0:
            raise ValueError("No free space found in map!")
        idx = rng.choice(range(len(valid_indices[0])))
        x, y = valid_indices[0][idx], valid_indices[1][idx]
        return x, y

    def get_sensor_data(self):
        """Simulates LiDAR sensor data based on robot position and orientation."""

        # Define sensor angles relative to robot orientation
        angles = np.linspace(
            -self.field_of_view/2 + self.theta,
            self.field_of_view/2 + self.theta,
            self.sensor_range * 2 + 1)

        for i, angle in enumerate(angles):
            # Check for obstacles within sensor range
            for dist in range(1, self.sensor_range + 1):
                x_sensor = int(self.x - dist * np.sin(angle))
                y_sensor = int(self.y + dist * np.cos(angle))

                # Check if sensor ray goes out of map bounds
                if not (0 <= x_sensor < self.occ_map.shape[0] and 0 <= y_sensor < self.occ_map.shape[1]):
                    break

                # Check if obstacle is detected
                if self.occ_map[x_sensor, y_sensor] == 1:
                     # Update predicted map with obstacle
                    self.pred_map[x_sensor, y_sensor] = 1 

                    # Update planner with obstacle
                    self.planner.update_obstacle(x_sensor, y_sensor)
                    
                    # Stop checking for this angle when obstacle found
                    break  
                else:
                    self.pred_map[x_sensor, y_sensor] = 0
    
    def step(self, cur_step):
        """Takes an action and updates the map.
        Action Space:
            0: Move forward
            1: Turn left
            2: Turn right
        """
        pred_occ = None

        # Deterministically navigate to the goal if it is visible
        if self.gt_goal_found:
            self.goal_x, self.goal_y = self.gt_goal_x, self.gt_goal_y

        # Check if gt_goal is visible
        elif self.pred_map[self.gt_goal_x, self.gt_goal_y] != 0.5:
            self.gt_goal_found = True
            self.goal_x, self.goal_y = self.gt_goal_x, self.gt_goal_y

        # Replan path if current goal is observed as obstacle or reached
        elif (
            (cur_step % self.replan_freq == 0) or 
            ((self.x, self.y) == (self.goal_x, self.goal_y)) or
            (self.pred_map[self.goal_x, self.goal_y] == 1)
        ): 
            # use the exploration model to get the next goal
            exp_output = self.exp_method.get_goal({
                "x": self.x,
                "y": self.y,
                "pred_map": self.pred_map
            })

            if "abort" in exp_output and exp_output["abort"]:
                print(f"[ERROR] Aborting due to no frontiers found!!")
                return False

            self.goal_x, self.goal_y = exp_output["goal_x"], exp_output["goal_y"]
            if exp_output["roll_back"]:
                # move the agent to a free cell
                if len(self.path_history) > 0:
                    self.x, self.y = self.path_history[max(-len(self.path_history), -5)]

                # Update planner weights
                self.planner.update_weights()

            if "pred_occ" in exp_output:
                pred_occ = exp_output["pred_occ"]

        vanilla_plan = True
        # Replan path using pred_occ if available
        if pred_occ is not None:
            tmp_planner = AstarPlanner(pred_occ)
            self.path = tmp_planner.plan_path((self.x, self.y), (self.goal_x, self.goal_y))

            if self.path is not None:
                vanilla_plan = False
        
        # Replan path using current pred_map
        if vanilla_plan:
            self.path = self.planner.plan_path((self.x, self.y), (self.goal_x, self.goal_y))

        # Abort if no path found
        if self.path is None:
            return False
        
        next_x, next_y = self.path[1]

        # Move to next position in planned path if free else replan
        if self.pred_map[next_x, next_y] == 0:
            
            # take action to move to next position
            angle_to_next = np.arctan2(self.x - next_x, next_y - self.y) % (2 * np.pi)
            angle_diff = (angle_to_next - self.theta) % (2 * np.pi)

            # Move to next position 
            if min(angle_diff, 2 * np.pi - angle_diff) < np.radians(10):
                self.x, self.y = next_x, next_y
                self.pathlen += 1

                # Check if reached gt_goal
                if (self.x, self.y) == (self.gt_goal_x, self.gt_goal_y):
                    return True

            # Turn towards next position
            else:
                turn_dir = 1 if angle_diff < np.pi else -1
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                mag = min(self.turn_angle, angle_diff)
                self.theta += turn_dir * mag
                self.theta = self.theta % (2 * np.pi)
        
        elif self.pred_map[next_x, next_y] == 0.5:
            # Rotate in place to get more sensor data
            self.theta -= self.turn_angle
            self.theta = self.theta % (2 * np.pi)
            self.cur_rotation += 1

            if self.cur_rotation >= self.max_rotation:
                self.cur_rotation = 0

                # check if next position is diagonal
                # if yes and adjacent cells are not free, mark as obstacle and replan
                if abs(self.x - next_x) == 1 or abs(self.y - next_y) == 1:
                    if self.pred_map[self.x, next_y] == 1 or self.pred_map[next_x, self.y] == 1:
                        self.pred_map[next_x, next_y] = 1
                        self.planner.update_obstacle(next_x, next_y)
                        self.path = self.planner.plan_path((self.x, self.y), (self.goal_x, self.goal_y))
        else:
            # Ideally should not reach here - next position should be free or unknown
            raise ValueError("Invalid next position in path!")

        # Abort if no path found
        if self.path is None:
            return False
        
        self.get_sensor_data()
        self.path_history.append((self.x, self.y))

        # Update planner with new observations [NOT NEEDED], as planner is updated in get_sensor_data
        # self.planner.update_map(self.pred_map)

        return None

    def plot_initial_map(self, ret_img=False):
        """Plots the initial map with waypoints, start position and path."""

        # Plot initial map with waypoints, start position and path
        img = cv2.cvtColor(self.occ_map * 255, cv2.COLOR_GRAY2BGR)

        # Plot waypoints
        pt = (self.gt_goal_x, self.gt_goal_y)
        cv2.circle(img, pt[::-1], 5, (0, 0, 255), -1)

        # Plot start position
        cv2.circle(img, (self.y, self.x), 5, (0, 255, 0), -1)

        # Plot path
        for i in range(len(self.gt_path) - 1):
            cv2.line(img, self.gt_path[i][::-1], self.gt_path[i + 1][::-1], (255, 0, 0), 2)
        
        if ret_img:
            return img

        pth = os.path.join(self.debug_dir, 'initial_map.png')
        cv2.imwrite(pth, img)
    

    def plot_debug_map(self, pred_img, idx, ret_img=False):
        """Plots the debug map with robot position and path.
        """
        img = cv2.cvtColor((pred_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.circle(img, (self.y, self.x), 5, (0, 255, 0), -1)
        
        # Plot path history
        for i in range(len(self.path_history) - 1):
            cv2.line(img, self.path_history[i][::-1], self.path_history[i + 1][::-1], (255, 0, 0), 2)

        # plot orientation
        x2 = int(self.x - 10 * np.sin(self.theta))
        y2 = int(self.y + 10 * np.cos(self.theta))
        cv2.line(img, (self.y, self.x), (y2, x2), (0, 255, 0), 2)

        if ret_img:
            return img

        pth = os.path.join(self.debug_dir, f'{idx}.png')
        cv2.imwrite(pth, img)
        

    def run_simulation(self):
        """Runs the simulation for a specified number of steps."""

        goal_reached = None

        # for idx in tqdm(range(self.max_steps)):
        for idx in range(self.max_steps):
            
            goal_reached = self.step(idx)

            if idx == 0:
                # Save initial map, goal and robot position, and path
                if self.debug:
                    self.plot_initial_map()
                if self.log:
                    img = self.plot_initial_map(ret_img=True)
                    wandb.log({f"{self.log_prefix}/initial_map": [wandb.Image(img)]})

            # Goal Reached or stuck during plan
            if goal_reached is not None:
                if goal_reached:
                    print(f"[LOG] Goal reached in {idx} steps!")
                else:
                    print(f"[LOG] Stuck at {idx} steps!")
                print(f"[LOG] Path length: {self.pathlen}, Best path length: {self.best_pathlen}")
                if self.log:
                    img = self.plot_debug_map(self.pred_map, idx, ret_img=True)
                    wandb.log({f"{self.log_prefix}/final_map": [wandb.Image(img)]})
                
                d2goal = np.sqrt((self.x - self.gt_goal_x) ** 2 + (self.y - self.gt_goal_y) ** 2)
                spl = self.best_pathlen / max(self.best_pathlen, self.pathlen)
                spl = 0 if not goal_reached else spl
                return {
                    "SPL": spl,
                    "D2Goal": d2goal,
                    "SuccessRate": goal_reached,
                    "Time": idx
                }
            
            if self.debug_freq and self.debug_dir and idx % self.debug_freq == 0:
                # Plot robot position and path
                self.plot_debug_map(self.pred_map, idx)
                
        if self.log:
            img = self.plot_debug_map(self.pred_map, idx, ret_img=True)
            wandb.log({f"{self.log_prefix}/final_map": [wandb.Image(img)]})

        print("Simulation complete!")

        if goal_reached is None:
            print("Goal not reached! Max steps reached!")
            goal_reached = False

        d2goal = np.sqrt((self.x - self.gt_goal_x) ** 2 + (self.y - self.gt_goal_y) ** 2)
        spl = self.best_pathlen / max(self.best_pathlen, self.pathlen)
        spl = 0 if not goal_reached else spl

        return {
            "SPL": spl,
            "D2Goal": d2goal,
            "SuccessRate": goal_reached,
            "Time": idx
        }