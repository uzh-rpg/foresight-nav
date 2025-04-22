import warnings
import random
import argparse
import os
from tqdm import tqdm
import pickle

import numpy as np
import cv2
import networkx

from datagen_imagination.occupancy_generation.stru3d_utils import invalid_scenes_ids

# set random seed
random.seed(0)
np.random.seed(0)

class AstarPlanner:
    def __init__(self, cur_map):

        # Initialize current map (fully unobserved at start)
        self.cur_map = cur_map

        # Create NetworkX graph representation of the map
        self.G = networkx.grid_2d_graph(*cur_map.shape)

        # Add diagonal edges
        for i in range(cur_map.shape[0] - 1):
            for j in range(cur_map.shape[1] - 1):
                self.G.add_edge((i, j), (i + 1, j + 1))
                self.G.add_edge((i + 1, j), (i, j + 1))

        for edge in self.G.edges():
            self.G.edges[edge]['weight'] = 1
        
        self.update_weights()

    def plan_path(self, start, goal):
        try:
            if start == goal:
                return [start, goal]
            path = networkx.shortest_path(self.G, start, goal, weight='weight')
            return path
        except (networkx.NetworkXNoPath, ValueError):
            # print("No path found to goal!")
            return None
    
    def update_map(self, new_map):
        self.cur_map = new_map
        self.update_weights()

    def update_weights(self):
        rm_edges = []
        for edge in self.G.edges():
            x1, y1 = edge[0]
            x2, y2 = edge[1]
            if self.cur_map[x1, y1] == 1 or self.cur_map[x2, y2] == 1:
                # remove edge if any of the nodes is an obstacle
                rm_edges.append(edge)
        self.G.remove_edges_from(rm_edges)
    
    def update_obstacle(self, i, j):
        cur_node = (i, j)
        rm_edges = []
        for edge in self.G.edges(cur_node):
            rm_edges.append(edge)
        self.G.remove_edges_from(rm_edges)
    
    def plan_waypoints(self, waypoints):
        path = []
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            cur_path = self.plan_path(start, goal)
            if cur_path is None:
                return path
            path += cur_path
        return path

class Robot:
    def __init__(self, scene_path, args):

        # Initialize scene data
        self.scene_path = scene_path
        occ_path = os.path.join(self.scene_path, 'occupancy_map.png')
        gt_occ_path = os.path.join(self.scene_path, 'gt_occupancy_map.png')
        map_data_path = os.path.join(self.scene_path, 'map_data.pkl')

        self.occ_map = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE)
        self.occ_map = (self.occ_map // 255).astype(np.uint8)

        self.gt_map = cv2.imread(gt_occ_path, cv2.IMREAD_GRAYSCALE)
        self.gt_map = (self.gt_map // 255).astype(np.uint8)

        # Predicted map - initialize with unknown cells
        self.pred_map = np.ones_like(self.occ_map).astype(np.float32) * 0.5 
        
        # Load map data
        with open(map_data_path, 'rb') as f:
            self.map_data = pickle.load(f)

        # Initialize sensor parameters
        self.field_of_view = np.radians(args.field_of_view)
        self.save_freq = args.save_freq
        self.turn_angle = np.radians(args.turn_angle)
        self.act_prob = args.act_prob
        self.max_steps = args.max_steps

        # Turning parameters
        self.max_rotation = 2*np.pi / self.turn_angle
        self.cur_rotation = 0

        # Get sensor range in pixels
        self.sensor_range = args.sensor_range
        map_res = self.map_data['map_res']
        self.sensor_range = int(self.sensor_range / map_res)

        # Check save path
        if not args.save_path:
            self.save_path = os.path.join(self.scene_path, 'simulated_maps')
        else:
            self.save_path = args.save_path
        if os.path.exists(self.save_path): 
            os.system(f'rm -r {self.save_path}')
        os.makedirs(self.save_path)

        # Add debug mode paths
        if args.debug:
            self.debug_path = os.path.join(self.scene_path, 'debug_maps')
            if os.path.exists(self.debug_path):
                os.system(f'rm -r {self.debug_path}')
            os.makedirs(self.debug_path)
        else:
            self.debug_path = None

        # Initialize A* planner
        self.planner = AstarPlanner(self.pred_map)
        self.gt_planner = AstarPlanner(self.occ_map)

         # Initialize robot orientation
        self.theta = random.uniform(0, 2*np.pi) # Angle with respect to x-axis (image width axis)

        # Initialize goal position
        self.goal_x, self.goal_y = self.set_random_free()

        # Get waypoints 
        self.get_waypoints()
        self.waypoint_idx = 1

        # plan path to the first waypoint
        self.path = self.planner.plan_path(self.waypoints[0], self.waypoints[1])
        self.path_idx = 0
    
    def set_random_free(self):
        """Sets random position using gt_map and occupancy_map."""

        valid_indices = np.where(np.logical_and(self.gt_map == 1, self.occ_map == 0))
        if len(valid_indices[0]) == 0:
            raise ValueError("No free space found in map!")
        idx = random.choice(range(len(valid_indices[0])))
        x, y = valid_indices[0][idx], valid_indices[1][idx]
        return x, y
    
    def get_waypoints(self):
        """Returns waypoints for the robot to follow."""
        # Get waypoints 
        self.room_rand_pts = self.map_data['room_random_points']
        self.room_centroids = self.map_data['room_centroids']
        self.waypoints = []

        assert len(self.room_rand_pts) == len(self.room_centroids)
        attempts = 0
        while len(self.waypoints) <= 3:
            self.x, self.y = self.set_random_free()
            self.goal_x, self.goal_y = self.set_random_free()
            self.waypoints = [(self.x, self.y)]
            for i in np.random.permutation(len(self.room_rand_pts)):

                # Remove waypoints to which path cannot be planned
                tmp_path = self.gt_planner.plan_path(self.waypoints[-1], tuple(self.room_centroids[i]))
                if tmp_path is None:
                    continue
                self.waypoints.append(tuple(self.room_centroids[i]))

                tmp_path = self.gt_planner.plan_path(self.waypoints[-1], tuple(self.room_rand_pts[i]))
                if tmp_path is None:
                    continue
                self.waypoints.append(tuple(self.room_rand_pts[i]))

            tmp_path = self.gt_planner.plan_path(self.waypoints[-1], (self.goal_x, self.goal_y))
            if tmp_path is not None:
                self.waypoints.append((self.goal_x, self.goal_y))

            attempts += 1
            if attempts > 3:
                break
        
        if len(self.waypoints) == 1:
            warnings.warn("No suitable waypoints found in map!")
            self.waypoints.append((self.goal_x, self.goal_y))

        self.gt_path = self.gt_planner.plan_waypoints(self.waypoints)

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
    
    def step(self):
        """Takes an action and updates the map.
        Action Space:
            0: Move forward
            1: Turn left
            2: Turn right
        """
        # self.get_sensor_data()
        # self.planner.update_map(self.pred_map)

        next_x, next_y = self.path[self.path_idx + 1]

        # Move to next position in planned path if free else replan
        if self.pred_map[next_x, next_y] == 0:
            
            # take action to move to next position
            angle_to_next = np.arctan2(self.x - next_x, next_y - self.y) % (2 * np.pi)
            angle_diff = (angle_to_next - self.theta) % (2 * np.pi)


            # Move to next position 
            if min(angle_diff, 2 * np.pi - angle_diff) < np.radians(10):

                self.x, self.y = next_x, next_y
                self.path_idx += 1

                # Check if reached waypoint
                if (self.x, self.y) == self.waypoints[self.waypoint_idx]:
                    self.waypoint_idx += 1

                    if self.waypoint_idx == len(self.waypoints):
                        return True
                    self.path = self.planner.plan_path((self.x, self.y), self.waypoints[self.waypoint_idx])
                    self.path_idx = 0
            
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
                        self.path = self.planner.plan_path((self.x, self.y), self.waypoints[self.waypoint_idx])
                        self.path_idx = 0

        # Replan path if next position is not free
        else:
            self.path = self.planner.plan_path((self.x, self.y), self.waypoints[self.waypoint_idx])
            self.path_idx = 0
        
        if self.path is None:
            return True
        self.get_sensor_data()
        # self.planner.update_map(self.pred_map)

        return False

    def plot_initial_map(self):
        """Plots the initial map with waypoints, start position and path."""

        # Plot initial map with waypoints, start position and path
        img = cv2.cvtColor(self.occ_map * 255, cv2.COLOR_GRAY2BGR)

        # Plot waypoints
        for pt in self.waypoints:
            cv2.circle(img, pt[::-1], 5, (0, 0, 255), -1)

        # Plot start position
        cv2.circle(img, (self.y, self.x), 5, (0, 255, 0), -1)

        # Plot path
        for i in range(len(self.gt_path) - 1):
            cv2.line(img, self.gt_path[i][::-1], self.gt_path[i + 1][::-1], (255, 0, 0), 2)
        
        pth = os.path.join(self.save_path, 'initial_map.png')
        cv2.imwrite(pth, img)
        

    def run_simulation(self):
        """Runs the simulation for a specified number of steps."""
        # for idx in tqdm(range(self.max_steps)):
        for idx in range(self.max_steps):
            goal_reached = self.step()

            if idx == 0:
                # Save initial map, goal and robot position, and path
                self.plot_initial_map()

            if goal_reached:
                print("Goal reached!")
                break

            if idx % self.save_freq == 0:
                # mask out unknown cells outside the room using gt_map
                # self.pred_map[np.logical_and((self.gt_map == 0),(self.pred_map==0))] = 0.5
                pred_img = (self.pred_map * 255).astype(np.uint8)

                if self.debug_path:
                    # Plot robot position and path

                    img = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                    cv2.circle(img, (self.y, self.x), 5, (0, 255, 0), -1)
                    for i in range(self.path_idx):
                        cv2.circle(img, self.path[i][::-1], 3, (0, 0, 255), -1)
                    for i in range(len(self.path) - 1):
                        cv2.line(img, self.path[i][::-1], self.path[i + 1][::-1], (255, 0, 0), 2)

                    # plot orientation
                    x2 = int(self.x - 10 * np.sin(self.theta))
                    y2 = int(self.y + 10 * np.cos(self.theta))
                    cv2.line(img, (self.y, self.x), (y2, x2), (0, 255, 0), 2)

                    pth = os.path.join(self.debug_path, f'{idx}.png')
                    cv2.imwrite(pth, img)
                

                pth = os.path.join(self.save_path, f'{idx}.png')
                cv2.imwrite(pth, pred_img)

        print("Simulation complete!")

def config():
    args = argparse.ArgumentParser(description='Robot Simulation')
    args.add_argument('--data_dir', type=str, help='Path to data directory', required=True)
    args.add_argument('--num_scenes', type=int, default=None, help='Number of scenes to run simulation')
    args.add_argument('--field_of_view', type=int, default=80, help='Field of view in degrees')
    args.add_argument('--sensor_range', type=int, default=5, help='Sensor range in meters')
    args.add_argument('--save_freq', type=int, default=20, help='Frequency to save map')
    args.add_argument('--turn_angle', type=int, default=30, help='Angle to turn during random walk')
    args.add_argument('--act_prob', type=float, default=0.5, help='Probability of taking action from planned path')
    args.add_argument('--max_steps', type=int, default=1000, help='Maximum number of steps to run simulation')
    args.add_argument('--save_path', type=str, default=None, help='Path to save predicted maps')
    args.add_argument('--debug', action='store_true', help='Debug mode')
    args = args.parse_args()

    return args

def main():
    args = config()

    data_dir = args.data_dir
    scene_list_path = os.path.join(data_dir, 'scene_list_occ.txt')
    with open(scene_list_path, 'r') as f:
        scenes = f.readlines()
        scenes = [scene.strip() for scene in scenes]

    if args.num_scenes:
        scenes = scenes[:args.num_scenes]

    for scene in tqdm(scenes):
        scene_path = os.path.join(data_dir, scene)
        scene_id = scene.split('_')[-1]
        if int(scene_id) in invalid_scenes_ids:
            print('skip invalid {}'.format(scene))
            continue
        
        if os.path.exists(os.path.join(scene_path, 'simulated_maps')):
            print('skip done {}'. format(scene))
            continue

        print(f"Running simulation for {scene_path.split('/')[-1]}")
        agent = Robot(scene_path, args)    
        agent.run_simulation()

    print("Done!")

if __name__ == "__main__":
    main()
    