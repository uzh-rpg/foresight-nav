"""
Get the ground truth goal x,y from the current clipfeat_map and the given goal category.
"""
import numpy as np
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from omegaconf import OmegaConf
import wandb

from evaluation.clip_model import CLIP_Model

class GoalType(Enum):
    ARGMAX = 'argmax'
    HEATMAP = 'heatmap'


class GoalSetter:
    def __init__(self, conf: OmegaConf):
        self.conf = conf

        self.goal_type = GoalType(self.conf.get('goal_type', 'argmax'))
        if self.goal_type == GoalType.HEATMAP:
            self.heatmap_threshold = self.conf.get('heatmap_threshold', 0.95)

        self.random_state = self.conf.get('random_state', 0)
        self.random_seed = self.conf.get('random_seed', 0)

        self.gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=self.random_state)
        self.rng = np.random.default_rng(self.random_seed)


    def get_gt_goal(self, goal_category: str, occ_map:np.ndarray, clipfeat_map: np.ndarray, clip_model: CLIP_Model, ctop_down: np.ndarray):
        """
        Get the ground truth goal x,y from the current clipfeat_map and the given goal category.
        """
        if self.goal_type == GoalType.ARGMAX:
            return self.get_gt_goal_argmax(goal_category, occ_map, clipfeat_map, clip_model)
        elif self.goal_type == GoalType.HEATMAP:
            return self.get_gt_goal_heatmap(goal_category, occ_map, clipfeat_map, clip_model, ctop_down)
        
    
    def get_closest_free(self, gt_goal_x: int, gt_goal_y: int, occ_map: np.ndarray):
        """
        Find the closest free cell in the occupancy map to the goal.
        """
       # find the closest free cell in the occupancy map to the goal
        occ_map = occ_map.astype(bool)
        occ_map[gt_goal_x, gt_goal_y] = True
        free_cells = np.logical_not(occ_map)
        dist = np.linalg.norm(np.array(np.where(free_cells)).T - np.array([gt_goal_x, gt_goal_y]), axis=1)
        goal_idx = np.argmin(dist)
        goal_x, goal_y = np.where(free_cells)[0][goal_idx], np.where(free_cells)[1][goal_idx]

        return goal_x, goal_y
        

    def get_gt_goal_argmax(self, goal_category: str, occ_map:np.ndarray, clipfeat_map: np.ndarray, clip_model: CLIP_Model):
        """
        Get ground truth goal x,y from current clipfeat_map and goal_clip_feat.
        """
        goal_clip_feat = clip_model.get_text_feats([goal_category])
        map_feats = clipfeat_map.reshape(-1, clipfeat_map.shape[-1])
        goal_sim = map_feats @ goal_clip_feat.T
        goal_sim = goal_sim.reshape(clipfeat_map.shape[:2])

        goal_idx = np.argmax(goal_sim)
        gt_goal_x, gt_goal_y = np.unravel_index(goal_idx, clipfeat_map.shape[:2])

        goal_x, goal_y = self.get_closest_free(gt_goal_x, gt_goal_y, occ_map)

        return goal_x, goal_y


    def get_gt_goal_heatmap(self, goal_category: str, occ_map:np.ndarray, clipfeat_map: np.ndarray, clip_model: CLIP_Model, ctop_down: np.ndarray):
        """
        Localize the hot region of the similarity scores between the CLIP-Embedded Floorplan and the
        Open Vocabulary text query. Represent the hot region as an ellipse fitted to the hot coordinates.
        Visualize the similarity heatmap and the fitted ellipse on the top-down RGB map.
        """

        text_feats = clip_model.get_text_feats([goal_category])

        map_feats = clipfeat_map.reshape((-1, clipfeat_map.shape[-1]))
        scores_list = map_feats @ text_feats.T
        scores = scores_list.reshape(clipfeat_map.shape[:2])

        # Normalize scores to [0, 1] for better visualization
        scores = (scores - scores.min()) / (scores.max() - scores.min())

        # Apply a threshold to find the hot region
        hot_region = scores >= self.heatmap_threshold

        # Extract coordinates of the hot region
        hot_coords = np.column_stack(np.where(hot_region))

        # Check if visualization is enabled
        vis = ctop_down is not None

        if len(hot_coords) > 1:
            # Fit a Gaussian Mixture Model to the hot coordinates
            self.gmm.fit(hot_coords)

            # Get the parameters of the fitted Gaussian
            means = self.gmm.means_
            covariances = self.gmm.covariances_

            # Choose a random mean for the goal
            goal_mean = self.rng.choice(means)
            gt_goal_x, gt_goal_y = goal_mean
            gt_goal_x, gt_goal_y = int(gt_goal_x), int(gt_goal_y)
            gt_goal_x, gt_goal_y = self.get_closest_free(gt_goal_x, gt_goal_y, occ_map)

            if not vis:
                return gt_goal_x, gt_goal_y

            fig = plt.figure(figsize=(12,6))

            # Generate a grid of coordinates
            x, y = np.meshgrid(np.arange(scores.shape[1]), np.arange(scores.shape[0]))
            xy = np.column_stack([y.ravel(), x.ravel()])

            # Compute the PDF of the Gaussian at each point in the grid
            pdf = np.exp(self.gmm.score_samples(xy))
            pdf = pdf.reshape(scores.shape)

            # Normalize the PDF to [0, 1] for visualization
            pdf = (pdf - pdf.min()) / (pdf.max() - pdf.min())

            # Create an elliptical mask
            ellipse_mask = np.ones_like(pdf)


            # Plot the Gaussian as an ellipse
            for mean, covar in zip(means, covariances):
                eigenvalues, eigenvectors = np.linalg.eigh(covar)
                order = eigenvalues.argsort()[::-1]
                eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                angle = 90 - np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                width, height = 2 * np.sqrt(eigenvalues)

                ellipse_patch = Ellipse(mean[::-1], width, height, angle=angle, edgecolor='red', facecolor='none', lw=2, alpha=0.5)                
                plt.gca().add_patch(ellipse_patch)

                # Create grid of coordinates
                x, y = np.meshgrid(np.arange(scores.shape[1]), np.arange(scores.shape[0]))

                # Rotate coordinates to align with ellipse angle
                x_rot = x - mean[1]
                y_rot = y - mean[0]
                cos_angle = np.cos(np.radians(angle))
                sin_angle = np.sin(np.radians(angle))
                x_new = x_rot * cos_angle + y_rot * sin_angle
                y_new = -x_rot * sin_angle + y_rot * cos_angle

                # Compute ellipse equation
                ellipse_eq = (((x_new*2) / width) ** 2 + ((y_new*2) / height) ** 2 <= 1)

                # Apply the ellipse mask
                ellipse_mask = np.logical_and(ellipse_mask, ellipse_eq)

            ellipse_mask = np.logical_not(ellipse_mask)
            
            # Plot the top-down RGB map
            plt.imshow(ctop_down)

            # Overlay the heatmap using seaborn for better visualization
            sns.heatmap(pdf, cmap='jet', alpha=0.5, zorder=2, mask=ellipse_mask, cbar=False)

            plt.colorbar(cm.ScalarMappable(cmap='jet'), label='Similarity Score', ax=plt.gca())

            # Plot the gt goal
            plt.scatter(gt_goal_y, gt_goal_x, color='red', marker='x', s=100, zorder=3)

            plt.axis('off')
            plt.title(f"Goal Heatmap: {goal_category}")

            # Log the visualization to wandb
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            wandb.log({f"goal_heatmap_{goal_category}": wandb.Image(img)}, commit=False)

            plt.close()
    
        else:
            print("[GT Goal Setter] No hot region found, no goal set.")
            gt_goal_x, gt_goal_y = None, None

        return gt_goal_x, gt_goal_y

    

