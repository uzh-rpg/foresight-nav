"""
Load GeoSem Map and visualize the heatmap of the
similarity scores between the CLIP-Feature map and the 
Open Vocabulary text query.

Example Usage:
    python -m datagen_imagination.geosem_map_generation.viz_openlang_heatmap \
        --root_path=/scratch/hshah/ForeSightDataset/Structured3D/scene_00000
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import clip
import torch
import argparse

from PIL import Image
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from datagen_imagination.geosem_map_generation.utils.clip_utils import get_text_feats
from datagen_imagination.geosem_map_generation.utils.mp3dcat import mp3dcat
from datagen_imagination.geosem_map_generation.utils.viz_utils import get_new_pallete, get_new_mask_pallete

class GeoSemMap_Query_Visualizer():
    def __init__(self, root_path, clip_version="ViT-B/32", vis_path=None):
        
        # load GeoSemMap and other data
        self.root_path = root_path
        print(f'Loading CLIP-Embedded Floorplan data from {self.root_path}')
        self.clipfeat_map = self.load_npy(os.path.join(self.root_path, 'clipfeat_map.npy'))
        self.obstacles = self.load_npy(os.path.join(self.root_path, 'obstacles.npy'))
        self.c_top_down = self.load_npy(os.path.join(self.root_path, 'color_top_down.npy'))
        self.vis_obs = self.load_npy(os.path.join(self.root_path, 'vis_obs.npy'))
        self.weight = self.load_npy(os.path.join(self.root_path, 'weight.npy'))
        self.grid_size = self.clipfeat_map.shape[0]

        # visualization path
        if vis_path is None:
            self.vis_path = os.path.join(self.root_path, 'vis')
        else:
            self.vis_path = vis_path

        # load CLIP model for text embeddings of language queries
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_version = clip_version
        self.init_clip_model()

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

    def load_npy(self, path):
        with open(path, 'rb') as f:
            mat = np.load(f)
        return mat
    
    def visualize_topdown_semantic(self, query, mp3d_cat=False):
        obstacles = np.logical_not(self.vis_obs).astype(np.uint8)
        no_map_mask = obstacles > 0

        if mp3d_cat:
            lang = mp3dcat 
        else:
            if ',' in query:
                lang = query.split(',')
            else:
                # single category, add other categories
                lang = [query]
                lang.extend(mp3dcat)

        text_feats = get_text_feats(
            lang,
            self.clip_model,
            self.clip_feat_dim,
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
        plt.title("Top Down Semantic Segmentation")
        plt.imshow(seg)
        plt.savefig(os.path.join(self.vis_path, f'TopDown_Semantic_{query}.png'))
        plt.close()
    
    def visualize_heatmap(self, query):
        """
        Visualize the hot region of the similarity scores between the CLIP-Embedded Floorplan and the
        Open Vocabulary text query. Represent the hot region as an ellipse fitted to the hot coordinates.
        """

        text_feats = get_text_feats(
            [query],
            self.clip_model,
            self.clip_feat_dim,
        )

        grid = self.clipfeat_map
        map_feats = grid.reshape((-1, grid.shape[-1]))
        scores_list = map_feats @ text_feats.T
        scores = scores_list.reshape((self.grid_size, self.grid_size))

        # Normalize scores to [0, 1] for better visualization
        scores = (scores - scores.min()) / (scores.max() - scores.min())

        # Apply a threshold to find the hot region
        threshold = 0.95  # Adjust this threshold value as needed
        hot_region = scores >= threshold

        # Extract coordinates of the hot region
        hot_coords = np.column_stack(np.where(hot_region))

        if hot_coords.size > 0:
            # Fit a Gaussian Mixture Model to the hot coordinates
            gmm = GaussianMixture(n_components=1, covariance_type='full')
            gmm.fit(hot_coords)

            # Get the parameters of the fitted Gaussian
            means = gmm.means_
            covariances = gmm.covariances_

            # Generate a grid of coordinates
            x, y = np.meshgrid(np.arange(scores.shape[1]), np.arange(scores.shape[0]))
            xy = np.column_stack([y.ravel(), x.ravel()])

            # Compute the PDF of the Gaussian at each point in the grid
            pdf = np.exp(gmm.score_samples(xy))
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
            plt.imshow(self.c_top_down)

            # Overlay the heatmap using seaborn for better visualization
            sns.heatmap(pdf, cmap='jet', alpha=0.5, zorder=2, mask=ellipse_mask, cbar=False)

            plt.colorbar(cm.ScalarMappable(cmap='jet'), label='Similarity Score', ax=plt.gca())
        else:
            print("No hot region found.")

        plt.axis('off')
        plt.title(f"Similarity Heatmap: {query}")
        plt.savefig(os.path.join(self.vis_path, f'Heatmap_{query}.png'))
        plt.close()
    
    def visualize_heatmap_multiple_zones(self, query):
        """
        Visualize the hot region of the similarity scores between the CLIP-Embedded Floorplan and the
        Open Vocabulary text query. Represent the hot region as an ellipse fitted to the hot coordinates.
        """

        text_feats = get_text_feats(
            [query],
            self.clip_model,
            self.clip_feat_dim,
        )

        grid = self.clipfeat_map
        map_feats = grid.reshape((-1, grid.shape[-1]))
        scores_list = map_feats @ text_feats.T
        scores = scores_list.reshape((self.grid_size, self.grid_size))

        # Normalize scores to [0, 1] for better visualization
        scores = (scores - scores.min()) / (scores.max() - scores.min())

        # Apply a threshold to find the hot region
        threshold = 0.95  # Adjust this threshold value as needed
        hot_region = scores >= threshold

        # Extract coordinates of the hot region
        hot_coords = np.column_stack(np.where(hot_region))

        # scaling
        # scaler = StandardScaler()
        # hot_coords_norm = scaler.fit_transform(hot_coords)

        """
        # Outlier removal using DBSCAN
        dbscan = DBSCAN(eps=10, min_samples=100)
        labels = dbscan.fit_predict(hot_coords)

        # Filter out the outliers
        hot_coords = hot_coords[labels != -1]
        """

        if hot_coords.size == 0:
            print(f"No hot region found for query: {query}")
            return
        """
        # Find the optimal number of components
        lowest_bic = np.inf
        n_components_range = range(1, min(10, hot_coords.shape[0]))
        gmm = None

        for n_components in n_components_range:
            cur_gmm = GaussianMixture(n_components=n_components, covariance_type='full')
            cur_gmm.fit(hot_coords)
            bic = cur_gmm.bic(hot_coords)
            if bic < lowest_bic:
                lowest_bic = bic
                gmm = cur_gmm
        print(f"Optimal number of clusters: {gmm.n_components}")
        """
        # """
        # Determine the optimal number of clusters using the elbow method or silhouette score
        range_n_clusters = range(2, min(10, hot_coords.shape[0]))
        best_n_clusters = 2
        best_silhouette_score = -1

        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(hot_coords)
            silhouette_avg = silhouette_score(hot_coords, kmeans.labels_)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_n_clusters = n_clusters

        # Fit a Gaussian Mixture Model to the hot coordinates
        gmm = GaussianMixture(n_components=best_n_clusters, covariance_type='full', random_state=0)
        gmm.fit(hot_coords)

        # if best_n_clusters == 2:
        #     # club the two clusters if they are too close
        #     means = gmm.means_
        #     if np.linalg.norm(means[0] - means[1]) < 100:
        #         best_n_clusters = 1
        #         gmm = GaussianMixture(n_components=best_n_clusters, covariance_type='full')
        #         gmm.fit(hot_coords)

        print(f"Optimal number of clusters: {best_n_clusters}")
        # """
        # Get the parameters of the fitted Gaussian
        means = gmm.means_
        covariances = gmm.covariances_

        # Generate a grid of coordinates
        x, y = np.meshgrid(np.arange(scores.shape[1]), np.arange(scores.shape[0]))
        xy = np.column_stack([y.ravel(), x.ravel()])

        # Compute the PDF of the Gaussian at each point in the grid
        pdf = np.exp(gmm.score_samples(xy))
        pdf = pdf.reshape(scores.shape)

        # Normalize the PDF to [0, 1] for visualization
        pdf = (pdf - pdf.min()) / (pdf.max() - pdf.min())

        # Create an elliptical mask
        ellipse_mask = np.zeros_like(pdf)


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

            # if pdf[ellipse_eq].mean() < 0.5:
            #     continue

            # normalize the pdf in the ellipse region
            pdf[ellipse_eq] = (pdf[ellipse_eq] - pdf[ellipse_eq].min()) / (pdf[ellipse_eq].max() - pdf[ellipse_eq].min())

            # Apply the ellipse mask
            ellipse_mask[ellipse_eq] = 1

        ellipse_mask = np.logical_not(ellipse_mask)
        
        # Plot the top-down RGB map
        plt.imshow(self.c_top_down)

        # Overlay the heatmap using seaborn for better visualization
        sns.heatmap(pdf, cmap='jet', alpha=0.5, zorder=2, mask=ellipse_mask, cbar=True, label='Similarity Score')
        # sns.heatmap(pdf, cmap='jet', alpha=0.5, zorder=2, cbar=True)

        # plt.colorbar(cm.ScalarMappable(cmap='jet'), label='Similarity Score', ax=plt.gca())

        plt.axis('off')
        plt.title(f"Similarity Heatmap: {query}")
        plt.savefig(os.path.join(self.vis_path, f'Heatmap_{query}.png'))
        plt.close()

    def visualize_raw_scores(self, query):
        """
        Visualize the similarity scores between the CLIP-Embedded Floorplan and the
        Open Vocabulary text query. Overlay the heatmap on the top-down view of the floorplan.
        """

        text_feats = get_text_feats(
            [query],
            self.clip_model,
            self.clip_feat_dim,
        )

        grid = self.clipfeat_map
        map_feats = grid.reshape((-1, grid.shape[-1]))
        scores_list = map_feats @ text_feats.T
        scores = scores_list.reshape((self.grid_size, self.grid_size))

        valid_mask = scores > 0

        # Normalize scores to [0, 1] for better visualization
        scores[valid_mask] = (scores[valid_mask] - scores[valid_mask].min()) / \
            (scores[valid_mask].max() - scores[valid_mask].min())

        # Plot the top-down RGB map
        plt.imshow(self.c_top_down)

        # Overlay the heatmap using seaborn for better visualization
        sns.heatmap(scores, cmap='jet', alpha=0.5, zorder=2, cbar=False, mask=np.logical_not(valid_mask))

        plt.colorbar(cm.ScalarMappable(cmap='jet'), label='Similarity Score', ax=plt.gca())
        plt.axis('off')
        plt.title(f"Raw Similarity Scores: {query}")
        plt.savefig(os.path.join(self.vis_path, f'RawScores_{query}.png'))
        plt.close()

def config():
    parser = argparse.ArgumentParser(description='Visualize a language quey localized in a GeoSemMap')
    parser.add_argument('--root_path', type=str, required=True, help='Path to the GeoSemMap data')
    parser.add_argument('--clip_version', type=str, default='ViT-B/32', help='CLIP model version')
    parser.add_argument('--query', type=str, default=None, help='Open Vocabulary text query')
    parser.add_argument('--vis_path', type=str, default=None, help='Path to save the visualization')
    args = parser.parse_args()
    return args

def main():
    args = config()
    viz = GeoSemMap_Query_Visualizer(args.root_path, args.clip_version, args.vis_path)
    query = args.query
    if query is None:
        query = input("Enter Open Vocabulary text query: ")

    # viz.visualize_topdown_semantic(query, mp3d_cat=True)
    # viz.visualize_heatmap(query)
    viz.visualize_heatmap_multiple_zones(query)
    viz.visualize_raw_scores(query)

if __name__ == "__main__":
    main()