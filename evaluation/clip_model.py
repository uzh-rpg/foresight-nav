"""
CLIP model to get text features for categories and calculate class-to-class similarity matrix.
"""

import torch
import clip
import numpy as np

from evaluation.categories import coco_categories, objectnav_categories

class CLIP_Model:
    """
    CLIP model to get text features for categories and calculate similarity matrix.
    """
    def __init__(self, clip_version='ViT-B/32'):
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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = clip_models[self.clip_version]
        clip_model, preprocess = clip.load(self.clip_version)
        clip_model.to(self.device).eval()
        self.clip_model = clip_model

        self.get_c2c_matrix()

        # clip features for coco categories + "other"
        self.coco_feat_other = self.get_text_feats(coco_categories + ["other"])
    
    def get_text_feats(self, in_text, batch_size=64):
        text_tokens = clip.tokenize(in_text).cuda()
        text_id = 0
        text_feats = np.zeros((len(in_text), self.clip_feat_dim), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)
            text_batch = text_tokens[text_id : text_id + batch_size]
            with torch.no_grad():
                batch_feats = self.clip_model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            text_feats[text_id : text_id + batch_size, :] = batch_feats
            text_id += batch_size
        return text_feats
    
    def get_c2c_matrix(self):
        """
        Get the clip to clip similarity matrix for coco categories.
        """
        text_feats = self.get_text_feats(coco_categories)
        c2c_matrix = np.matmul(text_feats, text_feats.T)
        self.c2c_matrix = c2c_matrix
        self.c2c_matrix /= np.linalg.norm(c2c_matrix, axis=1, keepdims=True)

        # add 'other' category to c2c_matrix: -1 similarity with all other categories
        other_sim = np.ones((len(coco_categories) + 1, len(coco_categories) + 1)) * -1
        other_sim[:-1, :-1] = c2c_matrix
        self.c2c_matrix_other = other_sim        
