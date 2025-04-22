"""
Code to visualize a GeoSemMap as a top-down semantic map during training.
"""

from omegaconf import OmegaConf
import matplotlib
from matplotlib import pyplot as plt
import wandb

import cv2
from PIL import Image
import numpy as np
import torch

from datagen_imagination.geosem_map_generation.utils.viz_utils import get_new_pallete, get_new_mask_pallete
from datagen_imagination.geosem_map_generation.utils.mp3dcat import mp3dcat


def visualize_topdown_semantic(
        clip_fp: np.ndarray,
        text_feats: np.ndarray,
        vis_obs: np.ndarray = None,
        pred_mask: np.ndarray = None,
    ):

    no_map_mask = None
    if vis_obs is not None:
        obstacles = np.logical_not(vis_obs).astype(np.uint8)
        no_map_mask = obstacles > 0
    grid_size = clip_fp.shape[0]

    lang = mp3dcat 
    
    grid = clip_fp
    map_feats = grid.reshape((-1, grid.shape[-1]))
    scores_list = map_feats @ text_feats.T

    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((grid_size, grid_size))
    floor_mask = predicts == 2

    new_pallete = get_new_pallete(len(lang))
    mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang)
    seg = mask.convert("RGBA")
    seg = np.array(seg)

    if no_map_mask is not None:
        seg[no_map_mask] = [225, 225, 225, 255]

    # make floor mask white
    seg[floor_mask] = [225, 225, 225, 255]

    if pred_mask is not None:
        # make mask values gray
        seg[pred_mask == 1] = [127, 127, 127, 255]

    seg = Image.fromarray(seg)

    matplotlib.use("Agg")
    fig = plt.figure(figsize=(12,6), dpi=120)
    ax = fig.add_subplot(111)

    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
    ax.axis('off')
    plt.title("GeoSem Map: TopDown Semantic Segmentation")
    ax.imshow(seg)

    # convert plot to image
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return data

def log_geosem_maps(
        inp_geosem: torch.Tensor,
        pred_geosem: torch.Tensor,
        gt_geosem: torch.Tensor,
        text_feats: np.ndarray,
        phase: str,
        log_conf: OmegaConf,
        log_id: int,
    ):
    inp_geosem = inp_geosem.detach().cpu().numpy()
    gt_geosem = gt_geosem.detach().cpu().numpy()
    pred_geosem = pred_geosem.detach().cpu().numpy()

    # Get semantic visualization of clip map
    cur_clip = inp_geosem[:, :512, :, :].transpose(0, 2, 3, 1)
    gt_clip = gt_geosem[:, :512, :, :].transpose(0, 2, 3, 1)
    pred_clip = pred_geosem[:, :512, :, :].transpose(0, 2, 3, 1)

    vis_cur_clips = []
    vis_gt_clips = []
    vis_pred_clips = []

    bs = cur_clip.shape[0]
    n_imgs = min(bs, log_conf.num_log_imgs)

    for b in range(n_imgs):
        vis_cur_clip = visualize_topdown_semantic(cur_clip[b], text_feats)
        vis_gt_clip = visualize_topdown_semantic(gt_clip[b], text_feats)
        vis_pred_clip = visualize_topdown_semantic(pred_clip[b], text_feats)

        vis_cur_clips.append(vis_cur_clip)
        vis_gt_clips.append(vis_gt_clip)
        vis_pred_clips.append(vis_pred_clip)

    # occupancy
    cur_occ = inp_geosem[:n_imgs, 512:513, :, :].transpose(0, 2, 3, 1)
    gt_occ = gt_geosem[:n_imgs, 512:513, :, :].transpose(0, 2, 3, 1)
    pred_occ = pred_geosem[:n_imgs, 512:513, :, :].transpose(0, 2, 3, 1)
    pred_t = (pred_occ > log_conf.occ_thresh)

    cur_occ = (cur_occ * 255).astype('uint8')
    gt_occ = (gt_occ * 255).astype('uint8')
    pred_occ = (pred_occ * 255).astype('uint8')
    pred_t = (pred_t * 255).astype('uint8')

    # int_mask
    gt_int = gt_geosem[:n_imgs, 513:514, :, :].transpose(0, 2, 3, 1)
    pred_int = pred_geosem[:n_imgs, 513:514, :, :].transpose(0, 2, 3, 1)

    gt_int = (gt_int * 255).astype('uint8')
    pred_int = (pred_int * 255).astype('uint8')

    wandb.log({f"{phase}/sim_clip": [wandb.Image(img) for img in vis_cur_clips]}, step=log_id)
    wandb.log({f"{phase}/gt_clip": [wandb.Image(img) for img in vis_gt_clips]}, step=log_id)
    wandb.log({f"{phase}/pred_clip": [wandb.Image(img) for img in vis_pred_clips]}, step=log_id)
    wandb.log({f"{phase}/sim_occ": [wandb.Image(img) for img in cur_occ]}, step=log_id)
    wandb.log({f"{phase}/gt_occ": [wandb.Image(img) for img in gt_occ]}, step=log_id)
    wandb.log({f"{phase}/pred_occ": [wandb.Image(img) for img in pred_occ]}, step=log_id)
    wandb.log({f"{phase}/pred_occ_thresh": [wandb.Image(img) for img in pred_t]}, step=log_id)
    wandb.log({f"{phase}/gt_int": [wandb.Image(img) for img in gt_int]}, step=log_id)
    wandb.log({f"{phase}/pred_int": [wandb.Image(img) for img in pred_int]}, step=log_id)
