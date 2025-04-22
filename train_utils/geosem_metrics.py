import numpy as np
import torch
from torchmetrics.segmentation import MeanIoU

def topdown_semantic_iou(
        pred_geosem: torch.Tensor,
        gt_geosem: torch.Tensor,
        text_feats: np.ndarray,
    ):
    """
    Calculate IoU of the top-down semantic segmentation of the
    predicted and ground truth GeoSem maps.
    """
    iou_metric = MeanIoU(num_classes=text_feats.shape[0])
    ious = []

    pred_clip = pred_geosem[:, :512, :, :].detach().cpu().numpy().transpose(0, 2, 3, 1)
    gt_clip = gt_geosem[:, :512, :, :].detach().cpu().numpy().transpose(0, 2, 3, 1)
    for b in range(pred_clip.shape[0]):
        map_feats = pred_clip[b].reshape(-1, pred_clip.shape[-1])
        scores = map_feats @ text_feats.T
        pred_sem = np.argmax(scores, axis=1)
        pred_sem = pred_sem.reshape(pred_clip[b].shape[:2])

        map_feats = gt_clip[b].reshape(-1, gt_clip.shape[-1])
        scores = map_feats @ text_feats.T
        gt_sem = np.argmax(scores, axis=1)
        gt_sem = gt_sem.reshape(gt_clip[b].shape[:2])

        cur_iou = iou_metric(torch.tensor(pred_sem), torch.tensor(gt_sem))
        ious.append(cur_iou)
    ious = torch.stack(ious)
    return ious
