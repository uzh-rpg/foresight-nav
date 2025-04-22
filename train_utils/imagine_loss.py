from enum import Enum
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccLossType(Enum):
    BINARY_CROSS_ENTROPY = 'binary_cross_entropy'
    MSE = 'mse'

class ClipLossType(Enum):
    COSINE_SIM = 'cosine_sim'
    CATEGORY_COSINE_SIM = 'category_cosine_sim'
    MSE = 'mse'


def clip_loss(
        pred_clip: torch.Tensor,
        target_clip: torch.Tensor,
        clip_loss_type: ClipLossType,
        category_wts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
    """
    Cosine Similarity loss between predicted and target embeddings (for non-zero norm pixels).
    MSE loss for zero norm pixels.
    pred: [N, D, H, W]
    target: [N, D, H, W]
    """
    losses = {}

    if clip_loss_type == 'cosine_sim' or clip_loss_type == 'cat_cosine_sim':
        pred_norm = nn.functional.normalize(pred_clip, p=2, dim=1)
        target_norm = nn.functional.normalize(target_clip, p=2, dim=1)

        norm_gt = torch.norm(target_clip, p=2, dim=1)

        mask_cosine = (norm_gt > 0)
        mask_mse = ~mask_cosine

        # compute cosine similarity loss
        cosine_similarity = torch.sum(pred_norm * target_norm, dim=1)
        cosine_loss_raw = 1 - cosine_similarity

        if clip_loss_type == ClipLossType.CATEGORY_COSINE_SIM:
            cosine_loss = cosine_loss_raw * category_wts
            losses['cosine_loss_raw'] = cosine_loss_raw[mask_cosine].mean()
        else:
            cosine_loss = cosine_loss_raw
        
        # Apply the cosine loss where norm(gt) > 0
        loss = torch.where(mask_cosine, cosine_loss, 0.0)

        # Calculate the mean of cosine loss
        mean_cosine_loss = cosine_loss[mask_cosine].mean()
        losses['cosine_loss'] = mean_cosine_loss

        # Apply the MSE loss where norm(gt) = 0
        mse_loss = ((pred_clip - target_clip) ** 2).mean(dim=1)
        
        # Apply the mse loss where norm(gt) == 0
        loss = torch.where(mask_mse, mse_loss, loss)

        # Calculate the mean of MSE loss
        mean_mse_loss = mse_loss[mask_mse].mean()
        losses['mse_loss'] = mean_mse_loss

        # Calculate the total loss
        losses['clip_loss'] = mean_cosine_loss

        return losses
    
    elif clip_loss_type == ClipLossType.MSE:
        # Use MSE loss for all pixels
        mse_loss = ((pred_clip - target_clip) ** 2).mean(dim=1)
        mean_mse_loss = mse_loss.mean()
        losses['mse_loss'] = mean_mse_loss
        losses['clip_loss'] = mean_mse_loss

        return losses
    
    else:
        raise ValueError(f"Unsupported clip loss type: {clip_loss_type}."
                         "Supported types are: {ClipLossType.COSINE_SIM}, "
                         "{ClipLossType.CATEGORY_COSINE_SIM}, {ClipLossType.MSE}")
    

def geosem_imagine_loss(
        pred_geosem: torch.Tensor,
        target_geosem: torch.Tensor,
        clip_loss_type: ClipLossType,
        occ_wt: Optional[float] = 1.0,
    ) -> Dict[str, torch.Tensor]:
    """
    CLIP loss for clip embeddings, BCE loss for occupancy map and internal mask.
    pred: [N, D+2, H, W]
    target: [N, D+2, H, W]
    """
    losses = {}
    loss = 0

    pred_clip = pred_geosem[:, :512, :, :]
    pred_occ = pred_geosem[:, 512, :, :]
    pred_int_mask = pred_geosem[:, 513, :, :]

    target_clip = target_geosem[:, :512, :, :]
    target_occ = target_geosem[:, 512, :, :]
    target_int_mask = target_geosem[:, 513, :, :]

    if clip_loss_type == ClipLossType.CATEGORY_COSINE_SIM:
        category_wts = target_geosem[:, 514, :, :] 
    else:
        category_wts = None

    # Loss for clip map
    clip_losses = clip_loss(pred_clip, target_clip, clip_loss_type, category_wts)
    losses.update(clip_losses)

    loss += losses['clip_loss']

    # We use the MSE loss just for monitoring purposes
    # We do not backprop through it. Uncomment to ablate.
    # loss += losses['cosine_loss'] + losses['mse_loss']

    # Loss for occupancy map
    occ_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_occ,
        target_occ,
        pos_weight=torch.tensor(occ_wt)
    )
    losses['occ_loss'] = occ_loss
    loss += occ_loss

    # Loss for internal mask
    int_mask_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_int_mask, target_int_mask)
    losses['int_mask_loss'] = int_mask_loss
    loss += int_mask_loss

    losses['summed_loss'] = loss

    return losses


def occ_imagine_loss(
    pred_occ: torch.Tensor,
    target_occ: torch.Tensor,
    occ_loss_type: OccLossType,
    occ_wt: Optional[float] = 1.0,
    ) -> Dict[str, torch.Tensor]:
    """
    Loss for only occupancy map prediction.
    pred: [N, 1, H, W]
    target: [N, 1, H, W]
    """
    losses = {}

    if occ_loss_type == OccLossType.BINARY_CROSS_ENTROPY:
        loss = nn.functional.binary_cross_entropy_with_logits(
            pred_occ, target_occ, pos_weight=occ_wt)
    elif occ_loss_type == OccLossType.MSE:
        loss = nn.functional.mse_loss(pred_occ, target_occ, reduction='none')
        loss = loss[target_occ > 0].mean() * occ_wt 
        loss += loss[target_occ == 0].mean()

    else:
        raise ValueError(f"Unsupported occ loss type: {occ_loss_type}."
                         "Supported types are: {OccLossType.BINARY_CROSS_ENTROPY}, "
                         "{OccLossType.MSE}")
    losses['summed_loss'] = loss
    return losses
