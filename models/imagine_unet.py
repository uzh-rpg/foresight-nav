import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet2D

class UNet2D_GeoSem_Pred(UNet2D):
    """
    UNet2D model for predicting GeoSemMap.
    """
    
    def forward(self, input: torch.Tensor, inference: bool=False) -> torch.Tensor:
        """
        Forward pass of the UNet model.
        During inference, the model normalizes the predicted CLIP features and
        applies sigmoid activation to the occupancy and interior mask predictions.
        The output tensor is reshaped to [N, D+2, H, W] format.
        Args:
            input: Input GeoSemMap tensor of shape [N, D+2, H, W].
            inference: Boolean flag indicating whether to perform inference.
        """
        pred_geosem = self.forward_pass(input) # [N, D+2, H, W]

        if inference:
            pred_clip = pred_geosem[:, :512, :, :]
            pred_occ = pred_geosem[:, 512, :, :]
            pred_int_mask = pred_geosem[:, 513, :, :]

            pred_clip = nn.functional.normalize(pred_clip, p=2, dim=1)
            pred_occ = nn.functional.sigmoid(pred_occ)
            pred_occ = pred_occ.unsqueeze(1)
            pred_int_mask = nn.functional.sigmoid(pred_int_mask)
            pred_int_mask = pred_int_mask.unsqueeze(1)
            
            pred_geosem_inf = torch.cat([pred_clip, pred_occ, pred_int_mask], dim=1)
            return pred_geosem_inf

        else:
            return pred_geosem


class UNet2D_Occ_Pred(UNet2D):
    """
    UNet2D model for predicting occupancy map.
    """

    def forward(self, input: torch.Tensor, inference: bool=False) -> torch.Tensor:
        """
        Forward pass of the UNet model.
        During inference, the model applies sigmoid activation to the occupancy predictions.
        The output tensor is in the [N, 1, H, W] format.
        Args:
            input: Input tensor of shape [N, 1, H, W].
            inference: Boolean flag indicating whether to perform inference.
        
        Note: Use inference=False for inference of a model trained with
        the MSE loss.
        """
        pred_occ = self.forward_pass(input) # [N, 1, H, W]
        
        if inference:
            pred_occ = nn.functional.sigmoid(pred_occ)
            
        return pred_occ
