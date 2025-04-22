from functools import partial
import torch
import torch.nn as nn

from models.third_party.mae.models_mae import MaskedAutoencoderViT


class MAE_GeoSem_Pred(MaskedAutoencoderViT):
    """
    MAE model for predicting GeoSemMap.
    """
    def forward(self, input: torch.Tensor, inference: bool=False) -> torch.Tensor:
        latent, mask, ids_restore = self.forward_encoder(input, mask_ratio=0.0)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*D]
        dim = self.out_chans
        pred_geosem = self.unpatchify_clip(pred, dim) # [N, D+2, H, W]
        
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


class MAE_Occ_Pred(MaskedAutoencoderViT):
    """
    MAE model for predicting occupancy map.
    """
    def forward(self, input: torch.Tensor, inference: bool=False) -> torch.Tensor:
        latent, mask, ids_restore = self.forward_encoder(input, mask_ratio=0.0)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*D]
        dim = self.out_chans
        pred_occ = self.unpatchify_clip(pred, dim) # [N, 1, H, W]
        
        if inference:
            pred_occ = nn.functional.sigmoid(pred_occ)
            
        return pred_occ
    

def mae_occ_vit_base_patch16_dec512d8b(**kwargs):
    model = MAE_Occ_Pred(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_geosem_vit_base_patch16_dec512d8b(**kwargs):
    model = MAE_GeoSem_Pred(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_geosem_vit_giga(**kwargs):
    model = MAE_GeoSem_Pred(
        patch_size=14, in_chans=513, out_chans=514, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


model_factory = {
    'mae_occ_vit_base_patch16_dec512d8b': mae_occ_vit_base_patch16_dec512d8b,
    'mae_geosem_vit_base_patch16_dec512d8b': mae_geosem_vit_base_patch16_dec512d8b,
    'mae_geosem_vit_giga': mae_geosem_vit_giga
}
