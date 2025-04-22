"""
Parts of this file and the code from 'third_party/lseg' is originally 
from the VLMaps project:
https://github.com/vlmaps/vlmaps

Original Author(s): Huang et al.
Licensed under the MIT License.

Modifications may have been made in this version.
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import clip

from datagen_imagination.geosem_map_generation.third_party.lseg.modules.models.lseg_net import LSegEncNet
from datagen_imagination.geosem_map_generation.third_party.lseg.additional_utils.models import (
    resize_image, pad_image, crop_image
)
from datagen_imagination.geosem_map_generation.utils.viz_utils import (
    get_new_pallete, get_new_mask_pallete)

class LSegEncoder():
    def __init__(
            self,
            ckpt_path,
            clip_version="ViT-B/32",
            crop_size=480,
            base_size=640,
        ):

        self.crop_size = crop_size # 480
        self.base_size = base_size # 520
        lang = "door,chair,ground,ceiling,other"
        self.labels = lang.split(",")
        self.clip_version = clip_version
        
        # loading models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        self.device = device

        # initialize CLIP model
        self.init_clip_model()
        
        model = LSegEncNet(
            self.labels,
            arch_option=0,
            block_depth=0,
            activation='lrelu',
            crop_size=self.crop_size
        )
        model_state_dict = model.state_dict()
        pretrained_state_dict = torch.load(ckpt_path, map_location=device)
        pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

        model.eval()
        self.model = model.cuda()

        self.norm_mean= [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]
        self.padding = [0.0] * 3
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    
    def get_lseg_feat(
            self,
            image: np.array,
            vis=False
        ):
        
        vis_image = image.copy()
        image = self.transform(image).unsqueeze(0).cuda()
        img = image[0].permute(1,2,0)
        img = img * 0.5 + 0.5
        
        batch, _, h, w = image.size()
        stride_rate = 2.0/3.0
        stride = int(self.crop_size * stride_rate)

        long_size = self.base_size
        if h > w:
            height = long_size
            width = int(1.0 * w * long_size / h + 0.5)
            short_size = width
        else:
            width = long_size
            height = int(1.0 * h * long_size / w + 0.5)
            short_size = height


        cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

        if long_size <= self.crop_size:
            pad_img = pad_image(cur_img, self.norm_mean,
                                self.norm_std, self.crop_size)
            print(pad_img.shape)
            with torch.no_grad():
                outputs, logits = self.model(pad_img, self.labels)
            outputs = crop_image(outputs, 0, height, 0, width)
        else:
            if short_size < self.crop_size:
                # pad if needed
                pad_img = pad_image(cur_img, self.norm_mean,
                                    self.norm_std, self.crop_size)
            else:
                pad_img = cur_img
            _,_,ph,pw = pad_img.shape #.size()
            assert(ph >= height and pw >= width)
            h_grids = int(math.ceil(1.0 * (ph-self.crop_size)/stride)) + 1
            w_grids = int(math.ceil(1.0 * (pw-self.crop_size)/stride)) + 1
            with torch.cuda.device_of(image):
                with torch.no_grad():
                    outputs = image.new().resize_(batch, self.model.out_c,ph,pw).zero_().cuda()
                    logits_outputs = image.new().resize_(batch, len(self.labels),ph,pw).zero_().cuda()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
            # grid evaluation
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + self.crop_size, ph)
                    w1 = min(w0 + self.crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # pad if needed
                    pad_crop_img = pad_image(crop_img, self.norm_mean,
                                                self.norm_std, self.crop_size)
                    with torch.no_grad():
                        output, logits = self.model(pad_crop_img, self.labels)
                    cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                    cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                    outputs[:,:,h0:h1,w0:w1] += cropped
                    logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                    count_norm[:,:,h0:h1,w0:w1] += 1
            assert((count_norm==0).sum()==0)
            outputs = outputs / count_norm
            logits_outputs = logits_outputs / count_norm
            outputs = outputs[:,:,:height,:width]
            logits_outputs = logits_outputs[:,:,:height,:width]
        outputs = outputs.cpu()
        outputs = outputs.numpy() # B, D, H, W
        predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
        pred = predicts[0]

        vis_fig = None
        if vis:
            new_palette = get_new_pallete(len(self.labels))
            mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=self.labels)
            seg = mask.convert("RGBA")
            
            # show image and segmentation side by side
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(vis_image)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(seg)
            plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 8})
            plt.axis("off")
            plt.tight_layout()

            vis_fig = fig.canvas.draw()
            fig_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            fig_data = fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            vis_fig = fig_data
            plt.close(fig)

        return outputs, vis_fig
    
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
