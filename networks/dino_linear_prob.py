# coding=utf-8
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import Dinov2Model
from transformers import AutoImageProcessor

logger = logging.getLogger(__name__)

def swish(x):
    return x * torch.sigmoid(x)

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class Decoder(nn.Module):
    def __init__(self, input_hidden_size):
        super().__init__()
        head_channels = 512
        self.conv1 = Conv2dReLU(
            input_hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv2 = Conv2dReLU(
            head_channels,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class VisionTransformer(nn.Module):
    '''
    Docstring for VisionTransformer
    Vision Transformer model for semantic segmentation using DINOv2 as backbone.
    1. Uses a pre-trained DINOv2 model as the feature extractor.
    '''
    def __init__(self, input_hidden_size, num_classes=17, chip_size=224):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes

        local_path = "/root/autodl-tmp/DINOv"  # 你上传模型文件的目录

        # ✅ 只从本地读取，不联网
        self.processor = AutoImageProcessor.from_pretrained(
            local_path,
            use_fast=False,
            local_files_only=True
        )
        self.backbone = Dinov2Model.from_pretrained(
            local_path,
            local_files_only=True
        ).eval()

        self.upsampler = torch.hub.load(
            "/root/autodl-tmp/anyup-main/", "anyup", verbose=False, source="local"
        ).eval()

        # frozen the ViT weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.segmentation_head = nn.Conv2d(
            input_hidden_size, num_classes, kernel_size=1, padding=0
        )
        self.CHIP_SIZE = chip_size
        
    def forward(self, x):
        inputs = self.processor(
            images=x,
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt",
        )

        hr_image = inputs["pixel_values"].to(next(self.backbone.parameters()).device)

        with torch.no_grad():
            out_tokens = self.backbone(hr_image)  # (B, n_patch, hidden)

        hidden_states = out_tokens.last_hidden_state[:, 1:, :] # drop [CLS]

        # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        (
            B,
            n_patch,
            hidden
        ) = (
            hidden_states.size()
        ) 
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = hidden_states.contiguous().view(B, hidden, h, w)

        # upsample
        with torch.no_grad():
            feat_up = self.upsampler(hr_image, hidden_states, q_chunk_size=256)

        # branch 1
        logits_up = self.segmentation_head(feat_up)
        logits_ori = self.segmentation_head(hidden_states)

        return logits_up, feat_up, logits_ori, hidden_states