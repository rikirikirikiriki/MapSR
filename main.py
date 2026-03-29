import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from tqdm import tqdm

# Ensure Paraformer project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config

# Configure GPU environment (set before allocating large tensors)
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID

# Backport torch.nn.RMSNorm for older torch versions (<2.5)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.weight is not None:
            x = x * self.weight
        return x

nn.RMSNorm = RMSNorm

import utils
from networks.dino_linear_prob import VisionTransformer
from dataset import build_dataset_and_loader
from prototypes import get_prototypes
from refinement import refine_output
from visualize import debug_single_chip_visualization

def run_inference_and_save(model, prototypes, image_fns, gt_fns):
    for image_idx in range(len(image_fns)):
        image_fn = image_fns[image_idx]
        gt_fn = gt_fns[image_idx]

        print(f"({image_idx}/{len(image_fns)}) Processing {os.path.basename(image_fn)}")

        dataset, dataloader, input_profile, input_width, input_height = build_dataset_and_loader(
            image_fn=image_fn,
            gt_fn=gt_fn,
            batch_size=4
        )

        output = np.zeros((config.num_classes, input_height, input_width), dtype=np.float32)
        refined_output = np.zeros((config.PRED_NUM_CLASSES, input_height, input_width), dtype=np.float32)

        kernel = np.ones((config.CHIP_SIZE, config.CHIP_SIZE), dtype=np.float32)
        kernel[config.HALF_PADDING:-config.HALF_PADDING, config.HALF_PADDING:-config.HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, label, coords) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.cuda()
            with torch.no_grad():
                coarse_logits, feat, _, _ = model(data)

                coarse_pred_hard = F.softmax(coarse_logits, dim=1)
                t_refined_output = torch.zeros(data.shape[0], config.PRED_NUM_CLASSES, config.CHIP_SIZE, config.CHIP_SIZE).cuda()

                for b_idx in range(coarse_pred_hard.shape[0]):
                    _refined_dict = refine_output(
                        coarse_logits=coarse_logits[b_idx].unsqueeze(0),
                        prototypes=prototypes,
                        dino_feats=feat[b_idx].unsqueeze(0),
                        image_rgb=data[b_idx].unsqueeze(0) / 255.0
                    )
                    t_refined_output[b_idx] = _refined_dict["logits_s1"][0] if config.only_s1 else _refined_dict["logits_s2"][0]

            for j in range(coarse_pred_hard.shape[0]):
                y, x = coords[j]
                output[:, y:y + config.CHIP_SIZE, x:x + config.CHIP_SIZE] += coarse_pred_hard[j].cpu().numpy() * kernel
                counts[y:y + config.CHIP_SIZE, x:x + config.CHIP_SIZE] += kernel
                refined_output[:, y:y + config.CHIP_SIZE, x:x + config.CHIP_SIZE] += t_refined_output[j].cpu().numpy() * kernel

        output = output / counts
        refined_output = refined_output / counts

        output_hard = output.argmax(axis=0).astype(np.uint8)
        refined_output_hard = refined_output.argmax(axis=0).astype(np.uint8)

        output_profile = input_profile.copy()
        output_profile.pop("photometric", None)
        output_profile.update({
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "nodata": 0
        })

        output_fn = image_fn.split("/")[-1]
        output_fn = output_fn.replace("naip", "predictions")

        output_fn_save_path = os.path.join(config.test_save_path, output_fn)
        with rasterio.open(output_fn_save_path, "w", **output_profile) as f:
            f.write(output_hard, 1)
            f.write_colormap(1, utils.LABEL_IDX_COLORMAP)

        output_fn_refined = os.path.join(config.test_save_path_refined, output_fn)
        with rasterio.open(output_fn_refined, "w", **output_profile) as f:
            f.write(refined_output_hard, 1)
            f.write_colormap(1, utils.LABEL_IDX_COLORMAP)

def main():
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    input_dataframe = pd.read_csv(config.list_dir)
    image_fns = input_dataframe["image_fn"].values[:config.image_num]
    gt_fns = input_dataframe["label_fn"].values[:config.image_num]

    model = VisionTransformer(
        input_hidden_size=768,
        num_classes=config.num_classes,
    ).cuda()
    model.load_state_dict(torch.load(config.snapshot))
    model.eval()

    # --- Build a dataloader on the first image to compute prototypes ---
    image_idx = 0
    image_fn = image_fns[image_idx]
    gt_fn = gt_fns[image_idx]

    print(f"({image_idx}/{len(image_fns)}) Processing {os.path.basename(image_fn)} (for prototype dataloader)")

    dataset, dataloader, _, _, _ = build_dataset_and_loader(
        image_fn=image_fn,
        gt_fn=gt_fn,
        batch_size=1
    )

    prototypes = get_prototypes(model, dataloader)

    # --- Optional: single-chip visualization for debugging ---
    # Uncomment below to visualize a single chip
    # debug_single_chip_visualization(model, dataset, prototypes, idx=82)

    # --- Batch inference and save ---
    run_inference_and_save(model, prototypes, image_fns, gt_fns)

if __name__ == "__main__":
    main()
