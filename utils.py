import numpy as np
import importlib
import torch
import torch.nn as nn
import os
from PIL import Image

def setup_environment(gpu_index="0"):
    """Set up GPU visibility and apply compatibility patches for older PyTorch versions."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    print(f"GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Compatibility for older torch versions (< 2.5) to use torch.nn.RMSNorm
    if not hasattr(nn, 'RMSNorm'):
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

IMAGE_MEANS = np.array(
    [117.67, 130.39, 121.52, 162.92]
)  # The setting here is for Chesapeake dataset
IMAGE_STDS = np.array([39.25, 37.82, 24.24, 60.03])
LABEL_CLASSES = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
LABEL_CLASS_COLORMAP = {  # Color map for Chesapeake dataset
    0: (0, 0, 0),
    11: (70, 107, 159),
    12: (209, 222, 248),
    21: (222, 197, 197),
    22: (217, 146, 130),
    23: (235, 0, 0),
    24: (171, 0, 0),
    31: (179, 172, 159),
    41: (104, 171, 95),
    42: (28, 95, 44),
    43: (181, 197, 143),
    52: (204, 184, 121),
    71: (223, 223, 194),
    81: (220, 217, 57),
    82: (171, 108, 40),
    90: (184, 217, 235),
    95: (108, 159, 184),
}

LABEL_IDX_COLORMAP = {
    idx: LABEL_CLASS_COLORMAP[c] for idx, c in enumerate(LABEL_CLASSES)
}


def get_label_class_to_idx_map():
    label_to_idx_map = []
    idx = 0
    # for i in range(LABEL_CLASSES[-1]+1):
    for i in range(256):
        if i in LABEL_CLASSES:
            label_to_idx_map.append(idx)
            idx += 1
        else:
            label_to_idx_map.append(0)
    label_to_idx_map = np.array(label_to_idx_map).astype(np.int64)
    return label_to_idx_map


LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()

# ---------------------- 1. Class Mapping Relations ----------------------
# HR label original 6 classes -> base 4 classes
truth_to_base = {
    1: 1,  # Water
    2: 2,  # Tree canopy
    3: 3,  # Low vegetation
    4: 3,  # Low vegetation (barren)
    5: 4,  # Built-up (impervious other)
    6: 4   # Built-up (impervious road)
}

# Model output (NLCD 16 classes) -> base 4 classes
pred_to_base = {
    1: 1,   # Open Water
    2: 1,   # Perennial Ice/Snow
    3: 4,   # Developed, Open Space
    4: 4,   # Developed, Low Intensity
    5: 4,   # Developed, Medium Intensity
    6: 4,   # Developed, High Intensity
    7: 3,   # Barren Land
    8: 2,   # Deciduous Forest
    9: 2,   # Evergreen Forest
    10: 2,  # Mixed Forest
    11: 3,  # Shrub/Scrub
    12: 3,  # Grassland
    13: 3,  # Pasture/Hay
    14: 3,  # Cultivated Crops
    15: 1,  # Woody Wetlands
    16: 1   # Herbaceous Wetlands
}

# ADE20K (150 classes) -> Base (4 classes)
pred_ade20k_to_base = {
    21: 1, 26: 1, 60: 1, 84: 1, 103: 1, 99: 1,
    4: 2, 72: 2,
    9: 3, 17: 3, 29: 3, 66: 3, 13: 3, 16: 3, 46: 3, 93: 3,
    0: 4, 1: 4, 3: 4, 6: 4, 11: 4, 25: 4, 32: 4, 48: 4, 52: 4, 53: 4, 
    54: 4, 61: 4, 80: 4, 83: 4, 20: 4, 126: 4,
}

# ---------------------- 2. Utility Functions ----------------------
def read_tif(file_path):
    return np.array(Image.open(file_path))

def map_classes(array, mapping):
    mapped = np.zeros_like(array, dtype=np.uint8)
    for k, v in mapping.items():
        mapped[array == k] = v
    return mapped

def map_classes_torch(array, mapping):
    mapped = torch.zeros_like(array, dtype=torch.uint8)
    for k, v in mapping.items():
        mapped[array == k] = v
    return mapped

def compute_confusion_matrix(truth, pred, num_classes=4, ignore_index=0, device="cuda"):
    truth = truth.to(device)
    pred = pred.to(device)
    
    mask = (truth != ignore_index)
    truth = truth[mask]
    pred = pred[mask]
    
    if truth.numel() == 0:
        k = num_classes + 1
        return torch.zeros((k, k), dtype=torch.int64, device=device)

    k = num_classes + 1
    truth = torch.clamp(truth, 0, k - 1)
    pred = torch.clamp(pred, 0, k - 1)
    
    indices = k * truth + pred
    cm = torch.bincount(indices, minlength=k*k).reshape(k, k)
    return cm

def compute_iou(confusion):
    ious = []
    confusion = confusion.float()
    for i in range(0, confusion.shape[0]):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        denom = tp + fp + fn
        
        if denom > 0:
            iou = tp / denom
            ious.append(iou.item())
        else:
            truth_count = confusion[i, :].sum()
            if truth_count == 0:
                 ious.append(np.nan)
            else:
                 ious.append(0.0)
    return ious