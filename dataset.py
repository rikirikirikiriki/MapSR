import numpy as np
import torch
import rasterio
import utils
from data.streaming_geo_spatial_dataset import TileInferenceDataset
import config

def image_transform(img):
    # Convert HWC to CHW and cast to float32 tensor (use first 3 channels)
    img = img[:, :, :3]
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img

def label_transforms_GT(labels):
    # Pass-through labels (GT indices preserved)
    labels = torch.from_numpy(labels)
    return labels

def label_transforms_vanilla(labels):
    # Map class names to indices using utils and convert to tensor
    labels = utils.LABEL_CLASS_TO_IDX_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels

def get_label_transform():
    # Choose label transform based on prototype mode
    if config.get_prototypes_mode == "GT":
        return label_transforms_GT
    else:
        return label_transforms_vanilla

def build_dataset_and_loader(image_fn, gt_fn, batch_size=1):
    # Build dataset and DataLoader; also return input raster profile and size
    label_transform = get_label_transform()
    with rasterio.open(image_fn) as f:
        input_width, input_height = f.width, f.height
        input_profile = f.profile.copy()

    dataset = TileInferenceDataset(
        image_fn,
        chip_size=config.CHIP_SIZE,
        stride=config.CHIP_STRIDE,
        gt=gt_fn,
        transform=image_transform,
        label_transform=label_transform,
        verbose=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return dataset, dataloader, input_profile, input_width, input_height

def get_test_data(dataset, imgage_idx=82):
    # Fetch a single chip for debugging/visualization
    data, label, coord = dataset[imgage_idx]
    return data, label, coord
