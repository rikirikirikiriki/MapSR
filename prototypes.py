import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import calc_metric
import config

@torch.no_grad()
def get_prototypes(model, dataloader):
    # Load cached prototypes if available; otherwise compute and save
    prototypes_dir = config.get_abs_path("prototypes")
    os.makedirs(prototypes_dir, exist_ok=True)
    prototype_path = os.path.join(prototypes_dir, f"{config.dataset_name}_prototypes_{config.get_prototypes_mode}_{config.to_base}.pth")
    
    try:
        prototypes = torch.load(prototype_path, map_location='cuda')
        print("Loaded prototypes from file.")
        return prototypes
    except FileNotFoundError:
        print("Prototypes file not found. Computing prototypes...")

    model.eval()
    device = next(model.parameters()).device

    if config.get_prototypes_mode == "GT" and config.to_base:
        num_classes = 5
    elif config.get_prototypes_mode == "GT" and (not config.to_base):                                                
        num_classes = 7
    elif config.get_prototypes_mode == "pred" and config.to_base:
        num_classes = 5
    else:
        num_classes = 17

    feature_dim = 768

    sum_features = torch.zeros(num_classes, feature_dim, device=device)
    count = torch.zeros(num_classes, device=device)

    for data, label, *_ in tqdm(dataloader, desc="Computing prototypes"):
        # Align feature/label spatial sizes and flatten
        label = label.squeeze(3)
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            logits, feat, _, _ = model(data)

        B, D, H, W = feat.shape
        if (H, W) != (data.shape[2], data.shape[3]):
            print(f"Warning: Feature size ({H}, {W}) does not match input size. Interpolating...")
            feat = F.interpolate(feat, size=(data.shape[2], data.shape[3]), mode='bilinear', align_corners=False)

        if label.shape[-2:] != feat.shape[-2:]:
            label = F.interpolate(label.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()

        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, D)
        label_flat = label.reshape(-1)
        pred = logits.argmax(dim=1)
        pred_flat = pred.reshape(-1)
        
        if config.to_base and config.get_prototypes_mode == "GT":
            label_flat = calc_metric.map_classes_torch(label_flat, calc_metric.truth_to_base)
            pred_flat = calc_metric.map_classes_torch(pred_flat, calc_metric.pred_to_base)

        if config.to_base and config.get_prototypes_mode == "pred":
            label_flat = calc_metric.map_classes_torch(label_flat, calc_metric.pred_to_base)
            pred_flat = calc_metric.map_classes_torch(pred_flat, calc_metric.pred_to_base)

        if config.get_prototypes_mode == "GT":
            # Use all non-ignore pixels from GT
            valid_mask = label_flat != config.ignore_index
            feat_flat = feat_flat[valid_mask]
            label_flat = label_flat[valid_mask]
        elif config.get_prototypes_mode == "pred":
            # Use pixels where prediction equals label (and not ignore)
            valid_mask = (label_flat != config.ignore_index) & (pred_flat == label_flat)
            feat_flat = feat_flat[valid_mask]
            label_flat = label_flat[valid_mask]

        for cls in range(num_classes):
            cls_mask = label_flat == cls
            if cls_mask.any():
                sum_features[cls] += feat_flat[cls_mask].sum(dim=0)
                count[cls] += cls_mask.sum()

    count[count == 0] = 1
    prototypes = sum_features / count.unsqueeze(1)

    torch.save(prototypes, prototype_path)

    return prototypes
