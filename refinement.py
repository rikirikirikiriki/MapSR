import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from lposs.lposs_eval import get_lposs_laplacian, perform_lp
import config

def generate_similarity_and_labels(prototypes, feat):
    C, D = prototypes.shape
    _, _, H, W = feat.shape

    # Normalize features and prototypes; compute similarity via 1x1 conv
    class_feat = F.normalize(prototypes, p=2, dim=-1).unsqueeze(2).unsqueeze(3)
    feat_norm = F.normalize(feat, p=2, dim=1)

    similarity = F.conv2d(feat_norm, weight=class_feat, bias=None)
    labels = torch.argmax(similarity, dim=1)

    heatmap = similarity - similarity.min(dim=1, keepdim=True)[0]
    heatmap = heatmap / (heatmap.max(dim=1, keepdim=True)[0] + 1e-8)
    return labels, heatmap

def compute_pca(features):
    # Compute top-3 PCA components for visualization
    features_mean = torch.mean(features, dim=0, keepdim=True)
    features_centered = features - features_mean
    covariance_matrix = torch.matmul(features_centered.t(), features_centered) / (features.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvectors = eigenvectors[:, sorted_indices[:3]]
    pca_features = torch.matmul(features_centered, top_eigenvectors)
    return pca_features

def refine_label_propogation(coarse_pred, dino_feats, image_rgb):
    device = coarse_pred.device
    B, C = coarse_pred.shape[:2]
    _, D, H, W = dino_feats.shape
    assert B == 1, 'Loop over batch dimension when B > 1'

    dino = dino_feats[0].permute(1, 2, 0)
    dino = F.normalize(dino.view(-1, D), p=2, dim=1)

    img_rgb = (image_rgb[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    segments = slic(img_as_float(img_rgb), n_segments=config.n_segments, compactness=config.compactness, start_label=0)

    n_sp = int(segments.max()) + 1
    segments = torch.from_numpy(segments).long().to(device)

    sp_mask = torch.zeros(n_sp, H * W, dtype=torch.bool, device=device)
    flat_seg = segments.view(-1)
    sp_mask.scatter_(0, flat_seg.unsqueeze(0), True)

    coarse_flat = coarse_pred[0].permute(1, 2, 0).view(-1, C)
    sp_pred = torch.mm(sp_mask.float(), coarse_flat)
    cnt = sp_mask.sum(1, keepdim=True).clamp_min(1e-8)
    sp_pred = sp_pred / cnt

    sp_feat = torch.mm(sp_mask.float(), dino)
    sp_feat = F.normalize(sp_feat, p=2, dim=1)

    centers = sp_mask.float() @ torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device), indexing='ij'), dim=0).view(2, -1).t().float()
    centers = centers / cnt
    coords = torch.cat([centers, torch.zeros(n_sp, 2, device=device)], 1)

    L = get_lposs_laplacian(
        sp_feat, 
        coords, 
        [(n_sp, 1)],
        sigma=config.LP_SIGMA,
        pix_dist_pow=config.LP_PIX_DIST_POW,
        k=config.LP_K,
        gamma=config.LP_GAMMA,
        alpha=config.LP_ALPHA,
        patch_size=1
    )
    sp_refined = perform_lp(L, sp_pred)

    refined_flat = sp_refined[flat_seg]
    refined = refined_flat.view(H, W, C).permute(2, 0, 1).unsqueeze(0)

    return refined

def reshape_windows(x):
    height_width = [(y.shape[0], y.shape[1]) for y in x]
    dim = x[0].shape[-1]
    x = [torch.reshape(y, (-1, dim)) for y in x]
    return torch.cat(x, dim=0), height_width

def refine_label_propogation_with_slide(coarse_pred, dino_feats):
    h_stride, w_stride = 112, 112
    h_crop, w_crop = 112, 112
    batch_size, _, h_img, w_img = coarse_pred.size()
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    num_classes = coarse_pred.shape[1]

    preds = coarse_pred.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = coarse_pred.new_zeros((batch_size, 1, h_img, w_img))

    idx_window = []
    window_counter = 0

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            _feat = dino_feats[:, :, y1:y2, x1:x2].permute(0, 2, 3, 1).reshape(-1, dino_feats.shape[1])
            height_width = [(y2 - y1, x2 - x1)]
            _feat = F.normalize(_feat, p=2, dim=1)

            L = get_lposs_laplacian(
                _feat,
                torch.zeros((1, 4), device=_feat.device),
                height_width,
                sigma=config.LP_SIGMA,
                pix_dist_pow=config.LP_PIX_DIST_POW,
                k=config.LP_K,
                gamma=config.LP_GAMMA,
                alpha=config.LP_ALPHA,
                patch_size=1
            )

            lp_preds = perform_lp(
                L,
                coarse_pred[:, :, y1:y2, x1:x2].permute(0, 2, 3, 1).reshape(-1, num_classes)
            )

            crop_seg_logit = torch.reshape(lp_preds, (height_width[0][0], height_width[0][1], num_classes))
            crop_seg_logit = torch.unsqueeze(crop_seg_logit, 0)
            crop_seg_logit = torch.permute(crop_seg_logit, (0, 3, 1, 2))

            preds += F.pad(
                crop_seg_logit,
                (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2))
            )

            count_mat[:, :, y1:y2, x1:x2] += 1
            idx_window.extend([window_counter] * ((y2 - y1) * (x2 - x1)))
            window_counter += 1

    preds = preds / count_mat
    return preds

def refine_output(coarse_logits, prototypes, dino_feats, image_rgb):
    device = coarse_logits.device
    B, C = coarse_logits.shape[:2]
    
    refined_labels_s1, refined_logits_s1 = generate_similarity_and_labels(prototypes, dino_feats)

    if config.only_s1:
        return {"pred_s1": refined_labels_s1, "logits_s1": refined_logits_s1}

    if config.use_slide: # Sliding window refinement
        refined_logits_s2 = refine_label_propogation_with_slide( 
            coarse_pred=refined_logits_s1,
            dino_feats=dino_feats
        )
    else: # Superpixel refinement
        refined_logits_s2 = refine_label_propogation(
            coarse_pred=refined_logits_s1, 
            dino_feats=dino_feats,
            image_rgb=image_rgb
        )

    refined_labels_s2 = torch.argmax(refined_logits_s2, dim=1)

    return {
        "pred_s1": refined_labels_s1,
        "logits_s1": refined_logits_s1,
        "pred_s2": refined_labels_s2,
        "logits_s2": refined_logits_s2,
    }
