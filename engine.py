import os
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from lposs.lposs_eval import get_lposs_laplacian, perform_lp
import utils
import calc_metric
import pandas as pd
from config import DATASET_CONFIG
from networks.dino_linear_prob import VisionTransformer
from data.streaming_geo_spatial_dataset import TileInferenceDataset

def image_transform(img):
    img = img[:, :, :3]
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img

def label_transforms_GT(labels):
    labels = torch.from_numpy(labels)
    return labels

def label_transforms_vanilla(labels):
    labels = utils.LABEL_CLASS_TO_IDX_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels

def run_pipeline(args):
    """
    Complete inference pipeline: setup paths, load model, get/compute prompts, and run inference.
    """
    num_classes_raw = DATASET_CONFIG[args.dataset_name]["num_classes"]
    if args.get_prompts_mode == "ZeroShot":
        list_dir = DATASET_CONFIG[args.dataset_name]["list_dir_wgt"]
    else:
        list_dir = DATASET_CONFIG[args.dataset_name]["list_dir"]

    if args.to_base:
        pred_num_classes = 5
    elif args.get_prompts_mode == "OneShot":
        pred_num_classes = 7
    else:
        pred_num_classes = 17

    isbase = "tobase" if args.to_base else None
    if args.only_s1:
        isS1 = "onlyS1" 
    elif args.use_slide:
        isS1 = "S2slide"
    else:
        isS1 = "S2propogation"

    test_save_path = f"results/MapSR_{args.get_prompts_mode}/{args.dataset_name}/linear_prob_{isS1}_{args.get_prompts_mode}_{isbase}"
    test_save_path_refined = f"results/MapSR_{args.get_prompts_mode}/{args.dataset_name}/refined_{isS1}_{args.get_prompts_mode}_{isbase}"
    
    os.makedirs(test_save_path, exist_ok=True) 
    os.makedirs(test_save_path_refined, exist_ok=True)

    input_dataframe = pd.read_csv(list_dir)
    if args.image_num == -1:
        image_fns = input_dataframe["image_fn"].values
        gt_fns = input_dataframe["label_fn"].values
    else:
        image_fns = input_dataframe["image_fn"].values[:args.image_num]
        gt_fns = input_dataframe["label_fn"].values[:args.image_num]

    model = VisionTransformer(input_hidden_size=768, num_classes=num_classes_raw).cuda()
    model.load_state_dict(torch.load(args.snapshot))
    model.eval()

    prompt_file = f"prompts/ChesapeakeBay_prompts_{args.get_prompts_mode}_{args.to_base}.pth"
    if os.path.exists(prompt_file):
        prompts = torch.load(prompt_file, map_location='cuda')
        print("Loaded prompts from file.")
    else:
        label_folder = "dataset/ChesapeakeBay/LR_label" if args.get_prompts_mode == "ZeroShot" else "dataset/ChesapeakeBay/HR_label_truth"

        if not os.path.exists(label_folder) or len(os.listdir(label_folder)) == 0:
            raise FileNotFoundError(f"{label_folder} folder is empty, cannot compute prompt")

        label_files = [f for f in os.listdir(label_folder) if f.endswith('.tif') or f.endswith('.tiff')]
        if len(label_files) == 0:
            raise FileNotFoundError(f"{label_folder} has no tif files, cannot compute prompt")

        first_label_name = label_files[0]
        prompt_gt_fn = os.path.join(label_folder, first_label_name)
        base_name = first_label_name.replace('_lc.tif', '').replace('_nlcd.tif', '')
        prompt_image_fn = os.path.join("dataset/ChesapeakeBay/HR_image", base_name + "_naip-new.tif")
        if not os.path.exists(prompt_image_fn):
            prompt_image_fn = os.path.join("dataset/ChesapeakeBay/HR_image", first_label_name)

        print(f"Using {os.path.basename(prompt_image_fn)} for prompt dataloader")

        label_transform_prompt = label_transforms_GT if args.get_prompts_mode == "OneShot" else label_transforms_vanilla
        _, prompt_dataloader, _, _, _ = build_dataset_and_loader(
            image_fn=prompt_image_fn, gt_fn=prompt_gt_fn, batch_size=1, label_transform=label_transform_prompt
        )

        prompts = get_prompts(
            model, prompt_dataloader, args.ignore_index, mode=args.get_prompts_mode, 
            to_base=args.to_base, dataset_name=args.dataset_name, num_classes_raw=num_classes_raw
        )

    label_transform_infer = label_transforms_GT if args.get_prompts_mode == "OneShot" else label_transforms_vanilla
    run_inference_and_save(
        model, prompts, image_fns, gt_fns, args, label_transform_infer,
        num_classes_raw, pred_num_classes, test_save_path, test_save_path_refined
    )

def build_dataset_and_loader(image_fn, gt_fn, batch_size=1, label_transform=None, chip_size=448, chip_stride=448):
    with rasterio.open(image_fn) as f:
        input_width, input_height = f.width, f.height
        input_profile = f.profile.copy()

    dataset = TileInferenceDataset(
        image_fn,
        chip_size=chip_size,
        stride=chip_stride,
        gt=gt_fn,
        transform=image_transform,
        label_transform=label_transform,
        verbose=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )

    return dataset, dataloader, input_profile, input_width, input_height

@torch.no_grad()
def get_prompts(model, dataloader, ignore_index=0, mode="ZeroShot", to_base=True, dataset_name="", num_classes_raw=17):
    try:
        prompts = torch.load(f"prompts/ChesapeakeBay_prompts_{mode}_{to_base}.pth", map_location='cuda')
        print("Loaded prompts from file.")
        return prompts
    except FileNotFoundError:
        print("Prompts file not found. Computing prompts...")

    model.eval()
    device = next(model.parameters()).device

    if mode == "OneShot" and to_base:
        num_classes = 5
    elif mode == "OneShot" and (not to_base):
        num_classes = 7
    elif mode == "ZeroShot" and to_base:
        num_classes = 5
    else:
        num_classes = num_classes_raw

    feature_dim = 768

    sum_features = torch.zeros(num_classes, feature_dim, device=device)
    count = torch.zeros(num_classes, device=device)

    for data, label, *_ in tqdm(dataloader, desc="Computing prompts"):
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

        if to_base and mode == "OneShot" :
            label_flat = utils.map_classes_torch(label_flat, utils.truth_to_base)
            pred_flat = utils.map_classes_torch(pred_flat, utils.pred_to_base)

        if to_base and mode == "ZeroShot":
            label_flat = utils.map_classes_torch(label_flat, utils.pred_to_base)
            pred_flat = utils.map_classes_torch(pred_flat, utils.pred_to_base)

        if mode == "OneShot":
            valid_mask = label_flat != ignore_index
            feat_flat = feat_flat[valid_mask]
            label_flat = label_flat[valid_mask]
        elif mode == "ZeroShot":
            valid_mask = (label_flat != ignore_index) & (pred_flat == label_flat)
            print(f"{valid_mask.sum()/len(valid_mask):.4f} of pixels used for prompt computation.")
            feat_flat = feat_flat[valid_mask]
            label_flat = label_flat[valid_mask]

        for cls in range(num_classes):
            cls_mask = label_flat == cls
            if cls_mask.any():
                sum_features[cls] += feat_flat[cls_mask].sum(dim=0)
                count[cls] += cls_mask.sum()

    count[count == 0] = 1
    prompts = sum_features / count.unsqueeze(1)

    if not os.path.exists("prompts"):
        os.makedirs("prompts", exist_ok=True)
    torch.save(prompts, f"prompts/ChesapeakeBay_prompts_{mode}_{to_base}.pth")  

    return prompts

def generate_similarity_and_labels(prompts, feat):
    C, D = prompts.shape
    _, _, H, W = feat.shape
    class_feat = F.normalize(prompts, p=2, dim=-1).unsqueeze(2).unsqueeze(3)    
    feat_norm = F.normalize(feat, p=2, dim=1)
    similarity = F.conv2d(feat_norm, weight=class_feat, bias=None)
    labels = torch.argmax(similarity, dim=1)
    heatmap = similarity - similarity.min(dim=1, keepdim=True)[0]
    heatmap = heatmap / (heatmap.max(dim=1, keepdim=True)[0] + 1e-8)
    return labels, heatmap

def refine_label_propogation(coarse_pred, dino_feats, image_rgb, args):     
    device = coarse_pred.device
    B, C = coarse_pred.shape[:2]
    _, D, H, W = dino_feats.shape
    assert B == 1, 'Loop when batch size >1'

    dino = dino_feats[0].permute(1, 2, 0)
    dino = F.normalize(dino.view(-1, D), p=2, dim=1)

    img_rgb = (image_rgb[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    segments = slic(img_as_float(img_rgb), n_segments=args.n_segments, compactness=args.compactness, start_label=0)

    n_sp = int(segments.max()) + 1
    segments = torch.from_numpy(segments).long().to(device)

    sp_mask = torch.zeros(n_sp, H * W, dtype=torch.bool, device=device)
    flat_seg = segments.view(-1)
    sp_mask.scatter_(0, flat_seg.unsqueeze(0), True)

    coarse_flat = coarse_pred[0].permute(1, 2, 0).view(-1, C)
    sp_pred = torch.mm(sp_mask.float(), coarse_flat)
    cnt = sp_mask.sum(1, keepdim=True).clamp_min(1e-8)
    sp_pred = sp_pred / cnt

    # To avoid OOM during large matrix multiplication, process sp_feat in smaller chunks if needed,
    # but here we can just delete variables we no longer need to free up memory before the next big allocation.
    del coarse_flat
    torch.cuda.empty_cache()

    sp_feat = torch.mm(sp_mask.float(), dino)
    sp_feat = F.normalize(sp_feat, p=2, dim=1)

    del dino
    torch.cuda.empty_cache()

    centers = sp_mask.float() @ torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device), indexing='ij'), dim=0).view(2, -1).t().float()
    centers = centers / cnt
    coords = torch.cat([centers, torch.zeros(n_sp, 2, device=device)], 1)       

    L = get_lposs_laplacian(
        sp_feat, coords, [(n_sp, 1)],
        sigma=args.lp_sigma, pix_dist_pow=args.lp_pix_dist_pow,
        k=args.lp_k, gamma=args.lp_gamma, alpha=args.lp_alpha, patch_size=1
    )
    sp_refined = perform_lp(L, sp_pred)

    refined_flat = sp_refined[flat_seg]
    refined = refined_flat.view(H, W, C).permute(2, 0, 1).unsqueeze(0)

    return refined

def refine_label_propogation_with_slide(coarse_pred, dino_feats, args):
    h_stride, w_stride = 112, 112
    h_crop, w_crop = 112, 112
    batch_size, _, h_img, w_img = coarse_pred.size()
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    num_classes = coarse_pred.shape[1]

    preds = coarse_pred.new_zeros((batch_size, num_classes, h_img, w_img))      
    count_mat = coarse_pred.new_zeros((batch_size, 1, h_img, w_img))

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
                _feat, torch.zeros((1, 4), device=_feat.device), height_width,
                sigma=args.lp_sigma, pix_dist_pow=args.lp_pix_dist_pow,
                k=args.lp_k, gamma=args.lp_gamma, alpha=args.lp_alpha, patch_size=1
            )

            lp_preds = perform_lp(L, coarse_pred[:, :, y1:y2, x1:x2].permute(0, 2, 3, 1).reshape(-1, num_classes))

            crop_seg_logit = torch.reshape(lp_preds, (height_width[0][0], height_width[0][1], num_classes))
            crop_seg_logit = torch.unsqueeze(crop_seg_logit, 0)
            crop_seg_logit = torch.permute(crop_seg_logit, (0, 3, 1, 2))        

            preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

    preds = preds / count_mat
    return preds

def refine_output(coarse_logits, prompts, dino_feats, image_rgb, args):
    device = coarse_logits.device
    B, C = coarse_logits.shape[:2]
    assert B == 1, 'Loop when batch size >1'

    refined_labels_s1, refined_logits_s1 = generate_similarity_and_labels(prompts, dino_feats)

    if args.only_s1:
        return {"pred_s1": refined_labels_s1, "logits_s1": refined_logits_s1}   

    if args.use_slide:
        refined_logits_s2 = refine_label_propogation_with_slide(refined_logits_s1, dino_feats, args)
    else:
        refined_logits_s2 = refine_label_propogation(refined_logits_s1, dino_feats, image_rgb, args)

    refined_labels_s2 = torch.argmax(refined_logits_s2, dim=1)
    return {
        "pred_s1": refined_labels_s1,
        "logits_s1": refined_logits_s1,
        "pred_s2": refined_labels_s2,
        "logits_s2": refined_logits_s2,
    }

def run_inference_and_save(model, prompts, image_fns, gt_fns, args, label_transform, num_classes, pred_num_classes, test_save_path, test_save_path_refined, chip_size=448, padding=0): 
    half_padding = padding // 2
    chip_stride = chip_size - padding
    
    for image_idx in range(len(image_fns)):
        image_fn = image_fns[image_idx]
        gt_fn = gt_fns[image_idx]

        print(f"({image_idx + 1}/{len(image_fns)}) Processing {os.path.basename(image_fn)}")

        dataset, dataloader, input_profile, input_width, input_height = build_dataset_and_loader(
            image_fn=image_fn, gt_fn=gt_fn, batch_size=2, label_transform=label_transform, chip_size=chip_size, chip_stride=chip_stride
        )

        output = np.zeros((num_classes, input_height, input_width), dtype=np.float32)
        refined_output = np.zeros((pred_num_classes, input_height, input_width), dtype=np.float32)

        kernel = np.ones((chip_size, chip_size), dtype=np.float32)
        kernel[half_padding:chip_size-half_padding, half_padding:chip_size-half_padding] = 5      
        counts = np.zeros((input_height, input_width), dtype=np.float32)        

        for i, (data, label, coords) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.cuda()
            with torch.no_grad():
                coarse_logits, feat, _, _ = model(data)
                coarse_pred_hard = F.softmax(coarse_logits, dim=1)
                t_refined_output = torch.zeros(data.shape[0], pred_num_classes, chip_size, chip_size).cuda()

                for b_idx in range(coarse_pred_hard.shape[0]):
                    _refined_dict = refine_output(
                        coarse_logits=coarse_logits[b_idx].unsqueeze(0),        
                        prompts=prompts,
                        dino_feats=feat[b_idx].unsqueeze(0),
                        image_rgb=data[b_idx].unsqueeze(0) / 255.0,
                        args=args
                    )
                    t_refined_output[b_idx] = _refined_dict["logits_s1"][0] if args.only_s1 else _refined_dict["logits_s2"][0]

            for j in range(coarse_pred_hard.shape[0]):
                y, x = coords[j]
                output[:, y:y + chip_size, x:x + chip_size] += coarse_pred_hard[j].cpu().numpy() * kernel
                counts[y:y + chip_size, x:x + chip_size] += kernel
                refined_output[:, y:y + chip_size, x:x + chip_size] += t_refined_output[j].cpu().numpy() * kernel
            
            # Free memory at the end of each batch
            del data, coarse_logits, feat, coarse_pred_hard, t_refined_output
            torch.cuda.empty_cache()

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

        output_fn_save_path = os.path.join(test_save_path, output_fn)
        with rasterio.open(output_fn_save_path, "w", **output_profile) as f:    
            f.write(output_hard, 1)
            f.write_colormap(1, utils.LABEL_IDX_COLORMAP)

        output_fn_refined = os.path.join(test_save_path_refined, output_fn)     
        with rasterio.open(output_fn_refined, "w", **output_profile) as f:      
            f.write(refined_output_hard, 1)
            f.write_colormap(1, utils.LABEL_IDX_COLORMAP)
