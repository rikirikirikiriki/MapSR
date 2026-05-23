'''
--dataset:
West_Virginia, Virginia, Pennsylvania, New_York, Maryland, Delaware
Some require the prefix 10test_
--method:
OneShot, ZeroShot
python calc_metric.py --dataset 10test_New_York --method OneShot
'''
import os
from time import time
import rasterio
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse

import utils

# ---------------------- Main Evaluation Function ----------------------
def batch_compute_miou(truth_dir, pred_dir, output_csv, mode, first_k_files=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(('.tif', '.tiff'))])
    
    if first_k_files:
        pred_files = pred_files[:first_k_files]
        
    print(f"📊 Found {len(pred_files)} prediction files to process.")

    confusion_total = torch.zeros((5, 5), dtype=torch.int64, device=device)
    csv_records = []

    for fname in tqdm(pred_files, desc="Processing"):
        pred_path = os.path.join(pred_dir, fname)
        truth_fname = fname
        truth_path = os.path.join(truth_dir, truth_fname)
        
        if not os.path.exists(truth_path):
            candidates = [
                fname.replace("predictions-new", "lc"),
                fname.replace("_pred", ""),
                fname.replace("_naip-new", "_lc"),
                fname
            ]
            found = False
            for cand in candidates:
                t_path = os.path.join(truth_dir, cand)
                if os.path.exists(t_path):
                    truth_path = t_path
                    found = True
                    break
            
            if not found:
                print(f"❌ Missing truth file for: {fname} (checked in {truth_dir})")
                continue

        try:
            truth_np = utils.read_tif(truth_path)
            truth_mapped = utils.map_classes(truth_np, utils.truth_to_base)
            truth = torch.from_numpy(truth_mapped).long().to(device)
            
            pred_np = utils.read_tif(pred_path)
            
            if mode == "base":
                pred = torch.from_numpy(pred_np).long().to(device)
            elif mode == "OneShot":
                pred_mapped = utils.map_classes(pred_np, utils.truth_to_base)
                pred = torch.from_numpy(pred_mapped).long().to(device)
            elif mode == "ade20k":
                pred_mapped = utils.map_classes(pred_np, utils.pred_ade20k_to_base)
                pred = torch.from_numpy(pred_mapped).long().to(device)
            else: # ZeroShot / nlcd mode
                pred_mapped = utils.map_classes(pred_np, utils.pred_to_base)
                pred = torch.from_numpy(pred_mapped).long().to(device)
            
            if truth.shape != pred.shape:
                 continue

            cm = utils.compute_confusion_matrix(truth, pred, num_classes=4, device=device)
            confusion_total += cm
            
            iou_list = utils.compute_iou(cm)

            valid_ious_img = [x for x in iou_list[1:] if not np.isnan(x)]
            miou = np.mean(valid_ious_img) if valid_ious_img else 0

            csv_records.append({
                "file": fname,
                "mIoU": miou,
                "IoU_background": iou_list[0] if len(iou_list)>0 else 0,
                "IoU_water": iou_list[1] if len(iou_list)>1 else 0,
                "IoU_tree": iou_list[2] if len(iou_list)>2 else 0,
                "IoU_low_veg": iou_list[3] if len(iou_list)>3 else 0,
                "IoU_built-up": iou_list[4] if len(iou_list)>4 else 0
            })
            
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            continue

    final_iou = utils.compute_iou(confusion_total)
    
    valid_ious = [x for x in final_iou[1:] if not np.isnan(x)]
    final_miou = np.mean(valid_ious) if valid_ious else 0
    
    print(f"\n📈 Final mean IoU (excluding background): {final_miou:.4f}")
    categories = ["Background", "Water", "Tree canopy", "Low vegetation", "Built-up"]
    for i, name in enumerate(categories):
        val = final_iou[i] if i < len(final_iou) else 0
        if np.isnan(val):
             print(f"  {name:<15}: N/A (No GT)")
        else:
             print(f"  {name:<15}: {val:.4f}")
        
    # Add average row to records
    csv_records.append({
        "file": "AVERAGE",
        "mIoU": final_miou,
        "IoU_background": final_iou[0] if len(final_iou)>0 else 0,
        "IoU_water": final_iou[1] if len(final_iou)>1 else 0,
        "IoU_tree": final_iou[2] if len(final_iou)>2 else 0,
        "IoU_low_veg": final_iou[3] if len(final_iou)>3 else 0,
        "IoU_built-up": final_iou[4] if len(final_iou)>4 else 0
    })

    df = pd.DataFrame(csv_records)
    df.to_csv(output_csv, index=False)
    print(f"📁 Saved to {output_csv}")

def count_tif_files(folder_path):
    """Count the number of .tif / .tiff files in a specified folder (non-recursive)"""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a valid folder: {folder_path}")
    count = 0
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.tif', '.tiff')):
            count += 1
    return count

# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mIoU between ground truth and predictions.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="10test_New_York",
        help="Dataset name (e.g., 10test_New_York). Used to locate GT and prediction paths."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "OneShot", "ZeroShot", "ade20k"],
        default="base",
        help="Mapping mode: 'base' (raw pred), 'OneShot' (map GT labels), 'ZeroShot' (map NLCD preds), or 'ade20k'."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["OneShot", "ZeroShot"],
        default="OneShot",
        help="Method to select base prediction directory: 'OneShot' or 'ZeroShot'"
    )
    
    args = parser.parse_args()

    # 1. Prediction path
    if args.method == "OneShot":
        base_pred_dir = "results/MapSR_OneShot"
        subdir = "refined_S2propogation_OneShot_tobase"
        pred_dir = os.path.join(base_pred_dir, args.dataset, subdir)
    elif args.method == "ZeroShot":
        base_pred_dir = "results/MapSR_ZeroShot"
        subdir = "refined_S2propogation_ZeroShot_tobase"
        pred_dir = os.path.join(base_pred_dir, args.dataset, subdir)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # 2. Truth path
    base_truth_dir = "dataset"
    truth_dir = os.path.join(base_truth_dir, args.dataset, "HR_label_truth")
    
    # Auto-fix: if dataset name has _wgt suffix but folder doesn't exist, try stripping it
    if not os.path.exists(truth_dir) and args.dataset.endswith("_wgt"):
        clean_name = args.dataset.replace("_wgt", "")
        alt_truth_dir = os.path.join(base_truth_dir, clean_name, "HR_label_truth")
        if os.path.exists(alt_truth_dir):
            print(f"⚠️ Redirecting truth dir from '{args.dataset}' to '{clean_name}'")
            truth_dir = alt_truth_dir

    # 3. Output CSV path
    output_csv = os.path.join(base_pred_dir, args.dataset, "miou_results.csv")
    
    # Verify paths exist
    if not os.path.exists(pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not os.path.exists(truth_dir):
        # Try alternative truth dir
        alt_truth_dir = os.path.join(base_truth_dir, args.dataset, "HR_truth")
        if os.path.exists(alt_truth_dir):
            truth_dir = alt_truth_dir
            print(f"⚠️ Using alternative truth dir: {truth_dir}")
        else:
            raise FileNotFoundError(f"Truth directory not found: {truth_dir}")
            
    print(f"📂 Dataset: {args.dataset}")
    print(f"📂 Pred Dir: {pred_dir}")
    print(f"📂 Truth Dir: {truth_dir}")
    print(f"💾 Output CSV: {output_csv}")
    
    # Count files
    try:
        num_files = count_tif_files(pred_dir)
    except Exception:
        num_files = 0
        
    batch_compute_miou(truth_dir, pred_dir, output_csv, args.mode, first_k_files=num_files)

