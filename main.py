import argparse
import utils
import engine

def parse_args():
    parser = argparse.ArgumentParser(description="MapSR Inference")
    parser.add_argument("--dataset_name", type=str, default="10test_New_York", help="Dataset name")
    parser.add_argument("--get_prompts_mode", type=str, choices=["OneShot", "ZeroShot"], default="ZeroShot", help="Mode for prompt computation")
    parser.add_argument("--ignore_index", type=int, default=0, help="Ignore index for labels")
    parser.add_argument("--to_base", action="store_true", default=True, help="Whether to map to base classes")
    parser.add_argument("--image_num", type=int, default=10, help="Number of images to process (-1 for all)")
    parser.add_argument("--snapshot", type=str, default="./networks/pre-train_model/epoch_10.pth", help="Model weights path")
    parser.add_argument("--only_s1", action="store_true", default=False, help="Only run S1 (prompt similarity)")
    parser.add_argument("--use_slide", action="store_true", default=False, help="Use sliding window for S2")
    parser.add_argument("--n_segments", type=int, default=8000, help="SLIC n_segments")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness")
    parser.add_argument("--lp_sigma", type=float, default=0, help="LP sigma")
    parser.add_argument("--lp_pix_dist_pow", type=float, default=1.0, help="LP pix dist pow")
    parser.add_argument("--lp_k", type=int, default=100, help="LP k")
    parser.add_argument("--lp_gamma", type=float, default=1.0, help="LP gamma")
    parser.add_argument("--lp_alpha", type=float, default=0.5, help="LP alpha")
    return parser.parse_args()

def main():
    utils.setup_environment(gpu_index="0")
    args = parse_args()
    engine.run_pipeline(args)

if __name__ == "__main__":
    main()
