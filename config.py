import os

# ==========================================
# 1. Hardware & Environment
# ==========================================
GPU_ID = "1"

# ==========================================
# 2. Path Management
# ==========================================
# Resolve project root (Paraformer/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_abs_path(rel_path):
    return os.path.join(PROJECT_ROOT, rel_path)

# ==========================================
# 3. Dataset & Path Configuration
# ==========================================
DATASET_CONFIG = {
    "Delaware": {
        "list_dir": get_abs_path("dataset/CSV_list/Delaware.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/Delaware_wgt.csv"),
        "num_classes": 17
    },
    "Maryland": {
        "list_dir": get_abs_path("dataset/CSV_list/Maryland.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/Maryland_wgt.csv"),
        "num_classes": 17
    },
    "New_York": {
        "list_dir": get_abs_path("dataset/CSV_list/New_York.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/New_York_wgt.csv"),
        "num_classes": 17
    },
    "Pennsylvania": {
        "list_dir": get_abs_path("dataset/CSV_list/Pennsylvania.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/Pennsylvania_wgt.csv"),
        "num_classes": 17
    },
    "Virginia": {
        "list_dir": get_abs_path("dataset/CSV_list/Virginia.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/Virginia_wgt.csv"),
        "num_classes": 17
    },
    "West_Virginia": {
        "list_dir": get_abs_path("dataset/CSV_list/West Virginia.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/West Virginia_wgt.csv"),
        "num_classes": 17
    },
    "All_States": {
        "list_dir": get_abs_path("dataset/CSV_list/All_States.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/All_States_wgt.csv"),
        "num_classes": 17
    },
    "10test_Delaware": {
        "list_dir": get_abs_path("dataset/CSV_list/10test_Delaware.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/10test_Delaware_wgt.csv"),
        "num_classes": 17
    },
    "10test_Maryland": {
        "list_dir": get_abs_path("dataset/CSV_list/10test_Maryland.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/10test_Maryland_wgt.csv"),
        "num_classes": 17
    },
    "10test_New_York": {
        "list_dir": get_abs_path("dataset/CSV_list/10test_New_York.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/10test_New_York_wgt.csv"),
        "num_classes": 17
    },
    "10test_Pennsylvania": {
        "list_dir": get_abs_path("dataset/CSV_list/10test_Pennsylvania.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/10test_Pennsylvania_wgt.csv"),
        "num_classes": 17
    },
    "10test_Virginia1": {
        "list_dir": get_abs_path("dataset/CSV_list/10test_Virginia1.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/10test_Virginia1_wgt.csv"),
        "num_classes": 17
    },
    "10test_West_Virginia": {
        "list_dir": get_abs_path("dataset/CSV_list/10test_West_Virginia.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/10test_West_Virginia_wgt.csv"),
        "num_classes": 17
    },
    "10test_All_States": {
        "list_dir": get_abs_path("dataset/CSV_list/10test_All_States.csv"),
        "list_dir_wgt": get_abs_path("dataset/CSV_list/10test_All_States_wgt.csv"),
        "num_classes": 17
    }
}

# Selected dataset to run
dataset_name = "10test_Virginia1"

# GT/pred: GT uses ground-truth labels for prototypes; pred uses predicted labels
get_prototypes_mode = "GT"
ignore_index = 0

# Whether to map to base classes
to_base = True

# Number of test images
image_num = 10

# Model snapshot path
snapshot = get_abs_path("networks/pre-train_model/DinoV2_LinearProb_exp_20251226_225930/epoch_10.pth")

# ==========================================
# 4. Model & Inference Parameters
# ==========================================
is_pretrain = True
vit_patches_size = 16

CHIP_SIZE = 448
PADDING = 0
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING

# ==========================================
# 5. LPOSS & Refinement Parameters
# ==========================================
# Whether to run only S1 (prototype similarity) without S2 refinement
only_s1 = False

# S2 refinement method: sliding window (True) or superpixel (False)
use_slide = True

# Superpixel parameters (used by refine_label_propogation)
n_segments = 8000      # smaller value → smoother
compactness = 10.0     # larger value → smoother

# LPOSS shared & sliding-window params (used by refine_label_propogation_with_slide)
LP_SIGMA = 0.01          # sigma for get_lposs_laplacian; larger → smoother (superpixel recommends 0)
LP_PIX_DIST_POW = 1.0    # pixel distance power; smaller → smoother
LP_K = 400               # kNN k; larger → smoother (superpixel recommends 100)
LP_GAMMA = 3.0           # similarity sharpening gamma; smaller → smoother (superpixel recommends 1.0)
LP_ALPHA = 0.95          # propagation strength alpha; larger → smoother (superpixel recommends 0.5)

# ==========================================
# 6. Derived Variables (auto-generated; do not modify)
# ==========================================
num_classes = DATASET_CONFIG[dataset_name]["num_classes"]
if get_prototypes_mode == "GT":
    list_dir = DATASET_CONFIG[dataset_name]["list_dir_wgt"]
else:
    list_dir = DATASET_CONFIG[dataset_name]["list_dir"]

if to_base:
    PRED_NUM_CLASSES = 5
elif get_prototypes_mode == "GT":
    PRED_NUM_CLASSES = 7
else:
    PRED_NUM_CLASSES = 17

isbase = "tobase" if to_base else None
if only_s1:
    isS1 = "onlyS1"
elif use_slide:
    isS1 = "S2slide"
else:
    isS1 = "S2propogation"

# Build output directories for predictions (linear_prob) and refined results
test_save_path = get_abs_path(f"GRSL_experiments/MapSR_{get_prototypes_mode}/{dataset_name}/linear_prob_{isS1}_{get_prototypes_mode}_{isbase}")
test_save_path_refined = get_abs_path(f"GRSL_experiments/MapSR_{get_prototypes_mode}/{dataset_name}/refined_{isS1}_{get_prototypes_mode}_{isbase}")

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True) 
if not os.path.exists(test_save_path_refined):
    os.makedirs(test_save_path_refined, exist_ok=True) 
