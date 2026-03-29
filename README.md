# MapSR Inference Pipeline

## Environment

Python 3.10 is recommended for this project:

```bash
conda create -n mapsr python=3.10
conda activate mapsr
pip install -r requirements.txt
```

> Note: For GPU support, please refer to the [PyTorch website](https://pytorch.org/) to install the appropriate CUDA version.

## Dataset

- **Download Link**: [Chesapeake Land Cover - LILA BC](https://lila.science/datasets/chesapeakelandcover)
- **Placement**: 
  - Place CSV list files into `dataset/CSV_list/`.
  - Place remote sensing `.tif` images according to the paths specified in the CSV files.

## Weights

- **Download Link**: <https://pan.baidu.com/s/13SfEEDTacGDd1AnsJHozqw?pwd=igdw>
- **Placement**: Place the `.pth` file in the `networks/pre-train_model/DinoV2_LinearProb_exp_20251226_225930/` directory (or modify the `snapshot` path in `config.py`).

## Parameters

Edit `config.py` to modify the following core parameters:

- `GPU_ID`: Specify the GPU to use, default is `"1"`.
- `dataset_name`: Name of the dataset for the current test (e.g., `"10test_Virginia1"`).
- `image_num`: Number of images to test.
- `use_slide`: `True` for sliding window method (more stable, medium speed), `False` for superpixel method (better boundary adherence but slower).
- `only_s1`: If `True`, skips the S2 smoothing step and only performs S1 prototype similarity inference.

## Execution

```bash
python main.py
```

Upon completion, the program will generate the following in the `GRSL_experiments` directory at the project root:

- `linear_prob_xxx`: S1 prediction results
- `refined_xxx`: S2 denoised prediction results
