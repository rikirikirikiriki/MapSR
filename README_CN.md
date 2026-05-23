# MapSR: 基于视觉基础模型和 Prompt 驱动的土地覆盖地图超分辨率

获取高空间分辨率（HR）的土地覆盖制图通常受到密集的 HR 标注成本高的限制。我们从地图超分辨率（MapSR）的角度重新审视这个问题，将粗糙的低空间分辨率（LR）土地覆盖产品提升为与输入图像分辨率一致的 HR 地图。MapSR 是一个由 Prompt 驱动的框架，它将监督信号与模型训练解耦。它利用 LR 标签，通过轻量级线性探测器（Linear Probe）从冻结的视觉基础模型特征中提取类别 Prompt，随后通过免训练的度量推理和基于图的预测优化来进行 HR 制图。

![MapSR 结构概览](Figure/Overview_of_MapSR.png)

## 环境配置

请使用 Python 3.10 和 PyTorch 2.0.1 创建 Conda 虚拟环境。

```bash
# 创建并激活 conda 虚拟环境
conda create -n mapsr python=3.10 -y
conda activate mapsr

# 安装 PyTorch (以 CUDA 11.8 为例，您可以根据自己的显卡驱动调整)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖包
pip install -r requirements.txt

# 使用纯 Python 模式安装 mmcv（跳过容易报错的 CUDA 编译过程）
MMCV_WITH_OPS=0 pip install mmcv==1.6.0 --no-build-isolation

# 固定 numpy 和 opencv 的版本以避免兼容性问题
pip install numpy==1.26.4 opencv-python==4.11.0.86
```

## 准备模型权重

本模型依赖于我们训练好的权重文件以及预训练的 DINOv2 权重。

1. 请从 [HuggingFace](https://huggingface.co/rikirikirikiriki/DinoV2_LinearProb/tree/main) 或 [百度网盘](https://pan.baidu.com/s/1eMn8fv9tvM0MNm2E0QBWpw)（提取码: `qhjx`）下载我们预训练好的模型权重 `epoch_10.pth`。
2. 将下载好的 `epoch_10.pth` 放入以下目录：
   ```text
   networks/pre-train_model/epoch_10.pth
   ```

*(注：HuggingFace 上的 DINOv2 基础模型权重会在您第一次运行代码时自动下载)。*

## 准备数据集

数据加载器会根据 CSV 文件中记录的路径来读取图像。

1. 请从 [HuggingFace](https://huggingface.co/datasets/rikirikirikiriki/ChesapeakeBay/tree/main) 或 [百度网盘](https://pan.baidu.com/s/1wBpycuYMR20Y9Ja10emqpw)（提取码: `peqd`）下载测试数据集（例如：Chesapeake Bay 数据集）。
2. *(可选)* 完整的 ChesapeakeBay 数据集请前往 [LILA BC](https://lila.science/datasets/chesapeakelandcover) 下载。
3. 将下载好的数据集解压放入 `dataset/` 目录中。
4. 请确保您 CSV 列表文件（位于 `dataset/CSV_list/` 目录下）中记录的 `.tif` 图像路径与您本地实际存放图像的绝对/相对路径一致。

## 运行推理与结果优化

```bash
python main.py --dataset_name 10test_New_York --get_prompts_mode OneShot
```
```bash
python main.py --dataset_name 10test_New_York --get_prompts_mode ZeroShot
```


## 计算评价指标 (mIoU)

```bash
python calc_metric.py --dataset 10test_New_York --method OneShot
```
```bash
python calc_metric.py --dataset 10test_New_York --method ZeroShot
```
