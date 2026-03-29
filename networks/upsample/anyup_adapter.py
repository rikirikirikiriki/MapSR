# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class AnyUpAdapter(nn.Module):
    """
    作用：
      - 从本地 anyup-main/ 加载 upsampler（hubconf.py）
      - 将【数据集归一化的输入】→ ImageNet 归一化的 RGB（仅取前三通道）
      - 调用 AnyUp 做特征上采样；显存不足时自动增大 q_chunk_size 重试
      - 兼容旧版 AnyUp（不支持 q_chunk_size/vis_attn 形参会自动降级）
      - hr_rgb 为空时自动退回双线性
    """

    def __init__(self, repo_dir, q_chunk_size: int = 16, vis_attn: bool = False, device: str = "cuda", **model_kwargs):
        super().__init__()
        repo_dir = Path(repo_dir)
        if not (repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(f"anyup hubconf.py 未找到: {repo_dir}")

        # —— 加载 AnyUp（本地 hub）；允许透传可选构造参数，若旧版不支持则回退 —— #
        try:
            self.upsampler = torch.hub.load(str(repo_dir), "anyup", source="local", **model_kwargs)
        except TypeError:
            # 旧版 anyup 构造不带这些 kwargs
            self.upsampler = torch.hub.load(str(repo_dir), "anyup", source="local")

        self.q_chunk_size = int(max(1, q_chunk_size))
        self.vis_attn = bool(vis_attn)
        self.to(device)
        self.upsampler.eval()
        for p in self.upsampler.parameters():
            p.requires_grad = False

        # —— 检测是否有 NATTEN 后端（仅用于日志提示） —— #
        has_natten = False
        try:
            import natten  # noqa: F401
            has_natten = True
        except Exception:
            has_natten = hasattr(torch.ops, "natten")
        self.has_natten = bool(has_natten)
        print(f"[AnyUpAdapter] NATTEN backend detected: {self.has_natten}; q_chunk_size={self.q_chunk_size}")

    @staticmethod
    def to_imagenet_rgb_from_dataset(x_ds: torch.Tensor, ds_mean, ds_std) -> torch.Tensor:
        """
        x_ds: (B,C,H,W) 已按【数据集】均值/方差归一化；例如 x=(raw255-mean)/std
        ds_mean, ds_std: 至少前三个数对应 RGB，单位仍为 0~255 标度
        返回: (B,3,H,W) 的 ImageNet 归一化 RGB
        """
        device = x_ds.device
        dm = torch.as_tensor(ds_mean[:3], device=device).view(1, 3, 1, 1).float()
        ds = torch.as_tensor(ds_std[:3],  device=device).view(1, 3, 1, 1).float()
        rgb01 = ((x_ds[:, :3] * ds + dm) / 255.0).clamp(0, 1)
        return (rgb01 - _IM_MEAN.to(device)) / _IM_STD.to(device)

    # -------- 内部：对 anyup 的一次调用，兼容旧版形参 -------- #
    def _call_anyup_once(self, hr_rgb_imgnet, lr_feats, out_hw, q_chunk_size, vis_attn):
        # 新版：支持 q_chunk_size / vis_attn
        try:
            return self.upsampler(
                hr_rgb_imgnet, lr_feats,
                output_size=out_hw,
                q_chunk_size=int(q_chunk_size),
                vis_attn=bool(vis_attn)
            )
        except TypeError:
            # 旧版：不支持这些关键字
            return self.upsampler(
                hr_rgb_imgnet, lr_feats,
                output_size=out_hw
            )

    def upsample(self, hr_rgb_imgnet: torch.Tensor, lr_feats: torch.Tensor, out_hw) -> torch.Tensor:
        """
        将低分辨率特征 lr_feats 上采样到 out_hw（H, W）。
        优先用 AnyUp；hr_rgb_imgnet 缺失时退回双线性。
        遇到 OOM 自动【减小】q_chunk_size（最多 4 次）；全部失败则回退双线性（保证不返回 None）。
        """
        # 没有引导图就退回双线性（兜底）
        if hr_rgb_imgnet is None:
            return F.interpolate(lr_feats, size=out_hw, mode="bilinear", align_corners=False)

        tries = 0
        qcs = int(max(1, self.q_chunk_size))

        while tries < 4:
            try:
                return self._call_anyup_once(
                    hr_rgb_imgnet=hr_rgb_imgnet,
                    lr_feats=lr_feats,
                    out_hw=out_hw,
                    q_chunk_size=qcs,
                    vis_attn=self.vis_attn
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                msg = str(e)
                if ("out of memory" in msg.lower()) or ("alloc" in msg.lower()):
                    tries += 1
                    new_qcs = max(1, qcs // 2)     # 关键：OOM 时减半
                    if new_qcs == qcs:
                        break                      # 已到极限，跳出改用双线性
                    qcs = new_qcs
                    torch.cuda.empty_cache()
                    print(f"[AnyUpAdapter] OOM -> retry with smaller q_chunk_size={qcs} (try {tries}/4)")
                    continue
                # 其他错误直接抛出
                raise

        # 全部重试失败：安全回退双线性，不返回 None
        print("[AnyUpAdapter] All retries failed; falling back to bilinear upsampling.")
        return F.interpolate(lr_feats, size=out_hw, mode="bilinear", align_corners=False)



    # 允许在外部动态调整分块大小
    def set_chunk_size(self, q_chunk_size: int):
        self.q_chunk_size = int(max(1, q_chunk_size))
        print(f"[AnyUpAdapter] set q_chunk_size = {self.q_chunk_size}")
