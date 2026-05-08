from __future__ import annotations

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2).clamp(min=1e-10)
    return -10.0 * torch.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return structural_similarity_index_measure(pred, target, data_range=1.0)


class LPIPSMetric:
    def __init__(self, device: torch.device):
        try:
            import lpips
        except ImportError as e:
            raise RuntimeError("LPIPS dependency missing. Install with `pip install lpips`.") from e
        self.net = lpips.LPIPS(net="alex").to(device)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = pred.clamp(0.0, 1.0) * 2.0 - 1.0
        target_n = target.clamp(0.0, 1.0) * 2.0 - 1.0
        v = self.net(pred_n, target_n)
        return v.mean()
