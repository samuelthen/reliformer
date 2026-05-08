from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    sobel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sobel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    mag_max = mag.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    norm = mag / mag_max.clamp(min=1e-8)
    target = torch.sigmoid(12.0 * (norm - 0.25))
    return target.clamp(1e-4, 1.0 - 1e-4)


def edge_distillation_loss(edge_preds: List[torch.Tensor], gt_l: torch.Tensor) -> torch.Tensor:
    if gt_l.dim() == 3:
        gt_l = gt_l.unsqueeze(1)
    gt_l = gt_l.float().clamp(0.0, 1.0)
    gt_edge_full = sobel_edges(gt_l)
    total = gt_edge_full.new_tensor(0.0)

    for pred in edge_preds:
        target = F.interpolate(gt_edge_full, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        pred = pred.float().clamp(1e-4, 1.0 - 1e-4)
        target = target.float().clamp(1e-4, 1.0 - 1e-4)
        total = total + F.binary_cross_entropy(pred, target)

    return total / max(len(edge_preds), 1)


LAMBDA_DEFAULT = {"rgb": 1.0, "edge": 0.05}


def compute_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    lambdas: Dict[str, float] | None = None,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if lambdas is None:
        lambdas = LAMBDA_DEFAULT

    if device is None:
        device = pred["rgb"].device
    device = torch.device(device)

    gt_rgb = batch["target_rgb"].to(device).float()
    gt_l = batch["target_lab_L"].to(device).float()
    if gt_l.dim() == 3:
        gt_l = gt_l.unsqueeze(1)

    crit = CharbonnierLoss()
    l_rgb = crit(pred["rgb"], gt_rgb)

    if getattr(model, "last_edge_preds", None):
        with torch.amp.autocast(device_type=device.type, enabled=False):
            l_edge = edge_distillation_loss([e.float() for e in model.last_edge_preds], gt_l.float())
    else:
        l_edge = pred["rgb"].new_zeros(())

    loss = lambdas["rgb"] * l_rgb + lambdas["edge"] * l_edge
    logs = {
        "rgb": float(l_rgb.detach().cpu()),
        "edge": float(l_edge.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
    return loss, logs
