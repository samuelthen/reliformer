#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliformer.losses import compute_loss
from reliformer.model import ColorCrossAttention, ReliFormer, StructureGate


def run_sanity_checks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReliFormer(t=11, base_c=32, n_blocks=2, n_fpm=2).to(device)

    b, h = 2, 64
    hh = h // 2

    spad = torch.rand(b, 11, h, h, device=device)
    cmos = torch.rand(b, 4, hh, hh, device=device)

    out = model(spad, cmos)
    assert out["rgb"].shape == (b, 3, h, h)
    assert out["luma"].shape == (b, 1, h, h)
    assert out["rgb"].min().item() >= 0.0
    assert out["rgb"].max().item() <= 1.0

    manual_luma = 0.2126 * out["rgb"][:, 0:1] + 0.7152 * out["rgb"][:, 1:2] + 0.0722 * out["rgb"][:, 2:3]
    assert (out["luma"] - manual_luma).abs().max().item() < 1e-5

    assert len(model.last_edge_preds) == 3
    for e in model.last_edge_preds:
        assert e.dim() == 4 and e.shape[1] == 1
        assert e.min().item() >= 0.0 and e.max().item() <= 1.0

    color_feats = model.color_pyr(cmos)
    assert color_feats[0].shape == (b, 32, hh, hh)
    assert color_feats[1].shape == (b, 64, hh // 2, hh // 2)
    assert color_feats[2].shape == (b, 128, hh // 4, hh // 4)

    feat = torch.randn(b, 32, hh, hh, device=device)
    color = torch.randn(b, 32, hh, hh, device=device)
    cca = ColorCrossAttention(32, 32).to(device)
    cca_out = cca(feat, color)
    assert (cca_out - feat).abs().max().item() < 1e-6

    gate = StructureGate(32).to(device)
    threshold = torch.sigmoid(gate.threshold_logit).mean().item()
    assert abs(threshold - 0.90) < 1e-3

    feat_col = torch.ones(b, 32, hh, hh, device=device)
    feat_ori = torch.zeros(b, 32, hh, hh, device=device)
    edge_high = torch.ones(b, 1, hh, hh, device=device)
    edge_low = torch.zeros(b, 1, hh, hh, device=device)

    out_edge = gate(feat_col, feat_ori, edge_high)
    out_flat = gate(feat_col, feat_ori, edge_low)
    assert out_edge.mean().item() < out_flat.mean().item()

    batch = {
        "target_rgb": torch.rand(b, 3, h, h, device=device),
        "target_lab_L": torch.rand(b, h, h, device=device),
    }
    loss, logs = compute_loss(out, batch, model, device=device)
    assert torch.isfinite(loss)
    assert "rgb" in logs and "edge" in logs and "total" in logs

    model.zero_grad(set_to_none=True)
    loss.backward()
    assert model.edge_pyr.luma_proj.weight.grad is not None
    assert torch.isfinite(model.edge_pyr.luma_proj.weight.grad).all()

    print("All sanity checks passed.")


if __name__ == "__main__":
    run_sanity_checks()
