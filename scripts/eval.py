#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliformer.data import VFIReliFormerDataset
from reliformer.losses import compute_loss
from reliformer.metrics import LPIPSMetric, psnr, ssim
from reliformer.model import ReliFormer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="vfi_dataset/i2-2kfps_v1_flat")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ppp", type=float, default=3.25)
    p.add_argument("--bins", type=int, default=7)
    p.add_argument("--cmos-sigma", type=float, default=2.0)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    margs = ckpt.get("args", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = VFIReliFormerDataset(
        args.data_root,
        split="test",
        crop=None,
        t=margs.get("t", 11),
        ppp=args.ppp,
        bins=args.bins,
        cmos_sigma=args.cmos_sigma,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ReliFormer(
        t=margs.get("t", 11),
        base_c=margs.get("base_c", 32),
        n_blocks=margs.get("n_blocks", 2),
        n_fpm=margs.get("n_fpm", 2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    lpips_metric = LPIPSMetric(device)
    total_loss, total_psnr, total_ssim, total_lpips, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            spad = batch["spad"].to(device)
            cmos = batch["cmos"].to(device)
            pred = model(spad, cmos)
            loss, _ = compute_loss(pred, batch, model, device=device)
            target = batch["target_rgb"].to(device)
            total_loss += float(loss.detach().cpu())
            total_psnr += float(psnr(pred["rgb"], target).detach().cpu())
            total_ssim += float(ssim(pred["rgb"], target).detach().cpu())
            total_lpips += float(lpips_metric(pred["rgb"], target).detach().cpu())
            n += 1

    print(
        f"test_loss={total_loss/max(n,1):.4f} "
        f"test_psnr={total_psnr/max(n,1):.2f}dB "
        f"test_ssim={total_ssim/max(n,1):.4f} "
        f"test_lpips={total_lpips/max(n,1):.4f}"
    )


if __name__ == "__main__":
    main()
