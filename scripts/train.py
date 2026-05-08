#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliformer.data import build_datasets
from reliformer.losses import LAMBDA_DEFAULT, compute_loss
from reliformer.metrics import LPIPSMetric, psnr as metric_psnr, ssim as metric_ssim
from reliformer.model import ReliFormer


def _to_image_hwc_uint8(x: torch.Tensor):
    x = x.detach().float().clamp(0.0, 1.0).cpu()
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    elif x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    elif x.dim() == 3 and x.shape[-1] in (1, 3):
        pass
    else:
        raise ValueError(f"Unsupported tensor shape for image conversion: {tuple(x.shape)}")
    x = (x.numpy() * 255.0).round().astype("uint8")
    return x


def evaluate(model, loader, device, lpips_metric, max_samples: int = 20, vis_samples: int = 4):
    import wandb

    model.eval()
    total_loss = 0.0
    total_rgb = 0.0
    total_edge = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    n = 0

    vis_pred_rgb = []
    vis_target_rgb = []
    vis_luma = []
    vis_luma_proxy = []
    vis_edge_1 = []
    vis_edge_2 = []
    vis_edge_3 = []

    with torch.no_grad():
        for batch in loader:
            if n >= max_samples:
                break

            spad = batch["spad"].to(device)
            cmos = batch["cmos"].to(device)
            pred = model(spad, cmos)
            loss, logs = compute_loss(pred, batch, model, device=device)
            target_rgb = batch["target_rgb"].to(device)

            v_psnr = metric_psnr(pred["rgb"], target_rgb)
            v_ssim = metric_ssim(pred["rgb"], target_rgb)
            v_lpips = lpips_metric(pred["rgb"], target_rgb)

            total_loss += float(loss.detach().cpu())
            total_rgb += logs["rgb"]
            total_edge += logs["edge"]
            total_psnr += float(v_psnr.detach().cpu())
            total_ssim += float(v_ssim.detach().cpu())
            total_lpips += float(v_lpips.detach().cpu())

            if len(vis_pred_rgb) < vis_samples:
                idx = 0
                vis_pred_rgb.append(wandb.Image(_to_image_hwc_uint8(pred["rgb"][idx]), caption=f"eval_{n:02d}_pred_rgb"))
                vis_target_rgb.append(wandb.Image(_to_image_hwc_uint8(target_rgb[idx]), caption=f"eval_{n:02d}_target_rgb"))
                vis_luma.append(wandb.Image(_to_image_hwc_uint8(pred["luma"][idx]), caption=f"eval_{n:02d}_pred_luma"))
                vis_luma_proxy.append(wandb.Image(_to_image_hwc_uint8(pred["luma_proxy"][idx]), caption=f"eval_{n:02d}_luma_proxy"))
                vis_edge_1.append(wandb.Image(_to_image_hwc_uint8(pred["edges"][0][idx]), caption=f"eval_{n:02d}_edge_s1"))
                vis_edge_2.append(wandb.Image(_to_image_hwc_uint8(pred["edges"][1][idx]), caption=f"eval_{n:02d}_edge_s2"))
                vis_edge_3.append(wandb.Image(_to_image_hwc_uint8(pred["edges"][2][idx]), caption=f"eval_{n:02d}_edge_s3"))

            n += 1

    denom = max(n, 1)
    return {
        "loss": total_loss / denom,
        "rgb": total_rgb / denom,
        "edge": total_edge / denom,
        "psnr": total_psnr / denom,
        "ssim": total_ssim / denom,
        "lpips": total_lpips / denom,
        "samples": n,
        "vis_pred_rgb": vis_pred_rgb,
        "vis_target_rgb": vis_target_rgb,
        "vis_luma": vis_luma,
        "vis_luma_proxy": vis_luma_proxy,
        "vis_edge_1": vis_edge_1,
        "vis_edge_2": vis_edge_2,
        "vis_edge_3": vis_edge_3,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="vfi_dataset/i2-2kfps_v1_flat")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=20)
    p.add_argument("--vis-samples", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--base-c", type=int, default=32)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--n-fpm", type=int, default=2)
    p.add_argument("--t", type=int, default=11)
    p.add_argument("--ppp", type=float, default=3.25)
    p.add_argument("--bins", type=int, default=7)
    p.add_argument("--cmos-sigma", type=float, default=2.0)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    p.add_argument("--wandb-project", default="reliformer")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    if args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError("wandb is required. Install with `pip install wandb`.") from e
    else:
        wandb = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = build_datasets(
        args.data_root,
        crop=args.crop,
        t=args.t,
        ppp=args.ppp,
        bins=args.bins,
        cmos_sigma=args.cmos_sigma,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)
    train_iter = iter(train_loader)

    model = ReliFormer(t=args.t, base_c=args.base_c, n_blocks=args.n_blocks, n_fpm=args.n_fpm).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.9))
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))
    start_epoch = 1
    lpips_metric = LPIPSMetric(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sch.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )
        wandb.watch(model, log="gradients", log_freq=100)

    global_step = (start_epoch - 1) * args.steps_per_epoch
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_rgb = 0.0
        running_edge = 0.0

        for step in range(1, args.steps_per_epoch + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            spad = batch["spad"].to(device, non_blocking=True)
            cmos = batch["cmos"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(spad, cmos)
                loss, logs = compute_loss(pred, batch, model, lambdas=LAMBDA_DEFAULT, device=device)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            running_total += logs["total"]
            running_rgb += logs["rgb"]
            running_edge += logs["edge"]

            if wandb is not None:
                wandb.log(
                    {
                        "train/loss_total": logs["total"],
                        "train/loss_rgb": logs["rgb"],
                        "train/loss_edge": logs["edge"],
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/step_in_epoch": step,
                    },
                    step=global_step,
                )

            if step % 20 == 0:
                print(f"epoch={epoch:03d} step={step:04d} loss={logs['total']:.4f} rgb={logs['rgb']:.4f} edge={logs['edge']:.4f}")

        sch.step()
        eval_out = evaluate(
            model,
            test_loader,
            device,
            lpips_metric=lpips_metric,
            max_samples=args.eval_samples,
            vis_samples=args.vis_samples,
        )
        avg_total = running_total / max(args.steps_per_epoch, 1)
        avg_rgb = running_rgb / max(args.steps_per_epoch, 1)
        avg_edge = running_edge / max(args.steps_per_epoch, 1)
        print(
            f"epoch={epoch:03d} "
            f"train_total={avg_total:.4f} train_rgb={avg_rgb:.4f} train_edge={avg_edge:.4f} "
            f"val_loss={eval_out['loss']:.4f} val_psnr={eval_out['psnr']:.2f}dB "
            f"val_ssim={eval_out['ssim']:.4f} val_lpips={eval_out['lpips']:.4f} eval_n={eval_out['samples']}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "eval/loss_total": eval_out["loss"],
                    "eval/loss_rgb": eval_out["rgb"],
                    "eval/loss_edge": eval_out["edge"],
                    "eval/psnr": eval_out["psnr"],
                    "eval/ssim": eval_out["ssim"],
                    "eval/lpips": eval_out["lpips"],
                    "eval/samples": eval_out["samples"],
                    "eval/vis_pred_rgb": eval_out["vis_pred_rgb"],
                    "eval/vis_target_rgb": eval_out["vis_target_rgb"],
                    "eval/vis_luma": eval_out["vis_luma"],
                    "eval/vis_luma_proxy": eval_out["vis_luma_proxy"],
                    "eval/vis_edge_s1": eval_out["vis_edge_1"],
                    "eval/vis_edge_s2": eval_out["vis_edge_2"],
                    "eval/vis_edge_s3": eval_out["vis_edge_3"],
                    "epoch": epoch,
                },
                step=global_step,
            )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / f"reliformer_epoch_{epoch:03d}.pt")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
