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
from reliformer.losses import CharbonnierLoss, edge_distillation_loss
from reliformer.model import GrayscaleBaseline

LAMBDA_GRAY = {"gray": 1.0, "edge": 0.05}


def _to_image_hwc_uint8(x: torch.Tensor):
    x = x.detach().float().clamp(0.0, 1.0).cpu()
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    elif x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    x = (x.numpy() * 255.0).round().astype("uint8")
    return x


def compute_loss_gray(pred, batch, model, device):
    gt_l = batch["target_lab_L"].to(device).float()
    if gt_l.dim() == 3:
        gt_l = gt_l.unsqueeze(1)

    crit = CharbonnierLoss()
    l_gray = crit(pred["gray"], gt_l)

    if getattr(model, "last_edge_preds", None):
        with torch.amp.autocast(device_type=device.type, enabled=False):
            l_edge = edge_distillation_loss(
                [e.float() for e in model.last_edge_preds], gt_l.float()
            )
    else:
        l_edge = pred["gray"].new_zeros(())

    loss = LAMBDA_GRAY["gray"] * l_gray + LAMBDA_GRAY["edge"] * l_edge
    logs = {
        "gray": float(l_gray.detach().cpu()),
        "edge": float(l_edge.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
    return loss, logs


def evaluate(model, loader, device, max_samples=20, vis_samples=4):
    import wandb

    model.eval()
    total_loss = total_gray = total_edge = total_psnr = 0.0
    n = 0

    vis_pred, vis_gt, vis_proxy, vis_e1, vis_e2, vis_e3 = [], [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            if n >= max_samples:
                break

            spad = batch["spad"].to(device)
            cmos = batch["cmos"].to(device)
            pred = model(spad, cmos)
            loss, logs = compute_loss_gray(pred, batch, model, device)

            gt_l = batch["target_lab_L"].to(device).float()
            if gt_l.dim() == 3:
                gt_l = gt_l.unsqueeze(1)

            mse = (pred["gray"] - gt_l).pow(2).mean().clamp(min=1e-10)
            psnr = -10.0 * torch.log10(mse)

            total_loss += float(loss.detach().cpu())
            total_gray += logs["gray"]
            total_edge += logs["edge"]
            total_psnr += float(psnr.detach().cpu())

            if len(vis_pred) < vis_samples:
                idx = 0
                vis_pred.append(wandb.Image(_to_image_hwc_uint8(pred["gray"][idx]), caption=f"eval_{n:02d}_pred_gray"))
                vis_gt.append(wandb.Image(_to_image_hwc_uint8(gt_l[idx]), caption=f"eval_{n:02d}_gt_gray"))
                vis_proxy.append(wandb.Image(_to_image_hwc_uint8(pred["luma_proxy"][idx]), caption=f"eval_{n:02d}_luma_proxy"))
                vis_e1.append(wandb.Image(_to_image_hwc_uint8(pred["edges"][0][idx]), caption=f"eval_{n:02d}_edge_s1"))
                vis_e2.append(wandb.Image(_to_image_hwc_uint8(pred["edges"][1][idx]), caption=f"eval_{n:02d}_edge_s2"))
                vis_e3.append(wandb.Image(_to_image_hwc_uint8(pred["edges"][2][idx]), caption=f"eval_{n:02d}_edge_s3"))

            n += 1

    d = max(n, 1)
    return {
        "loss": total_loss / d,
        "gray": total_gray / d,
        "edge": total_edge / d,
        "psnr": total_psnr / d,
        "samples": n,
        "vis_pred": vis_pred,
        "vis_gt": vis_gt,
        "vis_proxy": vis_proxy,
        "vis_e1": vis_e1,
        "vis_e2": vis_e2,
        "vis_e3": vis_e3,
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
    p.add_argument("--out", default="checkpoints_gray")
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
        args.data_root, crop=args.crop, t=args.t,
        ppp=args.ppp, bins=args.bins, cmos_sigma=args.cmos_sigma,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=max(1, args.num_workers // 2), pin_memory=True)
    train_iter = iter(train_loader)

    model = GrayscaleBaseline(t=args.t, base_c=args.base_c,
                               n_blocks=args.n_blocks, n_fpm=args.n_fpm).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.9))
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    if wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )
        wandb.watch(model, log="gradients", log_freq=100)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total = running_gray = running_edge = 0.0

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
                loss, logs = compute_loss_gray(pred, batch, model, device)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            running_total += logs["total"]
            running_gray += logs["gray"]
            running_edge += logs["edge"]

            if wandb is not None:
                wandb.log(
                    {
                        "train/loss_total": logs["total"],
                        "train/loss_gray": logs["gray"],
                        "train/loss_edge": logs["edge"],
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/step_in_epoch": step,
                    },
                    step=global_step,
                )

            if step % 20 == 0:
                print(f"epoch={epoch:03d} step={step:04d} loss={logs['total']:.4f} "
                      f"gray={logs['gray']:.4f} edge={logs['edge']:.4f}")

        sch.step()
        eval_out = evaluate(model, test_loader, device,
                            max_samples=args.eval_samples, vis_samples=args.vis_samples)
        avg_total = running_total / max(args.steps_per_epoch, 1)
        avg_gray  = running_gray  / max(args.steps_per_epoch, 1)
        avg_edge  = running_edge  / max(args.steps_per_epoch, 1)
        print(
            f"epoch={epoch:03d} "
            f"train_total={avg_total:.4f} train_gray={avg_gray:.4f} train_edge={avg_edge:.4f} "
            f"val_loss={eval_out['loss']:.4f} val_psnr={eval_out['psnr']:.2f}dB "
            f"eval_n={eval_out['samples']}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "eval/loss_total": eval_out["loss"],
                    "eval/loss_gray": eval_out["gray"],
                    "eval/loss_edge": eval_out["edge"],
                    "eval/psnr": eval_out["psnr"],
                    "eval/samples": eval_out["samples"],
                    "eval/vis_pred_gray": eval_out["vis_pred"],
                    "eval/vis_gt_gray": eval_out["vis_gt"],
                    "eval/vis_luma_proxy": eval_out["vis_proxy"],
                    "eval/vis_edge_s1": eval_out["vis_e1"],
                    "eval/vis_edge_s2": eval_out["vis_e2"],
                    "eval/vis_edge_s3": eval_out["vis_e3"],
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
        torch.save(ckpt, out_dir / f"gray_epoch_{epoch:03d}.pt")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
