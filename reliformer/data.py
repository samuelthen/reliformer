from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.distributions import Binomial


_FILE_EXTS = ("*.npz", "*.pt", "*.pth")


def _discover_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in _FILE_EXTS:
        files.extend(root.rglob(ext))
    return sorted(set(files))


def _discover_frame_dirs(root: Path) -> List[Path]:
    frame_dirs: List[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if any(d.glob("*.png")):
            frame_dirs.append(d)
    return frame_dirs


def _select_key(d: Dict, keys: Sequence[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)


class VFIReliFormerDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop: int | None = None,
        seed: int = 0,
        t: int = 11,
        ppp: float = 3.25,
        bins: int = 7,
        cmos_sigma: float = 2.0,
    ):
        self.root = Path(root)
        self.split = split
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        self.t = int(t)
        self.ppp = float(ppp)
        self.bins = int(bins)
        self.cmos_sigma = float(cmos_sigma)

        split_dir = self.root / split
        self.mode = "file"
        if split_dir.exists():
            file_samples = _discover_files(split_dir)
            frame_dirs = _discover_frame_dirs(split_dir)
            if file_samples:
                self.samples = file_samples
            elif frame_dirs:
                self.mode = "frame_dir"
                self.samples = frame_dirs
            else:
                self.samples = []
        else:
            all_samples = _discover_files(self.root)
            if not all_samples:
                raise FileNotFoundError(f"No dataset files found under {self.root}")
            pivot = int(0.9 * len(all_samples))
            self.samples = all_samples[:pivot] if split == "train" else all_samples[pivot:]

        if not self.samples:
            raise RuntimeError(f"No samples discovered for split='{split}' under {self.root}")

    @staticmethod
    def _read_png_rgb(path: Path, retries: int = 3) -> torch.Tensor:
        last_err: Exception | None = None
        for _ in range(max(retries, 1)):
            try:
                with Image.open(path) as img:
                    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            except (OSError, ValueError) as e:
                last_err = e
        raise OSError(f"Failed to read image after {retries} retries: {path}") from last_err

    def _candidate_centers(self, n_frames: int, half: int, max_tries: int = 12) -> List[int]:
        lo = half
        hi = n_frames - half - 1
        if lo > hi:
            return []

        if self.split == "train":
            span = hi - lo + 1
            n_pick = min(max_tries, span)
            picks = self.rng.choice(np.arange(lo, hi + 1), size=n_pick, replace=False)
            return [int(x) for x in picks.tolist()]

        center = n_frames // 2
        center = min(max(center, lo), hi)
        ordered = [center]
        for r in range(1, max_tries):
            left = center - r
            right = center + r
            if left >= lo:
                ordered.append(left)
            if right <= hi:
                ordered.append(right)
            if len(ordered) >= max_tries:
                break
        return ordered

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _rgb_to_lab_l(rgb: torch.Tensor) -> torch.Tensor:
        # rgb: (3,H,W) sRGB in [0,1], output: CIE L* normalized to [0,1]
        rgb = rgb.clamp(0.0, 1.0)
        a = 0.055
        rgb_lin = torch.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + a) / (1.0 + a)).pow(2.4),
        )
        x = 0.4124564 * rgb_lin[0] + 0.3575761 * rgb_lin[1] + 0.1804375 * rgb_lin[2]
        y = 0.2126729 * rgb_lin[0] + 0.7151522 * rgb_lin[1] + 0.0721750 * rgb_lin[2]
        z = 0.0193339 * rgb_lin[0] + 0.1191920 * rgb_lin[1] + 0.9503041 * rgb_lin[2]

        # D65 white
        xr = x / 0.95047
        yr = y / 1.00000
        zr = z / 1.08883
        eps = 216.0 / 24389.0
        kappa = 24389.0 / 27.0

        def f(t: torch.Tensor) -> torch.Tensor:
            return torch.where(t > eps, t.pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

        fy = f(yr)
        l_star = (116.0 * fy - 16.0).clamp(0.0, 100.0)
        return (l_star / 100.0).clamp(0.0, 1.0)

    @staticmethod
    def _pack_rggb(rgb: torch.Tensor) -> torch.Tensor:
        # rgb: (3,H,W) -> (4,H/2,W/2) packed RGGB
        r = rgb[0:1, 0::2, 0::2]
        g1 = rgb[1:2, 0::2, 1::2]
        g2 = rgb[1:2, 1::2, 0::2]
        b = rgb[2:3, 1::2, 1::2]
        return torch.cat([r, g1, g2, b], dim=0).clamp(0.0, 1.0)

    @staticmethod
    def _bayer_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
        m = torch.zeros(h, w, 3, dtype=torch.float32, device=device)
        m[0::2, 0::2, 0] = 1.0
        m[0::2, 1::2, 1] = 1.0
        m[1::2, 0::2, 1] = 1.0
        m[1::2, 1::2, 2] = 1.0
        return m

    @staticmethod
    def _pack_rggb_raw(raw: torch.Tensor) -> torch.Tensor:
        # raw: (H,W) -> (4,H/2,W/2) RGGB packed
        return torch.stack(
            [raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]],
            dim=0,
        ).clamp(0.0, 1.0)

    def _simulate_forward_model(self, rgb_burst: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Match quanta_fuse_v2_b simulate_mono + simulate_color_cmos path.
        burst = torch.stack(rgb_burst, dim=0).clamp(0.0, 1.0)  # (T,3,H,W), gamma-encoded
        x_lin = burst.pow(2.2)
        lum = x_lin.sum(dim=1)  # (T,H,W)
        mean_lum = lum.mean().clamp(min=1e-8)
        alpha = self.ppp / float(mean_lum.item())
        scaled = x_lin * alpha

        # SPAD mono burst: per-frame Binomial with p=1-exp(-lambda/bins)
        spad_frames: List[torch.Tensor] = []
        for t in range(scaled.shape[0]):
            lam = scaled[t].sum(dim=0).clamp(min=0.0)  # (H,W)
            p = (1.0 - torch.exp(-lam / max(float(self.bins), 1.0))).clamp(0.0, 1.0)
            cnt = Binomial(total_count=max(self.bins, 1), probs=p).sample()
            spad_frames.append((cnt / max(float(self.bins), 1.0)).clamp(0.0, 1.0))
        spad = torch.stack(spad_frames, dim=0)

        # CMOS packed RGGB: integrate all T in linear domain + Poisson-Gaussian readout.
        _, _, h, w = scaled.shape
        mask = self._bayer_mask(h, w, scaled.device)  # (H,W,3)
        scaled_hwc = scaled.permute(0, 2, 3, 1)  # (T,H,W,3)
        cmos_lam = (scaled_hwc.sum(dim=0) * mask).sum(dim=-1).clamp(min=0.0)  # (H,W)
        cmos_cnt = torch.poisson(cmos_lam)
        cmos_cnt = cmos_cnt + torch.randn_like(cmos_cnt) * self.cmos_sigma
        cmos_adc = torch.clamp(torch.round(cmos_cnt), 0.0, 255.0) / 255.0
        cmos = self._pack_rggb_raw(cmos_adc)
        return spad, cmos

    def _read_frame_dir_sample(self, seq_dir: Path) -> Dict[str, torch.Tensor]:
        frame_paths = sorted(seq_dir.glob("*.png"))
        if len(frame_paths) < self.t:
            raise RuntimeError(f"{seq_dir} has {len(frame_paths)} frames, but at least {self.t} are required")

        half = self.t // 2
        last_err: Exception | None = None
        rgbs: List[torch.Tensor] | None = None

        for center in self._candidate_centers(len(frame_paths), half, max_tries=12):
            sel = frame_paths[center - half : center + half + 1]
            if len(sel) != self.t:
                continue
            try:
                rgbs = [self._read_png_rgb(p, retries=3) for p in sel]
                break
            except (OSError, ValueError) as e:
                last_err = e
                rgbs = None
                continue

        if rgbs is None:
            raise RuntimeError(f"Could not assemble valid frame window for {seq_dir}") from last_err

        target_rgb = rgbs[half]
        spad, cmos = self._simulate_forward_model(rgbs)
        target_l = self._rgb_to_lab_l(target_rgb)

        return {
            "spad": spad.clamp(0.0, 1.0),
            "cmos": cmos.clamp(0.0, 1.0),
            "target_rgb": target_rgb.clamp(0.0, 1.0),
            "target_lab_L": target_l.clamp(0.0, 1.0),
        }

    def _read_sample(self, path: Path) -> Dict[str, torch.Tensor]:
        if path.suffix == ".npz":
            raw = dict(np.load(path, allow_pickle=False))
        else:
            raw = torch.load(path, map_location="cpu")
            if isinstance(raw, dict) and "state_dict" in raw:
                raw = raw["state_dict"]

        spad = _select_key(raw, ["spad", "spad_burst", "burst_spad"])
        cmos = _select_key(raw, ["cmos", "rggb", "bayer_packed"])
        target_rgb = _select_key(raw, ["target_rgb", "rgb", "gt_rgb"])
        target_l = _select_key(raw, ["target_lab_L", "target_L", "lab_L", "luma_target"])

        if spad is None or cmos is None or target_rgb is None:
            raise KeyError(f"{path} must include spad/cmos/target_rgb compatible keys")

        spad = _to_tensor(spad)
        cmos = _to_tensor(cmos)
        target_rgb = _to_tensor(target_rgb)

        if target_l is None:
            target_l = 0.2126 * target_rgb[0] + 0.7152 * target_rgb[1] + 0.0722 * target_rgb[2]
        target_l = _to_tensor(target_l)

        if spad.dim() != 3:
            raise ValueError(f"Expected spad shape (T,H,W), got {tuple(spad.shape)} in {path}")

        if cmos.dim() != 3 or cmos.shape[0] != 4:
            raise ValueError(f"Expected cmos shape (4,H/2,W/2), got {tuple(cmos.shape)} in {path}")

        if target_rgb.dim() != 3 or target_rgb.shape[0] != 3:
            raise ValueError(f"Expected target_rgb shape (3,H,W), got {tuple(target_rgb.shape)} in {path}")

        return {
            "spad": spad.clamp(0.0, 1.0),
            "cmos": cmos.clamp(0.0, 1.0),
            "target_rgb": target_rgb.clamp(0.0, 1.0),
            "target_lab_L": target_l.clamp(0.0, 1.0),
        }

    def _random_crop(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.crop is None:
            return sample
        spad = sample["spad"]
        _, h, w = spad.shape
        crop = min(self.crop, h, w)
        if crop == h and crop == w:
            return sample

        top = int(self.rng.integers(0, h - crop + 1))
        left = int(self.rng.integers(0, w - crop + 1))

        c_top, c_left, c_crop = top // 2, left // 2, crop // 2

        sample["spad"] = sample["spad"][:, top : top + crop, left : left + crop]
        sample["target_rgb"] = sample["target_rgb"][:, top : top + crop, left : left + crop]
        if sample["target_lab_L"].dim() == 2:
            sample["target_lab_L"] = sample["target_lab_L"][top : top + crop, left : left + crop]
        else:
            sample["target_lab_L"] = sample["target_lab_L"][:, top : top + crop, left : left + crop]
        sample["cmos"] = sample["cmos"][:, c_top : c_top + c_crop, c_left : c_left + c_crop]
        return sample

    def _augment(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.split != "train":
            return sample
        if float(self.rng.random()) < 0.5:
            sample["spad"] = torch.flip(sample["spad"], dims=[-1])
            sample["cmos"] = torch.flip(sample["cmos"], dims=[-1])
            sample["target_rgb"] = torch.flip(sample["target_rgb"], dims=[-1])
            sample["target_lab_L"] = torch.flip(sample["target_lab_L"], dims=[-1])
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_tries = min(12, len(self.samples))
        last_err: Exception | None = None

        for k in range(max_tries):
            if self.split == "train":
                pick = int(self.rng.integers(0, len(self.samples)))
            else:
                pick = (idx + k) % len(self.samples)
            path = self.samples[pick]

            try:
                if self.mode == "frame_dir":
                    sample = self._read_frame_dir_sample(path)
                else:
                    sample = self._read_sample(path)
                sample = self._random_crop(sample)
                sample = self._augment(sample)
                return sample
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                last_err = e
                continue

        raise RuntimeError(f"Failed to load sample after {max_tries} attempts (split={self.split})") from last_err


def build_datasets(
    root: str | Path,
    crop: int = 256,
    seed: int = 0,
    t: int = 11,
    ppp: float = 3.25,
    bins: int = 7,
    cmos_sigma: float = 2.0,
) -> Tuple[VFIReliFormerDataset, VFIReliFormerDataset]:
    train_ds = VFIReliFormerDataset(
        root=root,
        split="train",
        crop=crop,
        seed=seed,
        t=t,
        ppp=ppp,
        bins=bins,
        cmos_sigma=cmos_sigma,
    )
    test_ds = VFIReliFormerDataset(
        root=root,
        split="test",
        crop=None,
        seed=seed,
        t=t,
        ppp=ppp,
        bins=bins,
        cmos_sigma=cmos_sigma,
    )
    return train_ds, test_ds
