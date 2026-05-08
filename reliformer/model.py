import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def spad_physics_reliability(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clamp(0.0, 1.0)
    return 4.0 * obs * (1.0 - obs)


def cmos_motion_reliability(edge_map: torch.Tensor) -> torch.Tensor:
    return (1.0 - edge_map.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def cmos_exposure_validity(
    cmos: torch.Tensor,
    dark_thresh: float = 0.02,
    sat_thresh: float = 0.98,
    sharpness: float = 30.0,
) -> torch.Tensor:
    p = cmos.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    dark_ok = torch.sigmoid(sharpness * (p - dark_thresh))
    sat_ok = torch.sigmoid(sharpness * (sat_thresh - p))
    return (dark_ok * sat_ok).clamp(0.0, 1.0)


class NAFBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.pw1 = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.pw2 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw1(self.norm(x))
        h = self.dw(h)
        a, b = h.chunk(2, dim=1)
        h = self.pw2(a * torch.sigmoid(b))
        return x + self.beta * h


class DownBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, n_blocks: int):
        super().__init__()
        self.down = nn.Conv2d(in_c, out_c, 3, stride=2, padding=1)
        self.body = nn.Sequential(*[NAFBlock(out_c) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = x
        x = self.body(self.down(x))
        return x, skip


class UpBlockAdd(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, n_blocks: int):
        super().__init__()
        self.in_proj = nn.Conv2d(in_c, out_c, 1)
        self.skip_proj = nn.Conv2d(skip_c, out_c, 1)
        self.body = nn.Sequential(*[NAFBlock(out_c) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.in_proj(x) + self.skip_proj(skip)
        return self.body(x)


class PhotonReliabilityEncoder(nn.Module):
    def __init__(self, c1: int, n_fpm: int):
        super().__init__()
        self.stem = nn.Conv2d(8, c1, 3, padding=1)
        self.fpm = nn.Sequential(*[NAFBlock(c1) for _ in range(n_fpm)])

    def forward(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r_t = spad_physics_reliability(s_t)
        sr = torch.stack([s_t, r_t], dim=1)
        pus = F.pixel_unshuffle(sr, 2)
        feat = self.fpm(self.stem(pus))
        rel = F.pixel_unshuffle(r_t.unsqueeze(1), 2).mean(dim=1, keepdim=True)
        return feat, rel


class ReliabilityGatedTemporalAttention(nn.Module):
    def __init__(self, c: int, t: int):
        super().__init__()
        self.t = t
        self.center = t // 2
        self.n_heads = max(1, c // 8)
        assert c % self.n_heads == 0
        self.d_head = c // self.n_heads

        self.q_proj = nn.Linear(c, c)
        self.k_proj = nn.Linear(c, c)
        self.v_proj = nn.Linear(c, c)
        self.out_proj = nn.Linear(c, c)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.time_embed = nn.Embedding(t, c)
        nn.init.normal_(self.time_embed.weight, std=0.02)

    def forward(self, feat_list: List[torch.Tensor], rel_list: List[torch.Tensor]) -> torch.Tensor:
        b, c, hh, wh = feat_list[0].shape
        n = b * hh * wh
        feat_stack = torch.stack(feat_list, dim=1)
        rel_stack = torch.stack(rel_list, dim=1)

        feat_flat = feat_stack.permute(0, 3, 4, 1, 2).contiguous().reshape(n, self.t, c)
        rel_flat = rel_stack.permute(0, 3, 4, 1, 2).contiguous().reshape(n, self.t, 1)

        time_ids = torch.arange(self.t, device=feat_flat.device)
        feat_pos = feat_flat + self.time_embed(time_ids).unsqueeze(0)

        r_w = torch.softmax(rel_flat, dim=1)
        q_feat = (feat_flat * r_w).sum(dim=1)

        q = self.q_proj(q_feat).reshape(n, self.n_heads, self.d_head)
        k = self.k_proj(feat_pos).reshape(n, self.t, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(feat_pos).reshape(n, self.t, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        attn = (q.unsqueeze(2) * k).sum(-1) * (self.d_head ** -0.5)
        r_bias = torch.log(rel_flat.squeeze(-1).clamp(min=1e-6))
        attn = torch.softmax(attn + r_bias.unsqueeze(1), dim=-1)

        out = (attn.unsqueeze(-1) * v).sum(2).reshape(n, c)
        out = self.out_proj(out)
        out = out.reshape(b, hh, wh, c).permute(0, 3, 1, 2).contiguous()
        return feat_list[self.center] + out


class ColorPyramid(nn.Module):
    def __init__(self, c1: int, c2: int, c3: int, n_fpm: int):
        super().__init__()
        self.scale1 = nn.Sequential(nn.Conv2d(4, c1, 3, padding=1), *[NAFBlock(c1) for _ in range(n_fpm)])
        self.scale2 = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(c1 * 4, c2, 1), NAFBlock(c2))
        self.scale3 = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(c2 * 4, c3, 1), NAFBlock(c3))

    def forward(self, cmos: torch.Tensor) -> List[torch.Tensor]:
        c1 = self.scale1(cmos)
        c2 = self.scale2(c1)
        c3 = self.scale3(c2)
        return [c1, c2, c3]


class SPADEdgePyramid(nn.Module):
    def __init__(self, c1: int):
        super().__init__()
        self.luma_proj = nn.Conv2d(c1 + 1, 1, 1)
        self.edge1 = self._make_laplacian()
        self.edge2 = self._make_laplacian()
        self.edge3 = self._make_laplacian()

    @staticmethod
    def _make_laplacian() -> nn.Conv2d:
        conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[0, 0, 1, 1] = 4.0
            conv.weight[0, 0, 0, 1] = -1.0
            conv.weight[0, 0, 2, 1] = -1.0
            conv.weight[0, 0, 1, 0] = -1.0
            conv.weight[0, 0, 1, 2] = -1.0
        return conv

    @staticmethod
    def _edge_map(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        e = torch.abs(conv(x))
        emax = e.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        norm = e / emax.clamp(min=1e-8)
        edge = torch.sigmoid(12.0 * (norm - 0.25))
        return edge.clamp(1e-4, 1.0 - 1e-4)

    def forward(self, fused: torch.Tensor, mean_rel: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        luma_proxy = torch.sigmoid(self.luma_proj(torch.cat([fused, mean_rel], dim=1)))
        e1 = self._edge_map(self.edge1, luma_proxy)
        l2 = F.avg_pool2d(luma_proxy, 2).detach()
        l4 = F.avg_pool2d(luma_proxy, 4).detach()
        e2 = self._edge_map(self.edge2, l2)
        e3 = self._edge_map(self.edge3, l4)
        return [e1, e2, e3], luma_proxy


class AsymmetricCrossModalFusion(nn.Module):
    def __init__(self, c: int, use_cmos_exposure_mask: bool = False):
        super().__init__()
        r = max(1, c // 8)
        self.scale = r ** -0.5
        self.q_proj = nn.Conv2d(c, r, 1)
        self.k_proj = nn.Conv2d(c, r, 1)
        self.v_proj = nn.Conv2d(c, c, 1)
        self.out_proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.neutral_prior = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.use_cmos_exposure_mask = use_cmos_exposure_mask

    def forward(
        self,
        fused: torch.Tensor,
        f_cmos: torch.Tensor,
        i_cmos: torch.Tensor,
        r_spad: torch.Tensor,
        edge_map: torch.Tensor,
    ) -> torch.Tensor:
        if edge_map.shape[-2:] != fused.shape[-2:]:
            edge_map = F.interpolate(edge_map, fused.shape[-2:], mode="bilinear", align_corners=False)
        if f_cmos.shape[-2:] != fused.shape[-2:]:
            f_cmos = F.interpolate(f_cmos, fused.shape[-2:], mode="bilinear", align_corners=False)

        r_cmos = cmos_motion_reliability(edge_map)
        if self.use_cmos_exposure_mask:
            if i_cmos.shape[-2:] != fused.shape[-2:]:
                i_cmos = F.interpolate(i_cmos, fused.shape[-2:], mode="bilinear", align_corners=False)
            r_cmos = r_cmos * cmos_exposure_validity(i_cmos)
        r_cmos = r_cmos.clamp(0.0, 1.0)

        q = self.q_proj(fused)
        k = self.k_proj(f_cmos)
        v = self.v_proj(f_cmos)
        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) * self.scale)
        attn = attn * r_cmos

        colour_blend = attn * v + (1.0 - attn) * fused
        neutral = self.neutral_prior.expand_as(colour_blend)
        reliable_blend = r_spad * colour_blend + (1.0 - r_spad) * neutral
        return reliable_blend + self.out_proj(reliable_blend)


class ColorCrossAttention(nn.Module):
    def __init__(self, c_feat: int, c_color: int):
        super().__init__()
        r = max(1, c_feat // 4)
        self.scale = r ** -0.5
        self.q_proj = nn.Conv2d(c_feat, r, 1)
        self.k_proj = nn.Conv2d(c_color, r, 1)
        self.v_proj = nn.Conv2d(c_color, c_feat, 1)
        self.out_proj = nn.Conv2d(c_feat, c_feat, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, feat: torch.Tensor, color_feat: torch.Tensor) -> torch.Tensor:
        if color_feat.shape[-2:] != feat.shape[-2:]:
            color_feat = F.interpolate(color_feat, feat.shape[-2:], mode="bilinear", align_corners=False)
        q = self.q_proj(feat)
        k = self.k_proj(color_feat)
        v = self.v_proj(color_feat)
        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) * self.scale)
        delta = attn * (v - feat)
        return feat + self.out_proj(delta)


class StructureGate(nn.Module):
    def __init__(self, c_feat: int, init_threshold: float = 0.90):
        super().__init__()
        init_threshold = float(min(max(init_threshold, 1e-4), 1.0 - 1e-4))
        init_logit = math.log(init_threshold / (1.0 - init_threshold))
        self.threshold_logit = nn.Parameter(torch.full((1, c_feat, 1, 1), init_logit))
        self.sharpness = 20.0

    def forward(self, feat_coloured: torch.Tensor, feat_original: torch.Tensor, edge_map: torch.Tensor) -> torch.Tensor:
        if edge_map.shape[-2:] != feat_coloured.shape[-2:]:
            edge_map = F.interpolate(edge_map, feat_coloured.shape[-2:], mode="bilinear", align_corners=False)
        edge_map = edge_map.clamp(0.0, 1.0)
        threshold = torch.sigmoid(self.threshold_logit)
        gate = 1.0 - torch.sigmoid(self.sharpness * (edge_map - threshold))
        return gate * feat_coloured + (1.0 - gate) * feat_original


class GrayscaleBaseline(nn.Module):
    """
    SPAD-only grayscale reconstruction baseline.

    Same SPAD pathway and U-Net backbone as ReliFormer, but no CMOS branch,
    no colour injection, and a single-channel output head. Used as a benchmark
    to measure how much the CMOS colour pathway contributes.
    """

    def __init__(self, t: int = 11, base_c: int = 32, n_blocks: int = 2, n_fpm: int = 2):
        super().__init__()
        self.t = t
        self.center = t // 2

        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        self.pre = PhotonReliabilityEncoder(c1, n_fpm)
        self.rgta = ReliabilityGatedTemporalAttention(c1, t)
        self.edge_pyr = SPADEdgePyramid(c1)

        self.down1 = DownBlock(c1, c2, n_blocks)
        self.down2 = DownBlock(c2, c3, n_blocks)
        self.down3 = DownBlock(c3, c4, n_blocks)
        self.mid = nn.Sequential(*[NAFBlock(c4) for _ in range(n_blocks * 2)])
        self.up3 = UpBlockAdd(c4, c3, c3, n_blocks)
        self.up2 = UpBlockAdd(c3, c2, c2, n_blocks)
        self.up1 = UpBlockAdd(c2, c1, c1, n_blocks)

        self.gray_head = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(c1 // 4, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.last_edge_preds: List[torch.Tensor] = []
        self.last_luma_proxy: torch.Tensor | None = None

    def forward(self, spad: torch.Tensor, cmos: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        feat_list, rel_list = [], []
        for tt in range(self.t):
            feat_t, rel_t = self.pre(spad[:, tt])
            feat_list.append(feat_t)
            rel_list.append(rel_t)

        mean_rel = torch.stack(rel_list, dim=1).mean(dim=1)
        fused = self.rgta(feat_list, rel_list)

        edges, luma_proxy = self.edge_pyr(fused, mean_rel)
        self.last_edge_preds = edges
        self.last_luma_proxy = luma_proxy

        x, s1 = self.down1(fused)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x = self.mid(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        gray = self.gray_head(x)
        return {"gray": gray, "luma_proxy": luma_proxy, "edges": edges}


class ReliFormer(nn.Module):
    def __init__(self, t: int = 11, base_c: int = 32, n_blocks: int = 2, n_fpm: int = 2):
        super().__init__()
        self.t = t
        self.center = t // 2

        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        self.pre = PhotonReliabilityEncoder(c1, n_fpm)
        self.rgta = ReliabilityGatedTemporalAttention(c1, t)
        self.edge_pyr = SPADEdgePyramid(c1)

        self.color_pyr = ColorPyramid(c1, c2, c3, n_fpm)
        self.acmf = AsymmetricCrossModalFusion(c1)

        self.cca_enc = ColorCrossAttention(c1, c1)
        self.cca_d1 = ColorCrossAttention(c2, c2)
        self.cca_d2 = ColorCrossAttention(c3, c3)
        self.cca_u3 = ColorCrossAttention(c3, c3)
        self.cca_u2 = ColorCrossAttention(c2, c2)
        self.cca_u1 = ColorCrossAttention(c1, c1)

        self.gate_enc = StructureGate(c1)
        self.gate_d1 = StructureGate(c2)
        self.gate_d2 = StructureGate(c3)
        self.gate_u3 = StructureGate(c3)
        self.gate_u2 = StructureGate(c2)
        self.gate_u1 = StructureGate(c1)

        self.down1 = DownBlock(c1, c2, n_blocks)
        self.down2 = DownBlock(c2, c3, n_blocks)
        self.down3 = DownBlock(c3, c4, n_blocks)
        self.mid = nn.Sequential(*[NAFBlock(c4) for _ in range(n_blocks * 2)])
        self.up3 = UpBlockAdd(c4, c3, c3, n_blocks)
        self.up2 = UpBlockAdd(c3, c2, c2, n_blocks)
        self.up1 = UpBlockAdd(c2, c1, c1, n_blocks)

        self.rgb_head = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(c1 // 4, 3, 3, padding=1), nn.Sigmoid())

        self.last_edge_preds: List[torch.Tensor] = []
        self.last_luma_proxy: torch.Tensor | None = None

    @staticmethod
    def _inject_colour(
        x: torch.Tensor,
        cca: ColorCrossAttention,
        gate: StructureGate,
        color_feat: torch.Tensor,
        edge_map: torch.Tensor,
    ) -> torch.Tensor:
        x_coloured = cca(x, color_feat)
        return gate(x_coloured, x, edge_map)

    def forward(self, spad: torch.Tensor, cmos: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_list, rel_list = [], []
        for tt in range(self.t):
            feat_t, rel_t = self.pre(spad[:, tt])
            feat_list.append(feat_t)
            rel_list.append(rel_t)

        mean_rel = torch.stack(rel_list, dim=1).mean(dim=1)
        fused = self.rgta(feat_list, rel_list)

        edges, luma_proxy = self.edge_pyr(fused, mean_rel)
        self.last_edge_preds = edges
        self.last_luma_proxy = luma_proxy

        color_feats = self.color_pyr(cmos)
        f_cross = self.acmf(fused=fused, f_cmos=color_feats[0], i_cmos=cmos, r_spad=mean_rel, edge_map=edges[0])

        x = self._inject_colour(f_cross, self.cca_enc, self.gate_enc, color_feats[0], edges[0])

        x, s1 = self.down1(x)
        x = self._inject_colour(x, self.cca_d1, self.gate_d1, color_feats[1], edges[1])

        x, s2 = self.down2(x)
        x = self._inject_colour(x, self.cca_d2, self.gate_d2, color_feats[2], edges[2])

        x, s3 = self.down3(x)
        x = self.mid(x)

        x = self.up3(x, s3)
        x = self._inject_colour(x, self.cca_u3, self.gate_u3, color_feats[2], edges[2])

        x = self.up2(x, s2)
        x = self._inject_colour(x, self.cca_u2, self.gate_u2, color_feats[1], edges[1])

        x = self.up1(x, s1)
        x = self._inject_colour(x, self.cca_u1, self.gate_u1, color_feats[0], edges[0])

        rgb = self.rgb_head(x)
        luma = 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
        return {"rgb": rgb, "luma": luma, "luma_proxy": luma_proxy, "edges": edges}
