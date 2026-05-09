# model/readout.py

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helper: load and crop STAs ────────────────────────────────────────────────

def load_sta_crops(
    sta_dir:   "str | Path",
    centers_x: "array-like",
    centers_y: "array-like",
    crop_half: int,
    stride:    int = 1,
) -> np.ndarray:
    """
    Load STAs and crop them to (2*crop_half+1) x (2*crop_half+1) in feature space.

    centers_x/y in 108x108 pixels; crop_half and stride in feature pixels.
    File naming convention: cell_021_sta_z.npy ... cell_061_sta_z.npy.

    Returns: float32 [N, 2*crop_half+1, 2*crop_half+1]
    """
    sta_dir = Path(sta_dir)
    cs      = 2 * crop_half + 1
    crops   = []

    for i, (cx, cy) in enumerate(zip(centers_x, centers_y)):
        cell_id = 21 + i
        sta = np.load(sta_dir / f"cell_{cell_id:03d}_sta_z.npy").astype(np.float32)

        if stride > 1:
            sta_t = torch.from_numpy(sta).unsqueeze(0).unsqueeze(0)
            sta   = F.avg_pool2d(sta_t, kernel_size=stride, stride=stride).squeeze().numpy()

        H, W = sta.shape

        cy_int = int(round(float(cy) / stride))
        cx_int = int(round(float(cx) / stride))

        y0 = max(0, cy_int - crop_half);  y1 = y0 + cs
        if y1 > H: y1 = H;               y0 = y1 - cs

        x0 = max(0, cx_int - crop_half);  x1 = x0 + cs
        if x1 > W: x1 = W;               x0 = x1 - cs

        crops.append(sta[y0:y1, x0:x1])

    return np.stack(crops)   # [N, cs, cs]


# ── Readout ───────────────────────────────────────────────────────────────────

class FactorizedReadout(nn.Module):
    """
    Per-neuron factorized readout.

    For each neuron n:
        spatial_n = softmax(u_n) · x_n      [D]
        output_n  = ELU(spatial_n · v_n + bias_n) + 1  → always > 0

    u [N, T] is initialized from the cropped STA (per-neuron spatial mask).
    v [N, D] is initialized to small random values.

    Input:  [B, N, T, D]
    Output: [B, N]
    """

    def __init__(
        self,
        n_neurons: int,
        n_tokens:  int,
        emb_dim:   int,
        sta_crops: "np.ndarray | None" = None,   # [N, n_tokens]
    ):
        super().__init__()

        T = n_tokens

        u_init = torch.zeros(n_neurons, T)
        if sta_crops is not None:
            u_init = torch.tensor(sta_crops.reshape(n_neurons, T), dtype=torch.float32)
        self.u    = nn.Parameter(u_init)
        self.v    = nn.Parameter(torch.randn(n_neurons, emb_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, x: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:        [B, N, T, D]
        pad_mask: [N, T] bool, True = token outside this neuron's circle (ignore).
        output:   [B, N]
        """
        if pad_mask is not None:
            u_norm = F.softmax(self.u.masked_fill(pad_mask, float('-inf')), dim=-1)
        else:
            u_norm = F.softmax(self.u, dim=-1)                      # [N, T]
        spatial = torch.einsum('nt,bntd->bnd', u_norm, x)          # [B, N, D]
        out     = (spatial * self.v).sum(dim=-1) + self.bias        # [B, N]
        return F.elu(out) + 1.0


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.config import DEFAULT_CFG, N_CELLS

    cfg       = DEFAULT_CFG
    stride    = cfg.get('CNN_STRIDE', 1)
    radii_max = max(cfg['NEURON_RADII'])
    crop_half = max(1, math.ceil(radii_max / stride))
    emb_dim   = cfg['EMB_DIM'] if cfg['EMB_DIM'] is not None else cfg['CNN_DIM']

    rf        = np.load("ellipse_centers_exp2.npz")
    sta_crops = load_sta_crops("STA", rf['centers_x'], rf['centers_y'], crop_half, stride)
    T         = (2 * crop_half + 1) ** 2
    assert sta_crops.shape == (N_CELLS, 2*crop_half+1, 2*crop_half+1)
    print(f"STA crops: {sta_crops.shape}  min={sta_crops.min():.3f}  max={sta_crops.max():.3f}")

    readout = FactorizedReadout(N_CELLS, T, emb_dim, sta_crops)

    B   = 4
    x   = torch.randn(B, N_CELLS, T, emb_dim)
    out = readout(x)

    assert out.shape == (B, N_CELLS)
    assert out.min().item() > 0
    params = sum(p.numel() for p in readout.parameters())
    print(f"FactorizedReadout  input:  {tuple(x.shape)}")
    print(f"                   output: {tuple(out.shape)}")
    print(f"                   params: {params:,}  (u:{N_CELLS*T:,}  v:{N_CELLS*emb_dim:,}  bias:{N_CELLS})")
    print("OK")
