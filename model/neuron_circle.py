# model/neuron_circle.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronCircle(nn.Module):
    """
    Differentiable per-neuron circular crop.

    Extracts the T_circle tokens that fall inside a circle of radius r_max
    (the largest of the per-neuron radii). Since radii are fixed (from config),
    T_circle is constant and the per-neuron mask is precomputed at init.

    Input:  [B, C, H, W]
    Output: tokens   [B, N, T_circle, C]
            rel_row  [T_circle]   row offset relative to center  (for RoPE)
            rel_col  [T_circle]   col offset relative to center  (for RoPE)
            pad_mask [N, T_circle]  True = token outside this neuron's circle

    Trainable parameters:
        cx : [N]  x-coordinate of the center, in feature pixels
        cy : [N]  y-coordinate of the center, in feature pixels

    Buffers:
        neuron_mask [N, T_circle]  True  = token INSIDE this neuron's circle
        pad_mask    [N, T_circle]  True  = token OUTSIDE  (= ~neuron_mask)
    """

    def __init__(
        self,
        n_neurons:  int,
        img_h:      int,
        img_w:      int,
        centers_x:  "array-like",
        centers_y:  "array-like",
        radii:      "array-like",   # [N] radii in feature-space pixels
        crop_half:  int,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.img_h     = img_h
        self.img_w     = img_w
        self.crop_half = crop_half

        self.cx = nn.Parameter(torch.tensor(centers_x, dtype=torch.float32))
        self.cy = nn.Parameter(torch.tensor(centers_y, dtype=torch.float32))

        # ── precompute circle token positions ─────────────────────────────────
        h       = crop_half
        offsets = torch.arange(-h, h + 1, dtype=torch.float32)
        off_y, off_x = torch.meshgrid(offsets, offsets, indexing='ij')
        dist    = torch.sqrt(off_y ** 2 + off_x ** 2).reshape(-1)  # [T_sq]

        r_arr = torch.tensor(np.asarray(radii, dtype=np.float32))   # [N] feature px
        r_max = float(r_arr.max().item())

        # tokens in the union of all circles (set by the largest radius)
        idx          = (dist <= r_max).nonzero(as_tuple=True)[0]   # [T_circle]
        self.T_circle = int(idx.shape[0])

        self.register_buffer('circle_off_x',   off_x.reshape(-1)[idx])   # [T_circle]
        self.register_buffer('circle_off_y',   off_y.reshape(-1)[idx])   # [T_circle]
        self.register_buffer('dist_circle',    dist[idx])                 # [T_circle]
        self.register_buffer('circle_indices', idx)                       # [T_circle]

        # static per-neuron mask: True = token INSIDE this neuron's circle
        neuron_mask = r_arr.unsqueeze(1) >= dist[idx].unsqueeze(0)   # [N, T_circle] bool
        self.register_buffer('neuron_mask', neuron_mask)
        # pad_mask: True = token OUTSIDE (used as key_padding_mask in transformer)
        self.register_buffer('pad_mask', ~neuron_mask)

    # ── helper ────────────────────────────────────────────────────────────────
    def _to_normalized(self, px: torch.Tensor, size: int) -> torch.Tensor:
        return 2.0 * px / (size - 1) - 1.0

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        Returns: tokens [B, N, T_circle, C], rel_row [T], rel_col [T], pad_mask [N, T]
        """
        B, C, H, W = x.shape
        N = self.n_neurons
        T = self.T_circle

        cx_norm = self._to_normalized(self.cx, W)
        cy_norm = self._to_normalized(self.cy, H)

        off_x_norm = self.circle_off_x / (W / 2.0)
        off_y_norm = self.circle_off_y / (H / 2.0)

        grids_x = cx_norm.view(N, 1, 1) + off_x_norm.view(1, 1, T)  # [N, 1, T]
        grids_y = cy_norm.view(N, 1, 1) + off_y_norm.view(1, 1, T)  # [N, 1, T]
        grids   = torch.stack([grids_x, grids_y], dim=-1)             # [N, 1, T, 2]

        x_exp    = x.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)
        grid_exp = grids.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * N, 1, T, 2)

        feat = F.grid_sample(x_exp, grid_exp,
                             mode='bilinear', padding_mode='border',
                             align_corners=True)               # [B*N, C, 1, T]
        feat = feat.squeeze(2).permute(0, 2, 1)               # [B*N, T, C]
        feat = feat.reshape(B, N, T, C)                       # [B, N, T, C]

        return feat, self.circle_off_y, self.circle_off_x, self.pad_mask
