# model/tokenizer.py

import math
import torch
import torch.nn as nn


class CNNTokenizer(nn.Module):
    """
    Blocco 1: CNN che trasforma immagini [B, in_channels, H, W] in feature maps.

    Lo stride è applicato al primo layer Conv2d: stride=1 mantiene la risoluzione,
    stride=2 dimezza (come MAX_POOL), stride=s riduce a ceil(H/s) × ceil(W/s).

    Struttura per layer: Conv2d → BatchNorm2d → GELU (+ Dropout2d opzionale)

    Input:  [B, in_channels, H, W]       tipicamente [B, 1, 108, 108]
    Output: [B, cnn_dim, H', W']         H' = ceil(H / stride)
    """

    def __init__(
        self,
        in_channels: int,
        cnn_dim:     int,
        cnn_layers:  int,
        cnn_kernel:  int,
        dropout:     float = 0.0,
        stride:      int   = 1,
    ):
        super().__init__()

        assert cnn_kernel % 2 == 1, \
            f"cnn_kernel deve essere dispari per same-padding, ricevuto {cnn_kernel}"
        assert stride >= 1, f"stride deve essere >= 1, ricevuto {stride}"

        padding = cnn_kernel // 2

        layers = []
        in_ch  = in_channels
        for i in range(cnn_layers):
            s = stride if i == 0 else 1   # stride solo al primo layer
            layers.append(nn.Conv2d(in_ch, cnn_dim, kernel_size=cnn_kernel,
                                    padding=padding, stride=s))
            layers.append(nn.BatchNorm2d(cnn_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = cnn_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def feat_size(img_size: int, stride: int) -> int:
    """Dimensione della feature map per un'immagine img_size × img_size con dato stride."""
    return math.ceil(img_size / stride)


# ── test rapido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.config import DEFAULT_CFG, IMG_SIZE_ORIGINAL, IN_CHANNELS

    cfg = DEFAULT_CFG
    B   = 4
    H = W = IMG_SIZE_ORIGINAL   # 108

    for s in [1, 2, 3, 4]:
        tok = CNNTokenizer(
            in_channels = IN_CHANNELS,
            cnn_dim     = cfg['CNN_DIM'],
            cnn_layers  = cfg['CNN_LAYERS'],
            cnn_kernel  = cfg['CNN_KERNEL'],
            dropout     = cfg['DROPOUT'],
            stride      = s,
        )
        x   = torch.randn(B, IN_CHANNELS, H, W)
        out = tok(x)
        exp_h = feat_size(H, s)
        assert out.shape == (B, cfg['CNN_DIM'], exp_h, exp_h), \
            f"stride={s}: atteso {exp_h}×{exp_h}, ottenuto {out.shape[-2]}×{out.shape[-1]}"
        params = sum(p.numel() for p in tok.parameters())
        print(f"stride={s}  input: {tuple(x.shape)}  output: {tuple(out.shape)}  params: {params:,}")

    print("OK")
