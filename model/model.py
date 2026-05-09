# model/model.py

import math

import numpy as np
import torch
import torch.nn as nn

from model.config        import IMG_SIZE_ORIGINAL, IN_CHANNELS, N_CELLS
from model.tokenizer     import CNNTokenizer, feat_size
from model.neuron_circle import NeuronCircle
from model.transformer   import TransformerEncoder
from model.readout       import FactorizedReadout


class RetinalModel(nn.Module):
    """
    Full model: CNNTokenizer → NeuronCircle → TransformerEncoder → FactorizedReadout.

    Pipeline:
      [B, 1, 108, 108]
        → CNNTokenizer      [B, CNN_DIM, H', W']
        → NeuronCircle      [B, N, T, CNN_DIM]      T = T_circle (within max radius)
        → reshape           [B*N, T, CNN_DIM]
        → TransformerEncoder [B*N, T, EMB_DIM]
        → reshape           [B, N, T, EMB_DIM]
        → FactorizedReadout [B, N]
    """

    def __init__(self, cfg: dict, centers_x, centers_y, radii, sta_crops):
        """
        cfg       : configuration dictionary
        centers_x : [N] x-coordinate of RF centers in 108x108 pixels
        centers_y : [N] y-coordinate of RF centers in 108x108 pixels
        radii     : [N] per-neuron radii in 108x108 pixels
        sta_crops : [N, 2h+1, 2h+1] STAs cropped in feature space (from load_sta_crops)
        """
        super().__init__()

        cnn_dim = cfg['CNN_DIM']
        emb_dim = cfg['EMB_DIM']
        out_dim = emb_dim if emb_dim is not None else cnn_dim
        stride  = cfg.get('CNN_STRIDE', 1)

        fh = feat_size(IMG_SIZE_ORIGINAL, stride)

        cx     = np.asarray(centers_x, dtype=np.float32) / stride
        cy     = np.asarray(centers_y, dtype=np.float32) / stride
        r_feat = np.asarray(radii,     dtype=np.float32) / stride

        crop_half = max(1, math.ceil(float(np.asarray(radii).max()) / stride))

        self.n_cells  = N_CELLS
        self.crop_half = crop_half

        self.cnn = CNNTokenizer(
            in_channels = IN_CHANNELS,
            cnn_dim     = cnn_dim,
            cnn_layers  = cfg['CNN_LAYERS'],
            cnn_kernel  = cfg['CNN_KERNEL'],
            dropout     = cfg['DROPOUT'],
            stride      = stride,
        )

        self.circle = NeuronCircle(
            n_neurons = N_CELLS,
            img_h     = fh,
            img_w     = fh,
            centers_x = cx,
            centers_y = cy,
            radii     = r_feat,
            crop_half = crop_half,
        )

        # subset STA crops to the circular tokens only
        T_circle   = self.circle.T_circle
        T_sq       = (2 * crop_half + 1) ** 2
        idx_np     = self.circle.circle_indices.cpu().numpy()
        sta_circle = sta_crops.reshape(N_CELLS, T_sq)[:, idx_np]   # [N, T_circle]

        self.transformer = TransformerEncoder(
            in_dim     = cnn_dim,
            num_heads  = cfg['NUM_HEADS'],
            mlp_dim    = cfg['MLP_DIM'],
            num_blocks = cfg['NUM_BLOCKS'],
            emb_dim    = emb_dim,
            dropout    = cfg['DROPOUT'],
        )

        self.readout = FactorizedReadout(
            n_neurons = N_CELLS,
            n_tokens  = T_circle,
            emb_dim   = out_dim,
            sta_crops = sta_circle,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, 1, 108, 108]
        output: [B, N_cells]
        """
        B = x.shape[0]
        N = self.n_cells

        feat = self.cnn(x)                                        # [B, C, H', W']
        C    = feat.shape[1]

        tokens, rel_row, rel_col, pad_mask = self.circle(feat)
        T = tokens.shape[2]                                       # T_circle

        tokens  = tokens.reshape(B * N, T, C)
        rows    = rel_row.unsqueeze(0).expand(B * N, -1)
        cols    = rel_col.unsqueeze(0).expand(B * N, -1)
        pad_exp = pad_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * N, T)

        out = self.transformer(tokens, rows, cols, pad_exp)         # [B*N, T, D]

        out = out.reshape(B, N, T, out.shape[-1])
        return self.readout(out, pad_mask)                        # [B, N]
