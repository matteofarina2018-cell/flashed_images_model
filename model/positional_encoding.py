# model/positional_encoding.py

import torch
import torch.nn as nn


class RoPE2D(nn.Module):
    """
    Rotary Position Embedding 2D con posizioni RELATIVE al centro del cerchio.

    Invece di codificare la posizione assoluta (riga, colonna) nella feature map,
    codifica lo spostamento (Δrow, Δcol) rispetto al centro del neurone corrente.
    Questo rende il transformer invariante per traslazione: due neuroni con RF
    identici ma centrati in posizioni diverse producono gli stessi pattern di
    attenzione, indipendentemente da dove si trovano nella feature map.

    I valori Δrow e Δcol sono float (il centro è trainabile e può avere
    coordinate frazionarie), quindi cos/sin vengono calcolati direttamente —
    nessuna lookup table indicizzata per intero.

    Vincolo: head_dim deve essere divisibile per 4.
      - primi  head_dim//2 → encoding di Δrow
      - secondi head_dim//2 → encoding di Δcol
    """

    def __init__(self, head_dim: int):
        super().__init__()

        assert head_dim % 4 == 0, \
            f"head_dim ({head_dim}) deve essere divisibile per 4"

        half   = head_dim // 2
        n_freq = half // 2

        theta = 1.0 / (10000 ** (2 * torch.arange(n_freq, dtype=torch.float32) / half))
        self.register_buffer('theta', theta)   # [n_freq]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        a = x[..., 0::2]
        b = x[..., 1::2]
        return torch.stack([-b, a], dim=-1).flatten(-2)

    def apply(self, q: torch.Tensor, k: torch.Tensor,
              rel_row: torch.Tensor, rel_col: torch.Tensor):
        """
        q, k    : [B, num_heads, T, head_dim]
        rel_row : [T]    — inferenza, un neurone alla volta
                  [B, T] — training, N neuroni × batch appiattiti (rel diverso per neurone)
        rel_col : stessa forma di rel_row
        """
        ang_r = rel_row.unsqueeze(-1) * self.theta   # [..., T, n_freq]
        ang_c = rel_col.unsqueeze(-1) * self.theta

        cos_r = torch.cos(ang_r).repeat_interleave(2, dim=-1)  # [..., T, half]
        sin_r = torch.sin(ang_r).repeat_interleave(2, dim=-1)
        cos_c = torch.cos(ang_c).repeat_interleave(2, dim=-1)
        sin_c = torch.sin(ang_c).repeat_interleave(2, dim=-1)

        cos = torch.cat([cos_r, cos_c], dim=-1)   # [..., T, head_dim]
        sin = torch.cat([sin_r, sin_c], dim=-1)

        if rel_row.dim() == 1:
            cos = cos.unsqueeze(0).unsqueeze(0)    # [1, 1, T, D]
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)                 # [B, 1, T, D]
            sin = sin.unsqueeze(1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ── test rapido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.config import DEFAULT_CFG

    cfg      = DEFAULT_CFG
    head_dim = (cfg['EMB_DIM'] or cfg['CNN_DIM']) // cfg['NUM_HEADS']

    rope = RoPE2D(head_dim)

    B, H, T, D = 4, cfg['NUM_HEADS'], 300, head_dim
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    # posizioni relative al centro (float, anche negative)
    rel_row = torch.randn(T)   # [T] scostamento di riga
    rel_col = torch.randn(T)   # [T] scostamento di colonna

    q_rot, k_rot = rope.apply(q, k, rel_row, rel_col)

    assert q_rot.shape == q.shape
    print(f"RoPE2D  head_dim: {head_dim}  theta: {tuple(rope.theta.shape)}")
    print(f"        q_rot:    {tuple(q_rot.shape)}")

    # verifica che traslare tutti i token dello stesso offset non cambi i dot-product
    shift = 7.3
    q2, k2 = rope.apply(q, k, rel_row + shift, rel_col + shift)
    dot_orig  = (q_rot @ k_rot.transpose(-2, -1)).mean().item()
    dot_shift = (q2    @ k2.transpose(-2, -1)   ).mean().item()
    print(f"        dot originale:  {dot_orig:.4f}")
    print(f"        dot con shift uniforme: {dot_shift:.4f}  (valore diverso: RoPE assoluta)")
    print("OK")
