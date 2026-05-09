# model/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional_encoding import RoPE2D


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention con RoPE 2D su posizioni relative al centro del neurone.

    Input:  [B, T, emb_dim]   — T token, tutti validi (nessun padding)
    Output: [B, T, emb_dim]

    rel_row, rel_col : [T] float — spostamento rispetto al centro del cerchio,
    passati a RoPE per la codifica posizionale.
    """

    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        assert emb_dim % num_heads == 0, \
            f"emb_dim ({emb_dim}) deve essere divisibile per num_heads ({num_heads})"
        assert (emb_dim // num_heads) % 4 == 0, \
            f"head_dim ({emb_dim // num_heads}) deve essere divisibile per 4 (richiesto da RoPE)"

        self.num_heads = num_heads
        self.head_dim  = emb_dim // num_heads
        self.dropout_p = dropout

        self.W_q   = nn.Linear(emb_dim, emb_dim)
        self.W_k   = nn.Linear(emb_dim, emb_dim)
        self.W_v   = nn.Linear(emb_dim, emb_dim)
        self.W_out = nn.Linear(emb_dim, emb_dim)

        self.rope = RoPE2D(self.head_dim)

    def forward(self, x: torch.Tensor,
                rel_row: torch.Tensor,
                rel_col: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x               : [B, T, D]
        rel_row         : [T] o [B, T]  — Δrow dal centro
        rel_col         : [T] o [B, T]  — Δcol dal centro
        key_padding_mask: [B, T] bool, True = token di padding da ignorare
        """
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K = self.rope.apply(Q, K, rel_row, rel_col)

        # float additive mask: 0=valid, -inf=masked  (bool mask causes 20× SDPA slowdown
        # in bfloat16 on GB10 Blackwell / PyTorch 2.12 nightly — keep as float for safety)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(
                B, 1, 1, T, dtype=x.dtype, device=x.device
            ).masked_fill_(key_padding_mask[:, None, None, :], float('-inf'))

        drop_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V,
                                             attn_mask=attn_mask,
                                             dropout_p=drop_p)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_out(out)


class MLP(nn.Module):
    """Feed-forward applicato a ogni token indipendentemente."""

    def __init__(self, emb_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Singolo blocco: LN → Attention → residual → LN → MLP → residual.
    Propaga rel_row, rel_col all'attention; nessun padding/masking.
    """

    def __init__(self, emb_dim: int, num_heads: int, mlp_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = MultiHeadSelfAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp   = MLP(emb_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor,
                rel_row: torch.Tensor,
                rel_col: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rel_row, rel_col, key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Blocco 2: sequenza di TransformerBlock con RoPE 2D relativa.

    Riceve esattamente i T_n token validi di un singolo neurone (nessun padding).
    I pesi sono condivisi tra neuroni: model.py chiama questo modulo N volte
    in un loop, una per neurone, passando ogni volta la sequenza di lunghezza T_n.

      Input:  [B, T_n, in_dim]
      Output: [B, T_n, emb_dim]

    Se in_dim != emb_dim viene aggiunta una proiezione lineare in ingresso.
    """

    def __init__(self, in_dim: int, num_heads: int, mlp_dim: int, num_blocks: int,
                 emb_dim: int = None, dropout: float = 0.0):
        super().__init__()

        emb_dim = emb_dim if emb_dim is not None else in_dim

        self.proj = nn.Linear(in_dim, emb_dim) if emb_dim != in_dim else nn.Identity()

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor,
                rel_row: torch.Tensor,
                rel_col: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x               : [B, T_n, in_dim]
        rel_row         : [T_n] o [B, T_n]
        rel_col         : [T_n] o [B, T_n]
        key_padding_mask: [B, T_n] bool, True = padding
        output          : [B, T_n, emb_dim]
        """
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rel_row, rel_col, key_padding_mask)
        return x


# ── test rapido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.config import DEFAULT_CFG, N_CELLS

    cfg     = DEFAULT_CFG
    cnn_dim = cfg['CNN_DIM']
    emb_dim = cfg['EMB_DIM']
    out_dim = emb_dim if emb_dim is not None else cnn_dim
    B       = 4

    enc = TransformerEncoder(
        in_dim     = cnn_dim,
        num_heads  = cfg['NUM_HEADS'],
        mlp_dim    = cfg['MLP_DIM'],
        num_blocks = cfg['NUM_BLOCKS'],
        emb_dim    = emb_dim,
        dropout    = cfg['DROPOUT'],
    )

    # simula due neuroni con numero diverso di token
    for T_n, label in [(312, 'neurone A'), (587, 'neurone B')]:
        x       = torch.randn(B, T_n, cnn_dim)
        rel_row = torch.randn(T_n)   # posizioni relative float
        rel_col = torch.randn(T_n)
        out     = enc(x, rel_row, rel_col)
        assert out.shape == (B, T_n, out_dim)
        print(f"TransformerEncoder  {label}: in {tuple(x.shape)} → out {tuple(out.shape)}")

    params = sum(p.numel() for p in enc.parameters())
    print(f"                    params: {params:,}")
    print("OK")
