# model/losses.py

import torch


def poisson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Poisson negative log-likelihood loss.

    Derivata dalla distribuzione di Poisson:
        P(k spikes | rate λ) = λ^k * e^(-λ) / k!
        -log P = λ - k*log(λ) + costante

    In pratica:
        L = mean( y_pred - y_true * log(y_pred) )

    y_pred : [B, N_cells] — risposta predetta  (deve essere > 0, garantito da ELU+1)
    y_true : [B, N_cells] — risposta registrata (spike counts)
    """
    return (y_pred - y_true * torch.log(y_pred + 1e-8)).mean()


# ── test rapido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    y_pred = torch.tensor([[1.2, 0.5, 2.1],
                           [0.8, 1.9, 0.3]])
    y_true = torch.tensor([[1.0, 0.0, 2.0],
                           [1.0, 2.0, 0.0]])

    loss         = poisson_loss(y_pred, y_true)
    loss_perfect = poisson_loss(y_true, y_true)

    print(f"Poisson loss:  {loss.item():.4f}")
    print(f"Loss perfetta: {loss_perfect.item():.4f}  (deve essere < sopra)")
    assert loss_perfect.item() < loss.item()
    print("OK")
