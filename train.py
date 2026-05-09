# train.py

import math

import numpy as np
import torch
from torch.optim import Adam

from data          import get_dataloaders
from model.config  import DEFAULT_CFG
from model.model   import RetinalModel
from model.losses  import poisson_loss
from model.readout import load_sta_crops


# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "PNAS_paper_sorted_data.npz"
RF_PATH   = "ellipse_centers_exp2.npz"
STA_PATH  = "STA"
CKPT_PATH = "best_model.pt"

cfg = DEFAULT_CFG


# ── model construction ───────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device) -> RetinalModel:
    rf        = np.load(RF_PATH)
    centers_x = rf['centers_x']
    centers_y = rf['centers_y']

    radii     = np.asarray(cfg['NEURON_RADII'], dtype=np.float32)
    stride    = cfg.get('CNN_STRIDE', 1)
    crop_half = max(1, math.ceil(float(radii.max()) / stride))

    sta_crops = load_sta_crops(STA_PATH, centers_x, centers_y, crop_half, stride)
    model     = RetinalModel(cfg, centers_x, centers_y, radii, sta_crops)
    return model.to(device)


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict = cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        DATA_PATH,
        batch_size       = cfg['BATCH_SIZE'],
        img_noise_sigma  = cfg.get('IMG_NOISE_SIGMA', 0.0),
        poisson_resample = cfg.get('POISSON_RESAMPLE', False),
        aug_factor       = cfg.get('AUG_FACTOR', 1),
    )

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['LEARNING_RATE'], weight_decay=cfg['WEIGHT_DECAY'],
    )

    best_val_loss = float('inf')
    patience_cnt  = 0

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float32)

    for epoch in range(1, cfg['MAX_EPOCHS'] + 1):

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images    = batch['image'].to(device)
            responses = batch['response'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss = poisson_loss(model(images), responses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), amp_ctx:
            for batch in val_loader:
                val_loss += poisson_loss(
                    model(batch['image'].to(device)),
                    batch['response'].to(device),
                ).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:4d}  train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_loss':    best_val_loss,
                'cfg':         cfg,
            }, CKPT_PATH)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg['EARLY_STOP']:
                print(f"Early stopping at epoch {epoch}  (best val: {best_val_loss:.4f})")
                break

    print(f"Training complete — best val loss: {best_val_loss:.4f}  → {CKPT_PATH}")
    return best_val_loss


if __name__ == "__main__":
    train()
