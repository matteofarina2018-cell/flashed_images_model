#!/usr/bin/env python
"""Retrain trial #511 (best post-focused-search, mean_all=0.8043) for ONE seed.

Same structure as retrain_trial143.py but with trial #511's exact config:
- arch:   CNN_DIM=32 k=3 s=2 EMB=None heads=1 blocks=2 mlp=64
- radii:  BEST_RADII (from #143) with 5 free neurons overridden by #511's values
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from bayes_search import (
    _adjusted_r2_per_neuron,
    _compute_lsta_corr_per_neuron,
    _compute_scores,
    _predict_test,
    _train_config,
    count_params_per_block,
    BEST_RADII,
    DATA_PATH,
    LSTA_REF_PATH,
    RF_PATH,
    STA_PATH,
)
from data         import RetinalDataset, _NUM_WORKERS, _PIN_MEMORY
from model.config import BAYES_FIXED


# Trial #511 arch (DB indices → values via BAYES_SPACE)
TRIAL_511_ARCH = {
    'CNN_DIM':    32,    # CNN_DIM_exp=5
    'CNN_KERNEL': 3,     # idx 0 of [3,5,7]
    'CNN_STRIDE': 2,     # idx 0 of [2]
    'EMB_DIM':    None,  # idx 0 of [None,32]
    'NUM_HEADS':  1,     # idx 0 of [1,2,4,8]
    'NUM_BLOCKS': 2,     # idx 1 of [1,2]  ← differs from #143 (was 1)
    'MLP_DIM':    64,    # MLP_DIM_exp=6   ← differs from #143 (was 256)
}

# Radii: BEST_RADII (from #143) with 5 free neurons overridden by #511's values
TRIAL_511_RADII = list(BEST_RADII)
TRIAL_511_RADII[7]  = 19   # r_07
TRIAL_511_RADII[15] = 11   # r_15
TRIAL_511_RADII[21] = 15   # r_21
TRIAL_511_RADII[34] = 21   # r_34
TRIAL_511_RADII[40] = 8    # r_40

CFG_511 = {**BAYES_FIXED, **TRIAL_511_ARCH, 'NEURON_RADII': TRIAL_511_RADII}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',     type=int, required=True)
    ap.add_argument('--out_root', default='./bayes_search_results')
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, 'retrain_runs_511', f'seed_{args.seed:02d}')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[t511 seed {args.seed}] device={device}  out={out_dir}', flush=True)
    pc = count_params_per_block(CFG_511)
    print(f'[t511 seed {args.seed}] cfg: cnn={CFG_511["CNN_DIM"]} k={CFG_511["CNN_KERNEL"]} '
          f's={CFG_511["CNN_STRIDE"]} emb={CFG_511["EMB_DIM"]} '
          f'h={CFG_511["NUM_HEADS"]} b={CFG_511["NUM_BLOCKS"]} '
          f'mlp={CFG_511["MLP_DIM"]}  params={pc["total"]:,}', flush=True)

    # ── data ─────────────────────────────────────────────────────────────────
    data         = np.load(DATA_PATH)
    images_train = data['images_train'].squeeze(-1)
    images_val   = data['images_val'].squeeze(-1)
    images_test  = data['images_test'].squeeze(-1)
    resp_train   = data['responses_train']
    resp_val     = data['responses_val']
    test_rep     = data['responses_test'].transpose(1, 0, 2)

    rf        = np.load(RF_PATH)
    centers_x = rf['centers_x']
    centers_y = rf['centers_y']

    has_lsta = os.path.isfile(LSTA_REF_PATH)
    if has_lsta:
        _lr = np.load(LSTA_REF_PATH)
        lsta_exp, lsta_ellipses, lsta_img_idx = _lr['lsta'], _lr['ellipses'], _lr['image_indices']

    _dl_kw   = dict(num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY,
                    persistent_workers=_NUM_WORKERS > 0)
    train_ds = RetinalDataset(images_train, resp_train)
    val_ds   = RetinalDataset(images_val,   resp_val)
    train_loader = DataLoader(train_ds, batch_size=CFG_511['BATCH_SIZE'], shuffle=True,  **_dl_kw)
    val_loader   = DataLoader(val_ds,   batch_size=CFG_511['BATCH_SIZE'], shuffle=False, **_dl_kw)

    # ── train ────────────────────────────────────────────────────────────────
    _seed_all(args.seed)
    t0 = time.time()
    val_loss, history, state = _train_config(
        CFG_511, train_loader, val_loader, centers_x, centers_y, device)
    train_sec = time.time() - t0

    # ── evaluate on test ─────────────────────────────────────────────────────
    from model.model   import RetinalModel
    from model.readout import load_sta_crops
    radii_arr = np.asarray(CFG_511['NEURON_RADII'], dtype=np.float32)
    crop_half = max(1, math.ceil(float(radii_arr.max()) / CFG_511['CNN_STRIDE']))
    sta_crops = load_sta_crops(STA_PATH, centers_x, centers_y, crop_half, CFG_511['CNN_STRIDE'])
    eval_m = RetinalModel(CFG_511, centers_x, centers_y, radii_arr, sta_crops).to(device)
    eval_m.load_state_dict({k: v.to(device) for k, v in state.items()})
    y_pred = _predict_test(eval_m, images_test, device)
    adj_r2 = _adjusted_r2_per_neuron(y_pred, test_rep)
    scores = _compute_scores(adj_r2)
    lsta_scores = None
    if has_lsta:
        lsta_scores = _compute_scores(_compute_lsta_corr_per_neuron(
            eval_m, images_test, lsta_exp, lsta_img_idx, lsta_ellipses, device))
    del eval_m

    # ── persist ──────────────────────────────────────────────────────────────
    torch.save(state, os.path.join(out_dir, 'model.pt'))
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({'score': scores['mean_all'], 'cfg': CFG_511, 'params': pc,
                   'val_loss': val_loss, 'train_sec': train_sec,
                   'seed': args.seed, 'host': os.uname().nodename}, f, indent=2)
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, 'scores.json'), 'w') as f:
        json.dump({'scores': scores, 'lsta_scores': lsta_scores}, f, indent=2)

    lsta_str = f'  lsta_all={lsta_scores["mean_all"]:.4f}' if lsta_scores else ''
    print(f'[t511 seed {args.seed}] DONE  mean_all={scores["mean_all"]:.4f}  '
          f'val={val_loss:.4f}{lsta_str}  [{train_sec:.0f}s]  → {out_dir}', flush=True)


if __name__ == '__main__':
    main()
