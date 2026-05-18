#!/usr/bin/env python
"""Retrain trial #143 (best historical, mean_all=0.8140) for ONE seed.

Saves model + config + scores to bayes_search_results/retrain_runs/seed_<N>/.
Designed to be run in parallel across the 3 cluster nodes (different seed per
node). A separate script (pick_best_retrain.py) collects all seed outputs and
promotes the highest-scoring one to bayes_search_results/mean_all/.
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

# Force unbuffered stdout so per-epoch prints reach the log immediately.
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


# Trial #143 exact hyperparameters
TRIAL_143_ARCH = {
    'CNN_DIM':    32,
    'CNN_KERNEL': 3,
    'CNN_STRIDE': 2,
    'EMB_DIM':    None,
    'NUM_HEADS':  1,
    'NUM_BLOCKS': 1,
    'MLP_DIM':    256,
}
CFG_143 = {**BAYES_FIXED, **TRIAL_143_ARCH, 'NEURON_RADII': list(BEST_RADII)}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',     type=int, required=True)
    ap.add_argument('--out_root', default='./bayes_search_results',
                    help='Parent dir; output goes to <out_root>/retrain_runs/seed_<N>/')
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, 'retrain_runs', f'seed_{args.seed:02d}')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[seed {args.seed}] device={device}  out={out_dir}', flush=True)
    pc = count_params_per_block(CFG_143)
    print(f'[seed {args.seed}] cfg: cnn={CFG_143["CNN_DIM"]} k={CFG_143["CNN_KERNEL"]} '
          f's={CFG_143["CNN_STRIDE"]} emb={CFG_143["EMB_DIM"]} '
          f'h={CFG_143["NUM_HEADS"]} b={CFG_143["NUM_BLOCKS"]} '
          f'mlp={CFG_143["MLP_DIM"]}  params={pc["total"]:,}', flush=True)

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
    train_loader = DataLoader(train_ds, batch_size=CFG_143['BATCH_SIZE'], shuffle=True,  **_dl_kw)
    val_loader   = DataLoader(val_ds,   batch_size=CFG_143['BATCH_SIZE'], shuffle=False, **_dl_kw)

    # ── train ────────────────────────────────────────────────────────────────
    _seed_all(args.seed)
    t0 = time.time()
    val_loss, history, state = _train_config(
        CFG_143, train_loader, val_loader, centers_x, centers_y, device)
    train_sec = time.time() - t0

    # ── evaluate on test ─────────────────────────────────────────────────────
    from model.model   import RetinalModel
    from model.readout import load_sta_crops
    radii_arr = np.asarray(CFG_143['NEURON_RADII'], dtype=np.float32)
    crop_half = max(1, math.ceil(float(radii_arr.max()) / CFG_143['CNN_STRIDE']))
    sta_crops = load_sta_crops(STA_PATH, centers_x, centers_y, crop_half, CFG_143['CNN_STRIDE'])
    eval_m = RetinalModel(CFG_143, centers_x, centers_y, radii_arr, sta_crops).to(device)
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
        json.dump({'score': scores['mean_all'], 'cfg': CFG_143, 'params': pc,
                   'val_loss': val_loss, 'train_sec': train_sec,
                   'seed': args.seed, 'host': os.uname().nodename}, f, indent=2)
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, 'scores.json'), 'w') as f:
        json.dump({'scores': scores, 'lsta_scores': lsta_scores}, f, indent=2)

    lsta_str = f'  lsta_all={lsta_scores["mean_all"]:.4f}' if lsta_scores else ''
    print(f'[seed {args.seed}] DONE  mean_all={scores["mean_all"]:.4f}  '
          f'val={val_loss:.4f}{lsta_str}  [{train_sec:.0f}s]  → {out_dir}', flush=True)


if __name__ == '__main__':
    main()
