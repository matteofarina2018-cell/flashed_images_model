# bayes_search.py
#
# Bayesian search (Optuna TPE) over architecture hyperparameters and the 41
# per-neuron RF radii.
# - Constant hyperparameters live in BAYES_FIXED (model/config.py).
# - Architecture search space lives in BAYES_SPACE (categorical).
# - Radii are sampled as 41 integers in RADII_RANGE (one per neuron).
# - Optuna persists the study in <out_dir>/optuna.db (SQLite). Resuming is
#   automatic: re-run the script and it continues from the last trial.
#
# Evaluation criteria (42 total):
#   - mean_all      : mean Adj R² across all 41 neurons  (Optuna target)
#   - neuron_{NN}   : Adj R² for individual neuron n     (secondary tracker)
#
# Output layout under <out_dir>/:
#   mean_all/                   ← best trial overall (by mean Adj R²)
#     model.pt  config.json  history.json
#   per_neuron/
#     neuron_00/ ... neuron_40/
#       model.pt  config.json  history.json
#   lsta_corr/                  ← if LSTA reference is available
#     mean_all/  per_neuron/
#   all_results.csv             ← one row per completed trial
#   optuna.db                   ← Optuna storage (resume)

import ast
import math
import os
import json
import csv
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import optuna
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState as _TS

from data          import RetinalDataset, _NUM_WORKERS, _PIN_MEMORY
from model.config  import (IN_CHANNELS, N_CELLS, MAX_PARAMS, IMG_SIZE_ORIGINAL,
                            BAYES_FIXED, BAYES_SPACE, RADII_RANGE,
                            CNN_DIM_EXP_RANGE, MLP_DIM_EXP_RANGE,
                            N_TRIALS_DEFAULT)
from model.model   import RetinalModel
from model.losses  import poisson_loss
from model.readout import load_sta_crops


# ── fixed paths ───────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(_HERE, 'PNAS_paper_sorted_data.npz')
RF_PATH       = os.path.join(_HERE, 'ellipse_centers_exp2.npz')
STA_PATH      = os.path.join(_HERE, 'STA')
LSTA_REF_PATH = os.path.join(_HERE, 'lsta_ref.npz')   # optional; lsta_corr disabled if missing

# ── Optuna seed (reproducibility) ─────────────────────────────────────────────
# Hostname-derived seed so concurrent workers on different nodes don't
# generate identical "random" proposals (the seed=42 hard-coded value caused
# 3 workers to produce identical configs during random startup). Per-node
# determinism is preserved (same node → same seed → reproducible).
import hashlib, socket
SEED = int.from_bytes(hashlib.sha256(socket.gethostname().encode()).digest()[:4], 'big') % (2**31)


# ── Focused search: vary radii only for the hardest neurons ───────────────────
# Set LOCK_RADII = False to revert to the original full-41-radii search.
LOCK_RADII = True
# Indices of the 5 hardest neurons (lowest adj_R² in trial #143). TPE will
# explore radii ONLY for these — the other 36 are pinned to BEST_RADII[i].
FREE_NEURONS = {7, 15, 21, 34, 40}
# Radii from trial #143 (mean_all=0.8140). One value per neuron, indices 0..40.
BEST_RADII = [
    23, 8,  8,  4,  16, 14, 13, 25, 6,  4,
    9,  21, 12, 25, 7,  13, 4,  16, 17, 8,
    22, 8,  25, 8,  4,  18, 6,  4,  4,  25,
    9,  25, 16, 13, 20, 4,  13, 13, 21, 20,
    19,
]


# ─────────────────────────────────────────────────────────────────────────────
# Criteria
# ─────────────────────────────────────────────────────────────────────────────

CRITERIA = (
    ['mean_all'] +
    [f'neuron_{n:02d}' for n in range(41)]
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-block parameter count
# ─────────────────────────────────────────────────────────────────────────────

def count_params_per_block(cfg: dict) -> dict:
    """Analytical estimate of parameters per block: cnn, neuron_circle,
    transformer, readout, total."""
    cnn_dim    = cfg['CNN_DIM']
    emb_dim    = cfg['EMB_DIM'] if cfg['EMB_DIM'] is not None else cnn_dim
    mlp_dim    = cfg['MLP_DIM']
    num_blocks = cfg['NUM_BLOCKS']

    # ── CNN ──────────────────────────────────────────────────────────────────
    p_cnn = 0
    in_ch = IN_CHANNELS
    k     = cfg['CNN_KERNEL']
    for _ in range(cfg['CNN_LAYERS']):
        p_cnn += in_ch * cnn_dim * k * k + cnn_dim   # Conv2d (weight + bias)
        p_cnn += 2 * cnn_dim                          # BatchNorm2d (weight + bias)
        in_ch  = cnn_dim

    # ── NeuronCircle ─────────────────────────────────────────────────────────
    p_circle = 2 * N_CELLS   # cx, cy

    # ── Transformer ──────────────────────────────────────────────────────────
    p_trans = 0
    if cfg['EMB_DIM'] is not None and cfg['EMB_DIM'] != cnn_dim:
        p_trans += cnn_dim * emb_dim + emb_dim            # linear projection
    for _ in range(num_blocks):
        p_trans += 4 * (emb_dim * emb_dim + emb_dim)     # W_q W_k W_v W_out
        p_trans += emb_dim * mlp_dim + mlp_dim            # MLP fc1
        p_trans += mlp_dim * emb_dim + emb_dim            # MLP fc2
        p_trans += 2 * 2 * emb_dim                        # 2 × LayerNorm

    # ── Readout ──────────────────────────────────────────────────────────────
    p_read = N_CELLS * emb_dim + N_CELLS   # v + bias

    total = p_cnn + p_circle + p_trans + p_read
    return dict(cnn=p_cnn, neuron_circle=p_circle,
                transformer=p_trans, readout=p_read, total=total)


# ─────────────────────────────────────────────────────────────────────────────
# Single-config training
# ─────────────────────────────────────────────────────────────────────────────

def _train_config(cfg, train_loader, val_loader, centers_x, centers_y, device):
    """Single-phase training with fixed per-neuron radii.
    Returns (best_val_loss, history, best_state)."""
    radii     = np.asarray(cfg['NEURON_RADII'], dtype=np.float32)
    stride    = cfg['CNN_STRIDE']
    crop_half = max(1, math.ceil(float(radii.max()) / stride))

    sta_crops = load_sta_crops(STA_PATH, centers_x, centers_y, crop_half, stride)
    model     = RetinalModel(cfg, centers_x, centers_y, radii, sta_crops).to(device)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['LEARNING_RATE'], weight_decay=cfg['WEIGHT_DECAY'],
    )

    history       = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_cnt  = 0
    best_state    = None

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float32)

    for epoch in range(1, cfg['MAX_EPOCHS'] + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            resp = batch['response'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss = poisson_loss(model(imgs), resp)
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

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        best_mark = ' *' if val_loss < best_val_loss else ''
        print(f'    ep {epoch:4d}  train: {train_loss:.4f}  val: {val_loss:.4f}{best_mark}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg['EARLY_STOP']:
                break

    return best_val_loss, history, best_state


# ─────────────────────────────────────────────────────────────────────────────
# Test-set prediction
# ─────────────────────────────────────────────────────────────────────────────

def _predict_test(model, test_images_np, device):
    model.eval()
    with torch.no_grad():
        imgs = torch.tensor(test_images_np[:, np.newaxis], dtype=torch.float32).to(device)
        return model(imgs).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Score computation
# ─────────────────────────────────────────────────────────────────────────────

def _adjusted_r2_per_neuron(y_pred, y_true_repeated):
    """Adjusted R² (Goldin et al. PNAS 2023, Eq. 5)."""
    even_idx = list(range(0, y_true_repeated.shape[1], 2))
    odd_idx  = list(range(1, y_true_repeated.shape[1], 2))
    r_even   = y_true_repeated[:, even_idx, :].mean(axis=1)
    r_odd    = y_true_repeated[:, odd_idx,  :].mean(axis=1)

    adj_r2 = np.zeros(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        c_eo = np.corrcoef(r_even[:, i], r_odd[:, i])[0, 1]
        if c_eo <= 0:
            continue
        c_pe = np.corrcoef(y_pred[:, i], r_even[:, i])[0, 1]
        c_po = np.corrcoef(y_pred[:, i], r_odd[:, i])[0, 1]
        adj_r2[i] = max(((c_pe + c_po) / 2.0) ** 2 / c_eo, 0.0)
    return adj_r2


def _compute_scores(adj_r2):
    scores = {'mean_all': float(adj_r2.mean())}
    for n in range(41):
        scores[f'neuron_{n:02d}'] = float(adj_r2[n])
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# LSTA correlation (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _lsta_postprocess(arr, expon=2, vmax_thresh=1.0, to0_frac=0.2):
    arr = arr.copy().astype(float)
    if expon % 2 == 1:
        arr = arr ** expon
    else:
        arr = np.sign(arr) * np.exp(np.log(np.abs(arr) + 1e-12) * expon)
    vmax = max(arr.max(), -arr.min()) * vmax_thresh
    if vmax > 0:
        arr[np.abs(arr) < vmax * to0_frac] = 0.0
    return arr


def _compute_lsta_corr_per_neuron(model, images_np, lsta_exp, image_indices,
                                   ellipses, device,
                                   padding=8, expon=2, vmax_thresh=1.0, to0_frac=0.2):
    """Pearson r (averaged across N images) between model and experimental LSTA,
    computed on the crop around each neuron's RF ellipse."""
    from PIL import Image as PILImage

    N_neurons = lsta_exp.shape[0]
    N_images  = lsta_exp.shape[1]
    LSTA_SIZE = lsta_exp.shape[2]
    SCALE     = IMG_SIZE_ORIGINAL / LSTA_SIZE

    def _resize(arr):
        pil = PILImage.fromarray(arr.astype(np.float32), mode='F')
        pil = pil.resize((IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL), PILImage.BILINEAR)
        return np.array(pil)

    # ── compute model LSTA via autograd ──────────────────────────────────────
    model.eval()
    model_lsta = np.zeros(
        (N_neurons, N_images, IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL), dtype=np.float32)

    for j, idx in enumerate(image_indices):
        x = torch.tensor(
            images_np[idx][None, None], dtype=torch.float32,
        ).to(device).requires_grad_(True)
        out = model(x)
        for n in range(N_neurons):
            if x.grad is not None:
                x.grad.zero_()
            out[0, n].backward(retain_graph=(n < N_neurons - 1))
            model_lsta[n, j] = x.grad[0, 0].detach().cpu().numpy()

    # ── per-neuron correlation ───────────────────────────────────────────────
    lsta_corr = np.zeros(N_neurons)
    for n in range(N_neurons):
        ex = ellipses[n, 0, :] * SCALE
        ey = ellipses[n, 1, :] * SCALE
        x0 = int(max(0,                  ex.min() - padding))
        x1 = int(min(IMG_SIZE_ORIGINAL,  ex.max() + padding))
        y0 = int(max(0,                  ey.min() - padding))
        y1 = int(min(IMG_SIZE_ORIGINAL,  ey.max() + padding))

        corrs = []
        for i in range(N_images):
            pe = _lsta_postprocess(_resize(lsta_exp[n, i]), expon, vmax_thresh, to0_frac)
            pm = _lsta_postprocess(model_lsta[n, i],        expon, vmax_thresh, to0_frac)
            flat_e = pe[y0:y1, x0:x1].ravel()
            flat_m = pm[y0:y1, x0:x1].ravel()
            if flat_e.std() > 0 and flat_m.std() > 0:
                corrs.append(float(np.corrcoef(flat_e, flat_m)[0, 1]))
            else:
                corrs.append(0.0)
        lsta_corr[n] = float(np.mean(corrs))

    return lsta_corr


# ─────────────────────────────────────────────────────────────────────────────
# Saving helpers
# ─────────────────────────────────────────────────────────────────────────────

def _criterion_dir(out_dir, criterion):
    if criterion == 'mean_all':
        return os.path.join(out_dir, 'mean_all')
    return os.path.join(out_dir, 'per_neuron', criterion)


def _lsta_criterion_dir(out_dir, criterion):
    if criterion == 'mean_all':
        return os.path.join(out_dir, 'lsta_corr', 'mean_all')
    return os.path.join(out_dir, 'lsta_corr', 'per_neuron', criterion)


def _save_result(path, state_dict, score, cfg, param_counts, history):
    os.makedirs(path, exist_ok=True)
    torch.save(state_dict, os.path.join(path, 'model.pt'))
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump({'score': score, 'cfg': cfg, 'params': param_counts}, f, indent=2)
    with open(os.path.join(path, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)


def _save_csv(results, path):
    if not results:
        return
    all_keys = []
    seen     = set()
    for row in results:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore', restval='')
        writer.writeheader()
        writer.writerows(results)


def _import_prior_trials(study, csv_path, n_cells, radii_range):
    """Seed a fresh Optuna study with completed trials from all_results.csv.

    Old trials that predate NUM_BLOCKS in BAYES_SPACE have NUM_BLOCKS=1 stored
    in the CSV (it was in BAYES_FIXED), so they map cleanly to the new space.
    Radii from the old (4,25) range are clamped to the new (4,35) range — values
    ≤25 are unchanged, which is correct.
    """
    if not os.path.isfile(csv_path):
        return 0

    from optuna.distributions import CategoricalDistribution, IntDistribution

    dists = {k: CategoricalDistribution(v) for k, v in BAYES_SPACE.items()}
    dists['CNN_DIM_exp'] = IntDistribution(*CNN_DIM_EXP_RANGE)
    dists['MLP_DIM_exp'] = IntDistribution(*MLP_DIM_EXP_RANGE)
    for i in range(n_cells):
        dists[f'r_{i:02d}'] = IntDistribution(radii_range[0], radii_range[1])

    _MISS = object()

    def _coerce(raw, options):
        if raw in ('', 'None') and None in options:
            return None
        for opt in options:
            if opt is None:
                continue
            try:
                if type(opt)(raw) == opt:
                    return opt
            except (ValueError, TypeError):
                pass
        return _MISS

    imported = 0
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            params = {}
            skip = False
            for key, options in BAYES_SPACE.items():
                val = _coerce(row.get(key, ''), options)
                if val is _MISS:
                    skip = True
                    break
                params[key] = val
            if skip:
                continue

            # CNN_DIM and MLP_DIM stored as raw ints in CSV → convert to log2 exponents
            try:
                params['CNN_DIM_exp'] = int(round(math.log2(int(row['CNN_DIM']))))
                params['MLP_DIM_exp'] = int(round(math.log2(int(row['MLP_DIM']))))
            except (KeyError, ValueError, ZeroDivisionError):
                continue
            if not (CNN_DIM_EXP_RANGE[0] <= params['CNN_DIM_exp'] <= CNN_DIM_EXP_RANGE[1]):
                continue
            if not (MLP_DIM_EXP_RANGE[0] <= params['MLP_DIM_exp'] <= MLP_DIM_EXP_RANGE[1]):
                continue

            try:
                radii = ast.literal_eval(row['NEURON_RADII'])
                for i, r in enumerate(radii):
                    params[f'r_{i:02d}'] = max(radii_range[0],
                                               min(radii_range[1], int(round(float(r)))))
            except Exception:
                continue

            try:
                value = float(row['mean_all'])
            except (KeyError, ValueError):
                continue

            study.add_trial(optuna.trial.create_trial(
                params=params, distributions=dists, value=value,
            ))
            imported += 1

    return imported


def _load_existing_csv(csv_path, criteria):
    """Reload rows and previous bests from CSV. Optuna handles trial resume via
    its SQLite storage, but per-neuron / lsta best trackers live outside it."""
    if not os.path.isfile(csv_path):
        return [], {}, {}
    csv_rows    = []
    best_scores = {c: float('-inf') for c in criteria}
    best_lsta   = {c: float('-inf') for c in criteria}
    has_lsta_cols = False
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append(dict(row))
            for crit in criteria:
                if crit in row:
                    try:
                        v = float(row[crit])
                        if v > best_scores[crit]:
                            best_scores[crit] = v
                    except (ValueError, TypeError):
                        pass
                lsta_col = f'lsta_{crit}'
                if lsta_col in row and row[lsta_col] not in ('', None):
                    has_lsta_cols = True
                    try:
                        v = float(row[lsta_col])
                        if v > best_lsta[crit]:
                            best_lsta[crit] = v
                    except (ValueError, TypeError):
                        pass
    return csv_rows, best_scores, (best_lsta if has_lsta_cols else {})


# ─────────────────────────────────────────────────────────────────────────────
# Main Bayesian search
# ─────────────────────────────────────────────────────────────────────────────

def _sample_config(trial: optuna.Trial) -> dict:
    """Sample one full configuration from BAYES_FIXED + BAYES_SPACE + radii.

    CNN_DIM and MLP_DIM are sampled as log2 exponents (suggest_int) so the TPE
    treats them as ordinal — it knows CNN_DIM=32 is between 16 and 64.
    The actual values are always exact powers of 2.
    """
    arch = {
        key: trial.suggest_categorical(key, options)
        for key, options in BAYES_SPACE.items()
    }
    arch['CNN_DIM'] = 2 ** trial.suggest_int('CNN_DIM_exp', *CNN_DIM_EXP_RANGE)
    arch['MLP_DIM'] = 2 ** trial.suggest_int('MLP_DIM_exp', *MLP_DIM_EXP_RANGE)
    if LOCK_RADII:
        radii = [
            trial.suggest_int(f'r_{i:02d}', RADII_RANGE[0], RADII_RANGE[1])
            if i in FREE_NEURONS else BEST_RADII[i]
            for i in range(N_CELLS)
        ]
    else:
        radii = [trial.suggest_int(f'r_{i:02d}', RADII_RANGE[0], RADII_RANGE[1])
                 for i in range(N_CELLS)]
    return {**BAYES_FIXED, **arch, 'NEURON_RADII': radii}


def _validate_config(cfg: dict) -> None:
    """Raise optuna.TrialPruned if cfg violates RoPE or MAX_PARAMS constraints."""
    eff_emb = cfg['EMB_DIM'] if cfg['EMB_DIM'] is not None else cfg['CNN_DIM']
    if eff_emb % cfg['NUM_HEADS'] != 0:
        raise optuna.TrialPruned()
    if (eff_emb // cfg['NUM_HEADS']) % 4 != 0:        # RoPE constraint
        raise optuna.TrialPruned()
    if count_params_per_block(cfg)['total'] > MAX_PARAMS:
        raise optuna.TrialPruned()


def run_bayes_search(out_dir: str = os.path.join(_HERE, 'bayes_search_results'),
                     n_trials: int = N_TRIALS_DEFAULT,
                     seed: int = SEED):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── load data ────────────────────────────────────────────────────────────
    print('Loading data...')
    data         = np.load(DATA_PATH)
    images_train = data['images_train'].squeeze(-1)
    images_val   = data['images_val'].squeeze(-1)
    images_test  = data['images_test'].squeeze(-1)
    resp_train   = data['responses_train']
    resp_val     = data['responses_val']
    test_rep     = data['responses_test'].transpose(1, 0, 2)   # [n_img, n_reps, n_cells]

    rf        = np.load(RF_PATH)
    centers_x = rf['centers_x']
    centers_y = rf['centers_y']

    has_lsta = os.path.isfile(LSTA_REF_PATH)
    if has_lsta:
        _lsta_ref     = np.load(LSTA_REF_PATH)
        lsta_exp      = _lsta_ref['lsta']
        lsta_ellipses = _lsta_ref['ellipses']
        lsta_img_idx  = _lsta_ref['image_indices']
        print(f'LSTA reference loaded: {lsta_exp.shape[1]} reference images')
    else:
        print(f'LSTA reference not found ({LSTA_REF_PATH}) — lsta_corr disabled.')

    # ── DataLoader (BATCH_SIZE is fixed in BAYES_FIXED) ──────────────────────
    _dl_kw   = dict(num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY,
                    persistent_workers=_NUM_WORKERS > 0)
    val_ds   = RetinalDataset(images_val, resp_val)
    train_ds = RetinalDataset(
        images_train, resp_train,
        img_noise_sigma  = BAYES_FIXED.get('IMG_NOISE_SIGMA',  0.0),
        poisson_resample = BAYES_FIXED.get('POISSON_RESAMPLE', False),
        aug_factor       = BAYES_FIXED.get('AUG_FACTOR',       1),
    )
    train_loader = DataLoader(train_ds, batch_size=BAYES_FIXED['BATCH_SIZE'], shuffle=True,  **_dl_kw)
    val_loader   = DataLoader(val_ds,   batch_size=BAYES_FIXED['BATCH_SIZE'], shuffle=False, **_dl_kw)

    # ── best trackers (loaded from CSV if present) ───────────────────────────
    csv_path = os.path.join(out_dir, 'all_results.csv')
    csv_rows, prev_best, prev_best_lsta = _load_existing_csv(csv_path, CRITERIA)

    best = {
        crit: (prev_best.get(crit, float('-inf')), None, None, None, None)
        for crit in CRITERIA
    }
    best_lsta = (
        {crit: (prev_best_lsta.get(crit, float('-inf')), None, None, None, None)
         for crit in CRITERIA}
        if has_lsta else None
    )

    # ── persistent Optuna study ──────────────────────────────────────────────
    storage_url  = 'postgresql://optuna_user:optuna_pass@localhost:5432/optuna_db'
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(
            seed=seed, multivariate=True, n_startup_trials=50,
            # gamma must be callable: top 40% of trials counted as "good"
            gamma=lambda n: max(1, int(0.4 * n)),
        ),
        storage=storage_url,
        study_name='retinal_radii',
        load_if_exists=True,
    )
    n_done_prev = len(study.trials)
    if n_done_prev == 0:
        n_imp = _import_prior_trials(study, csv_path, N_CELLS, RADII_RANGE)
        if n_imp:
            n_done_prev = len(study.trials)
            print(f'Imported {n_imp} prior trials from CSV '
                  f'(TPE starts after {max(0, 50 - n_done_prev)} more random trials).')
    print(f'Optuna study: {storage_url}  '
          f'(prior trials: {n_done_prev}, target: {n_trials})')

    t_start = time.time()
    n_done  = [n_done_prev]

    # ── objective ────────────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        n_done[0] += 1
        n_this    = n_done[0] - n_done_prev

        cfg = _sample_config(trial)
        _validate_config(cfg)

        param_counts = count_params_per_block(cfg)
        radii_arr    = np.asarray(cfg['NEURON_RADII'], dtype=np.float32)
        eff_emb      = cfg['EMB_DIM'] if cfg['EMB_DIM'] is not None else cfg['CNN_DIM']

        print(f'\n[trial #{trial.number}  ({n_this}/{n_trials})]  '
              f"cnn={cfg['CNN_DIM']}×{cfg['CNN_LAYERS']}k{cfg['CNN_KERNEL']} "
              f"stride={cfg['CNN_STRIDE']}  "
              f"emb={eff_emb} heads={cfg['NUM_HEADS']} "
              f"blocks={cfg['NUM_BLOCKS']} mlp={cfg['MLP_DIM']}  "
              f"params={param_counts['total']:,}  "
              f"r=[{int(radii_arr.min())},{radii_arr.mean():.1f},{int(radii_arr.max())}]")

        t0 = time.time()
        try:
            val_loss, history, best_state = _train_config(
                cfg, train_loader, val_loader, centers_x, centers_y, device
            )
        except Exception as e:
            print(f'  TRAINING ERROR: {e} — pruning trial.')
            raise optuna.TrialPruned()

        # ── test-set evaluation ──────────────────────────────────────────────
        crop_half  = max(1, math.ceil(float(radii_arr.max()) / cfg['CNN_STRIDE']))
        sta_crops  = load_sta_crops(STA_PATH, centers_x, centers_y, crop_half, cfg['CNN_STRIDE'])
        eval_model = RetinalModel(cfg, centers_x, centers_y, radii_arr, sta_crops).to(device)
        eval_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        y_pred = _predict_test(eval_model, images_test, device)
        adj_r2 = _adjusted_r2_per_neuron(y_pred, test_rep)
        scores = _compute_scores(adj_r2)

        lsta_scores = None
        if has_lsta:
            lc = _compute_lsta_corr_per_neuron(
                eval_model, images_test, lsta_exp, lsta_img_idx, lsta_ellipses, device)
            lsta_scores = _compute_scores(lc)
        del eval_model

        elapsed = time.time() - t0
        eta_min = (time.time() - t_start) / max(n_this, 1) * (n_trials - n_this) / 60
        lsta_str = (f'  lsta_all={lsta_scores["mean_all"]:.4f}' if lsta_scores else '')
        print(f'  val={val_loss:.4f}  mean_all={scores["mean_all"]:.4f}'
              f'{lsta_str}  [{elapsed:.0f}s  ETA {eta_min:.0f}min]')

        # ── update per-criterion bests ───────────────────────────────────────
        for crit in CRITERIA:
            if scores[crit] > best[crit][0]:
                best[crit] = (scores[crit], cfg, param_counts, best_state, history)
                _save_result(_criterion_dir(out_dir, crit),
                             best_state, scores[crit], cfg, param_counts, history)
                print(f'  → new best adj_r2 [{crit}]: {scores[crit]:.4f}')

        if has_lsta and lsta_scores:
            for crit in CRITERIA:
                if lsta_scores[crit] > best_lsta[crit][0]:
                    best_lsta[crit] = (lsta_scores[crit], cfg, param_counts, best_state, history)
                    _save_result(_lsta_criterion_dir(out_dir, crit),
                                 best_state, lsta_scores[crit], cfg, param_counts, history)
                    print(f'  → new best lsta_corr [{crit}]: {lsta_scores[crit]:.4f}')

        # ── append CSV ───────────────────────────────────────────────────────
        row = {'trial':    trial.number,
               **cfg,
               'val_loss': val_loss,
               **{f'p_{k}': v for k, v in param_counts.items()},
               **scores,
               **({f'lsta_{k}': v for k, v in lsta_scores.items()}
                  if lsta_scores else {})}
        csv_rows.append(row)
        _save_csv(csv_rows, csv_path)

        return scores['mean_all']

    # ── run ──────────────────────────────────────────────────────────────────
    # Global stop: when the study reaches 800 COMPLETE trials, all workers exit.
    # Per-worker n_trials is the local budget; MaxTrialsCallback caps the global total.
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   gc_after_trial=True,
                   callbacks=[MaxTrialsCallback(800, states=(_TS.COMPLETE,))])

    # ── final summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('BAYESIAN SEARCH — SUMMARY')
    print('=' * 60)
    print(f'\n  Total trials: {len(study.trials)}')
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print(f'  Best mean_all (Optuna): {study.best_value:.4f}  (trial #{study.best_trial.number})')
        params = study.best_params
        arch_summary = ', '.join(f'{k}={params[k]}' for k in BAYES_SPACE.keys())
        print(f'  Best arch: {arch_summary}')
        radii_best = [params[f'r_{i:02d}'] for i in range(N_CELLS)]
        print(f'  Best radii:\n    {radii_best}')
        print(f'  → {out_dir}/mean_all/')
    else:
        print('  No completed trials yet.')

    if has_lsta:
        print('\n  ── LSTA Correlation ──')
        score, cfg_b, pc, _, _ = best_lsta['mean_all'][:5]
        if cfg_b is not None:
            print(f'  mean_all: {score:.4f}  '
                  f"params={pc['total']:,}  "
                  f"→ {out_dir}/lsta_corr/mean_all/")

    print(f'\nFull results : {csv_path}  ({len(csv_rows)} rows)')
    print(f'Optuna study : {storage_url}')
    return csv_rows


if __name__ == '__main__':
    run_bayes_search()
