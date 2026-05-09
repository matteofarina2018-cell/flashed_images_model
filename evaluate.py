# evaluate.py

from pathlib import Path

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from model.config  import N_CELLS, IMG_SIZE_ORIGINAL
from model.model   import RetinalModel
from model.readout import load_sta_crops


# ─────────────────────────────────────────────────────────────────────────────
# Metriche
# ─────────────────────────────────────────────────────────────────────────────

def pearson_per_neuron(y_pred, y_true):
    corrs = np.array([
        np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        for i in range(y_pred.shape[1])
    ])
    return np.nan_to_num(corrs)


def r2_per_neuron(y_pred, y_true):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / (ss_tot + 1e-8)


def mse_per_neuron(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean(axis=0)


def adjusted_r2_per_neuron(y_pred, y_true_repeated):
    """
    Adjusted R² — Goldin et al. PNAS 2023, Eq. 5.
    y_pred          : (n_images, n_neurons)
    y_true_repeated : (n_images, n_repeats, n_neurons)
    """
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


def explained_variance_per_neuron(y_pred, y_true_repeated):
    """
    Normalized explained variance = mean_accuracy / reliability.
    """
    even_idx = list(range(0, y_true_repeated.shape[1], 2))
    odd_idx  = list(range(1, y_true_repeated.shape[1], 2))
    r_even   = y_true_repeated[:, even_idx, :].mean(axis=1)
    r_odd    = y_true_repeated[:, odd_idx,  :].mean(axis=1)

    ev = np.zeros(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        c_eo = np.corrcoef(r_even[:, i], r_odd[:, i])[0, 1]
        if c_eo <= 0:
            continue
        c_pe = np.corrcoef(y_pred[:, i], r_even[:, i])[0, 1]
        c_po = np.corrcoef(y_pred[:, i], r_odd[:, i])[0, 1]
        ev[i] = max((c_pe + c_po) / 2.0 / c_eo, 0.0)
    return ev


def normalized_r2_per_neuron(y_pred, y_true_repeated):
    """
    R² normalizzato per il noise ceiling (reliability even/odd).
    """
    even_idx = list(range(0, y_true_repeated.shape[1], 2))
    odd_idx  = list(range(1, y_true_repeated.shape[1], 2))
    r_even   = y_true_repeated[:, even_idx, :].mean(axis=1)
    r_odd    = y_true_repeated[:, odd_idx,  :].mean(axis=1)
    y_mean   = (r_even + r_odd) / 2.0

    r2s         = r2_per_neuron(y_pred, y_mean)
    reliability = np.array([
        np.corrcoef(r_even[:, i], r_odd[:, i])[0, 1]
        for i in range(y_pred.shape[1])
    ])
    reliability = np.nan_to_num(reliability)
    return np.where(reliability > 0.01, r2s / np.maximum(reliability, 0.01), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Predizione
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model, images_np, device, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(images_np), batch_size):
            imgs = torch.tensor(
                images_np[i:i+batch_size, np.newaxis], dtype=torch.float32
            ).to(device)
            preds.append(model(imgs).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Plots — Metriche (line chart, ordine originale)
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_STYLES = {
    'pearson_r':          ('steelblue',      'Pearson r',          (-1, 1)),
    'r2':                 ('coral',          'R²',                 (None, None)),
    'normalized_r2':      ('mediumpurple',   'Normalized R²',      (None, None)),
    'adjusted_r2':        ('darkorange',     'Adjusted R²',        (0, 1)),
    'mse':                ('mediumseagreen', 'MSE',                (None, None)),
    'explained_variance': ('teal',           'Explained Variance', (None, 1)),
}


def plot_metrics(metrics_dict, out_dir):
    """
    Un line chart per metrica. Neuroni in ordine originale (0 … N_CELLS-1).
    Salva un PNG per metrica in out_dir/.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    neuron_ids = np.arange(N_CELLS)

    for name, values in metrics_dict.items():
        color, ylabel, ylim = _METRIC_STYLES.get(name, ('gray', name, (None, None)))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(neuron_ids, values, color=color, linewidth=1.6,
                marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.5)
        ax.axhline(values.mean(), color='red', linestyle='--',
                   label=f'mean = {values.mean():.3f}  median = {np.median(values):.3f}')
        ax.set_xlabel('Neuron index')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}  —  {N_CELLS} neurons')
        ax.set_xticks(neuron_ids)
        ax.tick_params(axis='x', labelsize=7)
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        if ylim[0] is not None:
            ax.set_ylim(ylim)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{name}.png', dpi=150)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plots — Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(y_pred, y_true_mean, adj_r2, out_dir, neuron_offset=0):
    """Un scatter plot per neurone (pred vs mean response).

    neuron_offset : offset added to the local index for plot titles / filenames,
                    used when evaluating single-neuron models (e.g. offset=5 →
                    loop index 0 is labelled as Neuron 05).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    WINDOW_S = 0.3

    for local_n in range(y_pred.shape[1]):
        n = local_n + neuron_offset
        true_fr = y_true_mean[:, local_n] / WINDOW_S
        pred_fr = y_pred[:, local_n]      / WINDOW_S

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(true_fr, pred_fr, alpha=0.5, s=15,
                   color='steelblue', edgecolors='none')
        lim = [min(true_fr.min(), pred_fr.min()),
               max(true_fr.max(), pred_fr.max())]
        ax.plot(lim, lim, 'r--', linewidth=1)
        ax.set_xlabel('True firing rate (spk/s)')
        ax.set_ylabel('Predicted firing rate (spk/s)')
        ax.set_title(f'Neuron {n:02d}  |  Adj R²={adj_r2[local_n]:.3f}')
        plt.tight_layout()
        plt.savefig(out / f'neuron_{n:02d}.png', dpi=120)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plots — Response rank
# ─────────────────────────────────────────────────────────────────────────────

def plot_response_rank(y_pred, y_true_mean, adj_r2, out_dir, neuron_offset=0):
    """Risposta reale vs predetta ordinate per rank della risposta reale.

    neuron_offset : same semantics as in plot_scatter.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    WINDOW_S = 0.3
    ranks    = np.arange(1, y_true_mean.shape[0] + 1)

    for local_n in range(y_pred.shape[1]):
        n = local_n + neuron_offset
        true_fr = y_true_mean[:, local_n] / WINDOW_S
        pred_fr = y_pred[:, local_n]      / WINDOW_S
        order   = np.argsort(true_fr)[::-1]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ranks, true_fr[order], color='steelblue', marker='o',
                markersize=4, linewidth=1.2, label='Real')
        ax.plot(ranks, pred_fr[order], color='tomato', marker='s',
                markersize=4, linewidth=1.2, linestyle='--', label='Predicted')
        ax.set_xlabel('Image rank (descending true response)')
        ax.set_ylabel('Firing rate (spk/s)')
        ax.set_title(f'Neuron {n:02d}  |  Adj R²={adj_r2[local_n]:.3f}')
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out / f'neuron_{n:02d}.png', dpi=130)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing LSTA
# ─────────────────────────────────────────────────────────────────────────────

def lsta_postprocess(arr, expon_treat=2, vmax_thresh=1.0, put_to0_frac=0.2):
    """
    Post-processing visivo per LSTA (NON modifica i dati salvati).
    a) Esponenziazione: sharpening che amplifica i valori forti.
    b) Thresholding: azzera il rumore sotto soglia.
    """
    arr = arr.copy().astype(float)
    if expon_treat % 2 == 1:
        arr = arr ** expon_treat
    else:
        arr = np.sign(arr) * np.exp(np.log(np.abs(arr) + 1e-12) * expon_treat)
    vmax = max(arr.max(), -arr.min()) * vmax_thresh
    if vmax > 0:
        arr[np.abs(arr) < vmax * put_to0_frac] = 0.0
    return arr, vmax


# ─────────────────────────────────────────────────────────────────────────────
# Plots — Confronto LSTA sperimentale vs modello
# ─────────────────────────────────────────────────────────────────────────────

def plot_lsta_comparison(
    lsta_exp_list,
    lsta_model_list,
    cell_nb,
    img_labels=None,
    out_dir='results/lsta',
    expon_treat=2,
    vmax_thresh=1.0,
    put_to0_frac=0.2,
    cmap='RdBu_r',
):
    """
    Per ogni immagine, affianca LSTA sperimentale e LSTA del modello.
    Salva out_dir/lsta_cell_XX.png
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_imgs = len(lsta_exp_list)
    if img_labels is None:
        img_labels = [f'img {i}' for i in range(n_imgs)]

    fig, axes = plt.subplots(2, n_imgs, figsize=(4 * n_imgs, 8))
    if n_imgs == 1:
        axes = axes[:, np.newaxis]

    for im in range(n_imgs):
        exp_proc, vmax_exp = lsta_postprocess(
            lsta_exp_list[im], expon_treat, vmax_thresh, put_to0_frac)
        axes[0, im].imshow(exp_proc, cmap=cmap,
                           vmin=-vmax_exp, vmax=vmax_exp,
                           interpolation='bicubic')
        axes[0, im].set_xticks([])
        axes[0, im].set_yticks([])
        if im == 0:
            axes[0, im].set_ylabel('LSTA sperimentale', fontsize=10)
        axes[0, im].set_title(img_labels[im], fontsize=9)

        mod_proc, vmax_mod = lsta_postprocess(
            lsta_model_list[im], expon_treat, vmax_thresh, put_to0_frac)
        axes[1, im].imshow(mod_proc, cmap=cmap,
                           vmin=-vmax_mod, vmax=vmax_mod,
                           interpolation='bicubic')
        axes[1, im].set_xticks([])
        axes[1, im].set_yticks([])
        if im == 0:
            axes[1, im].set_ylabel('LSTA modello', fontsize=10)

    fig.suptitle(
        f'Cell {cell_nb}  —  exp={expon_treat}  vmax_th={vmax_thresh}'
        f'  to0={put_to0_frac}  interp=bicubic',
        fontsize=10)
    plt.tight_layout()
    plt.savefig(out / f'lsta_cell_{cell_nb:03d}.png', dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# LSTA modello via autograd  +  confronto con LSTA sperimentale
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_lsta(model, images_np, image_indices, device, neuron_subset=None):
    """
    Calcola LSTA_n(X) = ∂r_n(X)/∂X per autograd.

    images_np     : (N, H, W) float32
    image_indices : indici delle immagini da usare
    neuron_subset : lista di indici globali; None → tutti i N_CELLS

    Restituisce float32 array  (len(neuron_subset), len(image_indices), H, W)
    """
    if neuron_subset is None:
        neuron_subset = list(range(N_CELLS))

    model.eval()
    N_ref = len(image_indices)
    lsta  = np.zeros((len(neuron_subset), N_ref,
                      IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL), dtype=np.float32)

    for j, idx in enumerate(image_indices):
        x = torch.tensor(
            images_np[idx][None, None], dtype=torch.float32,
            device=device, requires_grad=True)
        out = model(x)   # (1, N_CELLS)

        for local_n, global_n in enumerate(neuron_subset):
            if x.grad is not None:
                x.grad.zero_()
            out[0, global_n].backward(
                retain_graph=(local_n < len(neuron_subset) - 1))
            lsta[local_n, j] = x.grad[0, 0].detach().cpu().numpy()

    return lsta


def compute_attention_rollout_maps(model, images_np, image_indices, device, neuron_subset=None):
    """
    Calcola attention rollout maps per ogni neurone e immagine di riferimento.

    Usa forward pre-hook sul modulo Dropout interno a ogni MultiHeadSelfAttention
    per catturare i pesi softmax senza modificare il modello.

    Rollout (Abnar & Zuidema 2020):
      per ogni blocco l: Ã_l = (mean_heads(A_l) + I) / row_sum
      rollout = Ã_1 @ Ã_2 @ ... @ Ã_L  →  media sui query → (CS, CS) per neurone

    I risultati vengono poi riproiettati nello spazio immagine originale.

    Restituisce: (n_neurons, n_images, IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL) float32
    """
    from scipy.ndimage import zoom as nd_zoom

    if neuron_subset is None:
        neuron_subset = list(range(N_CELLS))

    n_sub = len(neuron_subset)
    n_ref = len(image_indices)
    model.eval()

    CS       = model.crop_size
    N_cells  = model.n_cells
    feat_h   = model.crop.img_h                      # 54 (MAX_POOL) o 108
    stride   = IMG_SIZE_ORIGINAL // feat_h            # 2 o 1
    cx_feat  = model.crop.cx.detach().cpu().numpy()  # (N_cells,) in feat coords
    cy_feat  = model.crop.cy.detach().cpu().numpy()

    # ── registra hook su Dropout di ogni blocco MHSA ─────────────────────────
    blocks = list(model.transformer.blocks)
    attn_store = [[] for _ in blocks]

    def make_pre_hook(store_list):
        def hook(module, inputs):
            store_list.append(inputs[0].detach().cpu())
        return hook

    hooks = [
        block.attn.drop.register_forward_pre_hook(make_pre_hook(attn_store[i]))
        for i, block in enumerate(blocks)
    ]

    result = np.zeros((n_sub, n_ref, IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL), dtype=np.float32)

    try:
        for j, idx in enumerate(image_indices):
            for store in attn_store:
                store.clear()

            x = torch.tensor(images_np[idx][None, None], dtype=torch.float32, device=device)
            with torch.no_grad():
                model(x)

            # attn_store[l][0]: (N_cells, heads, CS², CS²)  — batch=1 fuso in N_cells
            attn_list = [store[0] for store in attn_store]   # list of (N_cells, H, N, N)

            # ── rollout ────────────────────────────────────────────────────────
            rollout = None
            for attn in attn_list:
                A = attn.mean(dim=1).float()                     # (N_cells, CS², CS²)
                I = torch.eye(CS * CS).unsqueeze(0)              # (1, CS², CS²)
                A = A + I
                A = A / A.sum(dim=-1, keepdim=True)
                rollout = A if rollout is None else torch.bmm(rollout, A)

            # media sui query token → (N_cells, CS²) → (N_cells, CS, CS)
            rollout_cs = rollout.mean(dim=1).numpy().reshape(N_cells, CS, CS)

            # ── riproietta nello spazio immagine ───────────────────────────────
            half = (CS - 1) / 2.0
            ci_idx = np.arange(CS)
            cj_idx = np.arange(CS)
            ci_grid, cj_grid = np.meshgrid(ci_idx, cj_idx, indexing='ij')  # (CS, CS)

            for local_n, global_n in enumerate(neuron_subset):
                rm = rollout_cs[global_n]   # (CS, CS)

                # coordinate immagine di ogni posizione del crop
                img_y = np.round((cy_feat[global_n] + (ci_grid - half)) * stride).astype(int)
                img_x = np.round((cx_feat[global_n] + (cj_grid - half)) * stride).astype(int)

                full = np.zeros((IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL), dtype=np.float32)
                valid = ((img_x >= 0) & (img_x < IMG_SIZE_ORIGINAL) &
                         (img_y >= 0) & (img_y < IMG_SIZE_ORIGINAL))
                full[img_y[valid], img_x[valid]] = rm[valid]

                # interpola per riempire i buchi (stride > 1 lascia griglie rade)
                if stride > 1:
                    from scipy.ndimage import gaussian_filter
                    full = gaussian_filter(full, sigma=stride * 0.6)

                # normalizza 0-1
                v_max = full.max()
                if v_max > 0:
                    full /= v_max

                result[local_n, j] = full

    finally:
        for h in hooks:
            h.remove()

    return result


def plot_lsta_model_comparison(
    model,
    adj_r2,
    lsta_ref_path,
    images_np,        # (N, H, W) già caricato e squeezed
    out_dir,
    device,
    neuron_subset      = None,
    padding            = 8,
    expon_treat        = 2,
    vmax_thresh        = 1.0,
    put_to0_frac       = 0.2,
    lsta_model_all     = None,   # pre-computed (n_sub, N_img, H, W); skips autograd if provided
    attn_rollout_all   = None,   # pre-computed (n_sub, N_img, H, W); None → calcolato qui
):
    """
    Per ogni neurone produce una figura 7 righe × N_IMAGES colonne:
      riga 0  immagine stimulus              + ellisse RF
      riga 1  LSTA sperimentale  (raw)       + ellisse RF
      riga 2  LSTA modello       (raw)       + ellisse RF
      riga 3  LSTA sperimentale  (proc.)     + ellisse RF
      riga 4  LSTA modello       (proc.)     + ellisse RF
      riga 5  LSTA sperimentale  (gaussian)  + ellisse RF
      riga 6  LSTA modello       (gaussian)  + ellisse RF

    La LSTA del modello è il gradiente ∂r_n(X)/∂X calcolato via autograd.
    Il post-processing (righe 3-4) usa lsta_postprocess (esponente + soglia).

    lsta_ref_path : path a lsta_ref.npz  (lsta, ellipses, image_indices, cell_indices)
    images_np     : immagini test  (N, H, W)  già squeezed
    neuron_subset : lista di neuroni da plottare; None → tutti
    """
    from PIL import Image as PILImage

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lsta_ref      = np.load(lsta_ref_path)
    lsta_exp      = lsta_ref['lsta']           # (41, N_img, 72, 72)
    ellipses      = lsta_ref['ellipses']       # (41, 2, 360)
    image_indices = lsta_ref['image_indices']  # (N_img,)
    cell_indices  = lsta_ref['cell_indices']   # (41,)

    if neuron_subset is None:
        neuron_subset = list(range(N_CELLS))

    N_IMAGES  = lsta_exp.shape[1]
    LSTA_SIZE = lsta_exp.shape[2]
    SCALE     = IMG_SIZE_ORIGINAL / LSTA_SIZE

    def _resize(arr):
        pil = PILImage.fromarray(arr.astype(np.float32), mode='F')
        pil = pil.resize((IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL), PILImage.BILINEAR)
        return np.array(pil)

    def _bbox(ex, ey):
        x0 = int(max(0,                  ex.min() - padding))
        x1 = int(min(IMG_SIZE_ORIGINAL,  ex.max() + padding))
        y0 = int(max(0,                  ey.min() - padding))
        y1 = int(min(IMG_SIZE_ORIGINAL,  ey.max() + padding))
        return x0, x1, y0, y1

    if lsta_model_all is None:
        print(f'  Computing model LSTA  '
              f'({N_IMAGES} images × {len(neuron_subset)} neurons)...')
        lsta_model_all = compute_model_lsta(
            model, images_np, image_indices, device, neuron_subset=neuron_subset)

    if attn_rollout_all is None:
        print(f'  Computing attention rollout  '
              f'({N_IMAGES} images × {len(neuron_subset)} neurons)...')
        attn_rollout_all = compute_attention_rollout_maps(
            model, images_np, image_indices, device, neuron_subset=neuron_subset)

    row_labels = [
        'image',
        'LSTA exp\n(proc)',
        'LSTA model\n(proc)',
        'Attn\nRollout',
    ]

    for local_n, n in enumerate(neuron_subset):
        ex = ellipses[n, 0, :] * SCALE
        ey = ellipses[n, 1, :] * SCALE
        x0, x1, y0, y1 = _bbox(ex, ey)
        extent = [x0, x1, y1, y0]

        # ── calcola proc maps e correlazioni ─────────────────────────────────
        corrs = []
        proc_exp_list   = []
        proc_model_list = []
        vmax_pe_list    = []
        vmax_pm_list    = []

        for i in range(N_IMAGES):
            map_exp   = _resize(lsta_exp[n, i])
            map_model = lsta_model_all[local_n, i]

            pe, vmax_pe = lsta_postprocess(map_exp,   expon_treat, vmax_thresh, put_to0_frac)
            pm, vmax_pm = lsta_postprocess(map_model, expon_treat, vmax_thresh, put_to0_frac)

            proc_exp_list.append(pe)
            proc_model_list.append(pm)
            vmax_pe_list.append(vmax_pe)
            vmax_pm_list.append(vmax_pm)

            flat_e = pe[y0:y1, x0:x1].ravel()
            flat_m = pm[y0:y1, x0:x1].ravel()
            if flat_e.std() > 0 and flat_m.std() > 0:
                corr = float(np.corrcoef(flat_e, flat_m)[0, 1])
            else:
                corr = 0.0
            corrs.append(corr)

        mean_corr = float(np.mean(corrs))

        fig, axes = plt.subplots(
            4, N_IMAGES,
            figsize=(N_IMAGES * 2.4, 9.6),
            gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
        if N_IMAGES == 1:
            axes = axes[:, np.newaxis]

        corr_str = '  '.join(f'{c:.2f}' for c in corrs)
        fig.suptitle(
            f'Neuron {n}  (global {cell_indices[n]})  |  Adj R²={adj_r2[n]:.3f}'
            f'  |  mean corr={mean_corr:.3f}\ncorr per img: {corr_str}',
            fontsize=9, y=1.02)

        for i in range(N_IMAGES):
            img_idx  = image_indices[i]
            img      = images_np[img_idx]
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

            pe      = proc_exp_list[i]
            pm      = proc_model_list[i]
            vmax_pe = vmax_pe_list[i]
            vmax_pm = vmax_pm_list[i]
            ar      = attn_rollout_all[local_n, i]   # (H, W) in [0, 1]

            ax_img, ax_exp_p, ax_mod_p, ax_roll = axes[:, i]

            # riga 0: immagine
            ax_img.imshow(img_norm[y0:y1, x0:x1], cmap='gray',
                          origin='upper', extent=extent)
            ax_img.plot(ex, ey, color='lime', lw=1.2)
            ax_img.set_xlim(x0, x1); ax_img.set_ylim(y1, y0)
            ax_img.set_title(f'img {img_idx}\nr={corrs[i]:.2f}', fontsize=7)
            ax_img.axis('off')

            # riga 1: LSTA sperimentale post-processata
            ax_exp_p.imshow(pe[y0:y1, x0:x1], cmap='RdBu_r',
                            origin='upper', extent=extent,
                            vmin=-vmax_pe, vmax=vmax_pe,
                            interpolation='bicubic')
            ax_exp_p.plot(ex, ey, color='k', lw=1.2)
            ax_exp_p.set_xlim(x0, x1); ax_exp_p.set_ylim(y1, y0)
            ax_exp_p.axis('off')

            # riga 2: LSTA modello post-processata
            ax_mod_p.imshow(pm[y0:y1, x0:x1], cmap='RdBu_r',
                            origin='upper', extent=extent,
                            vmin=-vmax_pm, vmax=vmax_pm,
                            interpolation='bicubic')
            ax_mod_p.plot(ex, ey, color='k', lw=1.2)
            ax_mod_p.set_xlim(x0, x1); ax_mod_p.set_ylim(y1, y0)
            ax_mod_p.axis('off')

            # riga 3: attention rollout (0–1, colormap 'hot')
            ax_roll.imshow(ar[y0:y1, x0:x1], cmap='hot',
                           origin='upper', extent=extent,
                           vmin=0, vmax=1,
                           interpolation='bicubic')
            ax_roll.plot(ex, ey, color='lime', lw=1.2)
            ax_roll.set_xlim(x0, x1); ax_roll.set_ylim(y1, y0)
            ax_roll.axis('off')

        for ax, label in zip(axes[:, 0], row_labels):
            ax.axis('on')
            ax.set_ylabel(label, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.savefig(out / f'neuron_{n:02d}.png', dpi=130, bbox_inches='tight')
        plt.close(fig)

    print(f'  {len(neuron_subset)} LSTA figures saved to {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Plots — RF centers su STA (raw, senza post-processing)
# ─────────────────────────────────────────────────────────────────────────────

def _sym_vmax(arr):
    """Restituisce il valore massimo assoluto per una colormap simmetrica."""
    v = float(np.abs(arr).max())
    return v if v > 0 else 1e-6


def plot_sta_centers(model, cfg, centers_x_init, centers_y_init, adj_r2,
                     sta_dir, out_dir, sta_crops):
    """
    Per ogni neurone produce una figura a 3 pannelli:
      Sinistra : STA completa 108×108 con centri RF iniziali/trained, bbox del crop, freccia di drift.
      Centro   : Maschera spaziale iniziale del readout (= crop STA, raw).
      Destra   : Maschera spaziale trained del readout (raw).

    Salva out_dir/neuron_XX.png  +  out_dir/drift_overview.png
    Nessun filtro applicato: le immagini sono mostrate raw con colormap simmetrica.
    """
    out      = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sta_dir  = Path(sta_dir)
    CS       = cfg['CROP_SIZE']
    max_pool = cfg.get('MAX_POOL', False)
    scale    = 2 if max_pool else 1

    cx_trained = model.crop.cx.detach().cpu().numpy() * scale
    cy_trained = model.crop.cy.detach().cpu().numpy() * scale

    drifts  = np.sqrt((cx_trained - centers_x_init)**2 +
                      (cy_trained - centers_y_init)**2)
    CS_orig = CS * scale

    # Use the actual number of neurons in the model (supports N_CELLS_OVERRIDE=1)
    n_neurons_plot = model.n_cells
    u_trained_all = (model.readout.u.detach().cpu().numpy()
                     .reshape(n_neurons_plot, CS, CS))

    for n in range(n_neurons_plot):
        cell_id = 21 + n
        sta     = np.load(sta_dir / f'cell_{cell_id:03d}_sta_z.npy')

        cx_i = centers_x_init[n]
        cy_i = centers_y_init[n]
        cx_t = cx_trained[n]
        cy_t = cy_trained[n]

        half = CS_orig // 2
        y0 = int(np.clip(round(cy_t) - half, 0, IMG_SIZE_ORIGINAL - CS_orig))
        x0 = int(np.clip(round(cx_t) - half, 0, IMG_SIZE_ORIGINAL - CS_orig))

        u_init    = sta_crops[n]        # [CS, CS] — raw STA crop (init)
        u_trained = u_trained_all[n]    # [CS, CS] — trained readout weights

        vmax_sta     = _sym_vmax(sta)
        vmax_init    = _sym_vmax(u_init)
        vmax_trained = _sym_vmax(u_trained)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # ── pannello 1: STA completa con centri ──────────────────────────────
        ax1.imshow(sta, cmap='RdBu_r', vmin=-vmax_sta, vmax=vmax_sta,
                   interpolation='nearest', origin='upper')
        ax1.plot(cx_i, cy_i, 'o', color='limegreen', markersize=8,
                 markeredgecolor='black', markeredgewidth=0.8, label='Initial center')
        ax1.plot(cx_t, cy_t, '*', color='red', markersize=12,
                 markeredgecolor='black', markeredgewidth=0.5, label='Trained center')
        ax1.add_patch(Rectangle(
            (x0, y0), CS_orig, CS_orig,
            linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--',
            label=f'Crop {CS_orig}×{CS_orig}'))
        ax1.arrow(cx_i, cy_i, cx_t - cx_i, cy_t - cy_i,
                  color='yellow', width=0.3, head_width=1.5,
                  length_includes_head=True, alpha=0.8)
        ax1.set_title(
            f'Neuron {n:02d}  |  Adj R²={adj_r2[n]:.3f}  |  drift={drifts[n]:.2f} px')
        ax1.legend(fontsize=7, loc='upper right')
        ax1.axis('off')

        # ── pannello 2: maschera iniziale (raw) ──────────────────────────────
        im2 = ax2.imshow(u_init, cmap='RdBu_r',
                         vmin=-vmax_init, vmax=vmax_init,
                         interpolation='nearest', origin='upper')
        ax2.set_title('Readout u  —  init (STA crop, raw)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # ── pannello 3: maschera trained (raw) ───────────────────────────────
        im3 = ax3.imshow(u_trained, cmap='RdBu_r',
                         vmin=-vmax_trained, vmax=vmax_trained,
                         interpolation='nearest', origin='upper')
        ax3.set_title('Readout u  —  trained (raw)')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        plt.suptitle(f'Neuron {n:02d}  (cell_{cell_id:03d})  —  readout spatial mask',
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(out / f'neuron_{n:02d}.png', dpi=150)
        plt.close()

    # ── drift overview: tutti i neuroni in un unico plot ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(np.zeros((IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL)), cmap='gray',
              vmin=0, vmax=1, origin='upper')

    for n in range(n_neurons_plot):
        ax.annotate('', xy=(cx_trained[n], cy_trained[n]),
                    xytext=(centers_x_init[n], centers_y_init[n]),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2))
        ax.plot(centers_x_init[n], centers_y_init[n], 'o',
                color='limegreen', markersize=5)
        ax.plot(cx_trained[n],     cy_trained[n],     '*',
                color='red', markersize=7)

    ax.set_xlim(0, IMG_SIZE_ORIGINAL)
    ax.set_ylim(IMG_SIZE_ORIGINAL, 0)
    ax.set_title('RF center drift  (green = initial, red = trained)')
    handles = [mpatches.Patch(color='limegreen', label='Initial centers'),
               mpatches.Patch(color='red',       label='Trained centers')]
    ax.legend(handles=handles)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    plt.tight_layout()
    plt.savefig(out / 'drift_overview.png', dpi=150)
    plt.close()

    print(f'  Mean drift: {drifts.mean():.2f} px  '
          f'max: {drifts.max():.2f} px  (neuron {drifts.argmax():02d})')


# ─────────────────────────────────────────────────────────────────────────────
# Plots — Training history
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history, out_dir):
    """
    Plotta train/val loss dalla history distinguendo le due fasi di training.

    history : dict {'train_loss': [...], 'val_loss': [...],
                    'phase1_end': int}   ← epoch (1-based) in cui finisce la fase 1
              oppure path str/Path a un history.json
    out_dir : directory dove salvare training_history.png

    Se 'phase1_end' è assente o <= 0, il plot è single-phase (comportamento
    precedente).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(history, (str, Path)):
        with open(history) as f:
            history = json.load(f)

    train_loss  = history['train_loss']
    val_loss    = history['val_loss']
    n_epochs    = len(train_loss)
    epochs      = list(range(1, n_epochs + 1))
    phase1_end  = int(history.get('phase1_end', 0))   # 0 → no two-phase split

    # clamp: non può superare il numero effettivo di epoche
    phase1_end = min(phase1_end, n_epochs)
    two_phases = phase1_end > 0 and phase1_end < n_epochs

    fig, ax = plt.subplots(figsize=(10, 4))

    # ── shading delle fasi ────────────────────────────────────────────────────
    if two_phases:
        ax.axvspan(1, phase1_end + 0.5,
                   color='steelblue', alpha=0.07, label='Phase 1 (STA frozen)')
        ax.axvspan(phase1_end + 0.5, n_epochs,
                   color='darkorange', alpha=0.07, label='Phase 2 (all params)')
        ax.axvline(phase1_end + 0.5, color='dimgray', linewidth=1.4,
                   linestyle='--', zorder=3)
        # etichette di fase centrate nello shading
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(max(train_loss), max(val_loss))
        mid1  = (1 + phase1_end) / 2
        mid2  = (phase1_end + n_epochs) / 2
        ax.text(mid1, 1.0, 'Phase 1', ha='center', va='bottom',
                fontsize=8, color='steelblue', transform=ax.get_xaxis_transform())
        ax.text(mid2, 1.0, 'Phase 2', ha='center', va='bottom',
                fontsize=8, color='darkorange', transform=ax.get_xaxis_transform())

    # ── curve di loss ─────────────────────────────────────────────────────────
    ax.plot(epochs, train_loss, color='steelblue', linewidth=1.5, label='Train loss')
    ax.plot(epochs, val_loss,   color='coral',     linewidth=1.5, label='Val loss')

    # ── best epoch globale ────────────────────────────────────────────────────
    best_epoch = int(min(range(n_epochs), key=lambda i: val_loss[i])) + 1
    ax.axvline(best_epoch, color='crimson', linestyle=':', linewidth=1.2,
               label=f'Best epoch ({best_epoch})  val={min(val_loss):.4f}')

    # ── best epoch per fase (solo se two_phase) ───────────────────────────────
    if two_phases:
        p1_slice = val_loss[:phase1_end]
        p2_slice = val_loss[phase1_end:]
        if p1_slice:
            best1 = int(min(range(len(p1_slice)), key=lambda i: p1_slice[i])) + 1
            ax.axvline(best1, color='steelblue', linestyle=':', linewidth=1.0,
                       label=f'Best ph1 ({best1})  val={min(p1_slice):.4f}')
        if p2_slice:
            best2 = int(min(range(len(p2_slice)), key=lambda i: p2_slice[i])) + phase1_end + 1
            ax.axvline(best2, color='darkorange', linestyle=':', linewidth=1.0,
                       label=f'Best ph2 ({best2})  val={min(p2_slice):.4f}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Poisson Loss')
    ax.set_title('Training history' + ('  —  2-phase' if two_phases else ''))
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()

    path = Path(out_dir) / 'training_history.png'
    plt.savefig(path, dpi=150)
    plt.close()

    print(f'  Total epochs : {n_epochs}')
    if two_phases:
        print(f'  Phase 1 end  : epoch {phase1_end}')
    print(f'  Best epoch   : {best_epoch}')
    print(f'  Best val loss: {min(val_loss):.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# Plots — Per-neuron training histories (mean_all retrained models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_per_neuron(histories, out_dir):
    """
    Generates one training-history PNG per neuron.

    histories : list of 41 dicts, each with keys
                'train_loss', 'val_loss', 'best_epoch'  (produced by
                retrain_readout._retrain_mean_all)
                OR a list of paths to history_neuron_XX.json files.
    out_dir   : directory where neuron_00.png … neuron_40.png are saved.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for n, h in enumerate(histories):
        if isinstance(h, (str, Path)):
            with open(h) as f:
                h = json.load(f)

        train_loss = h['train_loss']
        val_loss   = h['val_loss']
        best_ep    = h.get('best_epoch', 0)
        n_ep       = len(train_loss)
        epochs     = list(range(1, n_ep + 1))

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(epochs, train_loss, color='steelblue', linewidth=1.4,
                label='Train loss')
        ax.plot(epochs, val_loss,   color='coral',     linewidth=1.4,
                label='Val loss')
        if best_ep > 0:
            ax.axvline(best_ep, color='crimson', linestyle=':', linewidth=1.2,
                       label=f'Best epoch ({best_ep})'
                             f'  val={min(val_loss):.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Poisson Loss')
        ax.set_title(f'Neuron {n:02d}  —  readout retrain'
                     f'  (best ep {best_ep}, {n_ep} total)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(out / f'neuron_{n:02d}.png', dpi=130)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# LSTA helper — computes arrays without plotting
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lsta_arrays(model, lsta_ref_path, images_np, device,
                          neuron_subset=None, padding=8,
                          expon=2, vmax_thresh=1.0, to0_frac=0.2):
    """
    Computes model LSTA (via autograd) and per-neuron LSTA correlation.

    Returns a dict with keys:
      lsta_model      float32  (n_sub, N_img, H, W)
      lsta_exp_raw    float32  (n_sub, N_img, H_lsta, W_lsta)  — original res
      lsta_corr       float32  (n_sub,)
      image_indices   int32    (N_img,)
      ellipses        float32  (n_sub, 2, 360)  — in LSTA pixel coords
      cell_indices    int32    (n_sub,)
      neuron_subset   int32    (n_sub,)
    """
    from PIL import Image as PILImage

    lsta_ref      = np.load(lsta_ref_path)
    lsta_exp      = lsta_ref['lsta']           # (41, N_img, H_lsta, W_lsta)
    ellipses      = lsta_ref['ellipses']       # (41, 2, 360)
    image_indices = lsta_ref['image_indices']  # (N_img,)
    cell_indices  = lsta_ref['cell_indices']   # (41,)

    if neuron_subset is None:
        neuron_subset = list(range(model.n_cells))

    N_IMAGES  = lsta_exp.shape[1]
    LSTA_SIZE = lsta_exp.shape[2]
    SCALE     = IMG_SIZE_ORIGINAL / LSTA_SIZE

    print(f'  Computing model LSTA  '
          f'({N_IMAGES} images × {len(neuron_subset)} neurons)...')
    lsta_model = compute_model_lsta(
        model, images_np, image_indices, device, neuron_subset=neuron_subset)
    # lsta_model: (n_sub, N_img, H, W) float32

    def _resize(arr):
        pil = PILImage.fromarray(arr.astype(np.float32), mode='F')
        pil = pil.resize((IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL), PILImage.BILINEAR)
        return np.array(pil)

    lsta_corr = np.zeros(len(neuron_subset), dtype=np.float32)
    for local_n, n in enumerate(neuron_subset):
        ex = ellipses[n, 0, :] * SCALE
        ey = ellipses[n, 1, :] * SCALE
        x0 = int(max(0,                 ex.min() - padding))
        x1 = int(min(IMG_SIZE_ORIGINAL, ex.max() + padding))
        y0 = int(max(0,                 ey.min() - padding))
        y1 = int(min(IMG_SIZE_ORIGINAL, ey.max() + padding))
        corrs = []
        for i in range(N_IMAGES):
            pe, _ = lsta_postprocess(_resize(lsta_exp[n, i]),
                                     expon, vmax_thresh, to0_frac)
            pm, _ = lsta_postprocess(lsta_model[local_n, i],
                                     expon, vmax_thresh, to0_frac)
            fe = pe[y0:y1, x0:x1].ravel()
            fm = pm[y0:y1, x0:x1].ravel()
            if fe.std() > 0 and fm.std() > 0:
                corrs.append(float(np.corrcoef(fe, fm)[0, 1]))
            else:
                corrs.append(0.0)
        lsta_corr[local_n] = float(np.mean(corrs))

    ns = np.array(neuron_subset, dtype=np.int32)
    return {
        'lsta_model':     lsta_model.astype(np.float32),
        'lsta_exp_raw':   lsta_exp[neuron_subset].astype(np.float32),
        'lsta_corr':      lsta_corr,
        'image_indices':  image_indices.astype(np.int32),
        'ellipses':       ellipses[neuron_subset].astype(np.float32),
        'cell_indices':   cell_indices[neuron_subset].astype(np.int32),
        'neuron_subset':  ns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Attribution operators — path generators, gradient computation, plotting
# ─────────────────────────────────────────────────────────────────────────────

def _interp_path_ig(image, n_steps=10):
    """Percorso lineare da grigio (0.5) all'immagine. Restituisce (n_steps, H, W)."""
    baseline = 0.5
    alphas = np.linspace(0.0, 1.0, n_steps)
    return np.stack([baseline + alpha * (image - baseline) for alpha in alphas], axis=0).astype(np.float32)



def _interp_path_wig(image, n_steps=10):
    """
    Percorso: nero → immagine (waypoint) → bianco (endpoint).
    Il bianco è ones × max(immagine, 1).
    Due segmenti: nero→immagine (metà passi), immagine→bianco (metà passi).
    Restituisce (n_steps, H, W).
    """
    white = np.ones_like(image) * max(float(image.max()), 1.0)
    n1    = n_steps // 2
    n2    = n_steps - n1
    seg1  = [t * image                  for t in np.linspace(0.0, 1.0, n1, endpoint=False)]
    seg2  = [image + t * (white - image) for t in np.linspace(0.0, 1.0, n2)]
    return np.stack(seg1 + seg2, axis=0).astype(np.float32)


def _operator_gradients_for_neuron(model, path_images_all, neuron_idx, device):
    """
    Calcola ∂output[neuron_idx]/∂input per ogni passo di percorso e immagine.

    path_images_all : (n_steps, N_img, H, W)  float32
    Restituisce     : (n_steps, N_img, H, W)  float32
    """
    model.eval()
    n_steps, N_img, H, W = path_images_all.shape
    grads = np.zeros((n_steps, N_img, H, W), dtype=np.float32)

    for k in range(n_steps):
        for j in range(N_img):
            x = torch.tensor(
                path_images_all[k, j][None, None],
                dtype=torch.float32, device=device, requires_grad=True
            )
            out = model(x)
            out[0, neuron_idx].backward()
            grads[k, j] = x.grad[0, 0].detach().cpu().numpy()

    return grads


def _compute_attribution(path_images_all, grads, x_start_all, x_end_all):
    """
    Attribuzione per integrazione dei gradienti:
      attr[j] = (x_end[j] - x_start[j]) × mean(grads[:, j], axis=0)

    Tutti gli array: (n_steps, N_img, H, W)  o  (N_img, H, W)
    Restituisce     : (N_img, H, W) float32
    """
    avg_grads = grads.mean(axis=0)       # (N_img, H, W)
    delta     = x_end_all - x_start_all  # (N_img, H, W)
    return (delta * avg_grads).astype(np.float32)


def plot_operator_figure(
    operator_label,
    neuron_idx,
    cell_index,
    adj_r2_val,
    path_imgs_ref0,  # (n_steps, H, W) — percorso per la prima immagine di riferimento
    stim_imgs,       # (N_img, H, W)   — stimoli originali
    image_indices,   # (N_img,)         — indici interi
    grad_maps,       # (n_steps, N_img, H, W)
    attr_maps,       # (N_img, H, W)
    ellipse,         # (2, 360)  coord x/y già in spazio IMG_SIZE_ORIGINAL
    out_path,
    padding          = 4,
    lsta_expon       = 2,
    lsta_vmax_thresh = 1.0,
    lsta_put_to0     = 0.2,
):
    """
    Genera la figura per un operatore di attribuzione per un singolo neurone.

    Layout orizzontale (N_img+1 righe × N_DISP+2 colonne):
      Riga 0 (top)  : [vuoto] | path_1 … path_N_DISP | [vuoto]
      Righe 1..N_img: stimolo  | grad_1 … grad_N_DISP | attributione

    Colorbar:
      - gradient maps: normalizzazione per riga (stessa vmax per tutti i passi
        di una stessa immagine)
      - attribution maps: normalizzazione per immagine (indipendente)
    """
    # ── numero fisso di pannelli da visualizzare ──────────────────────────────
    N_DISP  = 10
    n_steps = path_imgs_ref0.shape[0]
    N_img   = stim_imgs.shape[0]

    disp_idx  = np.round(np.linspace(0, n_steps - 1, N_DISP)).astype(int)
    path_disp = path_imgs_ref0[disp_idx]   # (N_DISP, H, W)
    grad_disp = grad_maps[disp_idx]         # (N_DISP, N_img, H, W)

    # ── bounding box intorno all'ellisse RF ───────────────────────────────────
    ex = ellipse[0]
    ey = ellipse[1]
    x0 = int(max(0,                  ex.min() - padding))
    x1 = int(min(IMG_SIZE_ORIGINAL,  ex.max() + padding))
    y0 = int(max(0,                  ey.min() - padding))
    y1 = int(min(IMG_SIZE_ORIGINAL,  ey.max() + padding))
    extent = [x0, x1, y1, y0]

    crop_h = max(y1 - y0, 1)
    crop_w = max(x1 - x0, 1)
    aspect = crop_h / crop_w
    cell_w = 2.0
    cell_h = cell_w * aspect

    def _show(ax, img, cmap, vmin, vmax, interp='nearest'):
        ax.imshow(img[y0:y1, x0:x1], cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation=interp, origin='upper', extent=extent)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

    def _tidy(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ── layout: (N_img+1) righe × (N_DISP+2) colonne ─────────────────────────
    n_rows = N_img + 1
    n_cols = N_DISP + 2   # col 0: stim/label | cols 1..N_DISP: step | col -1: attr

    fig_w = n_cols * cell_w
    fig_h = n_rows * cell_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(n_rows, n_cols, hspace=0.04, wspace=0.04)

    # ── riga 0 (top): immagini del percorso (prima img. di riferimento) ───────
    vmax_path = max(float(path_disp.max()), 1e-6)
    for d in range(N_DISP):
        ax = fig.add_subplot(gs[0, 1 + d])
        _show(ax, path_disp[d] / vmax_path, 'gray', 0, 1)
        frac = disp_idx[d] / max(n_steps - 1, 1)
        ax.set_title(f'α={frac:.2f}', fontsize=6, pad=2)
        _tidy(ax)
    # etichette angoli riga top
    ax_lbl = fig.add_subplot(gs[0, 0])
    ax_lbl.axis('off')
    ax_lbl.set_ylabel('percorso', fontsize=8)
    ax_lbl_r = fig.add_subplot(gs[0, N_DISP + 1])
    ax_lbl_r.axis('off')
    ax_lbl_r.set_title('attribut.', fontsize=7, pad=2)

    # ── pre-calcola vmax globale per tutti i gradient maps ───────────────────
    global_grad_vmax = max(
        (lsta_postprocess(grad_disp[d, j], lsta_expon, lsta_vmax_thresh, lsta_put_to0)[1]
         for d in range(N_DISP) for j in range(N_img)),
        default=1e-6
    ) or 1e-6

    # ── righe 1..N_img: stimolo | gradienti | attributione ────────────────────
    for j in range(N_img):
        row = 1 + j

        # col 0: stimolo
        ax_s = fig.add_subplot(gs[row, 0])
        vmax_s = max(float(stim_imgs[j].max()), 1e-6)
        _show(ax_s, stim_imgs[j] / vmax_s, 'gray', 0, 1)
        ax_s.plot(ex, ey, color='lime', lw=0.8)
        ax_s.set_ylabel(f'img {image_indices[j]}', fontsize=7)
        _tidy(ax_s)

        # cols 1..N_DISP: gradient maps (vmax globale condiviso)
        for d in range(N_DISP):
            ax_g = fig.add_subplot(gs[row, 1 + d])
            gp, _ = lsta_postprocess(grad_disp[d, j], lsta_expon, lsta_vmax_thresh, lsta_put_to0)
            _show(ax_g, gp, 'RdBu_r', -global_grad_vmax, global_grad_vmax, interp='bicubic')
            ax_g.plot(ex, ey, color='k', lw=0.8)
            _tidy(ax_g)

        # col N_DISP+1: attribution map (vmax per immagine)
        ax_a = fig.add_subplot(gs[row, N_DISP + 1])
        ap, va = lsta_postprocess(attr_maps[j], lsta_expon, lsta_vmax_thresh, lsta_put_to0)
        va = va or 1e-6
        _show(ax_a, ap, 'RdBu_r', -va, va, interp='bicubic')
        ax_a.plot(ex, ey, color='lime', lw=0.8)
        _tidy(ax_a)

    fig.suptitle(
        f'Neurone {neuron_idx}  (cell {cell_index})  |  Adj R²={adj_r2_val:.3f}'
        f'  |  {operator_label}'
        + (f'  [{n_steps} steps]' if n_steps != N_DISP else ''),
        fontsize=9)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _load_lsta_ref_for_operators(lsta_ref_path, images_np):
    """Carica lsta_ref.npz ed estrae le immagini di riferimento (N_img, H, W)."""
    lsta_ref      = np.load(lsta_ref_path)
    lsta_exp      = lsta_ref['lsta']            # (41, N_img, H_lsta, W_lsta)
    ellipses      = lsta_ref['ellipses']        # (41, 2, 360)  in coords LSTA
    image_indices = lsta_ref['image_indices']   # (N_img,)
    cell_indices  = lsta_ref['cell_indices']    # (41,)
    lsta_size     = lsta_exp.shape[2]
    scale         = IMG_SIZE_ORIGINAL / lsta_size
    ref_imgs      = images_np[image_indices.astype(int)]   # (N_img, H, W)
    return {
        'ellipses':      ellipses,
        'image_indices': image_indices,
        'cell_indices':  cell_indices,
        'scale':         scale,
        'ref_imgs':      ref_imgs,
        'lsta_exp':      lsta_exp,              # (41, N_img, H_lsta, W_lsta)
    }


def _precompute_op_paths(ref_imgs, n_steps, do_ig, do_wig):
    """
    Pre-calcola i percorsi per tutti gli operatori attivi e tutte le immagini.
    Restituisce: {op_key: (n_steps, N_img, H, W) float32}
    """
    N_img  = ref_imgs.shape[0]
    result = {}
    if do_ig:
        result['integrated_gradients'] = np.stack(
            [_interp_path_ig(ref_imgs[j], n_steps) for j in range(N_img)],
            axis=1)   # (n_steps, N_img, H, W)
    if do_wig:
        result['waypoint_integrated_gradients'] = np.stack(
            [_interp_path_wig(ref_imgs[j], n_steps) for j in range(N_img)],
            axis=1)
    return result


_OP_LABELS = {
    'integrated_gradients':          'Integrated Gradients',
    'waypoint_integrated_gradients': 'Waypoint Integrated Gradients',
}


def _run_operator_for_neuron(
    model,
    n,
    op_paths,
    ref_data,
    adj_r2_val,
    out_dir,
    device,
    padding          = 4,
    lsta_expon       = 2,
    lsta_vmax_thresh = 1.0,
    lsta_put_to0     = 0.2,
):
    """
    Calcola e salva le figure degli operatori per un singolo neurone.

    model       : RetinalModel caricato in eval mode
    n           : indice globale del neurone (0-40)
    op_paths    : {op_key: (n_steps, N_img, H, W)}  da _precompute_op_paths
    ref_data    : dict da _load_lsta_ref_for_operators
    adj_r2_val  : float scalare
    out_dir     : directory base; figure salvate in out_dir/<op_key>/neuron_XX.png
    """
    ellipses      = ref_data['ellipses']
    image_indices = ref_data['image_indices']
    cell_indices  = ref_data['cell_indices']
    scale         = ref_data['scale']
    ref_imgs      = ref_data['ref_imgs']
    N_img         = ref_imgs.shape[0]

    ex = ellipses[n, 0, :] * scale
    ey = ellipses[n, 1, :] * scale
    ellipse_scaled = np.stack([ex, ey], axis=0)   # (2, 360) in IMG coords

    attr_maps_by_op = {}
    for op_key, paths in op_paths.items():
        op_label = _OP_LABELS[op_key]
        grads    = _operator_gradients_for_neuron(model, paths, n, device)

        if op_key == 'waypoint_integrated_gradients':
            x_start = np.zeros_like(ref_imgs)
            x_end   = np.stack([
                np.ones_like(ref_imgs[j]) * max(float(ref_imgs[j].max()), 1.0)
                for j in range(N_img)
            ], axis=0)
        else:
            x_start = np.full_like(ref_imgs, 0.5)
            x_end   = ref_imgs.copy()

        attr     = _compute_attribution(paths, grads, x_start, x_end)
        out_path = Path(out_dir) / op_key / f'neuron_{n:02d}.png'

        plot_operator_figure(
            operator_label   = op_label,
            neuron_idx       = n,
            cell_index       = int(cell_indices[n]),
            adj_r2_val       = float(adj_r2_val),
            path_imgs_ref0   = paths[:, 0, :, :],
            stim_imgs        = ref_imgs,
            image_indices    = image_indices,
            grad_maps        = grads,
            attr_maps        = attr,
            ellipse          = ellipse_scaled,
            out_path         = str(out_path),
            padding          = padding,
            lsta_expon       = lsta_expon,
            lsta_vmax_thresh = lsta_vmax_thresh,
            lsta_put_to0     = lsta_put_to0,
        )
        attr_maps_by_op[op_key] = attr   # (N_img, H, W)

    return attr_maps_by_op


def plot_operator_recap(
    neuron_idx,
    cell_index,
    adj_r2_val,
    stim_imgs,          # (N_img, H, W)  stimoli originali
    image_indices,      # (N_img,)        indici interi
    lsta_exp_raw,       # (N_img, H_lsta, W_lsta)  LSTA sperimentale grezzo
    lsta_model_maps,    # (N_img, H, W)   gradiente modello ∂r/∂x
    attr_maps_dict,     # {op_key: (N_img, H, W)}  mappe di attribuzione
    ellipse,            # (2, 360)  in coords IMG_SIZE_ORIGINAL
    out_path,
    padding          = 4,
    lsta_expon       = 2,
    lsta_vmax_thresh = 1.0,
    lsta_put_to0     = 0.2,
):
    """
    Figura riassuntiva per neurone nella cartella operators/recap/.

    Righe: stimoli | LSTA exp (proc) | LSTA model (proc) | [IG attr] | [PIG attr] | [WIG attr]
    Colonne: N_img immagini di riferimento.

    Sul lato sinistro di ogni riga (tranne stimoli): coefficiente di correlazione
    medio tra quella mappa e la LSTA sperimentale, calcolato sul crop dell'ellisse RF.
    Per la riga LSTA exp il valore è corr(LSTA_exp, LSTA_model);
    per le righe modello e operatori è corr(mappa, LSTA_exp).
    """
    from PIL import Image as PILImage

    N_img   = stim_imgs.shape[0]
    H_lsta  = lsta_exp_raw.shape[1]
    SCALE   = IMG_SIZE_ORIGINAL / H_lsta

    ex = ellipse[0]
    ey = ellipse[1]
    x0 = int(max(0,                  ex.min() - padding))
    x1 = int(min(IMG_SIZE_ORIGINAL,  ex.max() + padding))
    y0 = int(max(0,                  ey.min() - padding))
    y1 = int(min(IMG_SIZE_ORIGINAL,  ey.max() + padding))
    extent = [x0, x1, y1, y0]

    def _resize(arr):
        pil = PILImage.fromarray(arr.astype(np.float32), mode='F')
        pil = pil.resize((IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL), PILImage.BILINEAR)
        return np.array(pil)

    def _pearson_crop(a, b):
        fa = a[y0:y1, x0:x1].ravel()
        fb = b[y0:y1, x0:x1].ravel()
        if fa.std() > 0 and fb.std() > 0:
            return float(np.corrcoef(fa, fb)[0, 1])
        return 0.0

    # ── post-processing LSTA exp ──────────────────────────────────────────────
    exp_proc  = []
    exp_vmax  = []
    for j in range(N_img):
        pe, ve = lsta_postprocess(
            _resize(lsta_exp_raw[j]), lsta_expon, lsta_vmax_thresh, lsta_put_to0)
        exp_proc.append(pe)
        exp_vmax.append(ve)

    # ── post-processing LSTA model ────────────────────────────────────────────
    mod_proc  = []
    mod_vmax  = []
    mod_corrs = []
    for j in range(N_img):
        pm, vm = lsta_postprocess(
            lsta_model_maps[j], lsta_expon, lsta_vmax_thresh, lsta_put_to0)
        mod_proc.append(pm)
        mod_vmax.append(vm)
        mod_corrs.append(_pearson_crop(exp_proc[j], pm))
    lsta_mean_corr = float(np.mean(mod_corrs))

    # ── post-processing operatori di attribuzione ─────────────────────────────
    op_results = {}   # {op_key: (proc_list, vmax_list, mean_corr)}
    for op_key, attr in attr_maps_dict.items():
        op_proc = []; op_vmax = []; op_corrs = []
        for j in range(N_img):
            pa, va = lsta_postprocess(
                attr[j], lsta_expon, lsta_vmax_thresh, lsta_put_to0)
            op_proc.append(pa)
            op_vmax.append(va)
            op_corrs.append(_pearson_crop(exp_proc[j], pa))
        op_results[op_key] = (op_proc, op_vmax, float(np.mean(op_corrs)))

    # ── layout figura ─────────────────────────────────────────────────────────
    n_rows   = 1 + 1 + 1 + len(attr_maps_dict)   # stim + exp + model + operatori
    crop_h   = max(y1 - y0, 1)
    crop_w   = max(x1 - x0, 1)
    aspect   = crop_h / crop_w
    cell_w   = 2.2
    cell_h   = cell_w * aspect

    fig, axes = plt.subplots(
        n_rows, N_img,
        figsize=(N_img * cell_w, n_rows * cell_h),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05},
    )
    if N_img == 1:
        axes = axes[:, np.newaxis]

    def _show(ax, img, cmap, vmin, vmax):
        ax.imshow(img[y0:y1, x0:x1], cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation='bicubic', origin='upper', extent=extent)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _ylabel(ax, label):
        ax.axis('on')
        ax.set_ylabel(label, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ── riga 0: stimoli ───────────────────────────────────────────────────────
    for j in range(N_img):
        img_n = (stim_imgs[j] - stim_imgs[j].min()) / (stim_imgs[j].max() - stim_imgs[j].min() + 1e-8)
        _show(axes[0, j], img_n, 'gray', 0, 1)
        axes[0, j].plot(ex, ey, color='lime', lw=0.8)
        axes[0, j].set_title(f'img {image_indices[j]}', fontsize=6, pad=2)
    _ylabel(axes[0, 0], 'stimolo')

    # vmax globale condiviso da tutte le LSTA (exp + model)
    global_lsta_vmax = max(max(exp_vmax), max(mod_vmax)) or 1

    # ── riga 1: LSTA sperimentale (proc) ─────────────────────────────────────
    for j in range(N_img):
        _show(axes[1, j], exp_proc[j], 'RdBu_r', -global_lsta_vmax, global_lsta_vmax)
        axes[1, j].plot(ex, ey, color='k', lw=0.8)
    _ylabel(axes[1, 0], f'LSTA exp\n(proc)\nr={lsta_mean_corr:.3f}')

    # ── riga 2: LSTA modello (proc) ───────────────────────────────────────────
    for j in range(N_img):
        _show(axes[2, j], mod_proc[j], 'RdBu_r', -global_lsta_vmax, global_lsta_vmax)
        axes[2, j].plot(ex, ey, color='k', lw=0.8)
    _ylabel(axes[2, 0], f'LSTA model\n(proc)\nr={lsta_mean_corr:.3f}')

    # ── righe operatori ───────────────────────────────────────────────────────
    for row_i, (op_key, (op_proc, op_vmax, op_mean_corr)) in enumerate(op_results.items()):
        row_idx   = 3 + row_i
        short_lbl = _OP_LABELS[op_key].replace(' ', '\n')
        for j in range(N_img):
            va = op_vmax[j]
            _show(axes[row_idx, j], op_proc[j], 'RdBu_r', -(va or 1), va or 1)
            axes[row_idx, j].plot(ex, ey, color='k', lw=0.8)
        _ylabel(axes[row_idx, 0], f'{short_lbl}\nr={op_mean_corr:.3f}')

    fig.suptitle(
        f'Neurone {neuron_idx}  (cell {cell_index})  |  Adj R²={adj_r2_val:.3f}',
        fontsize=9, y=1.01)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _run_operators(
    model,
    lsta_ref_path,
    images_np,
    device,
    adj_r2,
    neuron_subset,
    out_dir,
    do_ig,
    do_wig,
    n_steps          = 10,
    padding          = 4,
    lsta_expon       = 2,
    lsta_vmax_thresh = 1.0,
    lsta_put_to0     = 0.2,
):
    """
    Calcola e salva le figure degli operatori per un modello con neuron_subset neuroni.

    adj_r2 deve essere allineato con neuron_subset:
      adj_r2[local_n] corrisponde al neurone neuron_subset[local_n].
    """
    ref_data = _load_lsta_ref_for_operators(lsta_ref_path, images_np)
    op_paths = _precompute_op_paths(ref_data['ref_imgs'], n_steps, do_ig, do_wig)
    if not op_paths:
        return

    N_img = ref_data['ref_imgs'].shape[0]
    print(f'  Computing operators  '
          f'({len(op_paths)} op(s) × {len(neuron_subset)} neurons'
          f' × {N_img} imgs × {n_steps} steps)...')

    # Calcola LSTA model per il recap (gradiente ∂r/∂x valutato sull'immagine originale)
    print(f'  Computing model LSTA for recap ({N_img} imgs × {len(neuron_subset)} neurons)...')
    lsta_model_all = compute_model_lsta(
        model, images_np, ref_data['image_indices'], device,
        neuron_subset=neuron_subset)   # (n_sub, N_img, H, W)

    recap_out = Path(out_dir) / 'recap'
    ellipses  = ref_data['ellipses']
    scale     = ref_data['scale']

    for local_n, n in enumerate(neuron_subset):
        attr_dict = _run_operator_for_neuron(
            model            = model,
            n                = n,
            op_paths         = op_paths,
            ref_data         = ref_data,
            adj_r2_val       = float(adj_r2[local_n]),
            out_dir          = out_dir,
            device           = device,
            padding          = padding,
            lsta_expon       = lsta_expon,
            lsta_vmax_thresh = lsta_vmax_thresh,
            lsta_put_to0     = lsta_put_to0,
        )

        ellipse_scaled = np.stack([ellipses[n, 0, :] * scale,
                                   ellipses[n, 1, :] * scale], axis=0)
        plot_operator_recap(
            neuron_idx       = n,
            cell_index       = int(ref_data['cell_indices'][n]),
            adj_r2_val       = float(adj_r2[local_n]),
            stim_imgs        = ref_data['ref_imgs'],
            image_indices    = ref_data['image_indices'],
            lsta_exp_raw     = ref_data['lsta_exp'][n],
            lsta_model_maps  = lsta_model_all[local_n],
            attr_maps_dict   = attr_dict,
            ellipse          = ellipse_scaled,
            out_path         = str(recap_out / f'neuron_{n:02d}.png'),
            padding          = padding,
            lsta_expon       = lsta_expon,
            lsta_vmax_thresh = lsta_vmax_thresh,
            lsta_put_to0     = lsta_put_to0,
        )
        print(f'  neuron_{n:02d} operators + recap ✓', end='\r')

    print()
    for op_key in op_paths:
        print(f'  → operators/{op_key}/  saved')
    print(f'  → operators/recap/  saved')


# ─────────────────────────────────────────────────────────────────────────────
# NPZ serialiser
# ─────────────────────────────────────────────────────────────────────────────

def _save_eval_npz(out_path, y_pred, y_true_mean, test_rep, images_test,
                   metrics, centers_x_init, centers_y_init, sta_crops,
                   model, cfg, scale, target_n,
                   lsta_data, history, histories_per_neuron, model_label):
    """
    Serialises everything needed for offline metric replication and
    multi-model comparison into a single compressed .npz file.

    Arrays
    ──────
    Predictions / ground truth
      y_pred              float32  (n_imgs, n_cells)
      y_true_mean         float32  (n_imgs, n_cells)
      test_rep            float32  (n_imgs, n_rep, n_cells)
      images_test         float32  (n_imgs, H, W)

    Pre-computed metrics  — each float32 (n_cells,)
      pearson_r  r2  normalized_r2  adjusted_r2  mse  explained_variance

    RF centers
      centers_x_init / centers_y_init   float32  (n_cells,)  STA initialisation
      centers_x_trained / centers_y_trained  float32  (n_cells,)  after training
      rf_drift_px                        float32  (n_cells,)  Euclidean drift

    Readout masks (in crop space, shape (n_cells, CS, CS))
      readout_u_init      float32  — STA crop at init
      readout_u_trained   float32  — learned spatial mask

    LSTA  (present / non-empty only when LSTA was computed)
      has_lsta            int8     scalar 0/1
      lsta_model          float32  (n_sub, N_img, H, W)  autograd gradient
      lsta_exp_raw        float32  (n_sub, N_img, H_lsta, W_lsta)  original res
      lsta_corr           float32  (n_sub,)  Pearson r post-processed crops
      lsta_image_indices  int32    (N_img,)
      lsta_ellipses       float32  (n_sub, 2, 360)  in LSTA pixel coords
      lsta_cell_indices   int32    (n_sub,)
      lsta_neuron_subset  int32    (n_sub,)  local indices evaluated

    Training history
      has_history         int8     scalar
      train_loss          float32  (n_epochs,)
      val_loss            float32  (n_epochs,)
      phase1_end          int32    scalar  (0 = no two-phase split)

    Per-neuron retrain histories (mean_all retrained models only)
      has_retrain_history int8     scalar
      retrain_train_loss  float32  (n_cells, max_ep)  nan-padded
      retrain_val_loss    float32  (n_cells, max_ep)  nan-padded
      retrain_best_epoch  int32    (n_cells,)

    Metadata
      cfg_json            object   JSON string of the full cfg dict
      n_cells             int32    scalar
      target_neuron       int32    scalar  (-1 = full multi-neuron model)
      crop_size           int32    scalar
      model_label         object   human-readable identifier string
    """
    n_cells = model.n_cells
    CS      = cfg['CROP_SIZE']

    # RF centers
    cx_tr = model.crop.cx.detach().cpu().numpy() * scale
    cy_tr = model.crop.cy.detach().cpu().numpy() * scale
    drift = np.sqrt((cx_tr - centers_x_init) ** 2 +
                    (cy_tr - centers_y_init) ** 2)

    # readout masks
    u_trained = model.readout.u.detach().cpu().numpy().reshape(n_cells, CS, CS)
    u_init    = np.asarray(sta_crops).reshape(n_cells, CS, CS)

    # training history
    has_h = history is not None
    if has_h:
        if isinstance(history, (str, Path)):
            with open(history) as f:
                history = json.load(f)
        tl  = np.array(history['train_loss'], dtype=np.float32)
        vl  = np.array(history['val_loss'],   dtype=np.float32)
        ph1 = int(history.get('phase1_end', 0))
    else:
        tl  = np.empty(0, dtype=np.float32)
        vl  = np.empty(0, dtype=np.float32)
        ph1 = 0

    # per-neuron retrain histories
    has_rt = histories_per_neuron is not None
    if has_rt:
        loaded = []
        for h in histories_per_neuron:
            if isinstance(h, (str, Path)):
                with open(h) as f:
                    h = json.load(f)
            loaded.append(h)
        max_ep = max(len(h['train_loss']) for h in loaded)
        rt_tl = np.full((n_cells, max_ep), np.nan, dtype=np.float32)
        rt_vl = np.full((n_cells, max_ep), np.nan, dtype=np.float32)
        rt_be = np.zeros(n_cells, dtype=np.int32)
        for i, h in enumerate(loaded):
            ne = len(h['train_loss'])
            rt_tl[i, :ne] = h['train_loss']
            rt_vl[i, :ne] = h['val_loss']
            rt_be[i]      = int(h.get('best_epoch', 0))
    else:
        rt_tl = np.empty((0, 0), dtype=np.float32)
        rt_vl = np.empty((0, 0), dtype=np.float32)
        rt_be = np.empty(0, dtype=np.int32)

    # LSTA
    has_l = lsta_data is not None
    if has_l:
        lm  = lsta_data['lsta_model']
        le  = lsta_data['lsta_exp_raw']
        lc  = lsta_data['lsta_corr']
        li  = lsta_data['image_indices']
        lel = lsta_data['ellipses']
        lci = lsta_data['cell_indices']
        lns = lsta_data['neuron_subset']
    else:
        lm  = np.empty(0, dtype=np.float32)
        le  = np.empty(0, dtype=np.float32)
        lc  = np.zeros(n_cells, dtype=np.float32)
        li  = np.empty(0, dtype=np.int32)
        lel = np.empty(0, dtype=np.float32)
        lci = np.empty(0, dtype=np.int32)
        lns = np.empty(0, dtype=np.int32)

    np.savez_compressed(
        str(out_path),
        # predictions + ground truth
        y_pred              = y_pred.astype(np.float32),
        y_true_mean         = y_true_mean.astype(np.float32),
        test_rep            = test_rep.astype(np.float32),
        images_test         = images_test.astype(np.float32),
        # metrics
        pearson_r           = metrics['pearson_r'].astype(np.float32),
        r2                  = metrics['r2'].astype(np.float32),
        normalized_r2       = metrics['normalized_r2'].astype(np.float32),
        adjusted_r2         = metrics['adjusted_r2'].astype(np.float32),
        mse                 = metrics['mse'].astype(np.float32),
        explained_variance  = metrics['explained_variance'].astype(np.float32),
        # RF centers
        centers_x_init      = centers_x_init.astype(np.float32),
        centers_y_init      = centers_y_init.astype(np.float32),
        centers_x_trained   = cx_tr.astype(np.float32),
        centers_y_trained   = cy_tr.astype(np.float32),
        rf_drift_px         = drift.astype(np.float32),
        # readout masks
        readout_u_init      = u_init,
        readout_u_trained   = u_trained,
        # LSTA
        has_lsta            = np.int8(has_l),
        lsta_model          = lm,
        lsta_exp_raw        = le,
        lsta_corr           = lc,
        lsta_image_indices  = li,
        lsta_ellipses       = lel,
        lsta_cell_indices   = lci,
        lsta_neuron_subset  = lns,
        # training history
        has_history         = np.int8(has_h),
        train_loss          = tl,
        val_loss            = vl,
        phase1_end          = np.int32(ph1),
        # per-neuron retrain histories
        has_retrain_history = np.int8(has_rt),
        retrain_train_loss  = rt_tl,
        retrain_val_loss    = rt_vl,
        retrain_best_epoch  = rt_be,
        # metadata
        cfg_json            = np.array(json.dumps(cfg)),
        n_cells             = np.int32(n_cells),
        target_neuron       = np.int32(-1 if target_n is None else target_n),
        crop_size           = np.int32(CS),
        model_label         = np.array(str(model_label)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    state_dict,           # dict (già caricato) o path str/Path al .pt
    cfg:      dict,
    data_npz: str,
    sta_dir:  str,
    rf_path:  str,
    device:   torch.device,
    out_dir:  str,

    # ── flag selettivi ───────────────────────────────────────────────────────
    do_metrics:       bool = True,
    do_scatter:       bool = True,
    do_response_rank: bool = True,
    do_sta_centers:   bool = True,
    do_training:      bool = False,

    # ── LSTA ────────────────────────────────────────────────────────────────
    do_lsta:          bool  = False,
    lsta_ref_path:    str   = None,   # path a lsta_ref.npz
    lsta_padding:     int   = 8,
    lsta_expon:       float = 2,
    lsta_vmax_thresh: float = 1.0,
    lsta_put_to0:     float = 0.2,

    # ── training history ────────────────────────────────────────────────────
    # single-model history: dict o path a history.json (usato se do_training=True)
    history = None,

    # ── per-neuron training histories (mean_all retrained models) ─────────
    # list of 41 dicts or list of 41 paths to history_neuron_XX.json;
    # when provided, generates one training plot per neuron under
    # out_dir/training_per_neuron/
    histories_per_neuron = None,
    do_training_per_neuron: bool = False,

    # ── attribution operators ────────────────────────────────────────────────
    do_ig:  bool = False,   # Integrated Gradients
    do_wig: bool = False,   # Waypoint Integrated Gradients
    op_n_steps: int = 10,   # passi di interpolazione per ogni operatore

    # ── NPZ serialisation ───────────────────────────────────────────────────
    # Saves eval_data.npz in out_dir with all data needed for offline analysis
    # and multi-model comparison.
    do_save_npz: bool  = True,
    model_label: str   = '',   # human-readable tag stored in the NPZ metadata
):
    """
    Valuta un modello sul test set e genera i plot richiesti.

    state_dict : path a .pt (state_dict puro) oppure dict già caricato
    cfg        : dizionario di configurazione del run.
                 Se cfg contiene 'TARGET_NEURON' (int) e 'N_CELLS_OVERRIDE'=1
                 il modello è una versione single-neuron salvata da
                 retrain_readout.py; le metriche/plot riguardano solo quel
                 neurone.
    history    : history del training — dict o path a history.json
                 (usato solo se do_training=True)
    histories_per_neuron : lista di 41 history dict / path (mean_all retrained)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── single-neuron model? ──────────────────────────────────────────────────
    target_n = cfg.get('TARGET_NEURON', None)   # int or None

    # ── carica dati ───────────────────────────────────────────────────────────
    data        = np.load(data_npz)
    images_test = data['images_test'].squeeze(-1)            # (30, 108, 108)
    test_rep    = data['responses_test'].transpose(1, 0, 2)  # (30, n_rep, 41)
    y_true_mean = test_rep.mean(axis=1)                      # (30, 41)

    rf        = np.load(rf_path)
    centers_x = rf['centers_x']
    centers_y = rf['centers_y']

    # ── costruisce e carica il modello ────────────────────────────────────────
    all_sta_crops = load_sta_crops(sta_dir, centers_x, centers_y,
                                   cfg['CROP_SIZE'],
                                   max_pool=cfg.get('MAX_POOL', False))

    if target_n is not None:
        # Single-neuron model: build with only neuron target_n's parameters
        cx_m      = centers_x[target_n:target_n + 1]
        cy_m      = centers_y[target_n:target_n + 1]
        sta_crops = all_sta_crops[target_n:target_n + 1]
        # Slice test data to match
        test_rep_m    = test_rep[:, :, target_n:target_n + 1]
        y_true_mean_m = y_true_mean[:, target_n:target_n + 1]
    else:
        cx_m = centers_x; cy_m = centers_y
        sta_crops     = all_sta_crops
        test_rep_m    = test_rep
        y_true_mean_m = y_true_mean

    model = RetinalModel(cfg, cx_m, cy_m, sta_crops).to(device)

    if isinstance(state_dict, (str, Path)):
        state_dict = torch.load(state_dict, map_location=device,
                                weights_only=True)
    if isinstance(state_dict, dict) and 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    # strip _orig_mod. prefix added by torch.compile()
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # ── predizioni ────────────────────────────────────────────────────────────
    y_pred = get_predictions(model, images_test, device)  # (30, n_cells)

    # ── metriche ──────────────────────────────────────────────────────────────
    r       = pearson_per_neuron(y_pred, y_true_mean_m)
    r2      = r2_per_neuron(y_pred, y_true_mean_m)
    norm_r2 = normalized_r2_per_neuron(y_pred, test_rep_m)
    adj_r2  = adjusted_r2_per_neuron(y_pred, test_rep_m)
    mse     = mse_per_neuron(y_pred, y_true_mean_m)
    ev      = explained_variance_per_neuron(y_pred, test_rep_m)

    metrics = {
        'pearson_r':          r,
        'r2':                 r2,
        'normalized_r2':      norm_r2,
        'adjusted_r2':        adj_r2,
        'mse':                mse,
        'explained_variance': ev,
    }

    tag = (f'neuron_{target_n:02d}  ' if target_n is not None else '')
    print(f'\n{"="*55}')
    print(f'TEST SET  —  {tag}{out_dir.name}')
    print(f'{"="*55}')
    print(f'  Pearson r          mean={r.mean():.4f}  median={np.median(r):.4f}')
    print(f'  R²                 mean={r2.mean():.4f}  median={np.median(r2):.4f}')
    print(f'  Normalized R²      mean={norm_r2.mean():.4f}  median={np.median(norm_r2):.4f}')
    print(f'  Adjusted R²        mean={adj_r2.mean():.4f}  median={np.median(adj_r2):.4f}')
    print(f'  MSE                mean={mse.mean():.4f}')
    print(f'  Explained Variance mean={ev.mean():.4f}  median={np.median(ev):.4f}')
    print(f'{"="*55}')

    # ── save summary JSON ─────────────────────────────────────────────────────
    summary = {k: v.tolist() for k, v in metrics.items()}
    summary['cfg'] = cfg
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # offset used for scatter / response_rank titles when single-neuron model
    n_off = target_n if target_n is not None else 0

    # ── plots ─────────────────────────────────────────────────────────────────
    if do_metrics:
        plot_metrics(metrics, out_dir / 'metrics')
        print(f'  → metrics/  saved')

    if do_scatter:
        plot_scatter(y_pred, y_true_mean_m, adj_r2, out_dir / 'scatter',
                     neuron_offset=n_off)
        print(f'  → scatter/  saved')

    if do_response_rank:
        plot_response_rank(y_pred, y_true_mean_m, adj_r2,
                           out_dir / 'response_rank', neuron_offset=n_off)
        print(f'  → response_rank/  saved')

    if do_sta_centers:
        plot_sta_centers(model, cfg, cx_m, cy_m, adj_r2,
                         sta_dir, out_dir / 'sta_centers', sta_crops)
        print(f'  → sta_centers/  saved')

    # ── LSTA: compute once, reuse for plot and NPZ ────────────────────────────
    lsta_data = None
    _has_lsta_ref = lsta_ref_path is not None and Path(lsta_ref_path).exists()
    if (do_lsta or do_save_npz) and _has_lsta_ref:
        ns = ([target_n] if target_n is not None
              else list(range(model.n_cells)))
        lsta_data = _compute_lsta_arrays(
            model, lsta_ref_path, images_test, device,
            neuron_subset=ns, padding=lsta_padding,
            expon=lsta_expon, vmax_thresh=lsta_vmax_thresh, to0_frac=lsta_put_to0,
        )

    if do_lsta:
        if _has_lsta_ref:
            plot_lsta_model_comparison(
                model           = model,
                adj_r2          = adj_r2,
                lsta_ref_path   = lsta_ref_path,
                images_np       = images_test,
                out_dir         = out_dir / 'operators' / 'lsta_comparison',
                device          = device,
                neuron_subset   = lsta_data['neuron_subset'].tolist(),
                padding         = lsta_padding,
                expon_treat     = lsta_expon,
                vmax_thresh     = lsta_vmax_thresh,
                put_to0_frac    = lsta_put_to0,
                lsta_model_all  = lsta_data['lsta_model'],  # avoid recomputation
            )
            print(f'  → operators/lsta_comparison/  saved')
        else:
            print(f'  [do_lsta=True] lsta_ref.npz non trovato — skip')

    # ── attribution operators ─────────────────────────────────────────────────
    if do_ig or do_wig:
        if _has_lsta_ref:
            ns_ops = ([target_n] if target_n is not None
                      else list(range(model.n_cells)))
            _run_operators(
                model            = model,
                lsta_ref_path    = lsta_ref_path,
                images_np        = images_test,
                device           = device,
                adj_r2           = adj_r2,
                neuron_subset    = ns_ops,
                out_dir          = out_dir / 'operators',
                do_ig            = do_ig,
                do_wig           = do_wig,
                n_steps          = op_n_steps,
                padding          = lsta_padding,
                lsta_expon       = lsta_expon,
                lsta_vmax_thresh = lsta_vmax_thresh,
                lsta_put_to0     = lsta_put_to0,
            )
        else:
            print(f'  [do_operators] lsta_ref.npz non trovato — skip')

    if do_training:
        if history is not None:
            plot_training(history, out_dir / 'training')
            print(f'  → training/training_history.png  saved')
        else:
            print('  [do_training=True] history non fornita — skip')

    if do_training_per_neuron:
        if histories_per_neuron is not None:
            plot_training_per_neuron(histories_per_neuron,
                                     out_dir / 'training_per_neuron')
            print(f'  → training_per_neuron/  saved ({model.n_cells} plots)')
        else:
            print('  [do_training_per_neuron=True] histories_per_neuron non fornite — skip')

    # ── NPZ ───────────────────────────────────────────────────────────────────
    if do_save_npz:
        scale = 2 if cfg.get('MAX_POOL', False) else 1
        npz_path = out_dir / 'eval_data.npz'
        _save_eval_npz(
            out_path             = npz_path,
            y_pred               = y_pred,
            y_true_mean          = y_true_mean_m,
            test_rep             = test_rep_m,
            images_test          = images_test,
            metrics              = metrics,
            centers_x_init       = cx_m,
            centers_y_init       = cy_m,
            sta_crops            = sta_crops,
            model                = model,
            cfg                  = cfg,
            scale                = scale,
            target_n             = target_n,
            lsta_data            = lsta_data,
            history              = history,
            histories_per_neuron = histories_per_neuron,
            model_label          = model_label or str(out_dir),
        )
        print(f'  → eval_data.npz  saved')

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Best-per-neuron: sta_centers con cfg variabile per neurone
# ─────────────────────────────────────────────────────────────────────────────

def _plot_sta_centers_best_per_neuron(
    cx_trained, cy_trained,
    centers_x_init, centers_y_init,
    u_init_list, u_trained_list, cfg_list,
    adj_r2, sta_dir, out_dir,
):
    """
    Versione di plot_sta_centers per il modello best-per-neuron.
    Ogni neurone può avere un cfg diverso (CS e MAX_POOL diversi).
    u_init_list[n], u_trained_list[n] : array (CS_n, CS_n) o None se mancante.
    """
    out     = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sta_dir = Path(sta_dir)

    drifts = np.sqrt((cx_trained - centers_x_init)**2 +
                     (cy_trained - centers_y_init)**2)

    for n in range(N_CELLS):
        if cfg_list[n] is None:
            continue

        cfg_n   = cfg_list[n]
        CS_n    = cfg_n['CROP_SIZE']
        scale_n = 2 if cfg_n.get('MAX_POOL', False) else 1
        CS_orig = CS_n * scale_n

        cell_id = 21 + n
        sta     = np.load(sta_dir / f'cell_{cell_id:03d}_sta_z.npy')

        cx_i = centers_x_init[n]
        cy_i = centers_y_init[n]
        cx_t = cx_trained[n]
        cy_t = cy_trained[n]

        half = CS_orig // 2
        y0 = int(np.clip(round(cy_t) - half, 0, IMG_SIZE_ORIGINAL - CS_orig))
        x0 = int(np.clip(round(cx_t) - half, 0, IMG_SIZE_ORIGINAL - CS_orig))

        u_init    = u_init_list[n]
        u_trained = u_trained_list[n]

        vmax_sta     = _sym_vmax(sta)
        vmax_init    = _sym_vmax(u_init)
        vmax_trained = _sym_vmax(u_trained)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(sta, cmap='RdBu_r', vmin=-vmax_sta, vmax=vmax_sta,
                   interpolation='nearest', origin='upper')
        ax1.plot(cx_i, cy_i, 'o', color='limegreen', markersize=8,
                 markeredgecolor='black', markeredgewidth=0.8, label='Initial center')
        ax1.plot(cx_t, cy_t, '*', color='red', markersize=12,
                 markeredgecolor='black', markeredgewidth=0.5, label='Trained center')
        ax1.add_patch(Rectangle(
            (x0, y0), CS_orig, CS_orig,
            linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--',
            label=f'Crop {CS_orig}×{CS_orig}'))
        ax1.arrow(cx_i, cy_i, cx_t - cx_i, cy_t - cy_i,
                  color='yellow', width=0.3, head_width=1.5,
                  length_includes_head=True, alpha=0.8)
        ax1.set_title(
            f'Neuron {n:02d}  |  Adj R²={adj_r2[n]:.3f}  |  drift={drifts[n]:.2f} px')
        ax1.legend(fontsize=7, loc='upper right')
        ax1.axis('off')

        im2 = ax2.imshow(u_init, cmap='RdBu_r', vmin=-vmax_init, vmax=vmax_init,
                         interpolation='nearest', origin='upper')
        ax2.set_title(f'Readout u  —  init  (CS={CS_n}, raw)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        im3 = ax3.imshow(u_trained, cmap='RdBu_r', vmin=-vmax_trained, vmax=vmax_trained,
                         interpolation='nearest', origin='upper')
        ax3.set_title(f'Readout u  —  trained  (CS={CS_n}, raw)')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        plt.suptitle(f'Neuron {n:02d}  (cell_{cell_id:03d})  —  best-per-neuron model',
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(out / f'neuron_{n:02d}.png', dpi=150)
        plt.close()

    # drift overview
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(np.zeros((IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL)), cmap='gray',
              vmin=0, vmax=1, origin='upper')
    for n in range(N_CELLS):
        ax.annotate('', xy=(cx_trained[n], cy_trained[n]),
                    xytext=(centers_x_init[n], centers_y_init[n]),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2))
        ax.plot(centers_x_init[n], centers_y_init[n], 'o', color='limegreen', markersize=5)
        ax.plot(cx_trained[n],     cy_trained[n],     '*', color='red',       markersize=7)
    ax.set_xlim(0, IMG_SIZE_ORIGINAL)
    ax.set_ylim(IMG_SIZE_ORIGINAL, 0)
    ax.set_title('RF center drift — best-per-neuron  (green = initial, red = trained)')
    handles = [mpatches.Patch(color='limegreen', label='Initial centers'),
               mpatches.Patch(color='red',       label='Trained centers')]
    ax.legend(handles=handles)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    plt.tight_layout()
    plt.savefig(out / 'drift_overview.png', dpi=150)
    plt.close()

    print(f'  Mean drift: {drifts.mean():.2f} px  '
          f'max: {drifts.max():.2f} px  (neuron {drifts.argmax():02d})')


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate — best-per-neuron
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_best_per_neuron(
    grid_dir: str,
    data_npz: str,
    sta_dir:  str,
    rf_path:  str,
    device:   torch.device,
    out_dir:  str,
    do_metrics:       bool = True,
    do_scatter:       bool = True,
    do_response_rank: bool = True,
    do_sta_centers:   bool = True,
    do_lsta:          bool  = False,
    lsta_ref_path:    str   = None,
    lsta_padding:     int   = 8,
    lsta_expon:       float = 2,
    lsta_vmax_thresh: float = 1.0,
    lsta_put_to0:     float = 0.2,
    # ── attribution operators ────────────────────────────────────────────────
    do_ig:      bool = False,
    do_wig:     bool = False,
    op_n_steps: int  = 10,
):
    """
    Valutazione combinata best-per-neuron.

    Per ogni neurone n usa il modello ottimizzato specificamente per quel neurone:
      grid_dir/per_neuron/neuron_NN/avg_kK/model.pt

    Prende solo la colonna n della predizione di ciascun modello e assembla
    la matrice y_pred (n_images, N_CELLS). Poi produce gli stessi plot standard.

    Output: out_dir/  (tipicamente per_neuron/best_per_neuron/avg_kK/eval/)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── carica dati ───────────────────────────────────────────────────────────
    data        = np.load(data_npz)
    images_test = data['images_test'].squeeze(-1)
    test_rep    = data['responses_test'].transpose(1, 0, 2)
    y_true_mean = test_rep.mean(axis=1)

    rf        = np.load(rf_path)
    centers_x = rf['centers_x']
    centers_y = rf['centers_y']

    # ── assembla predizioni ───────────────────────────────────────────────────
    n_imgs = images_test.shape[0]
    y_pred = np.zeros((n_imgs, N_CELLS), dtype=np.float32)

    # per sta_centers: raccogli parametri per-neurone
    cx_trained_all = np.zeros(N_CELLS)
    cy_trained_all = np.zeros(N_CELLS)
    u_init_list    = [None] * N_CELLS
    u_trained_list = [None] * N_CELLS
    cfg_list       = [None] * N_CELLS

    print(f'\nBest-per-neuron  —  caricamento modelli...')
    for n in range(N_CELLS):
        model_dir   = Path(grid_dir) / 'per_neuron' / f'neuron_{n:02d}'
        model_path  = model_dir / 'model.pt'
        config_path = model_dir / 'config.json'

        if not model_path.exists():
            print(f'  [SKIP] neuron_{n:02d}: model.pt non trovato')
            continue

        with open(config_path) as f:
            cfg_n = json.load(f)['cfg']
        cfg_list[n] = cfg_n

        sta_crops_n = load_sta_crops(
            sta_dir, centers_x, centers_y,
            cfg_n['CROP_SIZE'], max_pool=cfg_n.get('MAX_POOL', False))

        model_n = RetinalModel(cfg_n, centers_x, centers_y, sta_crops_n).to(device)
        _sd = torch.load(model_path, map_location=device)
        _sd = {k.removeprefix('_orig_mod.'): v for k, v in _sd.items()}
        model_n.load_state_dict(_sd)
        model_n.eval()

        # predizione su tutte le immagini → prendi solo colonna n
        pred_n      = get_predictions(model_n, images_test, device)   # (n_imgs, N_CELLS)
        y_pred[:, n] = pred_n[:, n]

        # parametri per sta_centers
        scale_n = 2 if cfg_n.get('MAX_POOL', False) else 1
        CS_n    = cfg_n['CROP_SIZE']
        cx_trained_all[n] = model_n.crop.cx[n].detach().cpu().item() * scale_n
        cy_trained_all[n] = model_n.crop.cy[n].detach().cpu().item() * scale_n
        u_init_list[n]    = sta_crops_n[n]                                       # (CS_n, CS_n)
        u_trained_list[n] = (model_n.readout.u.detach().cpu().numpy()
                             .reshape(N_CELLS, CS_n, CS_n)[n])                   # (CS_n, CS_n)

        del model_n
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f'  neuron_{n:02d} ✓', end='\r')

    print(f'\n  Tutti i modelli caricati.')

    # ── metriche ──────────────────────────────────────────────────────────────
    r       = pearson_per_neuron(y_pred, y_true_mean)
    r2      = r2_per_neuron(y_pred, y_true_mean)
    norm_r2 = normalized_r2_per_neuron(y_pred, test_rep)
    adj_r2  = adjusted_r2_per_neuron(y_pred, test_rep)
    mse     = mse_per_neuron(y_pred, y_true_mean)
    ev      = explained_variance_per_neuron(y_pred, test_rep)

    metrics = {
        'pearson_r':          r,
        'r2':                 r2,
        'normalized_r2':      norm_r2,
        'adjusted_r2':        adj_r2,
        'mse':                mse,
        'explained_variance': ev,
    }

    print(f'\n{"="*55}')
    print(f'BEST PER NEURON')
    print(f'{"="*55}')
    print(f'  Pearson r          mean={r.mean():.4f}  median={np.median(r):.4f}')
    print(f'  R²                 mean={r2.mean():.4f}  median={np.median(r2):.4f}')
    print(f'  Normalized R²      mean={norm_r2.mean():.4f}  median={np.median(norm_r2):.4f}')
    print(f'  Adjusted R²        mean={adj_r2.mean():.4f}  median={np.median(adj_r2):.4f}')
    print(f'  MSE                mean={mse.mean():.4f}')
    print(f'  Explained Variance mean={ev.mean():.4f}  median={np.median(ev):.4f}')
    print(f'{"="*55}')

    import json as _json
    summary = {k: v.tolist() for k, v in metrics.items()}
    with open(out_dir / 'summary.json', 'w') as f:
        _json.dump(summary, f, indent=2)

    # ── plots ─────────────────────────────────────────────────────────────────
    if do_metrics:
        plot_metrics(metrics, out_dir / 'metrics')
        print(f'  → metrics/  saved')

    if do_scatter:
        plot_scatter(y_pred, y_true_mean, adj_r2, out_dir / 'scatter')
        print(f'  → scatter/  saved')

    if do_response_rank:
        plot_response_rank(y_pred, y_true_mean, adj_r2, out_dir / 'response_rank')
        print(f'  → response_rank/  saved')

    if do_sta_centers:
        _plot_sta_centers_best_per_neuron(
            cx_trained_all, cy_trained_all,
            centers_x, centers_y,
            u_init_list, u_trained_list, cfg_list,
            adj_r2, sta_dir, out_dir / 'sta_centers',
        )
        print(f'  → sta_centers/  saved')

    if do_lsta:
        if lsta_ref_path is not None and Path(lsta_ref_path).exists():
            lsta_out = out_dir / 'operators' / 'lsta_comparison'
            print(f'\n  Best-per-neuron LSTA — secondo passaggio (41 modelli)...')
            # Per ogni neurone ricarica il suo modello e calcola il gradiente
            # solo per quel neurone, poi genera la figura
            for n in range(N_CELLS):
                if cfg_list[n] is None:
                    continue
                cfg_n = cfg_list[n]
                model_path = (Path(grid_dir) / 'per_neuron'
                              / f'neuron_{n:02d}' / 'model.pt')
                sta_crops_n = load_sta_crops(
                    sta_dir, centers_x, centers_y,
                    cfg_n['CROP_SIZE'], max_pool=cfg_n.get('MAX_POOL', False))
                model_n = RetinalModel(cfg_n, centers_x, centers_y, sta_crops_n).to(device)
                _sd = torch.load(model_path, map_location=device)
                _sd = {k.removeprefix('_orig_mod.'): v for k, v in _sd.items()}
                model_n.load_state_dict(_sd)
                model_n.eval()

                plot_lsta_model_comparison(
                    model          = model_n,
                    adj_r2         = adj_r2,
                    lsta_ref_path  = lsta_ref_path,
                    images_np      = images_test,
                    out_dir        = lsta_out,
                    device         = device,
                    neuron_subset  = [n],
                    padding        = lsta_padding,
                    expon_treat    = lsta_expon,
                    vmax_thresh    = lsta_vmax_thresh,
                    put_to0_frac   = lsta_put_to0,
                )

                del model_n
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            print(f'  → operators/lsta_comparison/  saved')
        else:
            print(f'  [do_lsta=True] lsta_ref.npz non trovato — skip')

    # ── attribution operators ─────────────────────────────────────────────────
    if do_ig or do_wig:
        _has_lsta_bpn = lsta_ref_path is not None and Path(lsta_ref_path).exists()
        if _has_lsta_bpn:
            ref_data = _load_lsta_ref_for_operators(lsta_ref_path, images_test)
            op_paths = _precompute_op_paths(
                ref_data['ref_imgs'], op_n_steps, do_ig, do_wig)
            if op_paths:
                ops_out = out_dir / 'operators'
                N_img   = ref_data['ref_imgs'].shape[0]
                print(f'\n  Best-per-neuron operators — terzo passaggio'
                      f' ({len(op_paths)} op(s) × {N_CELLS} neuroni'
                      f' × {N_img} imgs × {op_n_steps} steps)...')
                recap_out_bpn = ops_out / 'recap'
                ellipses_bpn  = ref_data['ellipses']
                scale_bpn     = ref_data['scale']

                for n in range(N_CELLS):
                    if cfg_list[n] is None:
                        continue
                    cfg_n      = cfg_list[n]
                    model_path = (Path(grid_dir) / 'per_neuron'
                                  / f'neuron_{n:02d}' / 'model.pt')
                    sta_crops_n = load_sta_crops(
                        sta_dir, centers_x, centers_y,
                        cfg_n['CROP_SIZE'], max_pool=cfg_n.get('MAX_POOL', False))
                    model_n = RetinalModel(
                        cfg_n, centers_x, centers_y, sta_crops_n).to(device)
                    _sd = torch.load(model_path, map_location=device)
                    _sd = {k.removeprefix('_orig_mod.'): v for k, v in _sd.items()}
                    model_n.load_state_dict(_sd)
                    model_n.eval()

                    attr_dict_n = _run_operator_for_neuron(
                        model            = model_n,
                        n                = n,
                        op_paths         = op_paths,
                        ref_data         = ref_data,
                        adj_r2_val       = float(adj_r2[n]),
                        out_dir          = ops_out,
                        device           = device,
                        padding          = lsta_padding,
                        lsta_expon       = lsta_expon,
                        lsta_vmax_thresh = lsta_vmax_thresh,
                        lsta_put_to0     = lsta_put_to0,
                    )

                    # Recap: LSTA model calcolata con lo stesso modello per-neurone
                    lsta_model_n = compute_model_lsta(
                        model_n, images_test, ref_data['image_indices'],
                        device, neuron_subset=[n])   # (1, N_img, H, W)
                    ellipse_n = np.stack([ellipses_bpn[n, 0, :] * scale_bpn,
                                          ellipses_bpn[n, 1, :] * scale_bpn], axis=0)
                    plot_operator_recap(
                        neuron_idx       = n,
                        cell_index       = int(ref_data['cell_indices'][n]),
                        adj_r2_val       = float(adj_r2[n]),
                        stim_imgs        = ref_data['ref_imgs'],
                        image_indices    = ref_data['image_indices'],
                        lsta_exp_raw     = ref_data['lsta_exp'][n],
                        lsta_model_maps  = lsta_model_n[0],       # (N_img, H, W)
                        attr_maps_dict   = attr_dict_n,
                        ellipse          = ellipse_n,
                        out_path         = str(recap_out_bpn / f'neuron_{n:02d}.png'),
                        padding          = lsta_padding,
                        lsta_expon       = lsta_expon,
                        lsta_vmax_thresh = lsta_vmax_thresh,
                        lsta_put_to0     = lsta_put_to0,
                    )

                    del model_n
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    print(f'  neuron_{n:02d} operators + recap ✓', end='\r')

                print()
                for op_key in op_paths:
                    print(f'  → operators/{op_key}/  saved')
                print(f'  → operators/recap/  saved')
        else:
            print(f'  [do_operators] lsta_ref.npz non trovato — skip')

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    default='best_model.pt',
                        help='Checkpoint da train.py (contiene cfg)')
    parser.add_argument('--out',     default='results/eval')
    parser.add_argument('--no-metrics',       dest='metrics',       action='store_false')
    parser.add_argument('--no-scatter',       dest='scatter',       action='store_false')
    parser.add_argument('--no-response-rank', dest='response_rank', action='store_false')
    parser.add_argument('--no-sta-centers',   dest='sta_centers',   action='store_false')
    parser.add_argument('--training',         dest='training',      action='store_true',
                        help='Plotta la curva di training (richiede history nel checkpoint)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(args.ckpt, map_location=device)
    cfg    = ckpt['cfg']
    hist   = ckpt.get('history', None)

    evaluate_model(
        state_dict       = ckpt,
        cfg              = cfg,
        data_npz         = 'PNAS_paper_sorted_data.npz',
        sta_dir          = 'STA',
        rf_path          = 'ellipse_centers_exp2.npz',
        device           = device,
        out_dir          = args.out,
        do_metrics       = args.metrics,
        do_scatter       = args.scatter,
        do_response_rank = args.response_rank,
        do_sta_centers   = args.sta_centers,
        do_training      = args.training,
        history          = hist,
    )
