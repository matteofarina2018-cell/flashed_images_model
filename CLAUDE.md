# CLAUDE.md

## Project overview

PyTorch model predicting spike responses of 41 retinal ganglion cells (RGCs) to 108×108 grayscale natural images, based on Goldin et al. PNAS 2023. Pipeline: CNN tokenizer → differentiable per-neuron circular crop → shared transformer (RoPE 2D) → per-neuron factorized readout. Loss: Poisson NLL. Primary metric: adjusted R² (Goldin eq. 5). Secondary metric: LSTA correlation (gradient-based linearized STA vs experimental).

---

## Environment

```bash
conda activate retina   # always required — base env has no torch
cd ~/modello_TL3
```

**Runtime**: Python 3.11, PyTorch 2.12.0.dev (nightly, CUDA 12.8), Optuna 4.8.0, NumPy 2.4.2  
**Hardware**: NVIDIA GB10 Blackwell (121 GB VRAM), 121 GB RAM, 20 CPU cores, 3.7 TB NVMe  
VRAM is not a bottleneck. Model + batch 32 uses ~24 GB. Batch size and model size are unconstrained.  
**Training dtype**: `float32` — bf16 kernels are **3× slower** than fp32 on GB10/PyTorch 2.12 nightly (sm_121 not yet fully optimized). When a future nightly fixes bf16, switching back may yield ~2× speedup — change `dtype=torch.float32` → `torch.bfloat16` in `train.py` and `bayes_search.py` and re-benchmark.

---

## Commands

```bash
# Single training run with DEFAULT_CFG → best_model.pt
python train.py

# Bayesian hyperparameter search — Optuna TPE, auto-resumes from optuna.db
python bayes_search.py

# Evaluate / plot saved checkpoints (configure flags at top of file first)
python run_evaluate.py

# Module self-tests (each file has a __main__ block)
python data.py
python model/tokenizer.py
python model/transformer.py
python model/readout.py
python model/positional_encoding.py
```

---

## Architecture

```
Input:  [B, 1, 108, 108]  float32 → bfloat16 inside autocast

1. CNNTokenizer          → [B, CNN_DIM, H', W']
   CNN_STRIDE=1 → H'=108; CNN_STRIDE=2 → H'=54  (H' = ceil(108 / CNN_STRIDE))
   CNN_LAYERS stacked blocks: Conv2d(same-pad) + BatchNorm2d + GELU [+ Dropout2d]
   Stride applied to first layer only; all subsequent layers use stride=1.

2. NeuronCircle          → [B, N, T_circle, CNN_DIM]
   Bilinear grid_sample on [B*N, CNN_DIM, H', W'] — 41 circles sampled in one call.
   crop_half = max(1, ceil(max(NEURON_RADII) / CNN_STRIDE))  — derived, never set manually
   T_circle = #pixels ≤ r_max within (2*crop_half+1)² grid  — shared across all 41 neurons
   Typical T_circle: r_max=25, stride=1 → crop_half=25 → T_circle≈1963
                     r_max=25, stride=2 → crop_half=13 → T_circle≈489
   cx [N], cy [N]: trainable (initialized from ellipse_centers_exp2.npz / stride)
   r [N]: FIXED buffers (from NEURON_RADII); pad_mask [N, T_circle] precomputed at init
   Returns: tokens, rel_row [T_circle], rel_col [T_circle], pad_mask [N, T_circle]

3. TransformerEncoder    → [B*N, T_circle, EMB_DIM]
   Neurons folded into batch: effective batch = B×N = 32×41 = 1312 at BATCH_SIZE=32.
   Weights SHARED across all 41 neurons by design (not per-neuron).
   EMB_DIM=None → no projection, CNN_DIM flows through (effective_emb_dim = CNN_DIM)
   EMB_DIM≠None → linear projection CNN_DIM→EMB_DIM before transformer blocks
   Pre-norm: LN → MHSA(RoPE 2D) → residual → LN → MLP(GELU) → residual
   RoPE encodes (Δrow, Δcol) from each neuron's center as float offsets — NOT integer lookup.
   This makes attention translation-invariant: two neurons with the same RF shape
   but different center positions produce identical attention patterns.
   F.scaled_dot_product_attention with attn_mask = ~pad_mask (True = valid token).

4. FactorizedReadout     → [B, N]
   Per-neuron: spatial = softmax(u masked by pad_mask) · x  → [D]
               output  = ELU(spatial · v + bias) + 1        → always > 0
   u [N, T_circle]: initialized from cropped STA (not zeros — changes early training dynamics)
   v [N, EMB_DIM], bias [N]: initialized to small random / zero
   Output guaranteed > 0 — required by Poisson NLL.
```

---

## Key constraints

1. **RoPE divisibility** (hard crash if violated):
   `effective_emb_dim % NUM_HEADS == 0` AND `(effective_emb_dim // NUM_HEADS) % 4 == 0`
   where `effective_emb_dim = EMB_DIM if EMB_DIM is not None else CNN_DIM`
   Violations raise `TrialPruned` in `_validate_config()` before any training starts.

2. **CNN_KERNEL must be odd**: enforced by `assert` in `CNNTokenizer.__init__` (same-padding).

3. **Readout output > 0**: `ELU(x)+1` guarantees this. Never change the readout activation
   without also changing the Poisson loss — the loss calls `log(y_pred + 1e-8)` and requires
   y_pred > 0 to be meaningful.

4. **crop_half derived, never manual**: computed in both `train.py:build_model` and
   `bayes_search.py:_train_config`. If NEURON_RADII changes, crop_half updates automatically.

5. **bfloat16 AMP is load-bearing**: wraps forward + loss in both train and val loops.
   Do not remove autocast — it provides the primary speed benefit on GB10 Blackwell.

6. **STA file naming**: `STA/cell_021_sta_z.npy` … `cell_061_sta_z.npy`.
   Neuron index `i` → `cell_{21+i:03d}_sta_z.npy`. Neuron 0 = cell_021.

7. **pad_mask polarity**: `True = token OUTSIDE the circle` (PyTorch key_padding_mask convention).
   Inverted to `attn_mask = ~pad_mask` (True = valid) inside `MultiHeadSelfAttention`.
   Confusion here causes silent mismasking — the softmax in the readout also uses this convention.

8. **`RF_RANGE = None` in config.py**: defined but imported nowhere and used nowhere.
   Legacy stub — do not reference or rely on it.

9. **`count_params_per_block()` undercounts**: the readout's `u [N, T_circle]` is not included
   in the budget. At r_max=25, stride=1, N=41: T_circle≈1963 → ~80k missing params.
   The MAX_PARAMS=200M threshold is generous enough that this never causes incorrect pruning.

---

## Data

| Array | Raw npz shape | Notes |
|---|---|---|
| `images_train` | (2910, 108, 108, 1) | `.squeeze(-1)` → (2910, 108, 108) |
| `images_val` | (250, 108, 108, 1) | `.squeeze(-1)` → (250, 108, 108) |
| `images_test` | (30, 108, 108, 1) | `.squeeze(-1)` → (30, 108, 108) |
| `responses_train` | (2910, 41) | direct |
| `responses_val` | (250, 41) | direct |
| `responses_test` | **(n_reps, 30, 41)** | two different usages — see below |

**responses_test is used differently in two places:**
- `data.py` (used by `train.py`): `.mean(axis=0)` → (30, 41). Repetitions averaged. Adj R² cannot be computed from this path — no repetition structure.
- `bayes_search.py`: `.transpose(1, 0, 2)` → (30, n_reps, 41). Full repetitions kept for even/odd split in adj_r2 computation.

**Dataset augmentation** (`aug_factor`, `img_noise_sigma`, `poisson_resample` in `data.py`):
- Original indices `[0, N_orig)` always served clean (no noise, no resample).
- Augmented copies `[N_orig, aug_factor*N_orig)` get Gaussian image noise and/or Poisson resample.
- All three are disabled in `BAYES_FIXED` (defaults: aug_factor=1, sigma=0, resample=False).

---

## Performance bottlenecks (training time)

Each bayes_search trial is dominated by:

| Phase | Cost | Detail |
|---|---|---|
| Training loop | **2.7s/epoch** (fp32) | dominant cost — ~50 effective epochs ≈ 135s; see config dependence below |
| LSTA computation | ~0.8 s | 328 backward passes, negligible |
| Test prediction | <1 s | 30 images |
| Adj R² | <1 s | numpy only |

**Config dependence (fp32, clean GPU, BATCH_SIZE=32):**

| Config | T_circle | Params | ms/step | Epoch |
|---|---|---|---|---|
| CNN_DIM=16, stride=2, r=10 | 81 | 14k | 30 ms | 2.7s |
| CNN_DIM=8, stride=2, r=4 | 13 | 2k | 14 ms | 1.3s |
| CNN_DIM=64, stride=2, r=25 | 489 | 170k | 682 ms | 62s |

**At 50 effective epochs**: small configs ~2min/trial, large configs ~52min/trial.  
LSTA saves <0.1% of trial time — not worth disabling.

**SDPA mask**: `MultiHeadSelfAttention` now uses a **float additive mask** (`0=valid, -inf=masked`)
instead of a bool mask. Bool mask in bf16 caused a 20× SDPA slowdown on this GPU (8.5ms vs 0.4ms/call).
With fp32 the mask type doesn't affect performance, but the fix is future-proof for when bf16 returns.

**NeuronCircle memory traffic**: `x.expand(-1, N, ...)` then grid_sample on [B*N, C, H', W'].
At B=32, N=41, CNN_DIM=64, H'=108: tensor is ~1.1 GB per forward pass. Stride=2 cuts H'²
by 4x, reducing this to ~275 MB — a significant speed reason to prefer CNN_STRIDE=2.

**DataLoader workers**: `_NUM_WORKERS = min(4, os.cpu_count()) = 4` with 20 available cores.
Increasing to 8–12 in `data.py` would reduce CPU data-prep overhead with no side effects.

---

## Metrics

**Adjusted R²** — Goldin et al. PNAS 2023, Eq. 5. Computed on 30 test images × n_reps repetitions:
```python
r_even = responses_test[:, 0::2, i].mean(axis=1)   # even-rep average
r_odd  = responses_test[:, 1::2, i].mean(axis=1)   # odd-rep average
c_eo   = corr(r_even, r_odd)                         # noise ceiling / reliability
c_pe   = corr(y_pred[:, i], r_even)
c_po   = corr(y_pred[:, i], r_odd)
adj_r2[i] = max( ((c_pe + c_po) / 2)² / c_eo, 0 )  if c_eo > 0 else 0
```
Implemented identically in `evaluate.py:adjusted_r2_per_neuron` and `bayes_search.py:_adjusted_r2_per_neuron`.

**LSTA correlation**: Pearson r between `∂output_n/∂image` (autograd) and experimental LSTA,
evaluated on each neuron's RF ellipse crop. Averaged over 8 reference images from `lsta_ref.npz`.

---

## Bayesian search configuration (`model/config.py`)

**Fixed** (`BAYES_FIXED`): `CNN_LAYERS=1, NUM_BLOCKS=1, DROPOUT=0.1, BATCH_SIZE=32, LR=1e-3, WEIGHT_DECAY=0, MAX_EPOCHS=200, EARLY_STOP=30`

**Searched categorically** (`BAYES_SPACE`):
- `CNN_DIM ∈ {8, 16, 32, 64}`
- `CNN_KERNEL ∈ {3, 5, 7}`
- `CNN_STRIDE ∈ {1, 2}`
- `EMB_DIM ∈ {None, 32, 64, 128}`
- `NUM_HEADS ∈ {1, 2, 4, 8}`
- `MLP_DIM ∈ {32, 64, 128, 256}`

**Searched as integers**: 41 radii `r_00…r_40 ∈ [4, 25]` px (in 108×108 space)

Optuna objective: `mean_all` (mean adj R² across 41 neurons). Secondary best-trackers (per-neuron and per-LSTA) run outside Optuna, updated at every improvement, written immediately to disk.

Auto-resume: `create_study(..., load_if_exists=True)` + `_load_existing_csv()` reloads prior bests. Re-running `bayes_search.py` always continues from the last completed trial.

---

## Workflow

```bash
# Start / resume overnight search
conda activate retina
python bayes_search.py   # ctrl+C safe — resumes from optuna.db

# Check results next morning
# Sort all_results.csv by mean_all column — best config is in bayes_search_results/mean_all/
cat bayes_search_results/all_results.csv | python -c "
import csv,sys; rows=list(csv.DictReader(sys.stdin))
rows.sort(key=lambda r: float(r['mean_all']), reverse=True)
for r in rows[:5]: print(r['trial'], r['mean_all'], r['CNN_DIM'], r['CNN_STRIDE'], r['EMB_DIM'])
"

# Evaluate / generate plots for best model
# Edit run_evaluate.py: set RUN_MEAN_ALL=True, DO_LSTA=True, etc.
# IMPORTANT: also change GRID_DIR = 'bayes_search_results'  (file defaults to 'grid_search_results')
python run_evaluate.py
```

---

## Known issues & solutions

**bf16 is 3× SLOWER than fp32 on GB10/PyTorch 2.12 nightly**  
→ `train.py` and `bayes_search.py` use `dtype=torch.float32` in autocast. Measured: 90.6ms/step (bf16) vs 30.0ms/step (fp32). Reverts to bf16 when a future nightly optimizes sm_121 kernels. See `training_optimization_report.md`.

**SDPA bool mask causes 20× slowdown in bf16** (8.5ms vs 0.4ms/call)  
→ `model/transformer.py` now uses a float additive mask (0=valid, -inf=masked) instead of bool. Neutral in fp32, future-proof for bf16.

**`run_evaluate.py` hardcodes wrong output dir**: `GRID_DIR = 'grid_search_results'` but `bayes_search.py` writes to `bayes_search_results/`. Always update `GRID_DIR` before running evaluation on bayes results.

**Adj R² can be 0 for neurons with c_eo ≤ 0**: if a neuron's responses are anti-correlated between even/odd repetitions (very noisy neuron), the metric is set to 0 regardless of model quality. This is correct per the paper formula, not a bug.

**Trial 9 absent from CSV**: pruned trials (RoPE violation or param overflow) are not written to CSV. Gaps in trial numbers are normal.

---

## Off-limits

- **`model/losses.py`**: Poisson NLL is theoretically grounded for spike count data. Do not change.
- **`autocast` dtype**: currently `float32` — do NOT change to `bfloat16` until verified faster (bf16 is 3× slower on GB10/PyTorch 2.12 nightly). Change `float16` is also off-limits.
- **Radii as fixed buffers**: `r` is not a learnable parameter by design. The search explores radii between trials; within a trial they are frozen.
- **Adj R² formula in `bayes_search.py`**: implements Goldin eq. 5 exactly. Do not simplify or replace with standard R².
- **Data files** (`PNAS_paper_sorted_data.npz`, `STA/`, `ellipse_centers_exp2.npz`, `lsta_ref.npz`): read-only experimental data, never modify.
- **Transformer weight sharing**: weights are shared across neurons deliberately (parameter efficiency). Do not add per-neuron transformer weights without understanding the 41× parameter increase.

---

## Output layout

```
train.py:
  best_model.pt                   {epoch, model_state, val_loss, cfg}

bayes_search.py  → bayes_search_results/
  optuna.db                       Optuna TPE state (SQLite, auto-resume)
  all_results.csv                 one row/trial: hyperparams + val_loss + all scores
  mean_all/
    model.pt  config.json  history.json
  per_neuron/neuron_00/ … neuron_40/
    model.pt  config.json  history.json
  lsta_corr/mean_all/
  lsta_corr/per_neuron/neuron_00/ … neuron_40/

config.json structure:
  {"score": float, "cfg": {...}, "params": {cnn, neuron_circle, transformer, readout, total}}

run_evaluate.py  → <model_dir>/eval/
  metrics/  scatter/  response_rank/  sta_centers/
  training/  operators/lsta_comparison/
```
