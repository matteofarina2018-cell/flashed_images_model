# CLAUDE.md

## Project overview

PyTorch model predicting spike responses of 41 retinal ganglion cells (RGCs) to 108×108 grayscale natural images, based on Goldin et al. PNAS 2023. Pipeline: CNN tokenizer → differentiable per-neuron circular crop → shared transformer (RoPE 2D) → per-neuron factorized readout. Loss: Poisson NLL. Primary metric: adjusted R² (Goldin eq. 5). Secondary metric: LSTA correlation (gradient-based linearized STA vs experimental).

---

## Infrastructure (3-machine cluster)

The bayes_search runs distributed across three NVIDIA DGX Spark units sharing a single Optuna study via a PostgreSQL database.

### Machines

| Alias / MCP name | Hostname | IP | Role |
|---|---|---|---|
| **spark-09ab** (hub) | `spark-09ab` | `172.17.12.168` | hosts PostgreSQL server + repo + workers |
| `spark-fd4b` | `spark-fd4b` | `172.17.12.179` | worker only (DB via SSH tunnel) |
| `dgx_3` (MCP alias) | `spark-09bd` | `172.17.12.162` | worker only (DB via SSH tunnel) |

> The MCP alias `dgx_3` is **misleading**: the underlying machine is a DGX Spark named `spark-09bd`, not a DGX-3.

### Shared PostgreSQL database (Optuna storage)

- **Server**: native PostgreSQL **16.13** on `spark-09ab`, listening on `127.0.0.1:5432`
- **Connection URL** (used by `bayes_search.py:541`):
  ```
  postgresql://optuna_user:optuna_pass@localhost:5432/optuna_db
  ```
- **Database**: `optuna_db`
- **User / password**: `optuna_user` / `optuna_pass`
- **Port**: 5432

### How the two worker nodes reach the database

Each remote node opens a persistent SSH local-port-forward to spark-09ab and connects through it. From the worker's point of view, the URL `localhost:5432` resolves to the tunnel endpoint, which forwards to PostgreSQL on spark-09ab.

```bash
# Tunnel command running on spark-fd4b and spark-09bd (dgx_3):
ssh -L 5432:localhost:5432 matteo@172.17.12.168 -N -o ServerAliveInterval=60
```

Lives inside the `tunnel` tmux session on each worker node. **If this tmux dies, all bayes_search workers on that node lose the DB** and their currently running trial becomes a zombie in `RUNNING` state in PostgreSQL (Optuna has no heartbeat timeout).

### Verifying connectivity (any node)

```bash
PGPASSWORD=optuna_pass psql -U optuna_user -d optuna_db -h localhost \
  -c "SELECT inet_server_addr(), current_database(), current_user;"
```

### Shared filesystem (NFS)

`bayes_search_results/` is now a **single shared directory** exported via NFS from the hub. All `model.pt`/`config.json`/`history.json` checkpoints, plus `all_results.csv`, live in one canonical place. Workers across nodes write directly to it — no more per-node copies that diverge.

- **Server**: `nfs-kernel-server` on spark-09ab, listening on `:2049` and `:111`
- **Export** (`/etc/exports` on hub):
  ```
  /home/matteo/flashed_images_model/bayes_search_results 172.17.12.0/24(rw,sync,no_subtree_check,all_squash,anonuid=1002,anongid=1002)
  ```
  `all_squash` + `anonuid=1002/anongid=1002` maps every client UID to hub's `matteo` (UID 1002). This is necessary because the three nodes have **different `matteo` UIDs** (hub 1002, spark-09bd 1005, spark-fd4b 1001). On-disk ownership stays consistent (matteo:matteo on the hub); on the clients `ls` may show numeric UID or a coincidentally-mapped local user (e.g. spark-09bd's UID 1002 is `simoneazeglio`) — **cosmetic only, access works correctly**.

- **Client mount** (`/etc/fstab` on each worker that's NFS-attached):
  ```
  172.17.12.168:/home/matteo/flashed_images_model/bayes_search_results /home/matteo/flashed_images_model/bayes_search_results nfs defaults,_netdev 0 0
  ```
  `_netdev` ensures the mount waits for network availability at boot.

- **Mount status (as of 2026-05-12)**:
  - ✓ spark-09ab: serving (local path = export source)
  - ✓ spark-09bd (`dgx_3`): NFS-mounted, fstab persisted
  - ✓ spark-fd4b: NFS-mounted, fstab persisted

### Pre-NFS local backups (do not delete until NFS is proven stable)

The migration from per-node local dirs to shared NFS kept full backups on each node:
- spark-09ab: `bayes_search_results.bak_pre_nfs_20260511_153820/` (hub's pre-merge copy)
- spark-09bd: `bayes_search_results.pre_nfs_bak_20260511_154623/`
- spark-fd4b: `bayes_search_results.pre_nfs_bak_20260512_102329/` (the dominant source — 68/84 winning criteria came from this copy)

Hub also retains `.merge_staging/spark-09bd/` and `.merge_staging/spark-fd4b/` (the rsync snapshots used to compute the merge — ~85 MB total). Safe to delete once the new shared dir is validated.

---

## Git repository

Single GitHub remote, identical on all three machines.

- **Remote URL**: `https://github.com/matteofarina2018-cell/flashed_images_model.git`
- **Default branch**: `main`
- **Project path on every machine**: `/home/matteo/flashed_images_model`

### Workflow when modifying code

Code edits should happen on the hub (spark-09ab), then be propagated to the two worker nodes before they pick up the change. Workers do **not** auto-pull.

```bash
# 1. On the machine where you edited (typically spark-09ab):
cd /home/matteo/flashed_images_model
git add -p && git commit -m "..."
git push origin main

# 2. On each worker node — pull the new commit:
ssh matteo@172.17.12.179 'cd /home/matteo/flashed_images_model && git pull --ff-only'   # spark-fd4b
ssh matteo@172.17.12.162 'cd /home/matteo/flashed_images_model && git pull --ff-only'   # dgx_3 / spark-09bd

# 3. Restart the bayes_search workers on each node so they load the new code.
#    Running workers continue with the old code until restarted.
```

Worker processes on each node are launched inside the `training` tmux session as:
```
python bayes_search.py --out_dir ./bayes_search_results --n_trials 300
```
(multiple processes per node — all share the same Optuna study via PostgreSQL).

---

## Environment

```bash
conda activate retina   # always required — base env has no torch
cd /home/matteo/flashed_images_model
```

**Runtime**: Python 3.11, PyTorch 2.12.0.dev (nightly, CUDA 12.8), Optuna 4.8.0, NumPy 2.4.2
**Hardware (per node)**: NVIDIA GB10 Blackwell (121 GB VRAM), 121 GB RAM, 20 CPU cores, 3.7 TB NVMe
VRAM is not a bottleneck. Model + batch 32 uses ~24 GB. Batch size and model size are unconstrained.
**Training dtype**: `float32` — bf16 kernels are **3× slower** than fp32 on GB10/PyTorch 2.12 nightly (sm_121 not yet fully optimized). When a future nightly fixes bf16, switching back may yield ~2× speedup — change `dtype=torch.float32` → `torch.bfloat16` in `train.py` and `bayes_search.py` and re-benchmark.

---

## Commands

```bash
# Single training run with DEFAULT_CFG → best_model.pt
python train.py

# Bayesian hyperparameter search — Optuna TPE, auto-resumes from shared PG study
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

## Bayesian search configuration (`model/config.py` + `bayes_search.py`)

**Study identity (PostgreSQL)**:
- `study_name = 'retinal_radii'`
- `direction  = 'maximize'`
- `sampler    = TPESampler(seed, multivariate=True, n_startup_trials=50, gamma=0.4)`
- Objective: `mean_all` (mean adj R² across 41 neurons)

**Fixed** (`BAYES_FIXED`): `CNN_LAYERS=1, NUM_BLOCKS=1, DROPOUT=0.1, BATCH_SIZE=32, LR=1e-3, WEIGHT_DECAY=0, MAX_EPOCHS=200, EARLY_STOP=30`

**Searched categorically** (`BAYES_SPACE`):
- `CNN_DIM ∈ {8, 16, 32, 64}`
- `CNN_KERNEL ∈ {3, 5, 7}`
- `CNN_STRIDE ∈ {1, 2}`
- `EMB_DIM ∈ {None, 32, 64, 128}`
- `NUM_HEADS ∈ {1, 2, 4, 8}`
- `MLP_DIM ∈ {32, 64, 128, 256}`

**Searched as integers**: 41 radii `r_00…r_40 ∈ [4, 25]` px (in 108×108 space, `RADII_RANGE` in `config.py`)

**Trial budget**: `N_TRIALS_DEFAULT = 200` in `model/config.py`, but workers are currently invoked with `--n_trials 300`. The argument is a **per-worker budget** — workers stop when the global `study.trials` count reaches it, so multiple workers sharing the study finish collectively at the first target reached.

Secondary best-trackers (per-neuron and per-LSTA) run outside Optuna, updated at every improvement, written immediately to disk.

Auto-resume: `create_study(..., load_if_exists=True)` + `_load_existing_csv()` reloads prior bests. Re-running `bayes_search.py` always continues from the last completed trial.

---

## Current bayes_search status

Snapshot from PostgreSQL (`spark-09ab:5432/optuna_db`, study `retinal_radii`) after NFS migration cleanup:

| State | Count |
|---|---|
| COMPLETE | 44 |
| PRUNED | 4 |
| RUNNING | 0 |
| FAIL | 0 |

- **Best `mean_all`**: **0.7960** (held by a trial originally completed on spark-fd4b)
- **All workers stopped** during the NFS migration. To restart, on each NFS-attached node (hub + dgx_3 + fd4b):
  ```bash
  conda activate retina && cd /home/matteo/flashed_images_model
  python bayes_search.py --out_dir ./bayes_search_results --n_trials 300
  ```
  Launch inside the `training` tmux session. Multiple processes per node share the PG study.

### Zombie RUNNING trial cleanup (general procedure)

If workers are killed mid-trial (SSH tunnel down, `pkill`, OOM…), Optuna leaves the row as `RUNNING` indefinitely — there is no heartbeat or timeout. Cleanup via SQL transaction:
```sql
BEGIN;
CREATE TEMP TABLE _to_delete AS
  SELECT trial_id FROM trials WHERE state='RUNNING' AND datetime_start < NOW() - INTERVAL '2 hours';
DELETE FROM trial_heartbeats        WHERE trial_id IN (SELECT trial_id FROM _to_delete);
DELETE FROM trial_intermediate_values WHERE trial_id IN (SELECT trial_id FROM _to_delete);
DELETE FROM trial_values             WHERE trial_id IN (SELECT trial_id FROM _to_delete);
DELETE FROM trial_params             WHERE trial_id IN (SELECT trial_id FROM _to_delete);
DELETE FROM trial_system_attributes  WHERE trial_id IN (SELECT trial_id FROM _to_delete);
DELETE FROM trial_user_attributes    WHERE trial_id IN (SELECT trial_id FROM _to_delete);
DELETE FROM trials                   WHERE trial_id IN (SELECT trial_id FROM _to_delete);
COMMIT;
```
Tighten the `INTERVAL` threshold based on expected per-trial duration. Optuna's TPE ignores `RUNNING` for fitting anyway — cleanup is purely DB hygiene.

---

## Workflow

```bash
# Start / resume overnight search on a node
conda activate retina
cd /home/matteo/flashed_images_model
python bayes_search.py   # ctrl+C safe — resumes from shared PG study

# Check progress from anywhere (DB is shared)
PGPASSWORD=optuna_pass psql -U optuna_user -d optuna_db -h localhost -c \
  "SELECT state, COUNT(*) FROM trials GROUP BY state;"

# Inspect results CSV (written per-node into ./bayes_search_results/)
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

### Tmux layout on worker nodes (spark-fd4b, spark-09bd)

Each worker node keeps these long-lived tmux sessions:
- `tunnel`  → SSH local-port-forward to spark-09ab:5432 (must stay up)
- `training` → the bayes_search worker processes

There is no `claude` session anymore (closed manually). Re-create it with `tmux new -d -s claude` if you need to attach an interactive Claude Code shell on those nodes.

---

## Known issues & solutions

**bf16 is 3× SLOWER than fp32 on GB10/PyTorch 2.12 nightly**
→ `train.py` and `bayes_search.py` use `dtype=torch.float32` in autocast. Measured: 90.6ms/step (bf16) vs 30.0ms/step (fp32). Reverts to bf16 when a future nightly optimizes sm_121 kernels. See `training_optimization_report.md`.

**SDPA bool mask causes 20× slowdown in bf16** (8.5ms vs 0.4ms/call)
→ `model/transformer.py` now uses a float additive mask (0=valid, -inf=masked) instead of bool. Neutral in fp32, future-proof for bf16.

**`run_evaluate.py` hardcodes wrong output dir**: `GRID_DIR = 'grid_search_results'` but `bayes_search.py` writes to `bayes_search_results/`. Always update `GRID_DIR` before running evaluation on bayes results.

**Adj R² can be 0 for neurons with c_eo ≤ 0**: if a neuron's responses are anti-correlated between even/odd repetitions (very noisy neuron), the metric is set to 0 regardless of model quality. This is correct per the paper formula, not a bug.

**Pruned trials absent from CSV**: pruned trials (RoPE violation or param overflow) are not written to CSV. Gaps in trial numbers are normal.

**Zombie `RUNNING` trials in PostgreSQL**: if a worker process crashes (or its SSH tunnel dies) mid-trial, Optuna leaves the row as `RUNNING` indefinitely — no heartbeat / timeout. The TPE sampler ignores `RUNNING` rows for fitting but they pollute progress queries. Clean them up manually as shown in the "Current bayes_search status" section.

---

## Off-limits

- **`model/losses.py`**: Poisson NLL is theoretically grounded for spike count data. Do not change.
- **`autocast` dtype**: currently `float32` — do NOT change to `bfloat16` until verified faster (bf16 is 3× slower on GB10/PyTorch 2.12 nightly). Change to `float16` is also off-limits.
- **Radii as fixed buffers**: `r` is not a learnable parameter by design. The search explores radii between trials; within a trial they are frozen.
- **Adj R² formula in `bayes_search.py`**: implements Goldin eq. 5 exactly. Do not simplify or replace with standard R².
- **Data files** (`PNAS_paper_sorted_data.npz`, `STA/`, `ellipse_centers_exp2.npz`, `lsta_ref.npz`): read-only experimental data, never modify.
- **Transformer weight sharing**: weights are shared across neurons deliberately (parameter efficiency). Do not add per-neuron transformer weights without understanding the 41× parameter increase.
- **Shared PostgreSQL credentials**: `optuna_user` / `optuna_pass` are wired into the code. Don't rotate them without updating `bayes_search.py:541` and notifying all worker nodes.

---

## Output layout

```
train.py:
  best_model.pt                   {epoch, model_state, val_loss, cfg}

bayes_search.py  → bayes_search_results/   (NFS-shared from spark-09ab; same path on all attached nodes)
  all_results.csv                 one row/trial: hyperparams + val_loss + all scores
  mean_all/
    model.pt  config.json  history.json
  per_neuron/neuron_00/ … neuron_40/
    model.pt  config.json  history.json
  lsta_corr/mean_all/
  lsta_corr/per_neuron/neuron_00/ … neuron_40/

  NOTE 1: Optuna storage is NOT in this directory — it lives in PostgreSQL on spark-09ab.
  The legacy `optuna.db` SQLite file is no longer used; migration in commit ee8b158.

  NOTE 2: This directory is an NFS mount on all worker nodes (see "Shared filesystem"
  in the Infrastructure section). Multiple workers across multiple machines write to the
  same physical files. Concurrent writes to `all_results.csv` and to per-criterion best
  trackers are NOT currently protected by file locks — race conditions are possible at
  high writer concurrency. Mitigation (fcntl.flock around the hot paths) is a known
  follow-up if torn writes are observed.

config.json structure:
  {"score": float, "cfg": {...}, "params": {cnn, neuron_circle, transformer, readout, total}}

run_evaluate.py  → <model_dir>/eval/
  metrics/  scatter/  response_rank/  sta_centers/
  training/  operators/lsta_comparison/
```
