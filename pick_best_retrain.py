#!/usr/bin/env python
"""After parallel retrain_trial143.py invocations finish across nodes, pick the
seed with the highest mean_all and copy its model/config/history to
bayes_search_results/mean_all/, replacing the lower-scoring file currently there.

Also updates per_neuron/neuron_NN/ and lsta_corr/{...} only when a seed beats
the existing score for that criterion.
"""

import json
import os
import shutil
import sys

OUT_ROOT  = sys.argv[1] if len(sys.argv) > 1 else './bayes_search_results'
RUNS_NAME = sys.argv[2] if len(sys.argv) > 2 else 'retrain_runs'

retrain_root = os.path.join(OUT_ROOT, RUNS_NAME)
seed_dirs = sorted(d for d in os.listdir(retrain_root)
                   if d.startswith('seed_') and os.path.isdir(os.path.join(retrain_root, d)))
if not seed_dirs:
    sys.exit(f'No seed_* dirs under {retrain_root}')

runs = []
for sd in seed_dirs:
    p = os.path.join(retrain_root, sd)
    cfg = json.load(open(os.path.join(p, 'config.json')))
    scores = json.load(open(os.path.join(p, 'scores.json')))
    runs.append((sd, p, cfg, scores))
    print(f'{sd}: mean_all={cfg["score"]:.4f}  '
          f'val={cfg["val_loss"]:.4f}  host={cfg["host"]}  ({cfg["train_sec"]:.0f}s)')

# ── pick best by mean_all ────────────────────────────────────────────────────
best = max(runs, key=lambda r: r[2]['score'])
sd_b, path_b, cfg_b, scores_b = best
print(f'\n=== WINNER mean_all: {sd_b} (score={cfg_b["score"]:.4f}) ===')

mean_all_dir = os.path.join(OUT_ROOT, 'mean_all')
existing = -1.0
ex_cfg = os.path.join(mean_all_dir, 'config.json')
if os.path.exists(ex_cfg):
    try: existing = json.load(open(ex_cfg)).get('score') or -1.0
    except: pass

if cfg_b['score'] > existing:
    os.makedirs(mean_all_dir, exist_ok=True)
    for f in ['model.pt', 'history.json']:
        shutil.copy2(os.path.join(path_b, f), os.path.join(mean_all_dir, f))
    # write mean_all/config.json in the canonical bayes_search format
    cfg_out = {'score': cfg_b['score'], 'cfg': cfg_b['cfg'], 'params': cfg_b['params']}
    with open(os.path.join(mean_all_dir, 'config.json'), 'w') as f:
        json.dump(cfg_out, f, indent=2)
    print(f'mean_all/ updated  ({existing:.4f} → {cfg_b["score"]:.4f})')
else:
    print(f'mean_all/ NOT updated  (existing {existing:.4f} ≥ best retrain {cfg_b["score"]:.4f})')

# ── per_neuron + lsta criteria: update only if retrain beats existing ────────
for crit_kind in ['per_neuron', 'lsta_per_neuron', 'lsta_mean_all']:
    for n in (range(41) if 'per_neuron' in crit_kind else [None]):
        if crit_kind == 'per_neuron':
            key = f'neuron_{n:02d}'
            target = os.path.join(OUT_ROOT, 'per_neuron', key)
            getter = lambda r: r[3]['scores'].get(key, -1) if r[3]['scores'] else -1
        elif crit_kind == 'lsta_per_neuron':
            key = f'neuron_{n:02d}'
            target = os.path.join(OUT_ROOT, 'lsta_corr', 'per_neuron', key)
            getter = lambda r: r[3]['lsta_scores'].get(key, -1) if r[3]['lsta_scores'] else -1
        else:
            key = 'mean_all'
            target = os.path.join(OUT_ROOT, 'lsta_corr', 'mean_all')
            getter = lambda r: r[3]['lsta_scores'].get(key, -1) if r[3]['lsta_scores'] else -1

        if not any(getter(r) > -1 for r in runs):
            continue
        s_best = max(runs, key=getter)
        s_score = getter(s_best)
        if s_score <= 0:
            continue
        ex = -1.0
        if os.path.exists(os.path.join(target, 'config.json')):
            try: ex = json.load(open(os.path.join(target, 'config.json'))).get('score') or -1.0
            except: pass
        if s_score > ex:
            os.makedirs(target, exist_ok=True)
            for f in ['model.pt', 'history.json']:
                shutil.copy2(os.path.join(s_best[1], f), os.path.join(target, f))
            cfg_out = {'score': s_score, 'cfg': s_best[2]['cfg'], 'params': s_best[2]['params']}
            with open(os.path.join(target, 'config.json'), 'w') as f:
                json.dump(cfg_out, f, indent=2)
            print(f'{crit_kind}/{key}: {ex:.4f} → {s_score:.4f} (from {s_best[0]})')
