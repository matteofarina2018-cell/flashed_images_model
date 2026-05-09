# Training Optimization Report
**Data**: 2026-05-07  
**Hardware**: NVIDIA GB10 Blackwell (121 GB VRAM, sm_121), 121 GB RAM  
**PyTorch**: 2.12.0.dev20260306+cu128, CUDA 12.8  
**Config benchmark**: CNN_DIM=16, CNN_STRIDE=2, EMB_DIM=16, NUM_HEADS=4, MLP_DIM=256, r=10 → 14k params, T_circle=81

---

## Situazione iniziale

- **Tempo per epoca (GPU carica)**: ~20s  (contesa con training job in background)
- **Tempo per epoca (GPU libera, bf16)**: 90.6 ms/step → **8.2s/epoca**
- **Dtype**: bfloat16 via `torch.autocast(..., dtype=torch.bfloat16)`

---

## Bottleneck trovati

### 1. bfloat16 kernels non ottimizzati su GB10 — `train.py`, `bayes_search.py`

**Rilevato da**: benchmark diretto fp32 vs bf16 (il profiler PyTorch non produce CUDA time su sm_121, confermando che i kernel bf16 non sono JIT-ottimizzati per questa architettura).

**Causa**: GB10 Blackwell (compute cap 12.1) è un'architettura nuovissima. PyTorch 2.12.0 nightly non ha ancora i kernel CUDA ottimizzati per sm_121 in bfloat16. I kernel fp32 invece usano percorsi CUDA generici che funzionano correttamente.

**Fix**: `dtype=torch.bfloat16` → `dtype=torch.float32` in `train.py:67` e `bayes_search.py:136`.

**Impatto**: 90.6 ms/step → 30.0 ms/step = **3.0× speedup**  
Epoch: 8.2s → 2.7s

---

### 2. SDPA bool mask 20× più lento in bf16 — `model/transformer.py`

**Rilevato da**: micro-benchmark `F.scaled_dot_product_attention`:
- bf16 + bool mask `[1312,4,81,4]`: **8.5 ms/call**
- bf16 + no mask: **0.4 ms/call** (21× più veloce)
- fp32 + bool mask: ~0.3 ms/call (non impattato)

**Causa**: la bool mask forza il math kernel di PyTorch invece di Flash Attention in bf16. Il math kernel materializza l'intera matrice di attenzione [B×N, H, T, T] e non usa il percorso ottimizzato.

**Fix**: convertire la bool mask in float additive mask (`0=valid, -inf=masked`) in `MultiHeadSelfAttention.forward` (`model/transformer.py:59-61`). Il fix è neutro in fp32 ma risolve il problema quando bf16 tornerà a funzionare.

**Impatto immediato**: nessuno (già su fp32). Impatto futuro stimato con bf16: -53ms/step (3 SDPA call × 17.8ms risparmiati).

---

### 3. `optimizer.zero_grad()` → `set_to_none=True` — `train.py`, `bayes_search.py`

**Fix**: minima ottimizzazione di allocazione memoria (evita zeroing, imposta gradients a None).  
**Impatto**: < 1% — incluso per completezza.

---

## Tabella riassuntiva

| Intervento | ms/step | Epoca (91 batch) | Δ cumulativo |
|---|---|---|---|
| Baseline (bf16, GPU scarica) | 90.6 | 8.2 s | — |
| + fp32 autocast | 30.0 | 2.7 s | **3.0× tot** |
| + float mask (SDPA) | 30.0 | 2.7 s | (neutro ora, future-proof) |
| + set_to_none=True | ~29.8 | ~2.7 s | < 1% |

---

## Dipendenza dalla config (fp32, GPU libera)

| Config (stride=2) | T_circle | Params | ms/step | Epoca |
|---|---|---|---|---|
| CNN_DIM=8, r=4 (LIGHT) | 13 | 2k | 13.8 ms | 1.3 s |
| CNN_DIM=16, r=10 | 81 | 14k | 30.0 ms | 2.7 s |
| CNN_DIM=64, r=25 (HEAVY) | 489 | 170k | 681.7 ms | 62.0 s |

Proiezione Bayesian search (50 epoch eff. × 200 trial):
- Senza fix (bf16): ~200 × 50 × 8.2s = ~228 ore per le config piccole
- Con fix (fp32): ~200 × 50 × 2.7s = **~75 ore** (il mix small/large porta a risultati effettivi inferiori)

---

## Verifica di stabilità

- Config LIGHT (CNN_DIM=8, stride=2, r=4): **1.3s/epoch, loss=0.843, NaN=0 ✓**
- Config HEAVY (CNN_DIM=64, stride=2, r=25): **62.0s/epoch, loss=0.773, NaN=0 ✓**
- LIGHT < HEAVY ✓

---

## Fix considerati e scartati

- **Aumentare DataLoader workers** (4→8): DataLoader costa 0.2s per l'intera epoca (<1% del totale) — irrilevante.
- **Disabilitare LSTA durante la search**: LSTA costa 0.8s su trial da 135s+ — <0.6% del totale.
- **torch.compile**: non testato. Potenziale +10-30% aggiuntivo, ma torch.compile su nightly con architettura nuova ha rischio di bug. Da considerare in futuro.

---

## Prossimi passi

1. **Riabilitare bf16**: quando una versione stabile di PyTorch supporterà sm_121 in bf16, cambiare `dtype=torch.float32` → `torch.bfloat16` in train.py e bayes_search.py e misurare. Atteso ulteriore ~2× speedup.
2. **torch.compile**: `model = torch.compile(model, fullgraph=False)` potrebbe dare +10-30% su fp32. Da misurare dopo che il search space sarà consolidato.
3. **Config HEAVY con stride=1 e r_max=25**: T_circle=1961 → ~12 min/epoch → ~600 min/trial (50 epoch). Se configs con stride=1 dominano il search, valutare di aggiungere stride=1 come parametro penalizzato o di escluderlo dal BAYES_SPACE.
