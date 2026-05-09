# model/config.py

# ─────────────────────────────────────────
# Fixed constants (data-dependent, not searched)
# ─────────────────────────────────────────
IMG_SIZE_ORIGINAL = 108   # original input image size
IN_CHANNELS       = 1     # 1 = grayscale
N_CELLS           = 41    # number of neurons to predict

MAX_PARAMS = 200_000_000  # max parameter budget

RF_RANGE = None


# ─────────────────────────────────────────
# Default configuration
#
# Architecture:
#   1. CNN Tokenizer      (model/tokenizer.py)
#   2. NeuronCircle       (model/neuron_circle.py)
#   3. TransformerEncoder (model/transformer.py)
#   4. FactorizedReadout  (model/readout.py)
# ─────────────────────────────────────────

DEFAULT_CFG = {

    # ── Block 1: CNN Tokenizer ───────────────────────────────────────────
    'CNN_DIM':    16,
    'CNN_LAYERS': 1,
    'CNN_KERNEL': 3,
    'DROPOUT':    0.0,
    'CNN_STRIDE': 1,

    # ── Block 2: TransformerEncoder ──────────────────────────────────────
    'EMB_DIM':    None,
    'NUM_HEADS':  4,
    'NUM_BLOCKS': 1,
    'MLP_DIM':    32,

    # ── Training ─────────────────────────────────────────────────────────
    'BATCH_SIZE':    32,
    'LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY':  1e-4,
    'MAX_EPOCHS':    200,
    'EARLY_STOP':    30,

    # ── Per-neuron RF radii (px, in 108×108 space) ───────────────────────
    #
    # One value per neuron (cell_021…cell_061).
    # r is frozen during training; cx and cy stay trainable.
    # crop_half = ceil(max(NEURON_RADII) / CNN_STRIDE) — derived automatically.
    # Values derived from the major semi-axis of the experimental RF ellipses.
    'NEURON_RADII': [
         5.2,  6.3,  4.6,  5.1,  7.6,  6.0,  5.0,  4.7,  6.9,  7.0,
         6.3,  6.3,  6.1,  5.6,  5.5,  4.4,  5.4,  6.2,  6.1,  6.6,
         8.6,  7.4,  8.1,  6.3,  5.7,  5.9,  4.8,  6.1,  7.1,  6.1,
         5.1,  5.5,  5.1, 10.8,  3.5,  5.2,  3.6,  2.9,  7.7,  6.2,
         6.6,
    ],
}


# ─────────────────────────────────────────
# Bayesian search configuration (grid_search.py, Optuna TPE sampler)
#
# BAYES_FIXED  : hyperparameters held constant across all trials.
# BAYES_SPACE  : architecture hyperparameters with their candidate values;
#                each trial samples one option per key (categorical suggest).
# RADII_RANGE  : integer range (inclusive) for each of the 41 neuron radii.
#
# RoPE constraint: (effective_emb_dim // NUM_HEADS) % 4 == 0
#   effective_emb_dim = EMB_DIM if EMB_DIM is not None else CNN_DIM
# Trials that violate this constraint or exceed MAX_PARAMS are pruned.
# ─────────────────────────────────────────

BAYES_FIXED = {
    'CNN_LAYERS': 1,
    'DROPOUT':    0.1,

    'BATCH_SIZE':    32,
    'LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY':  0.0,
    'MAX_EPOCHS':    200,
    'EARLY_STOP':    30,
}

BAYES_SPACE = {
    # CNN_DIM and MLP_DIM are sampled as log2 integers (see CNN_DIM_EXP_RANGE /
    # MLP_DIM_EXP_RANGE below) so the TPE understands their ordinal structure
    # while the actual values remain exact powers of 2.
    'CNN_KERNEL': [3, 5, 7],
    'CNN_STRIDE': [1, 2],
    'EMB_DIM':    [None, 32, 64, 128],
    'NUM_HEADS':  [1, 2, 4, 8],
    'NUM_BLOCKS': [1, 2],
}

# CNN_DIM  = 2^exp  for exp in CNN_DIM_EXP_RANGE  → 8, 16, 32, 64
# MLP_DIM  = 2^exp  for exp in MLP_DIM_EXP_RANGE  → 32, 64, 128, 256
CNN_DIM_EXP_RANGE = (3, 6)
MLP_DIM_EXP_RANGE = (5, 8)

# Integer range (inclusive) sampled for each of the 41 neuron radii (px in 108×108).
RADII_RANGE = (4, 35)

# Default number of Optuna trials (override via run_bayes_search(n_trials=…)).
N_TRIALS_DEFAULT = 200
