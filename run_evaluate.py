# run_evaluate.py

import os
import json
import glob
import torch
from evaluate import evaluate_model, evaluate_best_per_neuron

# ─────────────────────────────────────────────────────────────────────────────
# Quali criteri valutare
# ─────────────────────────────────────────────────────────────────────────────
#
# RUN_MEAN_ALL   : True → valuta il modello ottimizzato per Adj R² medio su tutti i neuroni
#
# RUN_MEAN_TOP   : lista di k → valuta i modelli mean_top_k per ogni k specificato
#                  es. [10, 20]   oppure  list(range(1, 41))  per tutti i 40
#                  []  = nessuno
#
# RUN_PER_NEURON : lista di indici neurone → valuta il modello per-neurone per ogni n
#                  es. [0, 5, 10]  oppure  list(range(41))  per tutti i 41
#                  []  = nessuno
#
# AVG_KS         : lista di varianti di averaging da valutare per ogni criterio
#                  sottoinsieme di [1, 2, 3]   es. [1]  oppure  [1, 2, 3]

RUN_MEAN_ALL        = False   # valuta il modello mean_all
RUN_ALL_PER_NEURON  = False   # valuta tutti i 41 modelli neuron_XX singoli
RUN_BEST_PER_NEURON = False    # valuta il modello combinato best-per-neurone
                               # (per ogni neurone usa il modello neuron_XX ottimale)
                               # output: per_neuron/best_per_neuron/avg_kN/eval/

# ── modelli ottimizzati per LSTA correlation (sotto lsta_corr/) ───────────────
RUN_LSTA_MEAN_ALL        = False  # valuta lsta_corr/mean_all
RUN_LSTA_ALL_PER_NEURON  = False  # valuta lsta_corr/per_neuron/neuron_XX  (tutti 41)
RUN_LSTA_BEST_PER_NEURON = True  # best-per-neurone usando i modelli lsta_corr


GRID_DIR = 'grid_search_results'

# ─────────────────────────────────────────────────────────────────────────────
# Percorsi dati
# ─────────────────────────────────────────────────────────────────────────────

DATA_NPZ = 'PNAS_paper_sorted_data.npz'
STA_DIR  = 'STA'
RF_PATH  = 'ellipse_centers_exp2.npz'

# ─────────────────────────────────────────────────────────────────────────────
# Cosa plottare  (True = genera, False = salta)
# — si applica uguale a tutti i modelli valutati
# ─────────────────────────────────────────────────────────────────────────────

DO_METRICS       = False    # line chart per ogni metrica          → metrics/
DO_SCATTER       = False    # scatter pred vs true per ogni neurone → scatter/
DO_RESPONSE_RANK = False    # response-rank per ogni neurone        → response_rank/
DO_STA_CENTERS   = False    # RF center + readout mask              → sta_centers/
DO_TRAINING      = False   # curva di training loss                → training/
DO_LSTA          = True   # LSTA sperimentale + ellisse RF        → operators/lsta_comparison/

# ── operatori di attribuzione ─────────────────────────────────────────────────
# Tutti i plot degli operatori vanno in operators/<nome_operatore>/neuron_XX.png
# Richiedono lsta_ref.npz per le immagini di riferimento.
#
# DO_IG  : Integrated Gradients  — percorso lineare grigio → immagine
# DO_WIG : Waypoint Integrated Gradients — percorso nero → immagine → bianco
#          (l'immagine è un waypoint; spiega F(bianco)−F(nero))
#
# OP_N_STEPS : numero di passi di interpolazione (default 10)
DO_IG      = False
DO_WIG     = False
OP_N_STEPS = 100

# parametri LSTA (usati solo se DO_LSTA=True)
LSTA_REF      = 'lsta_ref.npz'
LSTA_PAD      = 4     # pixel di padding attorno all'ellisse
LSTA_EXPON    = 2     # esponente sharpening per righe post-processed
LSTA_VMAX     = 1.0   # saturazione colore (frazione del max)
LSTA_PUT_TO0  = 0.2   # soglia rumore (valori < vmax*frac → 0)

# ─────────────────────────────────────────────────────────────────────────────
# Logica interna
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_dir(grid_dir, criterion, lsta=False):
    """Costruisce il path del modello seguendo la struttura di grid_search.py."""
    if lsta:
        if criterion == 'mean_all':
            return os.path.join(grid_dir, 'lsta_corr', 'mean_all')
        else:
            return os.path.join(grid_dir, 'lsta_corr', 'per_neuron', criterion)
    if criterion == 'mean_all':
        return os.path.join(grid_dir, 'mean_all')
    else:  # neuron_XX
        return os.path.join(grid_dir, 'per_neuron', criterion)


def _collect_criteria():
    """Costruisce la lista di criteri (adj_r2) da valutare."""
    criteria = []
    if RUN_MEAN_ALL:
        criteria.append('mean_all')
    if RUN_ALL_PER_NEURON:
        for n in range(41):
            criteria.append(f'neuron_{n:02d}')
    return criteria


def _collect_lsta_criteria():
    """Costruisce la lista di criteri lsta_corr da valutare."""
    criteria = []
    if RUN_LSTA_MEAN_ALL:
        criteria.append('mean_all')
    if RUN_LSTA_ALL_PER_NEURON:
        for n in range(41):
            criteria.append(f'neuron_{n:02d}')
    return criteria


def _run_one(criterion, device, lsta=False):
    model_dir    = _build_model_dir(GRID_DIR, criterion, lsta=lsta)
    model_path   = os.path.join(model_dir, 'model.pt')
    config_path  = os.path.join(model_dir, 'config.json')
    history_path = os.path.join(model_dir, 'history.json')
    out_dir      = os.path.join(model_dir, 'eval')

    if not os.path.exists(model_path):
        print(f'[SKIP] {model_dir}  — model.pt non trovato')
        return

    state_dict = torch.load(model_path, map_location=device)
    with open(config_path) as f:
        cfg = json.load(f)['cfg']

    history = None
    do_training = DO_TRAINING
    if do_training:
        if os.path.exists(history_path):
            history = history_path
        else:
            print(f'  [WARN] history.json non trovato — training plot saltato')
            do_training = False

    lsta_tag = '  [lsta_corr]' if lsta else ''
    print(f'\n{"─"*60}')
    print(f'Criterio : {criterion}{lsta_tag}')
    print(f'Modello  : {model_dir}')
    print(f'Output   : {out_dir}')

    evaluate_model(
        state_dict       = state_dict,
        cfg              = cfg,
        data_npz         = DATA_NPZ,
        sta_dir          = STA_DIR,
        rf_path          = RF_PATH,
        device           = device,
        out_dir          = out_dir,
        do_metrics       = DO_METRICS,
        do_scatter       = DO_SCATTER,
        do_response_rank = DO_RESPONSE_RANK,
        do_sta_centers   = DO_STA_CENTERS,
        do_training      = do_training,
        history          = history,
        do_lsta          = DO_LSTA,
        lsta_ref_path    = LSTA_REF,
        lsta_padding     = LSTA_PAD,
        lsta_expon       = LSTA_EXPON,
        lsta_vmax_thresh = LSTA_VMAX,
        lsta_put_to0     = LSTA_PUT_TO0,
        do_ig            = DO_IG,
        do_wig           = DO_WIG,
        op_n_steps       = OP_N_STEPS,
    )


# ─────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criteria      = _collect_criteria()
lsta_criteria = _collect_lsta_criteria()

_any = criteria or lsta_criteria or RUN_BEST_PER_NEURON or RUN_LSTA_BEST_PER_NEURON
if not _any:
    print('Nessun criterio selezionato.')
else:
    if criteria:
        print(f'Modelli adj_r2 da valutare: {len(criteria)}')
        for criterion in criteria:
            _run_one(criterion, device)

    if RUN_BEST_PER_NEURON:
        out_dir = os.path.join(GRID_DIR, 'per_neuron', 'best_per_neuron', 'eval')
        print(f'\n{"─"*60}')
        print(f'Best-per-neuron  [adj_r2]')
        print(f'Output: {out_dir}')
        evaluate_best_per_neuron(
            grid_dir         = GRID_DIR,
            data_npz         = DATA_NPZ,
            sta_dir          = STA_DIR,
            rf_path          = RF_PATH,
            device           = device,
            out_dir          = out_dir,
            do_metrics       = DO_METRICS,
            do_scatter       = DO_SCATTER,
            do_response_rank = DO_RESPONSE_RANK,
            do_sta_centers   = DO_STA_CENTERS,
            do_lsta          = DO_LSTA,
            lsta_ref_path    = LSTA_REF,
            lsta_padding     = LSTA_PAD,
            do_ig            = DO_IG,
            do_wig           = DO_WIG,
            op_n_steps       = OP_N_STEPS,
        )

    if lsta_criteria:
        print(f'Modelli lsta_corr da valutare: {len(lsta_criteria)}')
        for criterion in lsta_criteria:
            _run_one(criterion, device, lsta=True)

    if RUN_LSTA_BEST_PER_NEURON:
        out_dir = os.path.join(GRID_DIR, 'lsta_corr', 'per_neuron', 'best_per_neuron', 'eval')
        print(f'\n{"─"*60}')
        print(f'Best-per-neuron  [lsta_corr]')
        print(f'Output: {out_dir}')
        evaluate_best_per_neuron(
            grid_dir         = os.path.join(GRID_DIR, 'lsta_corr'),
            data_npz         = DATA_NPZ,
            sta_dir          = STA_DIR,
            rf_path          = RF_PATH,
            device           = device,
            out_dir          = out_dir,
            do_metrics       = DO_METRICS,
            do_scatter       = DO_SCATTER,
            do_response_rank = DO_RESPONSE_RANK,
            do_sta_centers   = DO_STA_CENTERS,
            do_lsta          = DO_LSTA,
            lsta_ref_path    = LSTA_REF,
            lsta_padding     = LSTA_PAD,
            do_ig            = DO_IG,
            do_wig           = DO_WIG,
            op_n_steps       = OP_N_STEPS,
        )

    print(f'\n{"="*60}')
    print(f'Completato  ({len(criteria)} adj_r2'
          + (' + best-per-neuron' if RUN_BEST_PER_NEURON else '')
          + (f' | {len(lsta_criteria)} lsta_corr' if lsta_criteria else '')
          + (' + lsta best-per-neuron' if RUN_LSTA_BEST_PER_NEURON else '')
          + ')')


# ═════════════════════════════════════════════════════════════════════════════
# Valutazione modelli retrained  (models_retrained/)
# ═════════════════════════════════════════════════════════════════════════════

RETRAINED_DIR = 'models_retrained'

# ── cosa valutare ─────────────────────────────────────────────────────────────
RUN_RETRAINED_MEAN_ALL   = False   # valuta i modelli mean_all retrained
RUN_RETRAINED_PER_NEURON = False   # valuta i modelli per_neuron retrained

# ── cosa plottare  (True = genera, False = salta) ─────────────────────────────
RT_DO_METRICS            = False
RT_DO_SCATTER            = False
RT_DO_RESPONSE_RANK      = False
RT_DO_STA_CENTERS        = False
RT_DO_TRAINING           = False   # single history.json  (per_neuron models)
RT_DO_TRAINING_PER_NEURON = True   # 41 neuron-specific plots (mean_all models)
RT_DO_LSTA               = False


def _run_retrained_mean_all(device):
    src_dir = os.path.join(RETRAINED_DIR, 'mean_all')
    model_pt  = os.path.join(src_dir, 'model.pt')
    config_pt = os.path.join(src_dir, 'config.json')
    out_dir   = os.path.join(src_dir, 'eval')

    if not os.path.exists(model_pt):
        print(f'[SKIP] {src_dir}  — model.pt non trovato')
        return

    state_dict = torch.load(model_pt, map_location=device, weights_only=True)
    with open(config_pt) as f:
        cfg = json.load(f)['cfg']

    # Collect per-neuron histories (history_neuron_00.json … neuron_40.json)
    hist_paths = sorted(
        glob.glob(os.path.join(src_dir, 'history_neuron_??.json')))
    histories_pn = hist_paths if hist_paths else None

    print(f'\n{"─"*60}')
    print(f'Retrained  mean_all')
    print(f'Output: {out_dir}')

    evaluate_model(
        state_dict             = state_dict,
        cfg                    = cfg,
        data_npz               = DATA_NPZ,
        sta_dir                = STA_DIR,
        rf_path                = RF_PATH,
        device                 = device,
        out_dir                = out_dir,
        do_metrics             = RT_DO_METRICS,
        do_scatter             = RT_DO_SCATTER,
        do_response_rank       = RT_DO_RESPONSE_RANK,
        do_sta_centers         = RT_DO_STA_CENTERS,
        do_training            = False,
        do_training_per_neuron = RT_DO_TRAINING_PER_NEURON,
        histories_per_neuron   = histories_pn,
        do_lsta                = RT_DO_LSTA,
        lsta_ref_path          = LSTA_REF,
        lsta_padding           = LSTA_PAD,
        lsta_expon             = LSTA_EXPON,
        lsta_vmax_thresh       = LSTA_VMAX,
        lsta_put_to0           = LSTA_PUT_TO0,
    )


def _run_retrained_per_neuron(n, device):
    crit    = f'neuron_{n:02d}'
    src_dir = os.path.join(RETRAINED_DIR, 'per_neuron', crit)
    model_pt  = os.path.join(src_dir, 'model.pt')
    config_pt = os.path.join(src_dir, 'config.json')
    hist_pt   = os.path.join(src_dir, 'history.json')
    out_dir   = os.path.join(src_dir, 'eval')

    if not os.path.exists(model_pt):
        print(f'[SKIP] {src_dir}  — model.pt non trovato')
        return

    state_dict = torch.load(model_pt, map_location=device, weights_only=True)
    with open(config_pt) as f:
        cfg = json.load(f)['cfg']

    history     = hist_pt if os.path.exists(hist_pt) else None
    do_training = RT_DO_TRAINING and history is not None

    print(f'\n{"─"*60}')
    print(f'Retrained  {crit}')
    print(f'Output: {out_dir}')

    evaluate_model(
        state_dict       = state_dict,
        cfg              = cfg,
        data_npz         = DATA_NPZ,
        sta_dir          = STA_DIR,
        rf_path          = RF_PATH,
        device           = device,
        out_dir          = out_dir,
        do_metrics       = RT_DO_METRICS,
        do_scatter       = RT_DO_SCATTER,
        do_response_rank = RT_DO_RESPONSE_RANK,
        do_sta_centers   = RT_DO_STA_CENTERS,
        do_training      = do_training,
        history          = history,
        do_lsta          = RT_DO_LSTA,
        lsta_ref_path    = LSTA_REF,
        lsta_padding     = LSTA_PAD,
        lsta_expon       = LSTA_EXPON,
        lsta_vmax_thresh = LSTA_VMAX,
        lsta_put_to0     = LSTA_PUT_TO0,
    )


# ── run retrained evaluations ─────────────────────────────────────────────────
if RUN_RETRAINED_MEAN_ALL:
    _run_retrained_mean_all(device)

if RUN_RETRAINED_PER_NEURON:
    from model.config import N_CELLS
    for n in range(N_CELLS):
        _run_retrained_per_neuron(n, device)
