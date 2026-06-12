"""P17 -- Readout-validity / C-22 gate vs Kato HisCl + NeuroPAL(Atanas).

Grounds V7's AVA-containing command-fraction / 3.0 Hz quiescence readout against
held-out data. NON-DESTRUCTIVE: reads frozen phase_g_state_validator constants
(COMMAND_NEURONS, QUIESCENT_RATE_THRESHOLD_HZ) and held-out data read-only;
emits NEW artifacts under artifacts/p17_readout_validity/.

HARD MODALITY RULE (C-22): never compare a model firing-rate Hz to a data dF/F.
All cross-modality grounding is DIRECTIONAL/sign-based only. Within-modality valley
procedures derive a number from each modality's OWN histogram.

Prereg: audits/phase1/P17/prereg.json.

Usage:
  python p17_readout_validity.py --fast     # F1-F5 only (no big sim, no full Atanas pass)
  python p17_readout_validity.py --heavy    # full Atanas streaming pass + model Hz-valley
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ANESTH = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator')
ROOT = Path('/mnt/ssd4tb/Desktop/website/personalwebsite')
sys.path.insert(0, str(ANESTH / 'src'))

P17 = ANESTH / 'audits' / 'phase1' / 'P17'
OUT = ANESTH / 'artifacts' / 'p17_readout_validity'

KATO_DIR = ROOT / 'data' / 'external' / 'kato_zimmer2015'
ATANAS_DIR = ROOT / 'data' / 'external' / 'atanas2023'

# Frozen readout constants -- IMPORTED, never mutated.
from state_validation.phase_g_state_validator import (  # noqa: E402
    COMMAND_NEURONS, QUIESCENT_RATE_THRESHOLD_HZ,
)

# AVA-containing command set, intersected with names actually labeled in data.
AVA_NAMES = ['AVAL', 'AVAR']

# Kato 8-state immobility/active partition (frozen in prereg).
KATO8_ACTIVE = ['fwd']
KATO8_IMMOBILE = ['revsus', 'slow']
KATO8_REVERSAL = ['rev1', 'rev2', 'revsus']  # AVA reversal-tuning control
# AVA_HisCl integer state codes (States_key decoded from file):
HISCL_KEY = {0: 'reversal', 1: 'forward', 2: 'quiescence', 3: 'turn'}


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def _h5_str(h, ref) -> str:
    try:
        return ''.join(chr(int(c)) for c in h[ref][:].flatten())
    except Exception:
        return ''


def load_kato_wt_nostim(path: Path) -> list[dict]:
    """Full-load WT_NoStim (5 worms): dF/F (bleach-corrected) + 8 state vectors + names."""
    import h5py
    h = h5py.File(path, 'r')
    g = h['WT_NoStim']
    n_worms = g['NeuronNames'].shape[0]
    worms = []
    for w in range(n_worms):
        dff = h[g['deltaFOverF_bc'][w, 0]][:]            # (n_neurons, T)
        nn_grp = h[g['NeuronNames'][w, 0]]
        names = [_h5_str(h, nn_grp[i, 0]) for i in range(nn_grp.shape[0])]
        st_grp = h[g['States'][w, 0]]
        states = {k: st_grp[k][:].flatten().astype(bool) for k in st_grp.keys()}
        fps = float(h[g['fps'][w, 0]][:].flatten()[0])
        worms.append({'dff': np.asarray(dff, float), 'names': names,
                      'states': states, 'fps': fps})
    h.close()
    return worms


def load_kato_hiscl(path: Path) -> list[dict]:
    """Full-load AVA_HisCl (5 worms): traces + integer state codes (1-4 -> 0-3 key)."""
    import h5py
    h = h5py.File(path, 'r')
    g = h['AVA_HisCl']
    n_worms = g['traces'].shape[0]
    worms = []
    for w in range(n_worms):
        tr = h[g['traces'][w, 0]][:]                     # (n_neurons, T)
        st = h[g['States'][w, 0]][:].flatten()           # codes 1..4
        idg = h[g['IDs'][w, 0]]
        ids = [_h5_str(h, idg[i, 0]) for i in range(idg.shape[0])]
        fps = float(h[g['fps'][w, 0]][:].flatten()[0])
        worms.append({'traces': np.asarray(tr, float), 'ids': ids,
                      'state_code': st, 'fps': fps})
    h.close()
    return worms


def stream_atanas_processed(path: Path) -> dict:
    """STREAM ONLY the small processed datasets from a 27GB NWB via h5py partial
    reads. NEVER touches CalciumImageSeries (the raw movie)."""
    import h5py
    h = h5py.File(path, 'r')
    base = 'processing/CalciumActivity'
    act = h[f'{base}/SignalCalciumImResponseSeries/data'][:]      # (T, n_neurons) tiny
    labels_raw = h[f'{base}/NeuronIDs/labels'][:]
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in labels_raw]
    vel = h['processing/Behavior/velocity/velocity/data'][:]      # (T,)
    h.close()
    return {'act': np.asarray(act, float), 'labels': labels,
            'vel': np.asarray(vel, float)}


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------
def boot_ci(vals: np.ndarray, n_boot: int = 10000, seed: int = 20260612):
    """Worm-level bootstrap mean + 95% CI."""
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return float('nan'), (float('nan'), float('nan')), 0
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, len(vals), replace=True).mean()
                      for _ in range(n_boot)])
    return float(vals.mean()), (float(np.percentile(means, 2.5)),
                                float(np.percentile(means, 97.5))), len(vals)


def kde_valley(x: np.ndarray):
    """Antimode (deepest local minimum between the two largest peaks) of a 1-D
    distribution via Gaussian KDE. Returns (valley, is_bimodal)."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 10 or x.std() == 0:
        return float('nan'), False
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), 512)
    d = kde(grid)
    # local maxima / minima
    peaks = [i for i in range(1, len(d) - 1) if d[i] > d[i - 1] and d[i] >= d[i + 1]]
    if len(peaks) < 2:
        return float('nan'), False
    # two tallest peaks
    peaks = sorted(peaks, key=lambda i: -d[i])[:2]
    lo, hi = sorted(peaks)
    valley_idx = lo + int(np.argmin(d[lo:hi + 1]))
    return float(grid[valley_idx]), True


def delta_bic_1v2(x: np.ndarray):
    """deltaBIC = BIC(1-comp) - BIC(2-comp) Gaussian mixture (>=10 favors 2 comps)."""
    x = np.asarray([v for v in x if np.isfinite(v)], float).reshape(-1, 1)
    if len(x) < 20 or x.std() == 0:
        return float('nan')
    from sklearn.mixture import GaussianMixture
    g1 = GaussianMixture(1, covariance_type='full', random_state=0).fit(x)
    g2 = GaussianMixture(2, covariance_type='full', random_state=0,
                         n_init=3).fit(x)
    return float(g1.bic(x) - g2.bic(x))


def command_trace_kato(worm: dict, use_traces='dff') -> tuple[np.ndarray, list]:
    """Mean activity trace over AVA-containing command neurons present in worm.
    Returns (trace[T], names_used)."""
    names = worm['names'] if 'names' in worm else worm['ids']
    mat = worm[use_traces] if use_traces in worm else (
        worm['dff'] if 'dff' in worm else worm['traces'])
    cmd_idx = [i for i, n in enumerate(names) if n in COMMAND_NEURONS]
    if not cmd_idx:
        return np.array([]), []
    return mat[cmd_idx, :].mean(axis=0), [names[i] for i in cmd_idx]


def ava_idx(names: list) -> list:
    return [i for i, n in enumerate(names) if n in AVA_NAMES]


# ----------------------------------------------------------------------------
# Q1 -- data-side directional sign (Kato arm)
# ----------------------------------------------------------------------------
def q1_kato(worms: list[dict]) -> dict:
    per_worm_t, id_ctrl_pass, details = [], 0, []
    for w, worm in enumerate(worms):
        names = worm['names']
        ava = ava_idx(names)
        st = worm['states']
        # ID correctness: AVA higher in reversal than forward
        ok_id = False
        if ava:
            ava_tr = worm['dff'][ava, :].mean(axis=0)
            rev_mask = np.zeros(ava_tr.shape, bool)
            for k in KATO8_REVERSAL:
                if k in st:
                    rev_mask |= st[k]
            fwd_mask = st.get('fwd', np.zeros_like(rev_mask))
            if rev_mask.sum() > 0 and fwd_mask.sum() > 0:
                ok_id = ava_tr[rev_mask].mean() > ava_tr[fwd_mask].mean()
        cmd_tr, used = command_trace_kato(worm, 'dff')
        if not ava or not ok_id or len(cmd_tr) == 0:
            details.append({'worm': w, 'ava_labeled': bool(ava),
                            'id_control_pass': bool(ok_id), 'dropped': True})
            continue
        id_ctrl_pass += 1
        active = np.zeros(cmd_tr.shape, bool)
        immobile = np.zeros(cmd_tr.shape, bool)
        for k in KATO8_ACTIVE:
            if k in st:
                active |= st[k]
        for k in KATO8_IMMOBILE:
            if k in st:
                immobile |= st[k]
        if active.sum() == 0 or immobile.sum() == 0:
            details.append({'worm': w, 'dropped': True, 'reason': 'no active/immobile bins'})
            continue
        s = cmd_tr[active].mean() - cmd_tr[immobile].mean()
        t = -s  # sign-corrected: t>0 supports readout (command HIGH when immobile/reversal)
        per_worm_t.append(t)
        details.append({'worm': w, 'cmd_neurons': used, 'id_control_pass': True,
                        's_active_minus_immobile': float(s), 't_signcorr': float(t)})
    mean_t, ci, n = boot_ci(np.array(per_worm_t))
    excl0 = (ci[0] > 0) or (ci[1] < 0)
    side = 'supporting' if (excl0 and ci[0] > 0) else ('wrong' if excl0 else 'straddles')
    return {'modality': 'kato_labeled_states', 'n_worms_used': n,
            'n_id_controlled': id_ctrl_pass, 'mean_t_signcorr': mean_t,
            'ci95': ci, 'ci_excludes_0': bool(excl0), 'ci_side': side,
            'per_worm': details}


def q1_atanas(processed: list[dict]) -> dict:
    per_worm_t, id_ctrl_pass, details = [], 0, []
    for w, p in enumerate(processed):
        names = p['labels']
        ava = ava_idx(names)
        cmd_idx = [i for i, n in enumerate(names) if n in COMMAND_NEURONS]
        vel = p['vel']
        act = p['act']  # (T, n_neurons)
        # align lengths
        T = min(len(vel), act.shape[0])
        vel = vel[:T]
        act = act[:T]
        ok_id = False
        if ava:
            ava_tr = act[:, ava].mean(axis=1)
            rev = vel < 0
            fwd = vel > 0
            if rev.sum() > 0 and fwd.sum() > 0:
                ok_id = ava_tr[rev].mean() > ava_tr[fwd].mean()
        if not ava or not ok_id or not cmd_idx:
            details.append({'worm': w, 'ava_labeled': bool(ava),
                            'id_control_pass': bool(ok_id), 'dropped': True})
            continue
        valley, bimod = kde_valley(np.abs(vel))
        if not bimod:
            details.append({'worm': w, 'dropped': True, 'reason': 'velocity not bimodal'})
            continue
        id_ctrl_pass += 1
        cmd_tr = act[:, cmd_idx].mean(axis=1)
        active = np.abs(vel) >= valley
        immobile = np.abs(vel) < valley
        if active.sum() == 0 or immobile.sum() == 0:
            details.append({'worm': w, 'dropped': True})
            continue
        s = cmd_tr[active].mean() - cmd_tr[immobile].mean()
        t = -s
        per_worm_t.append(t)
        details.append({'worm': w, 'id_control_pass': True, 'vel_valley': float(valley),
                        's_active_minus_immobile': float(s), 't_signcorr': float(t)})
    mean_t, ci, n = boot_ci(np.array(per_worm_t))
    excl0 = (ci[0] > 0) or (ci[1] < 0)
    side = 'supporting' if (excl0 and ci[0] > 0) else ('wrong' if excl0 else 'straddles')
    return {'modality': 'atanas_pose', 'n_worms_used': n,
            'n_id_controlled': id_ctrl_pass, 'mean_t_signcorr': mean_t,
            'ci95': ci, 'ci_excludes_0': bool(excl0), 'ci_side': side,
            'per_worm': details}


# ----------------------------------------------------------------------------
# Q2 -- causal AVA-HisCl silencing (data half airtight)
# ----------------------------------------------------------------------------
def q2_data(hiscl: list[dict], wt: list[dict]) -> dict:
    # WT low-locomotion (reversal-dominated) occupancy: REVSUS as sustained-reversal class
    wt_rev = []
    for worm in wt:
        st = worm['states']
        rev = st.get('revsus', np.array([])).astype(float)
        if rev.size:
            wt_rev.append(float(rev.mean()))
    # HisCl reversal-class occupancy (state code 1 == 'reversal' after 1-indexing: codes are 1..4)
    # file codes: unique {1,2,3,4}; States_key index 0..3 -> code = index+1
    # 0:reversal->code1, 1:forward->code2, 2:quiescence->code3, 3:turn->code4
    hiscl_rev, hiscl_quiesc = [], []
    for worm in hiscl:
        c = worm['state_code']
        hiscl_rev.append(float((c == 1).mean()))      # reversal
        hiscl_quiesc.append(float((c == 3).mean()))   # quiescence
    wt_mean, wt_ci, _ = boot_ci(np.array(wt_rev))
    hc_mean, hc_ci, _ = boot_ci(np.array(hiscl_rev))
    # difference WT - HisCl reversal occupancy; pre-declared > 0 (silencing AVA reduces reversal)
    diffs = []
    rng = np.random.default_rng(20260612)
    a, b = np.array(wt_rev), np.array(hiscl_rev)
    for _ in range(10000):
        diffs.append(rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean())
    diffs = np.array(diffs)
    dmean = float(a.mean() - b.mean())
    dci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    excl0 = (dci[0] > 0) or (dci[1] < 0)
    side = 'declared' if (excl0 and dci[0] > 0) else ('wrong' if excl0 else 'straddles')
    return {'wt_revsus_occupancy_mean': wt_mean, 'wt_ci': wt_ci,
            'hiscl_reversal_occupancy_mean': hc_mean, 'hiscl_ci': hc_ci,
            'hiscl_quiescence_occupancy_mean': float(np.mean(hiscl_quiesc)),
            'diff_wt_minus_hiscl': dmean, 'diff_ci95': dci,
            'ci_excludes_0': bool(excl0), 'ci_side': side,
            'note': 'HisCl carries NO per-neuron AVA trace (numeric IDs); test is on '
                    'the labeled behavioral-state distribution, which is the readout variable.'}


# ----------------------------------------------------------------------------
# Q3 -- threshold valley
# ----------------------------------------------------------------------------
def q3_data_bimodality_kato(worms: list[dict]) -> dict:
    bics = []
    for worm in worms:
        cmd_tr, used = command_trace_kato(worm, 'dff')
        if len(cmd_tr) == 0:
            continue
        bics.append(delta_bic_1v2(cmd_tr))
    bics = [b for b in bics if np.isfinite(b)]
    med = float(np.median(bics)) if bics else float('nan')
    n_strong = int(sum(1 for b in bics if b >= 10))
    return {'per_worm_deltaBIC': bics, 'median_deltaBIC': med,
            'n_worms_deltaBIC_ge_10': n_strong, 'n_worms': len(bics),
            'PART_A_pass': bool(np.isfinite(med) and med >= 10.0)}


# ----------------------------------------------------------------------------
# Fast gates F1-F5
# ----------------------------------------------------------------------------
def run_fast() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {'block': 'P17', 'mode': 'fast', 'frozen_constants': {
        'COMMAND_NEURONS': COMMAND_NEURONS,
        'QUIESCENT_RATE_THRESHOLD_HZ': QUIESCENT_RATE_THRESHOLD_HZ}}

    # F1 availability
    kato_files = {n: (KATO_DIR / n) for n in
                  ['WT_NoStim.mat', 'WT_Stim.mat', 'AVA_HisCl.mat']}
    atanas_files = sorted(ATANAS_DIR.glob('atanas_worm_*.nwb'))
    import h5py
    kato_ok = all(p.exists() for p in kato_files.values())
    # open one of each lazily
    katotest = True
    try:
        for p in kato_files.values():
            with h5py.File(p, 'r') as _h:
                _ = list(_h.keys())
    except Exception as e:
        katotest = False
        res['kato_open_error'] = str(e)
    atanas_ok = len(atanas_files) >= 10
    atanas_lazy = True
    try:
        with h5py.File(atanas_files[0], 'r') as _h:
            _ = list(_h.keys())  # lazy, no movie read
    except Exception as e:
        atanas_lazy = False
        res['atanas_open_error'] = str(e)
    res['F1_data_availability'] = {
        'kato_files_exist': kato_ok, 'kato_open_v73': katotest,
        'n_atanas_nwb': len(atanas_files), 'atanas_ge_10': atanas_ok,
        'atanas_opens_lazily': atanas_lazy,
        'verdict': 'PASS' if (kato_ok and katotest and atanas_ok and atanas_lazy) else 'FAIL'}

    # F2 loaders + F3 AVA presence (Kato)
    t0 = time.time()
    wt = load_kato_wt_nostim(kato_files['WT_NoStim.mat'])
    hiscl = load_kato_hiscl(kato_files['AVA_HisCl.mat'])
    kato_load_s = time.time() - t0
    ava_in_wt = [bool(ava_idx(w['names'])) for w in wt]
    res['F2_kato_loader'] = {'n_wt_worms': len(wt), 'n_hiscl_worms': len(hiscl),
                             'wt0_dff_shape': list(wt[0]['dff'].shape),
                             'load_seconds': round(kato_load_s, 2),
                             'verdict': 'PASS' if (len(wt) and len(hiscl)) else 'FAIL'}

    # F2 Atanas streaming (one worm) + F3 AVA presence (Atanas)
    t0 = time.time()
    a0 = stream_atanas_processed(atanas_files[0])
    atanas_stream_s = time.time() - t0
    ava_in_atanas = bool(ava_idx(a0['labels']))
    res['F2_atanas_streamer'] = {
        'act_shape': list(a0['act'].shape), 'vel_len': len(a0['vel']),
        'n_labeled': int(sum(1 for l in a0['labels'] if l.strip())),
        'stream_seconds': round(atanas_stream_s, 3),
        'note': 'streamed ONLY processed traces+labels+velocity; 27GB raw movie untouched',
        'verdict': 'PASS' if a0['act'].size and len(a0['vel']) else 'FAIL'}
    res['F3_AVA_present'] = {
        'kato_wt_worms_with_AVA': sum(ava_in_wt), 'n_kato_wt': len(wt),
        'atanas_worm0_has_AVA': ava_in_atanas,
        'verdict': 'PASS' if (sum(ava_in_wt) >= 1 and ava_in_atanas) else 'FAIL'}

    # F4 Q1 within-Kato directional statistic (preview; full Q1 needs Atanas heavy pass)
    q1k = q1_kato(wt)
    res['F4_Q1_kato_preview'] = {**q1k,
        'verdict': 'PASS' if (q1k['ci_excludes_0'] and q1k['ci_side'] == 'supporting'
                              and q1k['n_id_controlled'] >= 3)
        else ('UNDERPOWERED' if q1k['n_id_controlled'] < 3 else 'FAIL_OR_AMBIGUOUS'),
        'note': 'within-Kato only; full Q1-data-sign PASS additionally requires the '
                'Atanas-pose arm (HEAVY) to exclude 0 on the supporting side.'}

    # F5 Q3 PART_A Kato bimodality
    q3a = q3_data_bimodality_kato(wt)
    res['F5_Q3_partA_kato_bimodality'] = {**q3a,
        'verdict': 'PASS' if q3a['PART_A_pass'] else 'FAIL'}

    # Bonus context (not a fast gate): Q2 data-half is pure-arithmetic on Kato states
    try:
        q2 = q2_data(hiscl, wt)
        res['Q2_data_half_preview'] = {**q2,
            'verdict': 'PASS' if (q2['ci_excludes_0'] and q2['ci_side'] == 'declared')
            else ('AMBIGUOUS' if q2['ci_side'] == 'straddles' else 'WRONG_DIRECTION')}
    except Exception as e:
        res['Q2_data_half_preview'] = {'error': str(e)}

    json.dump(res, open(OUT / 'p17_fast_gates.json', 'w'), indent=2, default=str)
    return res


# ----------------------------------------------------------------------------
# Heavy: full Atanas streaming pass + model Hz-valley (PART_B) + model AVA bonus
# ----------------------------------------------------------------------------
def run_heavy() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {'block': 'P17', 'mode': 'heavy'}
    wt = load_kato_wt_nostim(KATO_DIR / 'WT_NoStim.mat')
    hiscl = load_kato_hiscl(KATO_DIR / 'AVA_HisCl.mat')
    atanas_files = sorted(ATANAS_DIR.glob('atanas_worm_*.nwb'))

    # Stream all Atanas worms (small processed reads each)
    processed = []
    for p in atanas_files:
        try:
            processed.append(stream_atanas_processed(p))
        except Exception as e:
            print(f'  WARN {p.name}: {e}')

    q1k = q1_kato(wt)
    q1a = q1_atanas(processed)
    q1_pass = (q1k['ci_excludes_0'] and q1k['ci_side'] == 'supporting'
               and q1a['ci_excludes_0'] and q1a['ci_side'] == 'supporting'
               and q1k['n_id_controlled'] >= 3 and q1a['n_id_controlled'] >= 3)
    res['Q1_data_sign'] = {'kato': q1k, 'atanas': q1a,
                           'verdict': 'PASS' if q1_pass else 'FAIL_OR_AMBIGUOUS'}

    q2 = q2_data(hiscl, wt)
    res['Q2_causal_AVA_silencing'] = {'data_half': q2,
        'verdict': 'PASS' if (q2['ci_excludes_0'] and q2['ci_side'] == 'declared')
        else ('AMBIGUOUS' if q2['ci_side'] == 'straddles' else 'WRONG_DIRECTION')}

    q3a = q3_data_bimodality_kato(wt)
    # PART_B: model's OWN command-rate Hz valley vs 3.0 Hz (requires sim metrics --
    # placeholder reads any frozen per-bin command-rate artifact if present)
    model_valley, model_bimod = _model_command_rate_valley()
    partb_pass = (np.isfinite(model_valley) and
                  abs(model_valley - QUIESCENT_RATE_THRESHOLD_HZ) <= 1.0)
    res['Q3_threshold_valley'] = {
        'data_bimodality_kato': q3a,
        'model_command_rate_valley_hz': model_valley, 'model_bimodal': model_bimod,
        'threshold_hz': QUIESCENT_RATE_THRESHOLD_HZ,
        'PART_A_pass': q3a['PART_A_pass'], 'PART_B_pass': bool(partb_pass),
        'verdict': 'PASS' if (q3a['PART_A_pass'] and partb_pass) else 'FAIL'}

    overall = (res['Q1_data_sign']['verdict'] == 'PASS'
               and res['Q2_causal_AVA_silencing']['verdict'] == 'PASS'
               and res['Q3_threshold_valley']['verdict'] == 'PASS')
    res['overall'] = ('PASS_readout_validated_paper2_unblocked' if overall
                      else 'DEMOTE_readout_to_network_statistic_paper2_blocked')
    json.dump(res, open(OUT / 'p17_heavy_verdict.json', 'w'), indent=2, default=str)
    return res


MODEL_RATE_BINS = OUT / 'model_command_rate_bins.npy'


def _model_command_rate_valley():
    """Model's OWN command-rate Hz histogram valley (PART_B). Reads the
    per-bin command-rate artifact emitted by run_model_command_rate_bins()
    if available; else returns (nan, False) so PART_B is reported as
    NO-VALLEY rather than fabricated."""
    cand = [MODEL_RATE_BINS] if MODEL_RATE_BINS.exists() else \
        list((ANESTH / 'artifacts').rglob('*command_rate*bins*.npy'))
    if cand:
        rates = np.load(cand[0])
        return kde_valley(np.asarray(rates, float).flatten())
    return float('nan'), False


def _command_rate_bins_from_brain(brain, sim_duration_s, bin_dt_s=0.5,
                                  record_start_s=10.0):
    """Re-bin the per-bin command-set mean FIRING RATE (Hz) from a finished
    brain's spikes, reusing the frozen COMMAND_NEURONS set and compute_metrics'
    binning convention. Returns a 1-D array of per-bin command mean rates."""
    import brian2
    if hasattr(brain, 'spikes') and len(brain.spikes.t) > 0:
        st = np.asarray(brain.spikes.t / brian2.second)
        si = np.asarray(brain.spikes.i, dtype=int)
    else:
        return np.array([])
    names = brain.names
    cmd_idx = [i for i, n in enumerate(names) if n in COMMAND_NEURONS]
    if not cmd_idx:
        return np.array([])
    mask = (st >= record_start_s) & (st <= sim_duration_s)
    st, si = st[mask], si[mask]
    n_bins = int((sim_duration_s - record_start_s) / bin_dt_s)
    counts = np.zeros((len(names), n_bins))
    for nid, t in zip(si, st):
        if 0 <= nid < len(names):
            b = min(n_bins - 1, max(0, int((t - record_start_s) / bin_dt_s)))
            counts[nid, b] += 1
    rates = counts / bin_dt_s
    return rates[cmd_idx, :].mean(axis=0)


def run_model_command_rate_bins(doses_uM=None, seeds=(42, 137, 219),
                                sim_duration_s=40.0):
    """HEAVY (Brian2 LIF): build the worm SeededLIFBrain via run_single's path,
    sweep the V7 halothane dose ladder, and pool the per-bin command-set mean
    FIRING RATE (Hz) into one distribution ->
    artifacts/p17_readout_validity/model_command_rate_bins.npy.
    This is the model-OWN Hz histogram whose valley PART_B compares to 3.0 Hz.
    Runs (n_doses * n_seeds) short worm sims. NON-DESTRUCTIVE (new artifact).

    NOTE: re-builds the brain through run_single's exact construction
    (apply_genotype + apply_anesthetic with the frozen halothane profile) and
    re-bins command rates from its spikes; does NOT mutate any frozen operator."""
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ANESTH / 'src'))
    from state_validation.phase_g_state_validator import (
        apply_genotype, apply_anesthetic,
    )
    from state_validation.v7_random_ensemble import _get_full_halothane_profile
    profile = _get_full_halothane_profile('worm')
    if doses_uM is None:
        doses_uM = [0.0, 10.0, 30.0, 100.0, 200.0, 350.0, 500.0, 1000.0]
    pooled = []
    for dose in doses_uM:
        for sd in seeds:
            np.random.seed(sd)
            from brain.lif_brain import LIFBrain

            class SeededLIFBrain(LIFBrain):
                _brian2_seed = sd
            brain = SeededLIFBrain(use_per_edge_glu_signs=True)
            apply_genotype(brain, None, 1.0)
            apply_anesthetic(brain, profile, dose, 1.0)
            brain.run(sim_duration_s * 1000.0)
            cmr = _command_rate_bins_from_brain(brain, sim_duration_s)
            pooled.extend(list(np.asarray(cmr, float)))
            print(f'  dose={dose} seed={sd}: {len(cmr)} bins')
    pooled = np.asarray(pooled, float)
    np.save(MODEL_RATE_BINS, pooled)
    valley, bimod = kde_valley(pooled)
    print(f'  model command-rate bins: n={len(pooled)} valley={valley} bimodal={bimod}')
    return valley, bimod


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--heavy', action='store_true')
    ap.add_argument('--model-valley', action='store_true',
                    help='HEAVY Brian2 worm dose-ladder -> model command-rate bins (PART_B)')
    args = ap.parse_args()
    if args.model_valley:
        v, b = run_model_command_rate_bins()
        out = {'model_command_rate_valley_hz': v, 'model_bimodal': b,
               'artifact': str(MODEL_RATE_BINS)}
    elif args.heavy:
        out = run_heavy()
    else:
        out = run_fast()
    print(json.dumps(out, indent=2, default=str)[:4000])
