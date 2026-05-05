"""v7_m4_cross_cal — V5 M4 anchor-swap calibration cross-validation.

Pre-registration: AnestheticSimulator/docs/v7_preregistration.md
Hash: 533b624a00b5ff9efecee41fb549fcb9cf5f02810aefcde55e14c7981fd09ff4

Protocol per organism:
  1. Hold out the original halothane MAC anchor.
  2. Re-calibrate α on isoflurane MAC instead (target iso EC50 ≈ 290 µM).
  3. Predict halothane EC50 with the new (iso-calibrated) α.
  4. Compare:
     - new α vs original α (M4a)
     - predicted halothane EC50 (using iso-α) vs published 340/350 µM (M4b)

Pre-registered predictions:
  M4a: new α within 30% of original α; falsifies if > 50% off.
  M4b: predicted halothane EC50 within 2× of published; falsifies if > 2.5×.

Compute: alpha sweep ~10 candidates × 8 doses × 3 seeds × 30s = 240 sims/iso
         then halothane prediction at locked iso-α: 8 doses × 3 seeds = 24 sims
         × 3 organisms × 2 anesthetics = ~250 sims. ~30 min on Pool(12).
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/rohit/Desktop/website/personalwebsite')
ANESTH = ROOT / 'AnestheticSimulator'
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from state_validation.v7_subset_search import (  # noqa: E402
    ORG_CONFIG, SIM_DUR_S, N_WORKERS, CHUNKSIZE, PREREG_HASH,
)

ALPHA_SWEEP = {
    'worm':  [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40],
    'fly':   [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20],
    'mouse': [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40],
}

DOSES_ISO = [10.0, 30.0, 100.0, 200.0, 290.0, 500.0, 1000.0, 3000.0]
DOSES_HALO = [10.0, 30.0, 100.0, 200.0, 350.0, 500.0, 1000.0, 3000.0]
ISO_TARGET_UM = 290.0
SEEDS_M4 = [42, 137, 219]

OUT_DIR = ANESTH / 'artifacts' / 'v7_cross_cal'


def _alpha_sweep_worker(args):
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.phase_g_state_validator import (
        load_perturbation_table, run_single,
    )
    from state_validation.v7_subset_search import _organism_runtime
    organism, anesthetic, alpha, dose, seed = args
    table_path, factory, qf_thr, cmd_set = _organism_runtime(organism)
    profiles = load_perturbation_table(table_path)
    if anesthetic not in profiles:
        return (organism, anesthetic, alpha, dose, seed, None, None, None)
    profile = profiles[anesthetic]
    m = run_single(
        anesthetic=anesthetic, dose_uM=dose, seed=seed,
        sim_duration_s=SIM_DUR_S, profile=profile, mutant=None,
        alpha_calib=alpha, brain_factory=factory,
        quiescent_threshold_hz=qf_thr, command_set=cmd_set,
    )
    return (organism, anesthetic, alpha, dose, seed,
            float(m['quiescent_fraction']),
            float(m['command_mean_firing_rate_hz']),
            float(m['network_mean_firing_rate_hz']))


def _hill_fit(by_dose):
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import hill_fit_ec50
    ds = sorted(by_dose.keys())
    qfs = [float(np.mean(by_dose[d])) for d in ds]
    return hill_fit_ec50(np.array(ds), np.array(qfs), threshold=0.5)


def calibrate_iso_alpha(organism: str) -> tuple[float, dict]:
    """Sweep α on isoflurane; pick the α whose predicted iso EC50 is closest to 290."""
    cfg = ORG_CONFIG[organism]
    alphas = ALPHA_SWEEP[organism]
    print(f'\n  iso α sweep ({organism}): {alphas}', flush=True)

    tasks = [(organism, 'isoflurane', a, d, s)
             for a in alphas for d in DOSES_ISO for s in SEEDS_M4]
    print(f'    {len(tasks)} sims', flush=True)
    t0 = time.time()
    results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for r in pool.imap_unordered(_alpha_sweep_worker, tasks, chunksize=CHUNKSIZE):
            results.append(r)
    print(f'    iso sweep wall {(time.time()-t0)/60:.1f}m', flush=True)

    # Aggregate per (alpha) → dose → qf list
    by_alpha = defaultdict(lambda: defaultdict(list))
    for org, anest, alpha, dose, seed, qf, _cr, _nr in results:
        if qf is None:
            continue
        by_alpha[alpha][dose].append(qf)

    fold_per_alpha = {}
    ec50_per_alpha = {}
    for alpha, dose_qf in by_alpha.items():
        ec50 = _hill_fit(dose_qf)
        ec50_per_alpha[alpha] = ec50
        if ec50 is None:
            fold_per_alpha[alpha] = None
        else:
            fold_per_alpha[alpha] = max(ec50 / ISO_TARGET_UM, ISO_TARGET_UM / ec50)

    best_alpha = None
    best_fold = float('inf')
    for a, fold in fold_per_alpha.items():
        if fold is None:
            continue
        if fold < best_fold:
            best_fold = fold
            best_alpha = a
    print(f'    iso-anchored α = {best_alpha} (fold {best_fold:.2f}, '
          f'predicted iso EC50 = {ec50_per_alpha.get(best_alpha)})', flush=True)
    return best_alpha, {
        'alphas_tested': alphas,
        'iso_ec50_per_alpha': {str(a): ec50_per_alpha[a] for a in alphas},
        'iso_fold_per_alpha': {str(a): fold_per_alpha[a] for a in alphas},
        'iso_best_alpha': best_alpha,
        'iso_best_fold': best_fold,
    }


def predict_halothane_at(organism: str, alpha: float) -> dict:
    """At given α, run halothane dose-response and return predicted EC50 + fold-error."""
    cfg = ORG_CONFIG[organism]
    pub = cfg['halothane_pub']

    tasks = [(organism, 'halothane', alpha, d, s)
             for d in DOSES_HALO for s in SEEDS_M4]
    print(f'    halothane prediction at α={alpha}: {len(tasks)} sims', flush=True)
    results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for r in pool.imap_unordered(_alpha_sweep_worker, tasks, chunksize=CHUNKSIZE):
            results.append(r)

    by_dose = defaultdict(list)
    for _o, _a, _al, dose, _s, qf, _cr, _nr in results:
        if qf is not None:
            by_dose[dose].append(qf)
    ec50 = _hill_fit(by_dose)
    if ec50 is None:
        fold = None
    else:
        fold = float(max(ec50 / pub, pub / ec50))
    return {
        'predicted_halothane_EC50_uM': float(ec50) if ec50 is not None else None,
        'fold_error_vs_published': fold,
        'published_halothane_EC50_uM': pub,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=== V7 M4 Cross-Calibration (anchor swap) ===\n')
    out = {
        'preregistration_hash': PREREG_HASH,
        'pipeline': 'v7_m4_cross_cal',
        'iso_target_uM': ISO_TARGET_UM,
        'per_organism': {},
    }

    for organism in ORG_CONFIG:
        cfg = ORG_CONFIG[organism]
        original_alpha = cfg['alpha']
        print(f'--- {organism} ---', flush=True)
        print(f'  Original (halothane-anchored) α = {original_alpha}', flush=True)

        iso_alpha, iso_diag = calibrate_iso_alpha(organism)
        halo_pred = predict_halothane_at(organism, iso_alpha)

        if iso_alpha is None:
            alpha_pct_diff = None
        else:
            alpha_pct_diff = (iso_alpha - original_alpha) / original_alpha * 100.0

        # M4a: alpha within 30%; falsifies > 50%
        if alpha_pct_diff is None:
            m4a_verdict = 'NO_ALPHA_FOUND'
        elif abs(alpha_pct_diff) <= 30.0:
            m4a_verdict = 'PASS'
        elif abs(alpha_pct_diff) <= 50.0:
            m4a_verdict = 'DEVIATION'
        else:
            m4a_verdict = 'FAIL'

        # M4b: halothane fold within 2x; falsifies > 2.5x
        fold = halo_pred.get('fold_error_vs_published')
        if fold is None:
            m4b_verdict = 'NO_EC50_FOUND'
        elif fold <= 2.0:
            m4b_verdict = 'PASS'
        elif fold <= 2.5:
            m4b_verdict = 'DEVIATION'
        else:
            m4b_verdict = 'FAIL'

        org_record = {
            'original_alpha': original_alpha,
            'iso_anchored_alpha': iso_alpha,
            'alpha_pct_diff': alpha_pct_diff,
            'iso_calibration_diagnostic': iso_diag,
            'halothane_at_iso_alpha': halo_pred,
            'M4a_alpha_within_30pct': m4a_verdict,
            'M4b_halothane_fold_within_2x': m4b_verdict,
        }
        out['per_organism'][organism] = org_record
        print(f'  α: {original_alpha} → {iso_alpha} ({alpha_pct_diff:.1f}%)  '
              f'M4a {m4a_verdict}', flush=True)
        print(f'  halothane EC50 at iso-α: '
              f'{halo_pred["predicted_halothane_EC50_uM"]} '
              f'(pub {halo_pred["published_halothane_EC50_uM"]}, '
              f'fold {fold})  M4b {m4b_verdict}\n', flush=True)

    with open(OUT_DIR / 'v7_cross_cal_verdict.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nVerdict: {OUT_DIR / "v7_cross_cal_verdict.json"}')


if __name__ == '__main__':
    main()
