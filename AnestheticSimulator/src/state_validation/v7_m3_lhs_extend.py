"""v7_m3_lhs_extend — extend M3 LHS sensitivity to fly + mouse.

V7 §9.4 open item: the preregistered M3 LHS ran worm only (lead organism,
per the prereg's ~2,500-sim budget). This driver runs the identical LHS
protocol (100 joint ±50% samples, 8 doses × 3 seeds, frozen α) for fly and
mouse, then writes a combined three-organism CI summary.

This is an EXPLORATORY EXTENSION, not a preregistered gate. The original
`v7_sensitivity_verdict.json` (worm M3c) is left untouched — fly/mouse LHS
were explicitly scoped as optional follow-up in the prereg. The combined
summary is written to a separate artifact so the prereg-scoped verdict keeps
its meaning.

Performance note: this driver uses ONE persistent worker pool across all
samples (tasks tagged by sample index), instead of the per-sample pool
respawn in `v7_m3_sensitivity.run_lhs`. Each worker imports brian2 once
rather than once per sample — ~10× faster for the fly/mouse brains. The
LHS factor generation and EC50 fit are byte-identical to `run_lhs`, so the
numbers match the intended protocol.

Run: ml env (needs brian2 for fly/mouse brains).
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

ANESTH = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator')
sys.path.insert(0, str(ANESTH / 'src'))

from state_validation.v7_subset_search import ORG_CONFIG, N_WORKERS, CHUNKSIZE  # noqa: E402
from state_validation.v7_random_ensemble import (  # noqa: E402
    _get_full_halothane_profile, _profile_to_specs, _worker_random_ensemble,
)
from state_validation.phase_g_state_validator import hill_fit_ec50  # noqa: E402

DOSES_HALOTHANE = [10.0, 30.0, 100.0, 200.0, 350.0, 500.0, 1000.0, 3000.0]
SEEDS_M3 = [42, 137, 219]
PERTURB_FRAC = 0.5
LHS_N_SAMPLES = 100
LHS_SEED = 20260502  # matches v7_m3_sensitivity.run_lhs
ORGANISMS = ['fly', 'mouse']
PUB = {'worm': 340.0, 'fly': 340.0, 'mouse': 350.0}

OUT_DIR = ANESTH / 'artifacts' / 'v7_sensitivity'
WORM_CSV = OUT_DIR / 'v7_sensitivity_lhs.csv'


def _lhs_factors(n_dims: int, n_samples: int) -> np.ndarray:
    """Identical to run_lhs: LatinHypercube(seed=20260502) → ±50% factors."""
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_dims, seed=LHS_SEED)
        samples = sampler.random(n=n_samples)
    except ImportError:
        rng = np.random.default_rng(LHS_SEED)
        samples = rng.random((n_samples, n_dims))
    return (1.0 - PERTURB_FRAC) + samples * (2 * PERTURB_FRAC)


def _perturbed_specs(baseline_specs: dict, active: list, f_arr: np.ndarray) -> dict:
    ec50_factors = f_arr[:len(active)]
    max_factors = f_arr[len(active):]
    out = {}
    for j, cls in enumerate(active):
        ec50_b, max_b = baseline_specs[cls]
        out[cls] = (ec50_b * float(ec50_factors[j]),
                    max_b * float(max_factors[j]))
    return out


def run_lhs_pooled(organism: str, pool: mp.Pool, n_samples: int = LHS_N_SAMPLES) -> dict:
    """LHS for one organism using a shared persistent pool. Tasks across all
    samples are submitted at once; ensemble_id carries the sample index."""
    full_profile = _get_full_halothane_profile(organism)
    baseline_specs = _profile_to_specs(full_profile)
    active = list(baseline_specs.keys())
    n_dims = 2 * len(active)
    factors = _lhs_factors(n_dims, n_samples)

    print(f'\n=== M3 LHS (pooled) — {organism}, {n_samples} samples, '
          f'{n_dims} dims ±{int(PERTURB_FRAC*100)}% ===', flush=True)

    # Build every (sample, dose, seed) task up front.
    tasks = []
    for s_idx in range(n_samples):
        specs = _perturbed_specs(baseline_specs, active, factors[s_idx])
        for d in DOSES_HALOTHANE:
            for seed in SEEDS_M3:
                tasks.append((99, s_idx, organism, specs, d, seed))

    by_sample_dose = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    done = 0
    n_tasks = len(tasks)
    for r in pool.imap_unordered(_worker_random_ensemble, tasks, chunksize=CHUNKSIZE):
        _ml, s_idx, _o, dose, _seed, qf, _cr, _nr = r
        by_sample_dose[s_idx][dose].append(qf)
        done += 1
        if done % 240 == 0:
            el = (time.time() - t0) / 60.0
            eta = el / done * (n_tasks - done)
            print(f'  [{done}/{n_tasks} tasks]  {el:.1f}m  ETA {eta:.0f}m', flush=True)

    cfg = ORG_CONFIG[organism]
    pub = cfg['halothane_pub']
    rows = []
    for s_idx in range(n_samples):
        by_dose = by_sample_dose.get(s_idx, {})
        ds = sorted(by_dose.keys())
        if len(ds) < 2:
            ec50 = None
        else:
            qfs = np.array([float(np.mean(by_dose[d])) for d in ds])
            ec50 = hill_fit_ec50(np.array(ds), qfs, threshold=0.5)
        fold = max(ec50 / pub, pub / ec50) if ec50 else None
        rows.append({'sample_idx': s_idx, 'organism': organism,
                     'predicted_EC50_uM': ec50, 'fold_error': fold,
                     'ec50_factors': '|'.join(f'{f:.3f}' for f in factors[s_idx][:len(active)]),
                     'max_factors': '|'.join(f'{f:.3f}' for f in factors[s_idx][len(active):])})

    out_csv = OUT_DIR / f'v7_sensitivity_lhs_{organism}.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_idx', 'organism',
                           'predicted_EC50_uM', 'fold_error', 'ec50_factors', 'max_factors'])
        w.writeheader()
        w.writerows(rows)
    print(f'  wrote {out_csv}  ({(time.time()-t0)/60:.1f}m)', flush=True)
    return {'csv': out_csv}


def _ci_from_csv(path: Path) -> dict:
    vals = []
    with open(path) as f:
        for r in csv.DictReader(f):
            v = r.get('predicted_EC50_uM')
            if v not in (None, '', 'None'):
                vals.append(float(v))
    if not vals:
        return {'n_valid_samples': 0}
    return {'n_valid_samples': len(vals),
            'median_uM': float(np.median(vals)),
            '95pct_CI_low_uM': float(np.percentile(vals, 2.5)),
            '95pct_CI_high_uM': float(np.percentile(vals, 97.5))}


def main():
    with mp.Pool(processes=N_WORKERS) as pool:
        for org in ORGANISMS:
            run_lhs_pooled(org, pool, LHS_N_SAMPLES)

    summary = {
        'description': ('M3 LHS extended to fly + mouse (exploratory; not a '
                        'preregistered gate). 100 joint ±50% samples, 8 doses '
                        'x 3 seeds, frozen alpha. Worm is the preregistered '
                        'M3c case (see v7_sensitivity_verdict.json).'),
        'protocol': {'n_samples': LHS_N_SAMPLES, 'perturb_frac': PERTURB_FRAC,
                     'seeds': SEEDS_M3, 'lhs_seed': LHS_SEED},
        'organisms': {},
    }
    summary['organisms']['worm'] = _ci_from_csv(WORM_CSV)
    for org in ORGANISMS:
        summary['organisms'][org] = _ci_from_csv(OUT_DIR / f'v7_sensitivity_lhs_{org}.csv')
    for org, ci in summary['organisms'].items():
        if ci.get('median_uM') is not None:
            med = ci['median_uM']
            ci['published_uM'] = PUB[org]
            ci['median_fold_error'] = max(med / PUB[org], PUB[org] / med)
            ci['CI_width_uM'] = ci['95pct_CI_high_uM'] - ci['95pct_CI_low_uM']

    out = OUT_DIR / 'v7_sensitivity_lhs_allorganisms.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== Combined LHS summary: {out} ===')
    for org, ci in summary['organisms'].items():
        if ci.get('median_uM') is not None:
            print(f'  {org:6s} median {ci["median_uM"]:7.1f}  '
                  f'95% CI [{ci["95pct_CI_low_uM"]:7.1f}, {ci["95pct_CI_high_uM"]:7.1f}]  '
                  f'fold {ci["median_fold_error"]:.2f}  (pub {ci["published_uM"]:.0f})')


if __name__ == '__main__':
    main()
