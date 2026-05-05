"""v7_subset_search — Sub-Q2 Stage 1 minimum-sufficient mechanism subset search.

Pre-registration: AnestheticSimulator/docs/v7_preregistration.md
Hash recorded at: AnestheticSimulator/artifacts/v7_controls/V7_preregistration_hash.txt

Stage 1 protocol (locked):
  - 7 mech classes for worm + fly  → 127 non-empty subsets each
  - 6 mech classes for mouse        → 63 non-empty subsets
  - For each subset × organism: halothane Gate 1 dose-response at frozen α
    (8 doses × 5 seeds × 30s sims)
  - Hill-fit predicted EC50; pass = max(pred/pub, pub/pred) ≤ 2.0

Frozen α (V6 M0): worm 0.13, fly 0.060, mouse 0.10.
Published halothane MAC anchors: worm 340 µM, fly 340 µM, mouse 350 µM.

Outputs:
  artifacts/v7_subset_search/v7_stage1_halothane_raw.csv      — per-sim rows
  artifacts/v7_subset_search/v7_stage1_halothane.csv          — per-(org, subset) summary
  artifacts/v7_subset_search/v7_stage1_verdict.json           — pass-set + per-organism stats
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path('/home/rohit/Desktop/website/personalwebsite')
ANESTH = ROOT / 'AnestheticSimulator'
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


# ===== Mechanism classes per organism (frozen by V7 preregistration) =====

MECH_CLASSES_WORM = [
    'gaba_potentiation', 'k2p_potentiation', 'complex_i_block',
    'snare_cooperativity', 'nca_block', 'nachr_antagonism',
    'glucl_potentiation',
]
MECH_CLASSES_FLY = list(MECH_CLASSES_WORM)
MECH_CLASSES_MOUSE = [c for c in MECH_CLASSES_WORM if c != 'glucl_potentiation']

ORG_CONFIG = {
    'worm':  {'alpha': 0.13,  'halothane_pub': 340.0, 'mech_classes': MECH_CLASSES_WORM},
    'fly':   {'alpha': 0.060, 'halothane_pub': 340.0, 'mech_classes': MECH_CLASSES_FLY},
    'mouse': {'alpha': 0.10,  'halothane_pub': 350.0, 'mech_classes': MECH_CLASSES_MOUSE},
}

DOSES_HALOTHANE = [10.0, 30.0, 100.0, 200.0, 350.0, 500.0, 1000.0, 3000.0]
SEEDS = [42, 137, 219, 331, 443]
SIM_DUR_S = 30.0
PASS_FOLD_TOL = 2.0
N_WORKERS = 12
CHUNKSIZE = 4
PREREG_HASH = '533b624a00b5ff9efecee41fb549fcb9cf5f02810aefcde55e14c7981fd09ff4'


def enumerate_subsets(mech_classes):
    """Return list of all non-empty subsets as tuples (sorted within each subset)."""
    subsets = []
    for k in range(1, len(mech_classes) + 1):
        for combo in combinations(mech_classes, k):
            subsets.append(tuple(combo))
    return subsets


def _build_subset_profile(full_profile, subset_set):
    """Copy the full halothane profile, blanking out classes NOT in subset_set."""
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import PerturbationRow
    out = {}
    for cls, row in full_profile.items():
        if cls in subset_set:
            out[cls] = row
        else:
            out[cls] = PerturbationRow(
                anesthetic='halothane', mechanism_class=cls,
                target_EC50_uM=None, max_effect_factor=None, hill_n=row.hill_n,
                source_PMID=row.source_PMID, evidence_grade='SUBSET_DROPPED',
            )
    return out


def _organism_runtime(organism):
    """Return (table_path, brain_factory, qf_threshold, command_set) per organism."""
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    if organism == 'worm':
        from state_validation.phase_g_state_validator import QUIESCENT_RATE_THRESHOLD_HZ
        from brain.lif_brain import LIFBrain
        def factory(seed):
            class SeededLIF(LIFBrain):
                _brian2_seed = seed
            return SeededLIF(use_per_edge_glu_signs=True)
        return (
            ANESTH / 'data' / 'state_validation' / 'anesthetic_perturbation_table.csv',
            factory, QUIESCENT_RATE_THRESHOLD_HZ, None,
        )
    if organism == 'fly':
        from state_validation.fly_state_validator import (
            FLY_PERTURBATION_TABLE, FLY_QUIESCENT_THRESHOLD_HZ, make_fly_brain_factory,
        )
        return (FLY_PERTURBATION_TABLE, make_fly_brain_factory(),
                FLY_QUIESCENT_THRESHOLD_HZ, None)
    if organism == 'mouse':
        from state_validation.mouse_state_validator import (
            MOUSE_PERTURBATION_TABLE, MOUSE_QUIESCENT_THRESHOLD_HZ, make_mouse_brain_factory,
        )
        return (MOUSE_PERTURBATION_TABLE, make_mouse_brain_factory(),
                MOUSE_QUIESCENT_THRESHOLD_HZ, None)
    raise KeyError(organism)


# Per-process cache so we don't reload the perturbation table on every call.
_LOADED_PROFILE_CACHE: dict[str, dict] = {}


def _get_full_profile(organism: str):
    if organism in _LOADED_PROFILE_CACHE:
        return _LOADED_PROFILE_CACHE[organism]
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import load_perturbation_table
    table_path, _, _, _ = _organism_runtime(organism)
    profiles = load_perturbation_table(table_path)
    if 'halothane' not in profiles:
        raise KeyError(f'halothane not in {organism} table')
    _LOADED_PROFILE_CACHE[organism] = profiles['halothane']
    return profiles['halothane']


def _worker_v7_subset_stage1(args):
    """Run one (organism, subset_csv, dose, seed) sim; return tuple."""
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.phase_g_state_validator import run_single
    organism, subset_csv, dose, seed = args
    cfg = ORG_CONFIG[organism]
    _, factory, qf_thr, cmd_set = _organism_runtime(organism)
    full_profile = _get_full_profile(organism)
    subset_set = set(subset_csv.split('|')) if subset_csv else set()
    subset_profile = _build_subset_profile(full_profile, subset_set)
    m = run_single(
        anesthetic='halothane', dose_uM=dose, seed=seed,
        sim_duration_s=SIM_DUR_S, profile=subset_profile, mutant=None,
        alpha_calib=cfg['alpha'], brain_factory=factory,
        quiescent_threshold_hz=qf_thr, command_set=cmd_set,
    )
    return (organism, subset_csv, dose, seed,
            float(m['quiescent_fraction']),
            float(m['command_mean_firing_rate_hz']),
            float(m['network_mean_firing_rate_hz']))


def main(smoke: bool = False):
    out_dir = ANESTH / 'artifacts' / 'v7_subset_search'
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_index: dict[str, list[tuple[str, ...]]] = {}
    for organism, cfg in ORG_CONFIG.items():
        subset_index[organism] = enumerate_subsets(cfg['mech_classes'])

    if smoke:
        # Tiny sanity check: 1 subset per organism, 2 doses, 1 seed.
        subset_index = {org: [tuple(cfg['mech_classes'])] for org, cfg in ORG_CONFIG.items()}
        doses = [200.0, 1000.0]
        seeds = [42]
    else:
        doses = DOSES_HALOTHANE
        seeds = SEEDS

    tasks: list[tuple[str, str, float, int]] = []
    for organism, subsets in subset_index.items():
        for subset in subsets:
            subset_csv = '|'.join(subset)
            for dose in doses:
                for seed in seeds:
                    tasks.append((organism, subset_csv, dose, seed))

    n_subsets = sum(len(v) for v in subset_index.values())
    print(f'V7 Sub-Q2 Stage 1: {n_subsets} subsets total '
          f'({len(subset_index["worm"])} worm + {len(subset_index["fly"])} fly + '
          f'{len(subset_index["mouse"])} mouse)', flush=True)
    print(f'  doses={len(doses)}  seeds={len(seeds)}  sim={SIM_DUR_S}s  '
          f'workers={N_WORKERS}', flush=True)
    print(f'  Total {len(tasks)} sims', flush=True)
    print(f'  Frozen α: worm {ORG_CONFIG["worm"]["alpha"]}  '
          f'fly {ORG_CONFIG["fly"]["alpha"]}  '
          f'mouse {ORG_CONFIG["mouse"]["alpha"]}', flush=True)

    t0 = time.time()
    raw_path = out_dir / ('v7_stage1_halothane_raw.csv' if not smoke
                          else 'v7_stage1_halothane_smoke_raw.csv')
    results: list[tuple] = []
    with open(raw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['organism', 'subset', 'dose_uM', 'seed',
                         'quiescent_fraction', 'command_rate_hz', 'network_rate_hz'])
        with mp.Pool(processes=N_WORKERS) as pool:
            iterator = pool.imap_unordered(
                _worker_v7_subset_stage1, tasks, chunksize=CHUNKSIZE,
            )
            for i, r in enumerate(iterator):
                results.append(r)
                writer.writerow(r)
                if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                    elapsed_min = (time.time() - t0) / 60.0
                    eta_min = elapsed_min / (i + 1) * (len(tasks) - (i + 1))
                    f.flush()
                    print(f'    [{i+1}/{len(tasks)}]  {elapsed_min:.1f}m elapsed  '
                          f'ETA {eta_min:.0f}m', flush=True)

    # Aggregate
    import numpy as np
    from state_validation.phase_g_state_validator import hill_fit_ec50
    agg: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for org, subset_csv, dose, seed, qf, _cr, _nr in results:
        agg[(org, subset_csv)][dose].append(qf)

    summary_rows: list[dict] = []
    for (org, subset_csv), dose_qf in agg.items():
        cfg = ORG_CONFIG[org]
        ds = sorted(dose_qf.keys())
        qfs = [float(np.mean(dose_qf[d])) for d in ds]
        ec50 = hill_fit_ec50(np.array(ds), np.array(qfs), threshold=0.5)
        if ec50 is None:
            fold_err = None
            passes = False
        else:
            pub = cfg['halothane_pub']
            fold_err = float(max(ec50 / pub, pub / ec50))
            passes = fold_err <= PASS_FOLD_TOL
        n_classes = len(subset_csv.split('|')) if subset_csv else 0
        row = {
            'organism': org,
            'subset': subset_csv,
            'n_classes': n_classes,
            'predicted_EC50_uM': float(ec50) if ec50 is not None else None,
            'published_EC50_uM': cfg['halothane_pub'],
            'fold_error': fold_err,
            'passes_stage1': passes,
        }
        for d, q in zip(ds, qfs):
            row[f'qf_dose_{d:.0f}'] = q
        summary_rows.append(row)

    summary_path = out_dir / ('v7_stage1_halothane.csv' if not smoke
                              else 'v7_stage1_halothane_smoke.csv')
    all_keys = set()
    for r in summary_rows:
        all_keys.update(r.keys())
    static_keys = ['organism', 'subset', 'n_classes', 'predicted_EC50_uM',
                   'published_EC50_uM', 'fold_error', 'passes_stage1']
    dose_keys = sorted(
        (k for k in all_keys if k.startswith('qf_dose_')),
        key=lambda k: float(k.split('_')[2]),
    )
    fieldnames = static_keys + dose_keys
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    verdict = {
        'stage': 'stage1_halothane',
        'preregistration_hash': PREREG_HASH,
        'pass_criterion': f'max(predicted/published, published/predicted) <= {PASS_FOLD_TOL}',
        'sim_duration_s': SIM_DUR_S,
        'doses_halothane_uM': doses,
        'seeds': seeds,
        'workers': N_WORKERS,
        'n_total_subsets_tested': n_subsets,
        'n_total_sims': len(tasks),
        'wall_minutes': (time.time() - t0) / 60.0,
        'per_organism': {},
    }
    for org in ORG_CONFIG:
        org_rows = [r for r in summary_rows if r['organism'] == org]
        passers = [r for r in org_rows if r['passes_stage1']]
        size_hist = {
            str(k): sum(1 for r in passers if r['n_classes'] == k)
            for k in range(1, len(ORG_CONFIG[org]['mech_classes']) + 1)
        }
        verdict['per_organism'][org] = {
            'alpha_frozen': ORG_CONFIG[org]['alpha'],
            'halothane_pub_uM': ORG_CONFIG[org]['halothane_pub'],
            'n_subsets_tested': len(org_rows),
            'n_subsets_passing': len(passers),
            'min_passing_subset_size': (
                min((r['n_classes'] for r in passers), default=None)
            ),
            'passing_subsets_by_size': size_hist,
        }

    verdict_path = out_dir / ('v7_stage1_verdict.json' if not smoke
                              else 'v7_stage1_smoke_verdict.json')
    with open(verdict_path, 'w') as f:
        json.dump(verdict, f, indent=2)

    print(f'\n=== V7 Sub-Q2 Stage 1 complete ===')
    print(f'  wall:    {verdict["wall_minutes"]:.1f} min')
    print(f'  raw:     {raw_path}')
    print(f'  summary: {summary_path}')
    print(f'  verdict: {verdict_path}')
    for org, v in verdict['per_organism'].items():
        print(f'  {org:5s}: {v["n_subsets_passing"]}/{v["n_subsets_tested"]} pass; '
              f'min size {v["min_passing_subset_size"]}; '
              f'by-size {v["passing_subsets_by_size"]}')


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    main(smoke=smoke)
