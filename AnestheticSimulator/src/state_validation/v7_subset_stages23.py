"""v7_subset_stages23 — Sub-Q2 Stage 2 (isoflurane held-out) + Stage 3 (Eger).

Pre-registration: AnestheticSimulator/docs/v7_preregistration.md
Hash: 533b624a00b5ff9efecee41fb549fcb9cf5f02810aefcde55e14c7981fd09ff4

Stage 2:
  - Read Stage 1 passers from artifacts/v7_subset_search/v7_stage1_halothane.csv
  - For each passer × organism: isoflurane Gate 2 dose-response at frozen α
    (8 doses × 5 seeds × 30s sims)
  - Pass = max(pred/290, 290/pred) ≤ 2.0 (isoflurane published EC50 = 290 µM
    for all three organisms per V6 anchors)

Stage 3:
  - Read Stage 2 passers
  - For each passer × organism: 3 Eger compounds (cis-DCE, trans-DCE,
    hexafluoroethane) × 7 doses × 5 seeds × 30s sims
  - Pass = cis_DCE_max_qf ≥ 0.5 AND trans_DCE_max_qf < 0.5 AND
    hexafluoroethane_max_qf < 0.5

Outputs:
  artifacts/v7_subset_search/v7_stage2_isoflurane{_raw,}.csv + verdict
  artifacts/v7_subset_search/v7_stage3_eger{_raw,}.csv + verdict
  artifacts/v7_subset_search/v7_subset_verdict.json (final, all-stages)
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/rohit/Desktop/website/personalwebsite')
ANESTH = ROOT / 'AnestheticSimulator'
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from state_validation.v7_subset_search import (  # noqa: E402
    ORG_CONFIG, SIM_DUR_S, SEEDS, N_WORKERS, CHUNKSIZE, PREREG_HASH,
    PASS_FOLD_TOL, _organism_runtime, _build_subset_profile,
)

ISOFLURANE_PUB_UM = {'worm': 290.0, 'fly': 290.0, 'mouse': 290.0}
DOSES_ISOFLURANE = [10.0, 30.0, 100.0, 200.0, 290.0, 500.0, 1000.0, 3000.0]

EGER_COMPOUNDS = ['cis_12_dichloroethylene', 'trans_12_dichloroethylene',
                  'hexafluoroethane']
DOSES_EGER = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]
EGER_QF_THRESHOLD_CIS = 0.5    # cis-DCE must reach this
EGER_QF_THRESHOLD_NON = 0.5    # trans-DCE / hexafluoroethane must stay below

OUT_DIR = ANESTH / 'artifacts' / 'v7_subset_search'

# Per-process cache — same pattern as v7_subset_search
_LOADED_PROFILE_CACHE: dict[tuple[str, str], dict] = {}


def _get_full_profile_for(organism: str, anesthetic: str):
    key = (organism, anesthetic)
    if key in _LOADED_PROFILE_CACHE:
        return _LOADED_PROFILE_CACHE[key]
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import load_perturbation_table
    table_path, _, _, _ = _organism_runtime(organism)
    profiles = load_perturbation_table(table_path)
    if anesthetic not in profiles:
        raise KeyError(f'{anesthetic} not in {organism} table at {table_path}')
    _LOADED_PROFILE_CACHE[key] = profiles[anesthetic]
    return profiles[anesthetic]


def _worker_v7_subset_anesthetic(args):
    """Run one (organism, subset_csv, anesthetic, dose, seed) sim."""
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.phase_g_state_validator import run_single
    organism, subset_csv, anesthetic, dose, seed = args
    cfg = ORG_CONFIG[organism]
    _, factory, qf_thr, cmd_set = _organism_runtime(organism)
    full_profile = _get_full_profile_for(organism, anesthetic)
    subset_set = set(subset_csv.split('|')) if subset_csv else set()
    subset_profile = _build_subset_profile(full_profile, subset_set)
    m = run_single(
        anesthetic=anesthetic, dose_uM=dose, seed=seed,
        sim_duration_s=SIM_DUR_S, profile=subset_profile, mutant=None,
        alpha_calib=cfg['alpha'], brain_factory=factory,
        quiescent_threshold_hz=qf_thr, command_set=cmd_set,
    )
    return (organism, subset_csv, anesthetic, dose, seed,
            float(m['quiescent_fraction']),
            float(m['command_mean_firing_rate_hz']),
            float(m['network_mean_firing_rate_hz']))


def _read_stage1_passers() -> dict[str, list[str]]:
    path = OUT_DIR / 'v7_stage1_halothane.csv'
    if not path.exists():
        raise FileNotFoundError(f'Stage 1 summary missing: {path}')
    passers: dict[str, list[str]] = defaultdict(list)
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get('passes_stage1', '').strip().lower() in ('true', '1'):
                passers[row['organism']].append(row['subset'])
    return dict(passers)


def _read_stage2_passers() -> dict[str, list[str]]:
    path = OUT_DIR / 'v7_stage2_isoflurane.csv'
    if not path.exists():
        raise FileNotFoundError(f'Stage 2 summary missing: {path}')
    passers: dict[str, list[str]] = defaultdict(list)
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get('passes_stage2', '').strip().lower() in ('true', '1'):
                passers[row['organism']].append(row['subset'])
    return dict(passers)


def run_stage2():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    passers = _read_stage1_passers()
    n_pass = sum(len(v) for v in passers.values())
    print(f'\n=== V7 Sub-Q2 Stage 2 — isoflurane held-out ===')
    for org, lst in passers.items():
        print(f'  {org}: {len(lst)} Stage 1 passers')
    print(f'  Total: {n_pass} subsets to test on isoflurane')

    tasks = []
    for org, subset_list in passers.items():
        for subset_csv in subset_list:
            for d in DOSES_ISOFLURANE:
                for s in SEEDS:
                    tasks.append((org, subset_csv, 'isoflurane', d, s))
    print(f'  {len(DOSES_ISOFLURANE)} doses × {len(SEEDS)} seeds = '
          f'{len(tasks)} total sims')

    t0 = time.time()
    raw_path = OUT_DIR / 'v7_stage2_isoflurane_raw.csv'
    results: list[tuple] = []
    with open(raw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['organism', 'subset', 'anesthetic', 'dose_uM', 'seed',
                         'quiescent_fraction', 'command_rate_hz', 'network_rate_hz'])
        with mp.Pool(processes=N_WORKERS) as pool:
            for i, r in enumerate(pool.imap_unordered(
                    _worker_v7_subset_anesthetic, tasks, chunksize=CHUNKSIZE)):
                results.append(r)
                writer.writerow(r)
                if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                    e = (time.time() - t0) / 60.0
                    eta = e / (i + 1) * (len(tasks) - (i + 1))
                    f.flush()
                    print(f'    [{i+1}/{len(tasks)}]  {e:.1f}m  ETA {eta:.0f}m',
                          flush=True)

    import numpy as np
    from state_validation.phase_g_state_validator import hill_fit_ec50

    agg = defaultdict(lambda: defaultdict(list))
    for org, subset_csv, _a, dose, seed, qf, _cr, _nr in results:
        agg[(org, subset_csv)][dose].append(qf)

    summary = []
    for (org, subset_csv), dose_qf in agg.items():
        ds = sorted(dose_qf.keys())
        qfs = [float(np.mean(dose_qf[d])) for d in ds]
        ec50 = hill_fit_ec50(np.array(ds), np.array(qfs), threshold=0.5)
        pub = ISOFLURANE_PUB_UM[org]
        if ec50 is None:
            fold_err, passes = None, False
        else:
            fold_err = float(max(ec50 / pub, pub / ec50))
            passes = fold_err <= PASS_FOLD_TOL
        n_classes = len(subset_csv.split('|')) if subset_csv else 0
        row = {
            'organism': org, 'subset': subset_csv, 'n_classes': n_classes,
            'predicted_iso_EC50_uM': float(ec50) if ec50 is not None else None,
            'published_iso_EC50_uM': pub,
            'fold_error': fold_err, 'passes_stage2': passes,
        }
        for d, q in zip(ds, qfs):
            row[f'qf_dose_{d:.0f}'] = q
        summary.append(row)

    static_keys = ['organism', 'subset', 'n_classes', 'predicted_iso_EC50_uM',
                   'published_iso_EC50_uM', 'fold_error', 'passes_stage2']
    all_keys = set()
    for r in summary:
        all_keys.update(r.keys())
    dose_keys = sorted((k for k in all_keys if k.startswith('qf_dose_')),
                       key=lambda k: float(k.split('_')[2]))
    summary_path = OUT_DIR / 'v7_stage2_isoflurane.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=static_keys + dose_keys, extrasaction='ignore')
        w.writeheader()
        for r in summary:
            w.writerow(r)

    verdict = {
        'stage': 'stage2_isoflurane',
        'preregistration_hash': PREREG_HASH,
        'sim_duration_s': SIM_DUR_S,
        'doses_iso_uM': DOSES_ISOFLURANE,
        'seeds': SEEDS,
        'pass_criterion': f'max(pred/{ISOFLURANE_PUB_UM["worm"]}, '
                          f'pub/pred) <= {PASS_FOLD_TOL}',
        'wall_minutes': (time.time() - t0) / 60.0,
        'per_organism': {
            org: {
                'n_stage1_passers_tested': len([r for r in summary if r['organism'] == org]),
                'n_stage2_passers': len([r for r in summary
                                          if r['organism'] == org and r['passes_stage2']]),
            } for org in ORG_CONFIG
        },
    }
    with open(OUT_DIR / 'v7_stage2_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2)

    print(f'\n=== Stage 2 complete ===  wall {verdict["wall_minutes"]:.1f}m')
    for org, v in verdict['per_organism'].items():
        print(f'  {org}: {v["n_stage2_passers"]}/{v["n_stage1_passers_tested"]} pass iso')


def run_stage3():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    passers = _read_stage2_passers()
    n_pass = sum(len(v) for v in passers.values())
    print(f'\n=== V7 Sub-Q2 Stage 3 — Eger non-immobilizers ===')
    for org, lst in passers.items():
        print(f'  {org}: {len(lst)} Stage 2 passers')
    print(f'  {len(EGER_COMPOUNDS)} compounds × {len(DOSES_EGER)} doses × '
          f'{len(SEEDS)} seeds = {n_pass * 105} sims')

    tasks = []
    for org, subset_list in passers.items():
        for subset_csv in subset_list:
            for compound in EGER_COMPOUNDS:
                for d in DOSES_EGER:
                    for s in SEEDS:
                        tasks.append((org, subset_csv, compound, d, s))
    print(f'  Total: {len(tasks)} sims')

    t0 = time.time()
    raw_path = OUT_DIR / 'v7_stage3_eger_raw.csv'
    results: list[tuple] = []
    with open(raw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['organism', 'subset', 'compound', 'dose_uM', 'seed',
                         'quiescent_fraction', 'command_rate_hz', 'network_rate_hz'])
        with mp.Pool(processes=N_WORKERS) as pool:
            for i, r in enumerate(pool.imap_unordered(
                    _worker_v7_subset_anesthetic, tasks, chunksize=CHUNKSIZE)):
                results.append(r)
                writer.writerow(r)
                if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                    e = (time.time() - t0) / 60.0
                    eta = e / (i + 1) * (len(tasks) - (i + 1))
                    f.flush()
                    print(f'    [{i+1}/{len(tasks)}]  {e:.1f}m  ETA {eta:.0f}m',
                          flush=True)

    # Aggregate: per (org, subset, compound), max qf across doses (averaged across seeds per dose)
    import numpy as np
    by_compound = defaultdict(lambda: defaultdict(list))  # (org, subset, compound) → dose → list[qf]
    for org, subset_csv, compound, dose, seed, qf, _cr, _nr in results:
        by_compound[(org, subset_csv, compound)][dose].append(qf)

    # For each (org, subset), pick max qf per compound across doses (averaging seeds)
    per_subset: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (org, subset_csv, compound), dose_qf in by_compound.items():
        max_qf = 0.0
        for d, qfs in dose_qf.items():
            mean_qf = float(np.mean(qfs))
            if mean_qf > max_qf:
                max_qf = mean_qf
        per_subset[(org, subset_csv)][compound] = max_qf

    summary = []
    for (org, subset_csv), comp_max in per_subset.items():
        cis_qf = comp_max.get('cis_12_dichloroethylene', 0.0)
        trans_qf = comp_max.get('trans_12_dichloroethylene', 0.0)
        hex_qf = comp_max.get('hexafluoroethane', 0.0)
        passes = (cis_qf >= EGER_QF_THRESHOLD_CIS
                  and trans_qf < EGER_QF_THRESHOLD_NON
                  and hex_qf < EGER_QF_THRESHOLD_NON)
        n_classes = len(subset_csv.split('|')) if subset_csv else 0
        summary.append({
            'organism': org, 'subset': subset_csv, 'n_classes': n_classes,
            'cis_DCE_max_qf': cis_qf,
            'trans_DCE_max_qf': trans_qf,
            'hexafluoroethane_max_qf': hex_qf,
            'passes_stage3': passes,
        })

    summary_path = OUT_DIR / 'v7_stage3_eger.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=['organism', 'subset', 'n_classes',
                            'cis_DCE_max_qf', 'trans_DCE_max_qf',
                            'hexafluoroethane_max_qf', 'passes_stage3'])
        w.writeheader()
        for r in summary:
            w.writerow(r)

    verdict = {
        'stage': 'stage3_eger',
        'preregistration_hash': PREREG_HASH,
        'sim_duration_s': SIM_DUR_S,
        'doses_eger_uM': DOSES_EGER,
        'compounds': EGER_COMPOUNDS,
        'seeds': SEEDS,
        'pass_criterion': (
            f'cis_DCE_max_qf >= {EGER_QF_THRESHOLD_CIS} AND '
            f'trans_DCE_max_qf < {EGER_QF_THRESHOLD_NON} AND '
            f'hexafluoroethane_max_qf < {EGER_QF_THRESHOLD_NON}'
        ),
        'wall_minutes': (time.time() - t0) / 60.0,
        'per_organism': {
            org: {
                'n_stage2_passers_tested': len([r for r in summary if r['organism'] == org]),
                'n_stage3_passers': len([r for r in summary
                                          if r['organism'] == org and r['passes_stage3']]),
            } for org in ORG_CONFIG
        },
    }
    with open(OUT_DIR / 'v7_stage3_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2)

    print(f'\n=== Stage 3 complete ===  wall {verdict["wall_minutes"]:.1f}m')
    for org, v in verdict['per_organism'].items():
        print(f'  {org}: {v["n_stage3_passers"]}/{v["n_stage2_passers_tested"]} '
              f'pass Eger')


def write_final_subset_verdict():
    """Combine Stages 1-3 into a single all-stages verdict + redundancy analysis."""
    s1_path = OUT_DIR / 'v7_stage1_halothane.csv'
    s2_path = OUT_DIR / 'v7_stage2_isoflurane.csv'
    s3_path = OUT_DIR / 'v7_stage3_eger.csv'

    def _read(path, key):
        with open(path) as f:
            return list(csv.DictReader(f))

    s1 = _read(s1_path, 'passes_stage1')
    s2 = _read(s2_path, 'passes_stage2') if s2_path.exists() else []
    s3 = _read(s3_path, 'passes_stage3') if s3_path.exists() else []

    s1_pass = {(r['organism'], r['subset']) for r in s1
                if r.get('passes_stage1', '').lower() in ('true', '1')}
    s2_pass = {(r['organism'], r['subset']) for r in s2
                if r.get('passes_stage2', '').lower() in ('true', '1')}
    s3_pass = {(r['organism'], r['subset']) for r in s3
                if r.get('passes_stage3', '').lower() in ('true', '1')}

    final_pass = s1_pass & s2_pass & s3_pass

    # Redundancy: find necessary classes per organism (in 100% of all-pass subsets)
    necessary = {}
    sufficient = {}
    smallest_subset_size = {}
    for org in ORG_CONFIG:
        org_subsets = [s.split('|') for (o, s) in final_pass if o == org]
        all_classes = set()
        for s in org_subsets:
            all_classes.update(s)
        if not org_subsets:
            necessary[org] = []
            sufficient[org] = []
            smallest_subset_size[org] = None
            continue
        nec = [c for c in all_classes
               if all(c in subset for subset in org_subsets)]
        suff = sorted(all_classes)
        necessary[org] = sorted(nec)
        sufficient[org] = suff
        smallest_subset_size[org] = min(len(s) for s in org_subsets)

    final_verdict = {
        'preregistration_hash': PREREG_HASH,
        'pipeline': 'v7_subq2_subset_search',
        'stage_counts': {
            'stage1_pass': len(s1_pass),
            'stage2_pass': len(s2_pass),
            'stage3_pass': len(s3_pass),
            'all_stages_pass': len(final_pass),
        },
        'per_organism': {
            org: {
                'n_all_stages_passers': len([1 for (o, s) in final_pass if o == org]),
                'smallest_subset_size': smallest_subset_size[org],
                'necessary_classes_100pct': necessary[org],
                'sufficient_classes_at_least_one_passer': sufficient[org],
            } for org in ORG_CONFIG
        },
        'all_stages_passing_subsets': sorted(
            [{'organism': o, 'subset': s} for (o, s) in final_pass],
            key=lambda d: (d['organism'], d['subset']),
        ),
    }
    with open(OUT_DIR / 'v7_subset_verdict.json', 'w') as f:
        json.dump(final_verdict, f, indent=2)
    print(f'\nFinal verdict written: {OUT_DIR / "v7_subset_verdict.json"}')
    for org, v in final_verdict['per_organism'].items():
        print(f'  {org}: {v["n_all_stages_passers"]} all-stage passers; '
              f'min size {v["smallest_subset_size"]}; '
              f'necessary {v["necessary_classes_100pct"]}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: v7_subset_stages23.py {stage2|stage3|verdict|all}')
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == 'stage2':
        run_stage2()
    elif cmd == 'stage3':
        run_stage3()
    elif cmd == 'verdict':
        write_final_subset_verdict()
    elif cmd == 'all':
        run_stage2()
        run_stage3()
        write_final_subset_verdict()
    else:
        print(f'Unknown cmd: {cmd}')
        sys.exit(1)
