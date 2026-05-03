"""V4 fly ensemble runner — parallel across cores, 60s sim × 5 seeds.

Runs Gates 1-4 in one combined parallel sweep on the FlyLarvaBrain (Winding 2023).
Output structure mirrors worm V3 v3_ensemble for direct comparison.

α = 0.060 (calibrated by fly_m3_alpha_sweep on halothane WT).
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path('/home/rohit/Desktop/website/personalwebsite')
ANESTH = ROOT / 'AnestheticSimulator'
SIMV = ANESTH / 'src'
sys.path.insert(0, str(SIMV))
sys.path.insert(0, str(ROOT / 'scripts'))


def _worker(args):
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.fly_state_validator import run_fly_single
    anest, dose, seed, mutant_gene, alpha, sim_dur = args
    metrics = run_fly_single(
        anesthetic=anest, dose_uM=dose, seed=seed,
        sim_duration_s=sim_dur, mutant_gene=mutant_gene, alpha_calib=alpha,
    )
    metrics['mutant_gene'] = mutant_gene
    return metrics


# ===== task list =====
ALPHA = 0.060
SIM_DUR = 60.0
SEEDS = [42, 137, 219, 331, 443]
DOSES_VOLATILE = [30.0, 100.0, 200.0, 300.0, 340.0, 500.0, 1000.0, 2000.0]
DOSES_EGER = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]

VOLATILES = ['halothane', 'isoflurane']
HYPER_MUTANTS = ['Syx1A', 'unc-13', 'syt1', 'nSyb', 'na', 'dunc-79', 'dunc-80', 'ND-49', 'ND-75']
RESIST_MUTANTS = ['Sandman', 'ORK1', 'Goα47A', 'rdgA']
EGER_COMPOUNDS = ['cis_12_dichloroethylene', 'trans_12_dichloroethylene', 'hexafluoroethane']


def build_tasks():
    tasks = []
    for anest in VOLATILES:
        for d in DOSES_VOLATILE:
            for s in SEEDS:
                tasks.append((anest, d, s, 'WT', ALPHA, SIM_DUR))
    for g in HYPER_MUTANTS + RESIST_MUTANTS:
        for d in DOSES_VOLATILE:
            for s in SEEDS:
                tasks.append(('halothane', d, s, g, ALPHA, SIM_DUR))
    for c in EGER_COMPOUNDS:
        for d in DOSES_EGER:
            for s in SEEDS:
                tasks.append((c, d, s, 'WT', ALPHA, SIM_DUR))
    return tasks


def main():
    out_dir = ANESTH / 'artifacts' / 'state_validation_fly' / 'v4_ensemble'
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    print(f'V4 fly ensemble — {len(tasks)} sims  (α={ALPHA}, sim={SIM_DUR}s × {len(SEEDS)} seeds)')
    print(f'Workers: 8 of 16 cores')
    print()

    t_start = time.time()
    results = []
    with mp.Pool(processes=8) as pool:
        for i, m in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2)):
            results.append(m)
            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (len(tasks) - (i + 1))
                print(f'  [{i+1:>4d}/{len(tasks)}] {100*(i+1)/len(tasks):.0f}%  '
                      f'elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min')

    elapsed = time.time() - t_start
    print(f'\nAll {len(tasks)} sims complete in {elapsed/60:.1f} min')

    # Save raw rows
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(out_dir / 'all_results_raw.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in fieldnames})

    # === Aggregate per (anesthetic, mutant, dose) ===
    from collections import defaultdict
    import numpy as np
    agg = defaultdict(list)
    for r in results:
        agg[(r['anesthetic'], r['mutant_gene'], r['dose_uM'])].append(r)

    summary = []
    for (anest, mut, dose), runs in sorted(agg.items()):
        qf = np.array([x['quiescent_fraction'] for x in runs])
        cmd = np.array([x['command_mean_firing_rate_hz'] for x in runs])
        net = np.array([x['network_mean_firing_rate_hz'] for x in runs])
        summary.append({
            'anesthetic': anest, 'mutant_gene': mut, 'dose_uM': dose,
            'n_seeds': len(runs),
            'qf_mean': float(qf.mean()), 'qf_sd': float(qf.std(ddof=1)),
            'cmd_rate_mean': float(cmd.mean()),
            'net_rate_mean': float(net.mean()),
        })

    with open(out_dir / 'dose_response_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # === Hill fit per (anesthetic, mutant) ===
    from state_validation.phase_g_state_validator import hill_fit_ec50
    by_key = defaultdict(list)
    for r in summary:
        by_key[(r['anesthetic'], r['mutant_gene'])].append(r)
    ec50s = []
    for (anest, mut), rows in sorted(by_key.items()):
        rows = sorted(rows, key=lambda x: x['dose_uM'])
        doses = np.array([r['dose_uM'] for r in rows])
        qfs = np.array([r['qf_mean'] for r in rows])
        ec50 = hill_fit_ec50(doses, qfs, threshold=0.5)
        ec50s.append({'anesthetic': anest, 'mutant_gene': mut,
                      'predicted_EC50_uM': ec50, 'max_qf': float(qfs.max())})

    with open(out_dir / 'ec50_predictions.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ec50s[0].keys()))
        w.writeheader()
        w.writerows(ec50s)

    # === Verdict ===
    halothane_wt = next((r for r in ec50s if r['anesthetic'] == 'halothane' and r['mutant_gene'] == 'WT'), None)
    iso_wt = next((r for r in ec50s if r['anesthetic'] == 'isoflurane' and r['mutant_gene'] == 'WT'), None)

    verdict = {'alpha_calib': ALPHA, 'sim_duration_s': SIM_DUR, 'n_seeds': len(SEEDS), 'gates': {}}

    if halothane_wt and halothane_wt['predicted_EC50_uM']:
        ec50 = halothane_wt['predicted_EC50_uM']
        err = max(ec50 / 340.0, 340.0 / ec50)
        verdict['gates']['gate1_halothane_wt'] = {
            'predicted_EC50_uM': ec50, 'published_EC50_uM': 340.0, 'fold_error': err,
            'PASS': err <= 2.0,
        }
    if iso_wt and iso_wt['predicted_EC50_uM']:
        ec50 = iso_wt['predicted_EC50_uM']
        err = max(ec50 / 290.0, 290.0 / ec50)
        verdict['gates']['gate2_iso_wt'] = {
            'predicted_EC50_uM': ec50, 'published_EC50_uM': 290.0, 'fold_error': err,
            'PASS': err <= 2.0,
        }

    # Gate 3 — mutant directional
    if halothane_wt and halothane_wt['predicted_EC50_uM']:
        wt_ec50 = halothane_wt['predicted_EC50_uM']
        muts_table = {}
        with open(ANESTH / 'data' / 'state_validation_fly' / 'fly_mutant_baseline_perturbations.csv') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '': continue
                if line.startswith('gene,'): continue
                parts = next(csv.reader([line]))
                muts_table[parts[0]] = {'direction': parts[1], 'lit_ratio': parts[7] if len(parts) > 7 else ''}
        mutant_results = []
        for r in ec50s:
            if r['anesthetic'] != 'halothane' or r['mutant_gene'] in ('WT',):
                continue
            mut_data = muts_table.get(r['mutant_gene'])
            if not mut_data: continue
            ratio = (r['predicted_EC50_uM'] / wt_ec50) if r['predicted_EC50_uM'] else None
            expected = mut_data['direction']
            if ratio is None:
                dir_correct = False
            elif expected == 'HYPER':
                dir_correct = ratio < 1.0
            elif expected == 'RESISTANT':
                dir_correct = ratio > 1.0
            else:
                dir_correct = None
            mutant_results.append({
                'gene': r['mutant_gene'], 'expected': expected,
                'predicted_EC50_uM': r['predicted_EC50_uM'],
                'predicted_ratio': ratio, 'lit_ratio': mut_data['lit_ratio'],
                'direction_correct': bool(dir_correct),
            })
        n = len(mutant_results)
        n_dir = sum(1 for r in mutant_results if r['direction_correct'])
        verdict['gates']['gate3_mutant_directional'] = {
            'n_tested': n, 'n_correct': n_dir,
            'fraction': n_dir / n if n else 0,
            'PASS': (n_dir / n) >= 0.75 if n else False,
            'per_mutant': mutant_results,
        }

    # Gate 4 — Eger
    eger_results = []
    for r in ec50s:
        if r['anesthetic'] not in EGER_COMPOUNDS: continue
        max_qf = r['max_qf']
        expected = 'ANESTHETIC' if r['anesthetic'] == 'cis_12_dichloroethylene' else 'NON_IMMOBILIZER'
        verdict_eger = max_qf >= 0.5 if expected == 'ANESTHETIC' else max_qf < 0.5
        eger_results.append({
            'compound': r['anesthetic'], 'max_qf': max_qf, 'expected': expected,
            'CORRECT': verdict_eger,
        })
    n = len(eger_results)
    n_correct = sum(1 for r in eger_results if r['CORRECT'])
    verdict['gates']['gate4_eger_specificity'] = {
        'n_tested': n, 'n_correct': n_correct,
        'PASS': n_correct == n,
        'per_compound': eger_results,
    }

    n_passed = sum(1 for g in verdict['gates'].values() if g.get('PASS'))
    verdict['summary'] = f'{n_passed}/{len(verdict["gates"])} gates pass'

    with open(out_dir / 'v4_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2, default=str)

    print(f'\n=== V4 FLY VERDICT — {n_passed}/{len(verdict["gates"])} GATES PASS ===')
    for name, g in verdict['gates'].items():
        print(f'  {name:35s}  {"PASS" if g.get("PASS") else "FAIL"}')
    print(f'\nTotal wall: {(time.time()-t_start)/60:.1f} min')
    print(f'Saved → {out_dir}/')


if __name__ == '__main__':
    main()
