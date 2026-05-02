"""V2 ensemble runner — parallel across cores, 60s sim × 5 seeds (canonical protocol).

Re-runs Gate 1-4 in parallel using multiprocessing.Pool(8). Total ~545 sims;
leaves 8 cores for the C. elegans simulator gauntlet.

Outputs per-gate dose-response CSVs + a unified verdict JSON.

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python parallel_v2_ensemble.py
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
SIMV  = ANESTH / 'src'

sys.path.insert(0, str(SIMV))
sys.path.insert(0, str(ROOT / 'scripts'))


# ===== worker =====

def _worker(args):
    """Run one simulation. Imports happen inside worker to avoid pickle issues."""
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.phase_g_state_validator import (
        run_single, load_perturbation_table, load_mutant_table,
    )
    anest, dose, seed, mutant_gene, alpha, sim_dur = args
    prof = load_perturbation_table(
        '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/data/state_validation/anesthetic_perturbation_table.csv'
    )
    muts = load_mutant_table(
        '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/data/state_validation/mutant_baseline_perturbations.csv'
    )
    mut_obj = muts.get(mutant_gene) if mutant_gene not in (None, 'WT') else None
    metrics = run_single(anest, dose_uM=dose, seed=seed, sim_duration_s=sim_dur,
                          profile=prof[anest], mutant=mut_obj, alpha_calib=alpha)
    metrics['mutant_gene'] = mutant_gene
    return metrics


# ===== task list builder =====

ALPHA = 0.13  # V3: recalibrated after W_chem propagation bug fix activated SNARE perturbation
SIM_DUR = 60.0
SEEDS = [42, 137, 219, 331, 443]
DOSES_VOLATILE = [10.0, 30.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 3000.0]
DOSES_EGER     = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]

VOLATILES = ['halothane', 'isoflurane']
HYPER_MUTANTS = ['gas-1', 'gas-2', 'nduf-6', 'ndus-8', 'nuo-1', 'unc-79', 'unc-80']
RESIST_MUTANTS = ['goa-1', 'dgk-1']
EGER_COMPOUNDS = ['cis_12_dichloroethylene', 'trans_12_dichloroethylene', 'hexafluoroethane']


def build_tasks():
    tasks = []
    # Gates 1+2: WT volatiles
    for anest in VOLATILES:
        for d in DOSES_VOLATILE:
            for s in SEEDS:
                tasks.append((anest, d, s, 'WT', ALPHA, SIM_DUR))
    # Gate 3α: HYPER mutants × halothane
    for g in HYPER_MUTANTS:
        for d in DOSES_VOLATILE:
            for s in SEEDS:
                tasks.append(('halothane', d, s, g, ALPHA, SIM_DUR))
    # Gate 3β: RESISTANT mutants × halothane (will hit known W_chem propagation bug)
    for g in RESIST_MUTANTS:
        for d in DOSES_VOLATILE:
            for s in SEEDS:
                tasks.append(('halothane', d, s, g, ALPHA, SIM_DUR))
    # Gate 4: Eger
    for c in EGER_COMPOUNDS:
        for d in DOSES_EGER:
            for s in SEEDS:
                tasks.append((c, d, s, 'WT', ALPHA, SIM_DUR))
    return tasks


# ===== main =====

def main():
    out_dir = ANESTH / 'artifacts' / 'state_validation' / 'v3_ensemble'
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    print(f'V2 ensemble — {len(tasks)} sims  (alpha={ALPHA}, sim={SIM_DUR}s × {len(SEEDS)} seeds)')
    print(f'Workers: 8 of 16 cores (leaving 8 for gauntlet + system)')
    print(f'Estimated wall: {len(tasks) * 50 / 8 / 60:.0f} min  (assuming 50s per sim)')
    print()

    t_start = time.time()
    results = []
    with mp.Pool(processes=8) as pool:
        for i, m in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2)):
            results.append(m)
            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t_start
                eta = elapsed / (i+1) * (len(tasks) - (i+1))
                print(f'  [{i+1:>4d}/{len(tasks)}] {100*(i+1)/len(tasks):.0f}%  '
                      f'elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min')

    elapsed = time.time() - t_start
    print(f'\nAll {len(tasks)} sims complete in {elapsed/60:.1f} min')

    # Save raw rows
    raw_path = out_dir / 'all_results_raw.csv'
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'Raw rows: {raw_path}')

    # ===== Aggregate per (anesthetic, mutant_gene, dose) =====
    from collections import defaultdict
    import numpy as np

    agg = defaultdict(list)
    for r in results:
        key = (r['anesthetic'], r['mutant_gene'], r['dose_uM'])
        agg[key].append(r)

    summary_rows = []
    for (anest, mut, dose), runs in sorted(agg.items()):
        qf = np.array([x['quiescent_fraction'] for x in runs])
        cmd = np.array([x['command_mean_firing_rate_hz'] for x in runs])
        net = np.array([x['network_mean_firing_rate_hz'] for x in runs])
        ac = np.array([x['state_autocorrelation_lag1'] for x in runs])
        summary_rows.append({
            'anesthetic': anest, 'mutant_gene': mut, 'dose_uM': dose,
            'n_seeds': len(runs),
            'qf_mean': float(qf.mean()), 'qf_sd': float(qf.std(ddof=1)),
            'qf_sem': float(qf.std(ddof=1)/np.sqrt(len(qf))),
            'cmd_rate_mean': float(cmd.mean()), 'cmd_rate_sd': float(cmd.std(ddof=1)),
            'net_rate_mean': float(net.mean()),
            'autocorr_mean': float(ac.mean()),
        })
    summary_path = out_dir / 'dose_response_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f'Summary: {summary_path}')

    # ===== Hill-fit EC50 per (anesthetic, mutant_gene) =====
    from state_validation.phase_g_state_validator import hill_fit_ec50

    ec50_results = []
    by_key = defaultdict(list)
    for r in summary_rows:
        by_key[(r['anesthetic'], r['mutant_gene'])].append(r)
    for (anest, mut), rows in sorted(by_key.items()):
        rows = sorted(rows, key=lambda x: x['dose_uM'])
        doses = np.array([x['dose_uM'] for x in rows])
        qfs = np.array([x['qf_mean'] for x in rows])
        ec50 = hill_fit_ec50(doses, qfs, threshold=0.5)
        ec50_results.append({'anesthetic': anest, 'mutant_gene': mut,
                             'predicted_EC50_uM': ec50, 'max_qf': float(qfs.max())})
    ec50_path = out_dir / 'ec50_predictions.csv'
    with open(ec50_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ec50_results[0].keys()))
        w.writeheader()
        w.writerows(ec50_results)
    print(f'EC50 predictions: {ec50_path}')

    # ===== Verdict computation =====
    halothane_wt = next((r for r in ec50_results
                         if r['anesthetic'] == 'halothane' and r['mutant_gene'] == 'WT'), None)
    iso_wt = next((r for r in ec50_results
                   if r['anesthetic'] == 'isoflurane' and r['mutant_gene'] == 'WT'), None)

    verdict = {
        'alpha_calib': ALPHA, 'sim_duration_s': SIM_DUR, 'n_seeds': len(SEEDS),
        'gates': {},
    }
    # Gate 1
    if halothane_wt and halothane_wt['predicted_EC50_uM']:
        ec50 = halothane_wt['predicted_EC50_uM']
        err = max(ec50/340.0, 340.0/ec50)
        verdict['gates']['gate1_halothane_wt'] = {
            'predicted_EC50_uM': ec50, 'published_EC50_uM': 340.0, 'fold_error': err,
            'pass_criterion': '≤ 2× off published', 'PASS': err <= 2.0,
        }
    # Gate 2
    if iso_wt and iso_wt['predicted_EC50_uM']:
        ec50 = iso_wt['predicted_EC50_uM']
        err = max(ec50/290.0, 290.0/ec50)
        verdict['gates']['gate2_iso_wt'] = {
            'predicted_EC50_uM': ec50, 'published_EC50_uM': 290.0, 'fold_error': err,
            'pass_criterion': '≤ 2× off published', 'PASS': err <= 2.0,
        }
    # Gate 3
    if halothane_wt and halothane_wt['predicted_EC50_uM']:
        wt_ec50 = halothane_wt['predicted_EC50_uM']
        muts_table = load_mutant_table_simple()
        mutant_results = []
        for r in ec50_results:
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
            'fraction': n_dir/n if n else 0,
            'pass_criterion': '≥ 75% correct',
            'PASS': (n_dir/n) >= 0.75 if n else False,
            'per_mutant': mutant_results,
        }
    # Gate 4
    eger_results = []
    for r in ec50_results:
        if r['anesthetic'] not in EGER_COMPOUNDS: continue
        max_qf = r['max_qf']
        expected = 'ANESTHETIC' if r['anesthetic'] == 'cis_12_dichloroethylene' else 'NON_IMMOBILIZER'
        if expected == 'ANESTHETIC':
            verdict_eger = max_qf >= 0.5
        else:
            verdict_eger = max_qf < 0.5
        eger_results.append({
            'compound': r['anesthetic'], 'max_qf': max_qf, 'expected': expected,
            'CORRECT': verdict_eger,
        })
    n = len(eger_results)
    n_correct = sum(1 for r in eger_results if r['CORRECT'])
    verdict['gates']['gate4_eger_specificity'] = {
        'n_tested': n, 'n_correct': n_correct,
        'pass_criterion': '3/3 correct',
        'PASS': n_correct == n,
        'per_compound': eger_results,
    }

    # Final pass count
    n_gates_passed = sum(1 for g in verdict['gates'].values() if g.get('PASS'))
    n_gates_total = len(verdict['gates'])
    verdict['summary'] = f'{n_gates_passed}/{n_gates_total} gates pass'

    out_path = out_dir / 'v2_verdict.json'
    with open(out_path, 'w') as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f'Verdict: {out_path}')
    print(f'\n=== {n_gates_passed}/{n_gates_total} GATES PASS ===')
    for name, g in verdict['gates'].items():
        print(f'  {name:35s}  {"PASS" if g.get("PASS") else "FAIL"}')
    print(f'\nTotal wall: {(time.time()-t_start)/60:.1f} min')


def load_mutant_table_simple():
    """Simple mutant table loader for ratio metadata."""
    out = {}
    with open('/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/data/state_validation/mutant_baseline_perturbations.csv') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '': continue
            if line.startswith('gene,'): continue
            parts = next(csv.reader([line]))
            gene, direction = parts[0], parts[1]
            lit_ratio = parts[6]
            out[gene] = {'direction': direction, 'lit_ratio': lit_ratio}
    return out


if __name__ == '__main__':
    main()
