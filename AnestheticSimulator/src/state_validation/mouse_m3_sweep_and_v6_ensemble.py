"""V6 mouse — alpha sweep (M3) followed by full V6 ensemble.

Two-stage pipeline:
  Stage 1 — quick alpha sweep at 30s sims to find α calibrating halothane to 350 µM
  Stage 2 — full V6 ensemble (Gates 1-4) at the locked α with 60s × 5 seeds

Borrows speed-up lessons from V5: Pool(12), chunksize=4 to reduce overhead.
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
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def _alpha_sweep_worker(args):
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.mouse_state_validator import run_mouse_single
    a, d, s, sim_dur = args
    m = run_mouse_single('halothane', dose_uM=d, seed=s, sim_duration_s=sim_dur,
                         mutant_gene='WT', alpha_calib=a)
    return (a, d, s, m['quiescent_fraction'])


def _ensemble_worker(args):
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.mouse_state_validator import run_mouse_single
    anest, dose, seed, mutant_gene, alpha, sim_dur = args
    m = run_mouse_single(anest, dose_uM=dose, seed=seed, sim_duration_s=sim_dur,
                         mutant_gene=mutant_gene, alpha_calib=alpha)
    m['mutant_gene'] = mutant_gene
    return m


# ===== Sweep config =====

ALPHAS = [0.04, 0.06, 0.08, 0.10, 0.13]
SWEEP_DOSES = [100.0, 200.0, 300.0, 350.0, 500.0, 1000.0]
SWEEP_SEEDS = [42, 137, 219]
SWEEP_SIM_DUR = 30.0

# ===== Ensemble config =====

ENSEMBLE_SIM_DUR = 60.0
ENSEMBLE_SEEDS = [42, 137, 219, 331, 443]
DOSES_VOLATILE = [10.0, 30.0, 100.0, 200.0, 350.0, 500.0, 1000.0, 3000.0]
DOSES_EGER = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]

VOLATILES = ['halothane', 'isoflurane']
HYPER_MUTANTS = ['NDUFS4_cKO', 'Stx1A_hypo', 'Vamp2_cKO']
RESIST_MUTANTS = ['β3_N265M', 'α1_H101R', 'TREK1_KO', 'TASK1_KO', 'TASK3_KO',
                   'TASK13_dKO', 'GIRK2_KO']
EGER_COMPOUNDS = ['cis_12_dichloroethylene', 'trans_12_dichloroethylene', 'hexafluoroethane']


def main():
    out_dir = ANESTH / 'artifacts/state_validation_mouse'
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Stage 1: alpha sweep ===
    print(f'=== V6 Stage 1: mouse halothane WT alpha sweep ===')
    print(f'  alphas={ALPHAS}, doses={SWEEP_DOSES}, seeds={SWEEP_SEEDS}, sim={SWEEP_SIM_DUR}s')
    sweep_tasks = [(a, d, s, SWEEP_SIM_DUR) for a in ALPHAS for d in SWEEP_DOSES for s in SWEEP_SEEDS]
    print(f'  total {len(sweep_tasks)} sims')
    t0 = time.time()
    sweep_results = []
    with mp.Pool(processes=12) as pool:
        for i, r in enumerate(pool.imap_unordered(_alpha_sweep_worker, sweep_tasks, chunksize=4)):
            sweep_results.append(r)
            if (i+1) % 25 == 0 or (i+1) == len(sweep_tasks):
                print(f'    [{i+1}/{len(sweep_tasks)}] {(time.time()-t0)/60:.1f} min', flush=True)

    from collections import defaultdict
    import numpy as np
    agg_sweep = defaultdict(list)
    for a, d, s, qf in sweep_results:
        agg_sweep[(a, d)].append(qf)

    from state_validation.phase_g_state_validator import hill_fit_ec50
    print(f'\n{"alpha":>6s}  ' + '  '.join(f'{d:>5.0f}µM' for d in SWEEP_DOSES) + '  pred_EC50')
    ec50_by_alpha = {}
    for alpha in ALPHAS:
        qfs = [np.mean(agg_sweep[(alpha, d)]) for d in SWEEP_DOSES]
        ec50 = hill_fit_ec50(np.array(SWEEP_DOSES), np.array(qfs), threshold=0.5)
        ec50_by_alpha[alpha] = ec50
        ec50_str = f'{ec50:.0f} µM ({max(ec50/350, 350/ec50):.2f}×)' if ec50 else 'no cross'
        print(f'  {alpha:>4.2f}    ' + '  '.join(f'{q:>5.3f} ' for q in qfs) + f'   {ec50_str}')

    target = 350.0
    best_alpha = None
    best_err = float('inf')
    for a, ec50 in ec50_by_alpha.items():
        if ec50 is None: continue
        err = max(ec50/target, target/ec50)
        if err < best_err:
            best_err = err
            best_alpha = a
    if best_alpha is None:
        print('WARN: no alpha found a crossing; defaulting to 0.08')
        best_alpha = 0.08
    print(f'\n→ Best calibration: α = {best_alpha} (EC50 {ec50_by_alpha[best_alpha]:.0f} µM, '
          f'{best_err:.2f}× off published 350)')

    # === Stage 2: V6 ensemble at locked alpha ===
    ALPHA = best_alpha
    print(f'\n=== V6 Stage 2: full ensemble at α={ALPHA}, sim={ENSEMBLE_SIM_DUR}s × {len(ENSEMBLE_SEEDS)} seeds ===')
    tasks = []
    for anest in VOLATILES:
        for d in DOSES_VOLATILE:
            for s in ENSEMBLE_SEEDS:
                tasks.append((anest, d, s, 'WT', ALPHA, ENSEMBLE_SIM_DUR))
    for g in HYPER_MUTANTS + RESIST_MUTANTS:
        for d in DOSES_VOLATILE:
            for s in ENSEMBLE_SEEDS:
                tasks.append(('halothane', d, s, g, ALPHA, ENSEMBLE_SIM_DUR))
    for c in EGER_COMPOUNDS:
        for d in DOSES_EGER:
            for s in ENSEMBLE_SEEDS:
                tasks.append((c, d, s, 'WT', ALPHA, ENSEMBLE_SIM_DUR))
    print(f'  total {len(tasks)} sims')

    t_ens = time.time()
    results = []
    with mp.Pool(processes=12) as pool:
        for i, m in enumerate(pool.imap_unordered(_ensemble_worker, tasks, chunksize=4)):
            results.append(m)
            if (i+1) % 25 == 0 or (i+1) == len(tasks):
                elapsed = time.time() - t_ens
                eta = elapsed / (i+1) * (len(tasks) - (i+1))
                print(f'    [{i+1}/{len(tasks)}] {100*(i+1)/len(tasks):.0f}%  '
                      f'elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min', flush=True)

    print(f'\nEnsemble complete in {(time.time()-t_ens)/60:.1f} min')

    # Save raw + summary
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(out_dir / 'v6_all_results_raw.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in fieldnames})

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
            'cmd_rate_mean': float(cmd.mean()), 'net_rate_mean': float(net.mean()),
        })
    with open(out_dir / 'v6_dose_response_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # EC50 fits
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
    with open(out_dir / 'v6_ec50_predictions.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ec50s[0].keys()))
        w.writeheader()
        w.writerows(ec50s)

    # Verdict
    halothane_wt = next((r for r in ec50s if r['anesthetic']=='halothane' and r['mutant_gene']=='WT'), None)
    iso_wt = next((r for r in ec50s if r['anesthetic']=='isoflurane' and r['mutant_gene']=='WT'), None)

    verdict = {'alpha_calib': ALPHA, 'sim_duration_s': ENSEMBLE_SIM_DUR, 'n_seeds': len(ENSEMBLE_SEEDS),
               'gates': {}}

    if halothane_wt and halothane_wt['predicted_EC50_uM']:
        ec50 = halothane_wt['predicted_EC50_uM']
        err = max(ec50/350.0, 350.0/ec50)
        verdict['gates']['gate1_halothane_wt'] = {
            'predicted_EC50_uM': ec50, 'published_EC50_uM': 350.0, 'fold_error': err,
            'PASS': err <= 2.0,
        }
    if iso_wt and iso_wt['predicted_EC50_uM']:
        ec50 = iso_wt['predicted_EC50_uM']
        err = max(ec50/290.0, 290.0/ec50)
        verdict['gates']['gate2_iso_wt'] = {
            'predicted_EC50_uM': ec50, 'published_EC50_uM': 290.0, 'fold_error': err,
            'PASS': err <= 2.0,
        }

    if halothane_wt and halothane_wt['predicted_EC50_uM']:
        wt_ec50 = halothane_wt['predicted_EC50_uM']
        muts_table = {}
        with open(ANESTH / 'data/state_validation_mouse/mouse_mutant_baseline_perturbations.csv') as f:
            for line in f:
                if line.startswith('#') or line.strip()=='' or line.startswith('gene,'): continue
                parts = next(csv.reader([line]))
                muts_table[parts[0]] = {'direction': parts[1], 'lit_ratio': parts[7] if len(parts)>7 else ''}
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
            'fraction': n_dir/n if n else 0,
            'PASS': (n_dir/n) >= 0.75 if n else False,
            'per_mutant': mutant_results,
        }

    eger_results = []
    for r in ec50s:
        if r['anesthetic'] not in EGER_COMPOUNDS: continue
        max_qf = r['max_qf']
        expected = 'ANESTHETIC' if r['anesthetic'] == 'cis_12_dichloroethylene' else 'NON_IMMOBILIZER'
        verdict_e = max_qf >= 0.5 if expected == 'ANESTHETIC' else max_qf < 0.5
        eger_results.append({'compound': r['anesthetic'], 'max_qf': max_qf,
                              'expected': expected, 'CORRECT': verdict_e})
    n = len(eger_results)
    n_correct = sum(1 for r in eger_results if r['CORRECT'])
    verdict['gates']['gate4_eger_specificity'] = {
        'n_tested': n, 'n_correct': n_correct,
        'PASS': n_correct == n,
        'per_compound': eger_results,
    }

    n_passed = sum(1 for g in verdict['gates'].values() if g.get('PASS'))
    verdict['summary'] = f'{n_passed}/{len(verdict["gates"])} gates pass'

    with open(out_dir / 'v6_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2, default=str)

    print(f'\n=== V6 MOUSE VERDICT — {n_passed}/{len(verdict["gates"])} GATES PASS ===')
    for name, g in verdict['gates'].items():
        if name.startswith('gate1') or name.startswith('gate2'):
            ec50_str = f'{g["predicted_EC50_uM"]:.0f} µM ({g["fold_error"]:.2f}×)' if g.get("predicted_EC50_uM") else '—'
            print(f'  {name:35s}  {"PASS" if g.get("PASS") else "FAIL"}  {ec50_str}')
        elif name.startswith('gate3'):
            print(f'  {name:35s}  {"PASS" if g.get("PASS") else "FAIL"}  {g["n_correct"]}/{g["n_tested"]}')
        elif name.startswith('gate4'):
            print(f'  {name:35s}  {"PASS" if g.get("PASS") else "FAIL"}  {g["n_correct"]}/{g["n_tested"]}')


if __name__ == '__main__':
    main()
