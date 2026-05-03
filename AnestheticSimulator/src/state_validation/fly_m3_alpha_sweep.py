"""M3 — fly halothane WT alpha calibration sweep.

Find the α that produces predicted halothane EC50 ≈ 340 µM (van Swinderen 1999).
Worm V3 used α=0.13; fly smoke test shows that's too strong (saturates at 340 µM).
Sweep lower α values, identify the calibration point, lock for Gate 2-4.
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
    """Run one (alpha, dose, seed) sim. Imports inside worker to avoid pickle issues."""
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.fly_state_validator import run_fly_single
    alpha, dose, seed, sim_dur = args
    m = run_fly_single(
        anesthetic='halothane',
        dose_uM=dose,
        seed=seed,
        sim_duration_s=sim_dur,
        mutant_gene='WT',
        alpha_calib=alpha,
    )
    return {
        'alpha': alpha, 'dose_uM': dose, 'seed': seed,
        'qf': m['quiescent_fraction'],
        'cmd_rate': m['command_mean_firing_rate_hz'],
        'net_rate': m['network_mean_firing_rate_hz'],
    }


# === Sweep grid ===
SIM_DUR = 30.0  # 30s per sim for sweep (faster than 60s; tightens later)
SEEDS = [42, 137, 219]
DOSES = [50.0, 100.0, 200.0, 340.0, 500.0, 1000.0, 2000.0]
ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.13]


def main():
    out_dir = ANESTH / 'artifacts' / 'state_validation_fly'
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(a, d, s, SIM_DUR) for a in ALPHAS for d in DOSES for s in SEEDS]
    print(f'Fly Gate 1 alpha sweep — {len(tasks)} sims (8-core parallel)')
    print(f'Alphas: {ALPHAS}')
    print(f'Doses: {DOSES} µM')
    print(f'Seeds: {SEEDS}, sim duration: {SIM_DUR}s')
    print()

    t0 = time.time()
    results = []
    with mp.Pool(processes=8) as pool:
        for i, m in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2)):
            results.append(m)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(tasks) - (i + 1))
                print(f'  [{i+1:>3d}/{len(tasks)}] {100*(i+1)/len(tasks):.0f}%  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min')

    elapsed = time.time() - t0
    print(f'\nAll sims complete in {elapsed/60:.1f} min')

    # Aggregate per (alpha, dose)
    from collections import defaultdict
    import numpy as np
    agg = defaultdict(list)
    for r in results:
        agg[(r['alpha'], r['dose_uM'])].append(r)

    rows = []
    for (alpha, dose), runs in sorted(agg.items()):
        qfs = np.array([x['qf'] for x in runs])
        cmds = np.array([x['cmd_rate'] for x in runs])
        rows.append({
            'alpha': alpha, 'dose_uM': dose,
            'qf_mean': float(qfs.mean()), 'qf_sd': float(qfs.std(ddof=1)),
            'cmd_rate_mean': float(cmds.mean()),
            'n': len(runs),
        })

    # Write summary CSV
    with open(out_dir / 'gate1_alpha_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Compute predicted EC50 per alpha
    from state_validation.phase_g_state_validator import hill_fit_ec50
    print(f'\n{"alpha":>6s}  ' + '  '.join(f'{d:>4.0f}µM' for d in DOSES) + '  pred_EC50')
    ec50_by_alpha = {}
    for alpha in ALPHAS:
        qfs = [next((r['qf_mean'] for r in rows if r['alpha']==alpha and r['dose_uM']==d), 0.0) for d in DOSES]
        ec50 = hill_fit_ec50(np.array(DOSES), np.array(qfs), threshold=0.5)
        ec50_by_alpha[alpha] = ec50
        print(f'  {alpha:>4.2f}    ' + '  '.join(f'{q:>5.3f} ' for q in qfs) +
              (f'   {ec50:.0f} µM' if ec50 else '   no cross'))

    # Pick the alpha closest to target 340 µM
    target = 340.0
    best_alpha = None
    best_err = float('inf')
    for alpha, ec50 in ec50_by_alpha.items():
        if ec50 is None: continue
        err = max(ec50 / target, target / ec50)
        if err < best_err:
            best_err = err
            best_alpha = alpha
    if best_alpha is not None:
        print(f'\n  Best calibration: α = {best_alpha}  → predicted EC50 {ec50_by_alpha[best_alpha]:.0f} µM '
              f'({best_err:.2f}× off published 340)')

    json.dump({
        'sim_duration_s': SIM_DUR, 'n_seeds': len(SEEDS),
        'doses': DOSES, 'alphas': ALPHAS,
        'ec50_by_alpha': {f'{a:.2f}': ec50 for a, ec50 in ec50_by_alpha.items()},
        'best_alpha': best_alpha,
        'best_predicted_EC50_uM': ec50_by_alpha.get(best_alpha),
        'fold_error_vs_published': best_err if best_alpha is not None else None,
    }, open(out_dir / 'gate1_alpha_sweep_verdict.json', 'w'), indent=2, default=str)
    print(f'\nWrote → {out_dir}/gate1_alpha_sweep.csv + gate1_alpha_sweep_verdict.json')


if __name__ == '__main__':
    main()
