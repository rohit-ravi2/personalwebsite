"""v7_m3_sensitivity — V5 M3 sensitivity analysis (OAT + LHS).

Pre-registration: AnestheticSimulator/docs/v7_preregistration.md
Hash: 533b624a00b5ff9efecee41fb549fcb9cf5f02810aefcde55e14c7981fd09ff4

OAT (one-at-a-time):
  For each (organism, mech_class in halothane row, parameter), perturb ±50%
  (param ∈ {target_EC50_uM, max_effect_factor, hill_n} where applicable);
  run halothane Gate 1 dose-response (8 doses × 3 seeds × 30s); compute
  sensitivity index S = (ΔEC50 / EC50_baseline) / (Δparam / param_baseline).

LHS (Latin Hypercube Sampling):
  100 joint samples in the parameter space; each sample independently
  perturbs ALL parameters of all halothane-active classes within ±50%.
  Run dose-response; build 95% CI on predicted halothane EC50.

Scope (interpretation of prereg "~2500 sims" budget): OAT on all three
organisms + LHS on worm only (lead organism with most characterized
profile). Cross-organism LHS deferred to optional follow-up.

Pre-registered predictions (from prereg):
  M3a: no single ±50% OAT perturbation causes Gate 1 fold-error > 2×.
       Falsifies if any ±50% OAT yields fold-error > 3× (architecture
       fragile to single-parameter uncertainty).
  M3b: at least one parameter has |sensitivity index| > 0.3 (load-bearing).
       Falsifies if max |S| < 0.1 (over-determined).
  M3c: 95% LHS CI on halothane EC50 ⊂ [200, 600] µM (worm).
       Falsifies if 95% CI extends beyond [100, 1000] µM (brittle under
       realistic literature uncertainty).
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
    _organism_runtime,
)
from state_validation.v7_random_ensemble import (  # noqa: E402
    _get_full_halothane_profile, _profile_to_specs, _worker_random_ensemble,
)

DOSES_HALOTHANE = [10.0, 30.0, 100.0, 200.0, 350.0, 500.0, 1000.0, 3000.0]
SEEDS_M3 = [42, 137, 219]
PERTURB_FRAC = 0.5  # ±50%
LHS_N_SAMPLES = 100
LHS_ORGANISM = 'worm'

OUT_DIR = ANESTH / 'artifacts' / 'v7_sensitivity'


def _fit_ec50(by_dose: dict[float, list[float]]) -> float | None:
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import hill_fit_ec50
    ds = sorted(by_dose.keys())
    qfs = [float(np.mean(by_dose[d])) for d in ds]
    return hill_fit_ec50(np.array(ds), np.array(qfs), threshold=0.5)


def _profile_with_perturb(profile_specs: dict, perturb: dict) -> dict:
    """profile_specs: cls→(EC50, max). perturb: cls→{'param': new_value, ...}.

    Returns new profile_specs with perturbations applied. Drops classes whose
    EC50 is None after perturbation (defensive).
    """
    out = {}
    for cls, (ec50, mxf) in profile_specs.items():
        new_ec50, new_mxf = ec50, mxf
        if cls in perturb:
            for k, v in perturb[cls].items():
                if k == 'EC50':
                    new_ec50 = v
                elif k == 'max':
                    new_mxf = v
        if new_ec50 is None:
            continue
        out[cls] = (new_ec50, new_mxf)
    return out


def _run_dose_response(profile_specs: dict, organism: str,
                        seeds: list[int], doses: list[float],
                        match_level_tag: int = 99) -> tuple[float | None, dict]:
    """Run 8 doses × 3 seeds for one perturbed profile; return (EC50, by_dose_qf)."""
    tasks = [(match_level_tag, -1, organism, profile_specs, d, s)
             for d in doses for s in seeds]
    results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for r in pool.imap_unordered(_worker_random_ensemble, tasks,
                                       chunksize=CHUNKSIZE):
            results.append(r)
    by_dose = defaultdict(list)
    for _ml, _eid, _o, dose, _seed, qf, _cr, _nr in results:
        by_dose[dose].append(qf)
    ec50 = _fit_ec50(by_dose)
    return ec50, by_dose


def run_oat():
    """Run OAT sensitivity sweep across all three organisms."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for organism in ORG_CONFIG:
        cfg = ORG_CONFIG[organism]
        full_profile = _get_full_halothane_profile(organism)
        baseline_specs = _profile_to_specs(full_profile)
        active_classes = list(baseline_specs.keys())

        print(f'\n--- OAT: {organism} ({len(active_classes)} active classes) ---', flush=True)

        # Baseline run
        print(f'  Running baseline...', flush=True)
        t0 = time.time()
        ec50_base, _ = _run_dose_response(
            baseline_specs, organism, SEEDS_M3, DOSES_HALOTHANE)
        pub = cfg['halothane_pub']
        fold_base = max(ec50_base / pub, pub / ec50_base) if ec50_base else None
        print(f'  baseline EC50 = {ec50_base}  fold = {fold_base}  '
              f'wall {(time.time()-t0)/60:.1f}m', flush=True)

        # OAT: each (class, param, direction) → one perturbed run
        # Params: EC50, max
        for cls in active_classes:
            ec50_b, max_b = baseline_specs[cls]
            for param_name, baseline_val in (('EC50', ec50_b), ('max', max_b)):
                if baseline_val is None:
                    continue
                for direction, factor in (('plus50', 1.0 + PERTURB_FRAC),
                                            ('minus50', 1.0 - PERTURB_FRAC)):
                    new_val = baseline_val * factor
                    perturb = {cls: {param_name: new_val}}
                    perturbed = _profile_with_perturb(baseline_specs, perturb)
                    t1 = time.time()
                    ec50_p, _ = _run_dose_response(
                        perturbed, organism, SEEDS_M3, DOSES_HALOTHANE)
                    fold_p = max(ec50_p / pub, pub / ec50_p) if ec50_p else None
                    if ec50_p and ec50_base:
                        sensitivity = (
                            ((ec50_p - ec50_base) / ec50_base) /
                            ((new_val - baseline_val) / baseline_val)
                        )
                    else:
                        sensitivity = None
                    rows.append({
                        'organism': organism, 'class': cls,
                        'param': param_name, 'direction': direction,
                        'baseline_value': baseline_val, 'perturbed_value': new_val,
                        'baseline_EC50_uM': ec50_base,
                        'perturbed_EC50_uM': ec50_p,
                        'fold_error_baseline': fold_base,
                        'fold_error_perturbed': fold_p,
                        'sensitivity_index': sensitivity,
                        'wall_min': (time.time() - t1) / 60.0,
                    })
                    print(f'    {cls:25s} {param_name:5s} {direction:8s}  '
                          f'EC50 {ec50_p}  fold {fold_p}  S {sensitivity}',
                          flush=True)

    out_csv = OUT_DIR / 'v7_sensitivity_oat.csv'
    fieldnames = ['organism', 'class', 'param', 'direction',
                   'baseline_value', 'perturbed_value',
                   'baseline_EC50_uM', 'perturbed_EC50_uM',
                   'fold_error_baseline', 'fold_error_perturbed',
                   'sensitivity_index', 'wall_min']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nOAT complete: {out_csv}')

    return rows


def run_lhs(organism: str = LHS_ORGANISM, n_samples: int = LHS_N_SAMPLES):
    """LHS over halothane parameters (EC50 + max for each active class) within ±50%.

    Uses simple random uniform perturbation as Latin Hypercube approximation
    (adequate sample size for 8-parameter space; full LHS would use scipy.stats.qmc).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full_profile = _get_full_halothane_profile(organism)
    baseline_specs = _profile_to_specs(full_profile)
    active_classes = list(baseline_specs.keys())
    n_dims = 2 * len(active_classes)

    # Simple LHS via scipy if available, else uniform random
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_dims, seed=20260502)
        samples = sampler.random(n=n_samples)
    except ImportError:
        rng = np.random.default_rng(20260502)
        samples = rng.random((n_samples, n_dims))

    # Map [0, 1) → factor in [1-PERTURB_FRAC, 1+PERTURB_FRAC]
    factors = (1.0 - PERTURB_FRAC) + samples * (2 * PERTURB_FRAC)

    print(f'\n=== M3 LHS — {organism}, {n_samples} samples, {n_dims} dims ±{int(PERTURB_FRAC*100)}% ===',
          flush=True)

    cfg = ORG_CONFIG[organism]
    pub = cfg['halothane_pub']
    rows = []
    t0 = time.time()
    for sample_idx in range(n_samples):
        f_arr = factors[sample_idx]
        # First half = EC50 factors, second half = max factors
        ec50_factors = f_arr[:len(active_classes)]
        max_factors = f_arr[len(active_classes):]
        perturbed_specs = {}
        for j, cls in enumerate(active_classes):
            ec50_b, max_b = baseline_specs[cls]
            new_ec50 = ec50_b * float(ec50_factors[j])
            new_max = max_b * float(max_factors[j])
            perturbed_specs[cls] = (new_ec50, new_max)

        ec50_p, _ = _run_dose_response(
            perturbed_specs, organism, SEEDS_M3, DOSES_HALOTHANE)
        fold_p = max(ec50_p / pub, pub / ec50_p) if ec50_p else None
        rows.append({
            'sample_idx': sample_idx,
            'organism': organism,
            'predicted_EC50_uM': ec50_p,
            'fold_error': fold_p,
            'ec50_factors': '|'.join(f'{f:.3f}' for f in ec50_factors),
            'max_factors': '|'.join(f'{f:.3f}' for f in max_factors),
        })
        if (sample_idx + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60.0
            eta = elapsed / (sample_idx + 1) * (n_samples - sample_idx - 1)
            print(f'  [{sample_idx+1}/{n_samples}]  {elapsed:.1f}m  ETA {eta:.0f}m',
                  flush=True)

    out_csv = OUT_DIR / f'v7_sensitivity_lhs_{organism}.csv'
    fieldnames = ['sample_idx', 'organism', 'predicted_EC50_uM', 'fold_error',
                   'ec50_factors', 'max_factors']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nLHS complete: {out_csv}')

    # Compute 95% CI on predicted EC50
    valid = [r['predicted_EC50_uM'] for r in rows if r['predicted_EC50_uM'] is not None]
    if valid:
        ci_low = float(np.percentile(valid, 2.5))
        ci_high = float(np.percentile(valid, 97.5))
        median = float(np.median(valid))
    else:
        ci_low = ci_high = median = None

    return {
        'organism': organism,
        'n_samples': n_samples,
        'n_valid_samples': len(valid),
        '95pct_CI_low_uM': ci_low,
        '95pct_CI_high_uM': ci_high,
        'median_uM': median,
        'csv_path': str(out_csv),
    }


def write_verdict(oat_rows: list[dict], lhs_summary: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # M3a: any single OAT perturbation with fold-error > 2× ?
    oat_fold_max = None
    oat_failures_2x = []
    oat_failures_3x = []
    for r in oat_rows:
        f = r.get('fold_error_perturbed')
        if f is not None:
            if oat_fold_max is None or f > oat_fold_max:
                oat_fold_max = f
            if f > 2.0:
                oat_failures_2x.append(
                    f"{r['organism']}/{r['class']}/{r['param']}/{r['direction']}: fold {f:.2f}")
            if f > 3.0:
                oat_failures_3x.append(
                    f"{r['organism']}/{r['class']}/{r['param']}/{r['direction']}: fold {f:.2f}")

    # M3b: max |sensitivity index|
    sensitivities = [r['sensitivity_index'] for r in oat_rows
                     if r['sensitivity_index'] is not None]
    max_abs_sensitivity = max((abs(s) for s in sensitivities), default=None)
    n_above_03 = sum(1 for s in sensitivities if abs(s) > 0.3)

    # M3c: 95% LHS CI for worm
    ci_low = lhs_summary.get('95pct_CI_low_uM')
    ci_high = lhs_summary.get('95pct_CI_high_uM')
    m3c_in_200_600 = (ci_low is not None and ci_high is not None
                      and ci_low >= 200.0 and ci_high <= 600.0)
    m3c_in_100_1000 = (ci_low is not None and ci_high is not None
                       and ci_low >= 100.0 and ci_high <= 1000.0)

    verdict = {
        'preregistration_hash': PREREG_HASH,
        'pipeline': 'v7_m3_sensitivity',
        'predictions': {
            'M3a_no_OAT_fails_2x': {
                'expected': 'no single ±50% OAT yields fold-error > 2x',
                'observed_max_OAT_fold_error': oat_fold_max,
                'OAT_failures_above_2x': oat_failures_2x,
                'falsifies_threshold': 'any OAT > 3x falsifies',
                'OAT_failures_above_3x': oat_failures_3x,
                'verdict': ('PASS' if not oat_failures_2x else
                             ('DEVIATION' if not oat_failures_3x else 'FAIL')),
            },
            'M3b_load_bearing_param_exists': {
                'expected': 'at least one param with |S| > 0.3',
                'observed_max_abs_sensitivity': max_abs_sensitivity,
                'n_params_above_0.3': n_above_03,
                'falsifies_threshold': 'max |S| < 0.1',
                'verdict': ('PASS' if (max_abs_sensitivity is not None and
                                          max_abs_sensitivity > 0.3) else
                             ('FAIL' if (max_abs_sensitivity is not None and
                                            max_abs_sensitivity < 0.1) else 'MARGINAL')),
            },
            'M3c_LHS_95pct_CI_in_200_600': {
                'expected': '95% LHS CI ⊂ [200, 600] µM (worm)',
                'observed_95pct_CI_uM': [ci_low, ci_high],
                'organism': lhs_summary.get('organism'),
                'falsifies_threshold': '95% CI extending beyond [100, 1000] µM',
                'verdict': ('PASS' if m3c_in_200_600 else
                             ('FAIL' if not m3c_in_100_1000 else 'DEVIATION')),
            },
        },
        'lhs_summary': lhs_summary,
    }

    with open(OUT_DIR / 'v7_sensitivity_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f'\nVerdict: {OUT_DIR / "v7_sensitivity_verdict.json"}')
    for k, v in verdict['predictions'].items():
        print(f'  {k}: {v.get("verdict")}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=== V7 M3 Sensitivity ===')
    oat_rows = run_oat()
    lhs_summary = run_lhs(LHS_ORGANISM, LHS_N_SAMPLES)
    write_verdict(oat_rows, lhs_summary)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'oat':
            run_oat()
        elif cmd == 'lhs':
            org = sys.argv[2] if len(sys.argv) > 2 else LHS_ORGANISM
            n = int(sys.argv[3]) if len(sys.argv) > 3 else LHS_N_SAMPLES
            print(json.dumps(run_lhs(org, n), indent=2, default=str))
        else:
            main()
    else:
        main()
