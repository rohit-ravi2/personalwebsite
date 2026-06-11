"""v7_m5_bootstrap_ci — V5 M5 bootstrap 95% CIs on predicted EC50.

Reconstructs the M5 bootstrap CI computation (the original generating script
was never committed; only its output `artifacts/v5_controls/M5_bootstrap_CIs.json`
was) and extends it to the mouse V6 substrate, which the V7 closeout §9.4 flagged
as an open item: the page reports worm + fly WT CIs but only mouse point predictions.

Method (matches the committed M5 JSON `method` string):
  95% bootstrap CI (1000 resamples) on per-dose seed-mean qf, refit EC50
  each iteration. For each bootstrap iteration, the per-seed quiescent-fraction
  values at each dose are resampled with replacement (independently per dose),
  averaged to a seed-mean qf, and the EC50 is re-extracted via the same
  log-linear threshold-crossing fit (`hill_fit_ec50`, threshold 0.5) used
  everywhere else in the pipeline. Iterations whose resampled curve never
  crosses threshold are recorded but excluded from the CI.

Validation: run with `python v7_m5_bootstrap_ci.py validate` to reproduce the
already-published worm V3 + fly V4 halothane/isoflurane WT CIs from their raw
ensemble CSVs, confirming the reconstruction is faithful before trusting the
mouse numbers.

No model parameters, α, or perturbation tables are touched. This is a
post-hoc statistical summary of frozen V6 simulation output.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ANESTH = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator')
sys.path.insert(0, str(ANESTH / 'src'))

from state_validation.phase_g_state_validator import hill_fit_ec50  # noqa: E402

RNG_SEED = 20260503
N_BOOT = 1000
THRESHOLD = 0.5

# Raw per-seed ensemble CSVs (same schema across organisms:
# anesthetic, dose_uM, mutant_gene, seed, quiescent_fraction).
RAW_CSV = {
    'worm_V3': ANESTH / 'artifacts/state_validation/v3_ensemble/all_results_raw.csv',
    'fly_V4': ANESTH / 'artifacts/state_validation_fly/v4_ensemble/all_results_raw.csv',
    'mouse_V6': ANESTH / 'artifacts/state_validation_mouse/v6_all_results_raw.csv',
}


def _load_condition(csv_path: Path, anesthetic: str, mutant_gene: str
                    ) -> dict[float, list[float]]:
    """Return {dose_uM: [qf per seed]} for one (anesthetic, mutant_gene)."""
    by_dose: dict[float, list[float]] = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r['anesthetic'] != anesthetic:
                continue
            if r['mutant_gene'] != mutant_gene:
                continue
            by_dose[float(r['dose_uM'])].append(float(r['quiescent_fraction']))
    return dict(by_dose)


def bootstrap_ci(by_dose: dict[float, list[float]], rng: np.random.Generator,
                 n_boot: int = N_BOOT) -> dict:
    """Per-dose seed-resampling bootstrap; refit EC50 each iteration."""
    doses = np.array(sorted(by_dose.keys()))
    seed_qf = {d: np.array(by_dose[d]) for d in doses}
    ec50s = []
    n_attempted = 0
    for _ in range(n_boot):
        n_attempted += 1
        mean_qf = np.empty(len(doses))
        for i, d in enumerate(doses):
            vals = seed_qf[d]
            idx = rng.integers(0, len(vals), size=len(vals))
            mean_qf[i] = vals[idx].mean()
        ec50 = hill_fit_ec50(doses, mean_qf, threshold=THRESHOLD)
        if ec50 is not None:
            ec50s.append(ec50)
    ec50s = np.array(ec50s)
    if len(ec50s) == 0:
        return {'n_bootstrap_samples_with_crossing': 0,
                'n_bootstrap_attempted': n_attempted}
    return {
        'n_bootstrap_samples_with_crossing': int(len(ec50s)),
        'n_bootstrap_attempted': n_attempted,
        'ec50_median_uM': float(np.median(ec50s)),
        'ec50_ci_low_uM': float(np.percentile(ec50s, 2.5)),
        'ec50_ci_high_uM': float(np.percentile(ec50s, 97.5)),
        'ec50_log10_sd': float(np.std(np.log10(ec50s))),
    }


def run_conditions(organism_key: str, conditions: list[tuple[str, str]]) -> dict:
    """conditions = list of (anesthetic, mutant_gene). One RNG, shared seed."""
    rng = np.random.default_rng(RNG_SEED)
    csv_path = RAW_CSV[organism_key]
    out = {}
    for anesthetic, mutant_gene in conditions:
        by_dose = _load_condition(csv_path, anesthetic, mutant_gene)
        if not by_dose:
            print(f'  [skip] {organism_key} {anesthetic}/{mutant_gene}: no rows')
            continue
        ci = bootstrap_ci(by_dose, rng)
        ci = {'anesthetic': anesthetic, 'mutant_gene': mutant_gene, **ci}
        key = f'{anesthetic}__{mutant_gene}'
        out[key] = ci
        med = ci.get('ec50_median_uM')
        lo = ci.get('ec50_ci_low_uM')
        hi = ci.get('ec50_ci_high_uM')
        if med is not None:
            print(f'  {key:32s} median {med:7.1f}  CI [{lo:7.1f}, {hi:7.1f}]')
        else:
            print(f'  {key:32s} no threshold crossing')
    return out


def validate():
    """Reproduce published worm/fly WT CIs and compare to committed M5 JSON."""
    committed = json.load(open(ANESTH / 'artifacts/v5_controls/M5_bootstrap_CIs.json'))
    print('=== M5 reproduction validation (worm V3 + fly V4 WT) ===')
    print(f'rng_seed={RNG_SEED}  n_boot={N_BOOT}\n')
    for org in ('worm_V3', 'fly_V4'):
        print(f'--- {org} ---')
        repro = run_conditions(org, [('halothane', 'WT'), ('isoflurane', 'WT')])
        for cond in ('halothane__WT', 'isoflurane__WT'):
            c = committed['organisms'][org][cond]
            r = repro[cond]
            print(f'    {cond}: committed median {c["ec50_median_uM"]:.1f} '
                  f'[{c["ec50_ci_low_uM"]:.1f}, {c["ec50_ci_high_uM"]:.1f}]  '
                  f'| repro median {r["ec50_median_uM"]:.1f} '
                  f'[{r["ec50_ci_low_uM"]:.1f}, {r["ec50_ci_high_uM"]:.1f}]')
        print()


def run_mouse_and_update():
    """Compute mouse V6 WT CIs and add a mouse_V6 block to the M5 JSON."""
    print('=== Mouse V6 bootstrap CIs (halothane + isoflurane WT) ===')
    print(f'rng_seed={RNG_SEED}  n_boot={N_BOOT}\n')
    mouse = run_conditions('mouse_V6',
                           [('halothane', 'WT'), ('isoflurane', 'WT')])

    json_path = ANESTH / 'artifacts/v5_controls/M5_bootstrap_CIs.json'
    data = json.load(open(json_path))
    data['organisms']['mouse_V6'] = mouse
    note = ('mouse_V6 CIs added post-hoc (V7 §9.4 open item); same method, '
            'rng_seed, and n_boot as worm_V3/fly_V4; computed by '
            'src/state_validation/v7_m5_bootstrap_ci.py from frozen V6 raw qf.')
    data['mouse_V6_provenance'] = note
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nUpdated {json_path} with mouse_V6 block.')
    return mouse


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        validate()
    elif len(sys.argv) > 1 and sys.argv[1] == 'mouse':
        run_mouse_and_update()
    else:
        validate()
        print()
        run_mouse_and_update()


if __name__ == '__main__':
    main()
