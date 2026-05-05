"""v7_random_ensemble — Sub-Q1 random-ensemble controls (Match #1 + #2).

Pre-registration: AnestheticSimulator/docs/v7_preregistration.md
Hash: 533b624a00b5ff9efecee41fb549fcb9cf5f02810aefcde55e14c7981fd09ff4

Match definitions (stringency ladder):
  Match #1 — Count only.    Random class identity, random EC50 ∈ U(50, 1000),
                             random max_effect ∈ U(0.3, 3.0).  n_active matches
                             the conserved ensemble's active row count
                             (worm 8, fly 8, mouse 7 — NOTE: includes
                             complex_ii_block, distinct from Sub-Q2's 7-class
                             subset space which excludes it).
  Match #2 — Count + total magnitude.  Same as Match #1 plus rejection sampling
                             so that ensemble's aggregate pA at clinical dose
                             matches conserved within ±5%.
  Match #3 — Count + magnitude + cell-type spread.  Requires CeNGEN-resolved
                             targeting in the validator. NOT implemented in V1
                             (validator applies all perturbations globally).
                             Documented as v7-DEVIATION; deferred to V2.

Compute estimate: 50 ensembles × 2 match levels × 3 organisms × 8 doses × 3 seeds
                  × 30s sims = 7,200 sims, ~5-6h on Pool(12).

Outputs:
  artifacts/v7_random_ensemble/v7_match1_random_50.csv
  artifacts/v7_random_ensemble/v7_match2_random_50.csv
  artifacts/v7_random_ensemble/v7_match1_raw.csv (per-sim)
  artifacts/v7_random_ensemble/v7_match2_raw.csv
  artifacts/v7_random_ensemble/v7_random_ensemble_verdict.json
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

# Full mech-class universe for Sub-Q1 (8 worm/fly, 7 mouse — INCLUDES complex_ii_block)
ALL_MECH_CLASSES_WORM_FLY = [
    'gaba_potentiation', 'k2p_potentiation', 'complex_i_block',
    'snare_cooperativity', 'nca_block', 'nachr_antagonism',
    'glucl_potentiation', 'complex_ii_block',
]
ALL_MECH_CLASSES_MOUSE = [c for c in ALL_MECH_CLASSES_WORM_FLY if c != 'glucl_potentiation']

ORG_ALL_CLASSES = {
    'worm': ALL_MECH_CLASSES_WORM_FLY,
    'fly': ALL_MECH_CLASSES_WORM_FLY,
    'mouse': ALL_MECH_CLASSES_MOUSE,
}

# Sub-Q1 doses + seeds (use 3 seeds for sweep efficiency per prereg)
DOSES_HALOTHANE = [10.0, 30.0, 100.0, 200.0, 350.0, 500.0, 1000.0, 3000.0]
SEEDS_RANDOM = [42, 137, 219]
N_RANDOM_ENSEMBLES = 50
PASS_FOLD_TOL = 2.0
MATCH2_TOLERANCE_PCT = 5.0
MATCH2_REJECTION_CAP = 5000  # max random tries per match-2 sample
DEFAULT_PA_AT_SATURATION = {
    'complex_i_block':       60.0,
    'complex_ii_block':      20.0,
    'k2p_potentiation':      30.0,
    'nca_block':             40.0,
    'gaba_potentiation':     30.0,
    'glucl_potentiation':    20.0,
    'nachr_antagonism':      30.0,
    'snare_cooperativity':   50.0,
}

OUT_DIR = ANESTH / 'artifacts' / 'v7_random_ensemble'

# Per-process cache
_PROCESS_CACHE: dict = {}


def _get_full_halothane_profile(organism: str) -> dict:
    if ('halothane', organism) in _PROCESS_CACHE:
        return _PROCESS_CACHE[('halothane', organism)]
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import load_perturbation_table
    table_path, _, _, _ = _organism_runtime(organism)
    profiles = load_perturbation_table(table_path)
    _PROCESS_CACHE[('halothane', organism)] = profiles['halothane']
    return profiles['halothane']


def _engagement_at_dose(ec50: float, dose: float, hill_n: float = 1.0) -> float:
    if ec50 is None or dose <= 0:
        return 0.0
    return (dose ** hill_n) / (dose ** hill_n + ec50 ** hill_n)


def _aggregate_pa_at_dose(profile_classes: list[tuple[str, float]],
                          dose: float) -> float:
    """Sum of sat_pa[cls] × engagement(dose) over classes (excluding SNARE special).

    profile_classes: list of (class_name, ec50_uM) tuples.
    """
    total = 0.0
    for cls, ec50 in profile_classes:
        if ec50 is None:
            continue
        sat_pa = DEFAULT_PA_AT_SATURATION.get(cls, 0.0)
        e = _engagement_at_dose(ec50, dose)
        total += sat_pa * e
    return total


def conserved_aggregate_pa(organism: str) -> tuple[float, int, list[str]]:
    """For the conserved halothane profile, compute aggregate pA at clinical dose.

    Returns (aggregate_pa, n_active_classes, active_class_names).
    """
    profile = _get_full_halothane_profile(organism)
    cfg = ORG_CONFIG[organism]
    clinical_dose = cfg['halothane_pub']
    active = [(cls, row.target_EC50_uM) for cls, row in profile.items()
              if row.target_EC50_uM is not None]
    return _aggregate_pa_at_dose(active, clinical_dose), len(active), [c for c, _ in active]


def _draw_random_profile(organism: str, n_active: int, rng: np.random.Generator
                          ) -> dict:
    """Draw a random halothane profile with n_active classes from the organism's pool."""
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import PerturbationRow
    pool = ORG_ALL_CLASSES[organism]
    if n_active > len(pool):
        n_active = len(pool)
    chosen_classes = rng.choice(pool, size=n_active, replace=False).tolist()
    profile = {}
    # Build full profile dict — all known classes; non-chosen get None
    for cls in pool:
        if cls in chosen_classes:
            ec50 = float(rng.uniform(50.0, 1000.0))
            mxf = float(rng.uniform(0.3, 3.0))
            profile[cls] = PerturbationRow(
                anesthetic='halothane', mechanism_class=cls,
                target_EC50_uM=ec50, max_effect_factor=mxf, hill_n=1.0,
                source_PMID='RANDOM', evidence_grade='RANDOM_ENSEMBLE',
            )
        else:
            profile[cls] = PerturbationRow(
                anesthetic='halothane', mechanism_class=cls,
                target_EC50_uM=None, max_effect_factor=None, hill_n=1.0,
                source_PMID='RANDOM', evidence_grade='RANDOM_DROP',
            )
    return profile


def _draw_random_profile_match2(organism: str, n_active: int,
                                 conserved_pa: float, tol_pct: float,
                                 rng: np.random.Generator,
                                 cap: int = MATCH2_REJECTION_CAP) -> dict | None:
    """Rejection-sample for total-pA match within tol_pct of conserved_pa."""
    cfg = ORG_CONFIG[organism]
    clinical_dose = cfg['halothane_pub']
    tol = conserved_pa * (tol_pct / 100.0)
    for _ in range(cap):
        prof = _draw_random_profile(organism, n_active, rng)
        active_classes = [(cls, row.target_EC50_uM) for cls, row in prof.items()
                          if row.target_EC50_uM is not None]
        agg_pa = _aggregate_pa_at_dose(active_classes, clinical_dose)
        if abs(agg_pa - conserved_pa) <= tol:
            return prof
    return None  # failed to sample


def _profile_to_serializable(profile: dict) -> list[dict]:
    out = []
    for cls, row in profile.items():
        if row.target_EC50_uM is None:
            continue
        out.append({
            'class': cls,
            'EC50_uM': row.target_EC50_uM,
            'max_effect': row.max_effect_factor,
        })
    return out


# ===== Worker (must be top-level for pickling) =====

# Profiles are pre-generated and stored in module-level dict, keyed by ensemble_id.
# Workers read from this dict by id; we ship the profile content via args instead
# (pickling each profile per task).
def _worker_random_ensemble(args):
    """Run one (match_level, ensemble_id, organism, profile_dict, dose, seed) sim."""
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.phase_g_state_validator import (
        run_single, PerturbationRow,
    )
    match_level, ensemble_id, organism, profile_specs, dose, seed = args
    cfg = ORG_CONFIG[organism]
    _, factory, qf_thr, cmd_set = _organism_runtime(organism)
    # Reconstruct PerturbationRow profile from profile_specs (dict cls→(EC50, max))
    profile = {}
    for cls, (ec50, mxf) in profile_specs.items():
        profile[cls] = PerturbationRow(
            anesthetic='halothane', mechanism_class=cls,
            target_EC50_uM=ec50, max_effect_factor=mxf, hill_n=1.0,
            source_PMID='RANDOM', evidence_grade='RANDOM_ENSEMBLE',
        )
    m = run_single(
        anesthetic='halothane', dose_uM=dose, seed=seed,
        sim_duration_s=SIM_DUR_S, profile=profile, mutant=None,
        alpha_calib=cfg['alpha'], brain_factory=factory,
        quiescent_threshold_hz=qf_thr, command_set=cmd_set,
    )
    return (match_level, ensemble_id, organism, dose, seed,
            float(m['quiescent_fraction']),
            float(m['command_mean_firing_rate_hz']),
            float(m['network_mean_firing_rate_hz']))


def _profile_to_specs(profile: dict) -> dict:
    """Serialize profile → dict cls→(EC50, max) suitable for pickling."""
    out = {}
    for cls, row in profile.items():
        if row.target_EC50_uM is None:
            continue
        out[cls] = (row.target_EC50_uM, row.max_effect_factor)
    return out


def _conserved_run_halothane_doses(organism: str) -> tuple[float, dict]:
    """Run the CONSERVED halothane profile across DOSES_HALOTHANE × SEEDS_RANDOM.

    Returns (predicted_EC50, per_dose_qf_means) — to compare random ensembles
    against the conserved pipeline's actual fold-error at this seed protocol.
    """
    profile = _get_full_halothane_profile(organism)
    specs = _profile_to_specs(profile)
    tasks = [(0, -1, organism, specs, d, s)
             for d in DOSES_HALOTHANE for s in SEEDS_RANDOM]
    results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for r in pool.imap_unordered(_worker_random_ensemble, tasks,
                                       chunksize=CHUNKSIZE):
            results.append(r)
    by_dose = defaultdict(list)
    for _ml, _eid, _o, dose, _seed, qf, _cr, _nr in results:
        by_dose[dose].append(qf)
    qfs = {d: float(np.mean(by_dose[d])) for d in sorted(by_dose)}
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import hill_fit_ec50
    ds_arr = np.array(sorted(qfs.keys()))
    qf_arr = np.array([qfs[d] for d in sorted(qfs.keys())])
    ec50 = hill_fit_ec50(ds_arr, qf_arr, threshold=0.5)
    return (float(ec50) if ec50 is not None else None), qfs


def main_match_level(match_level: int, seed_master: int = 20260502):
    """Run all 50 random ensembles × 3 organisms × 8 doses × 3 seeds for one match level."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if match_level not in (1, 2):
        raise ValueError(f'match_level must be 1 or 2 (got {match_level}); '
                         f'Match #3 deferred to V2 — see header docstring.')

    print(f'\n=== V7 Sub-Q1 Match #{match_level} — {N_RANDOM_ENSEMBLES} ensembles × 3 organisms ===')

    # Pre-compute conserved aggregate pA per organism (for Match #2 rejection)
    conserved_pa = {}
    n_active_per_org = {}
    active_classes_per_org = {}
    for org in ORG_CONFIG:
        agg, n_a, names = conserved_aggregate_pa(org)
        conserved_pa[org] = agg
        n_active_per_org[org] = n_a
        active_classes_per_org[org] = names
        print(f'  {org}: conserved aggregate pA = {agg:.1f}  n_active = {n_a}  '
              f'classes = {names}')

    # Pre-generate all random profiles
    rng = np.random.default_rng(seed_master + match_level)
    random_profiles: dict[tuple[str, int], dict] = {}
    rejection_failures: list[tuple[str, int]] = []
    for org in ORG_CONFIG:
        n_active = n_active_per_org[org]
        for eid in range(N_RANDOM_ENSEMBLES):
            if match_level == 1:
                prof = _draw_random_profile(org, n_active, rng)
            else:
                prof = _draw_random_profile_match2(
                    org, n_active, conserved_pa[org], MATCH2_TOLERANCE_PCT, rng)
                if prof is None:
                    rejection_failures.append((org, eid))
                    continue
            random_profiles[(org, eid)] = prof
    if rejection_failures:
        print(f'  WARN: {len(rejection_failures)} Match #2 rejection-sampling failures '
              f'(>{MATCH2_REJECTION_CAP} attempts each)')

    # Build task list
    tasks = []
    for (org, eid), prof in random_profiles.items():
        specs = _profile_to_specs(prof)
        for d in DOSES_HALOTHANE:
            for s in SEEDS_RANDOM:
                tasks.append((match_level, eid, org, specs, d, s))
    print(f'  Total {len(tasks)} sims across {len(random_profiles)} ensembles')

    t0 = time.time()
    raw_path = OUT_DIR / f'v7_match{match_level}_raw.csv'
    results = []
    with open(raw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['match_level', 'ensemble_id', 'organism', 'dose_uM',
                         'seed', 'quiescent_fraction', 'command_rate_hz',
                         'network_rate_hz'])
        with mp.Pool(processes=N_WORKERS) as pool:
            for i, r in enumerate(pool.imap_unordered(
                    _worker_random_ensemble, tasks, chunksize=CHUNKSIZE)):
                results.append(r)
                writer.writerow(r)
                if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                    e = (time.time() - t0) / 60.0
                    eta = e / (i + 1) * (len(tasks) - (i + 1))
                    f.flush()
                    print(f'    [{i+1}/{len(tasks)}]  {e:.1f}m  ETA {eta:.0f}m',
                          flush=True)

    # Aggregate per (organism, ensemble_id) → Hill-fit EC50, fold-error
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    from state_validation.phase_g_state_validator import hill_fit_ec50
    by_ens = defaultdict(lambda: defaultdict(list))
    for _ml, eid, org, dose, _seed, qf, _cr, _nr in results:
        by_ens[(org, eid)][dose].append(qf)

    summary = []
    for (org, eid), dose_qf in by_ens.items():
        ds = sorted(dose_qf.keys())
        qfs = [float(np.mean(dose_qf[d])) for d in ds]
        ec50 = hill_fit_ec50(np.array(ds), np.array(qfs), threshold=0.5)
        pub = ORG_CONFIG[org]['halothane_pub']
        if ec50 is None:
            fold = None
        else:
            fold = float(max(ec50 / pub, pub / ec50))
        prof = random_profiles[(org, eid)]
        active = [c for c, row in prof.items() if row.target_EC50_uM is not None]
        summary.append({
            'match_level': match_level, 'ensemble_id': eid, 'organism': org,
            'n_active': len(active),
            'active_classes': '|'.join(sorted(active)),
            'predicted_EC50_uM': ec50, 'published_EC50_uM': pub,
            'fold_error': fold,
        })

    summary_path = OUT_DIR / f'v7_match{match_level}_random_50.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['match_level', 'ensemble_id', 'organism',
                                            'n_active', 'active_classes',
                                            'predicted_EC50_uM', 'published_EC50_uM',
                                            'fold_error'])
        w.writeheader()
        for r in summary:
            w.writerow(r)

    # Conserved fold-error per organism (for percentile rank)
    print(f'\n  Computing conserved-ensemble fold-error baseline...')
    conserved_fold = {}
    for org in ORG_CONFIG:
        ec50, qfs = _conserved_run_halothane_doses(org)
        pub = ORG_CONFIG[org]['halothane_pub']
        if ec50 is None:
            conserved_fold[org] = None
        else:
            conserved_fold[org] = float(max(ec50 / pub, pub / ec50))
        print(f'    {org}: conserved EC50 = {ec50}  fold = {conserved_fold[org]}')

    # Percentile ranks
    perc_per_org = {}
    for org in ORG_CONFIG:
        org_folds = [r['fold_error'] for r in summary
                      if r['organism'] == org and r['fold_error'] is not None]
        cf = conserved_fold[org]
        if cf is None or not org_folds:
            perc_per_org[org] = None
        else:
            n_better = sum(1 for f in org_folds if f < cf)
            perc_per_org[org] = float(n_better) / len(org_folds) * 100.0
        # smaller percentile rank = conserved is better than fewer randoms
        # (conserved being top-15% means percentile ≤ 15%)

    print(f'  Conserved-ensemble percentile rank (smaller = better than more randoms):')
    for org, p in perc_per_org.items():
        print(f'    {org}: {p}')

    return {
        'match_level': match_level,
        'rejection_failures': len(rejection_failures),
        'conserved_fold_error': conserved_fold,
        'percentile_rank': perc_per_org,
        'wall_minutes': (time.time() - t0) / 60.0,
        'summary_path': str(summary_path),
        'raw_path': str(raw_path),
    }


def write_random_ensemble_verdict(match1_verdict: dict, match2_verdict: dict | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verdict = {
        'preregistration_hash': PREREG_HASH,
        'pipeline': 'v7_subq1_random_ensemble',
        'doses_halothane_uM': DOSES_HALOTHANE,
        'seeds_random': SEEDS_RANDOM,
        'sim_duration_s': SIM_DUR_S,
        'n_random_ensembles': N_RANDOM_ENSEMBLES,
        'match_levels': {
            'match1_count_only': match1_verdict,
            'match2_count_plus_magnitude': match2_verdict,
            'match3_cell_type_spread': {
                'status': 'V7-DEVIATION',
                'reason': (
                    'V1 validator applies all perturbations globally '
                    '(resolve_target_neurons returns range(brain.N)). '
                    'CeNGEN-resolved targeting required for cell-type spread '
                    'discrimination is V2 work and is not in V7 scope. '
                    'In V1 architecture, Match #3 reduces mathematically to '
                    'Match #2 because all classes hit all neurons.'
                ),
                'P8_thresholds_left_unfilled': True,
            },
        },
        'pre_registered_predictions': {
            'P6_match1_percentile_le_50': {
                'threshold': '<= 50%',
                'falsifies_below': '<= 10% (conserved too special at this match)',
            },
            'P7_match2_percentile_le_30': {
                'threshold': '<= 30%',
                'falsifies_below': '<= 5% (total magnitude is the entire story)',
            },
            'P8_match3': {
                'status': 'NOT_TESTED — see match3_cell_type_spread above',
            },
        },
    }
    with open(OUT_DIR / 'v7_random_ensemble_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f'\nFinal verdict: {OUT_DIR / "v7_random_ensemble_verdict.json"}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m1 = main_match_level(1)
    m2 = main_match_level(2)
    write_random_ensemble_verdict(m1, m2)
    print('\n=== V7 Sub-Q1 complete ===')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'match1':
            v = main_match_level(1)
            print(json.dumps(v, indent=2, default=str))
        elif cmd == 'match2':
            v = main_match_level(2)
            print(json.dumps(v, indent=2, default=str))
        elif cmd == 'smoke':
            # Tiny: 2 ensembles × 1 organism × 2 doses × 1 seed
            global N_RANDOM_ENSEMBLES, DOSES_HALOTHANE, SEEDS_RANDOM
            N_RANDOM_ENSEMBLES = 2
            DOSES_HALOTHANE = [200.0, 1000.0]
            SEEDS_RANDOM = [42]
            v = main_match_level(1)
            print(json.dumps(v, indent=2, default=str))
        else:
            main()
    else:
        main()
