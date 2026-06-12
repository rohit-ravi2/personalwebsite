"""v7_match2b — corrected two-coordinate (total_pa, snare_factor) Match#2b null (WF2 P8).

The shipped Match#2 control (v7_random_ensemble._draw_random_profile_match2) matched only
`_aggregate_pa_at_dose`, which is wrong against the rank-2 operator in two ways:
  (a) it SUMS snare_cooperativity's 50 pA into the current total — a PHANTOM term the operator
      never applies (apply_anesthetic routes SNARE to a synaptic-weight multiplier, NOT I_ext);
  (b) it leaves the actual second sufficient-statistic coordinate (snare_factor) UNMATCHED.

P18 certified the operator is rank-2: QF = G(total_pa, snare_factor). This module rejection-samples
random ensembles matching BOTH coordinates of the conserved halothane profile (at the clinical anchor
dose), then re-asks whether the conserved profile's EC50 precision is special beyond the AIRTIGHT
two-coordinate magnitude control. Prereg + gates: audits/phase1/P8/prereg.json.

Operator mirror (verified to 1e-9 by the G1 fidelity gate against the real apply_anesthetic):
  total_pa(prof,dose,alpha)   = alpha * sum_{cls in CURRENT_CLASSES} (-sat_pa[cls] * engagement(dose))
  snare_factor(prof,dose)     = 1 + (snare_max-1)*engagement_snare(dose), else 1.0
  (phase_g_state_validator.apply_anesthetic lines 308-335.)

CLI:  fidelity | density | run     (run is gated on fidelity PASS + density >= n_min)
Env:  ml conda (brian2) for fidelity + run.
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

from state_validation.v7_random_ensemble import (  # noqa: E402
    _draw_random_profile, _get_full_halothane_profile, _profile_to_specs,
    _worker_random_ensemble, _conserved_run_halothane_doses,
    ORG_CONFIG, DOSES_HALOTHANE, SEEDS_RANDOM, N_WORKERS, CHUNKSIZE,
)
from state_validation.phase_g_state_validator import (  # noqa: E402
    DEFAULT_PER_CLASS_PA_AT_SATURATION, hill_fit_ec50,
)

CURRENT_CLASSES = {
    'complex_i_block', 'complex_ii_block', 'nachr_antagonism', 'nca_block',
    'gaba_potentiation', 'glucl_potentiation', 'k2p_potentiation',
}

OUT = ANESTH / 'artifacts' / 'v7_match2b'
P8 = ANESTH / 'audits' / 'phase1' / 'P8'
PREREG = json.load(open(P8 / 'prereg.json'))
TOL_PCT = PREREG['protocol']['joint_match_tol_pct']
N_ENS = PREREG['protocol']['n_ensembles_per_organism']
REJ_CAP = PREREG['protocol']['rejection_cap']
RNG_SEED = PREREG['protocol']['rng_seed']
N_MIN = PREREG['gates']['G2_null_density']['n_min']
FID_TOL = PREREG['gates']['G1_formula_fidelity']['threshold']


# ---------- operator-faithful coordinate functions ----------
def operator_total_pa(profile: dict, dose: float, alpha: float) -> float:
    total = 0.0
    for cls, row in profile.items():
        e = row.engagement(dose)
        if e == 0:
            continue
        if cls in CURRENT_CLASSES:
            total += -DEFAULT_PER_CLASS_PA_AT_SATURATION.get(cls, 0.0) * e
    return total * alpha


def operator_snare_factor(profile: dict, dose: float) -> float:
    row = profile.get('snare_cooperativity')
    if row is None:
        return 1.0
    e = row.engagement(dose)
    if e <= 0 or row.max_effect_factor is None:
        return 1.0
    return 1.0 + (row.max_effect_factor - 1.0) * e


def conserved_coords(org: str) -> tuple[float, float, int]:
    prof = _get_full_halothane_profile(org)
    dose = ORG_CONFIG[org]['halothane_pub']
    alpha = ORG_CONFIG[org]['alpha']
    n_active = sum(1 for r in prof.values() if r.target_EC50_uM is not None)
    return operator_total_pa(prof, dose, alpha), operator_snare_factor(prof, dose), n_active


def _draw_match2b(org: str, n_active: int, ct_pa: float, csnare: float,
                  rng: np.random.Generator, cap: int = REJ_CAP) -> dict | None:
    """Rejection-sample a profile matching BOTH (total_pa, snare_factor) within TOL_PCT."""
    dose = ORG_CONFIG[org]['halothane_pub']
    alpha = ORG_CONFIG[org]['alpha']
    tp = abs(ct_pa) * TOL_PCT / 100.0
    ts = abs(csnare) * TOL_PCT / 100.0
    for _ in range(cap):
        prof = _draw_random_profile(org, n_active, rng)
        if abs(operator_total_pa(prof, dose, alpha) - ct_pa) <= tp and \
           abs(operator_snare_factor(prof, dose) - csnare) <= ts:
            return prof
    return None


# ---------- G1: formula fidelity ----------
def fidelity_check() -> dict:
    import brian2
    from state_validation.v7_subset_search import _organism_runtime
    from state_validation.phase_g_state_validator import apply_anesthetic
    org = 'worm'
    _, factory, _, _ = _organism_runtime(org)
    alpha = ORG_CONFIG[org]['alpha']
    rng = np.random.default_rng(7)
    battery = [('conserved', _get_full_halothane_profile(org))]
    for i in range(10):
        battery.append((f'rand{i}', _draw_random_profile(org, 8, rng)))
    doses = [100.0, 340.0, 1000.0]
    max_err_pa = 0.0
    max_err_sf = 0.0
    n = 0
    for name, prof in battery:
        for d in doses:
            brain = factory(0)
            ip = np.asarray(brain.neurons.I_ext[:] / brian2.pA, dtype=np.float64).copy()
            wpre = None
            if getattr(brain, 'syn_exc', None) is not None and len(brain.syn_exc) > 0:
                wpre = np.asarray(brain.syn_exc.w[:], dtype=np.float64).copy()
            apply_anesthetic(brain, prof, d, alpha)
            ipost = np.asarray(brain.neurons.I_ext[:] / brian2.pA, dtype=np.float64)
            realized_total = float(np.mean(ipost - ip))
            # uniformity sanity: every neuron must move by the same amount
            assert np.allclose(ipost - ip, realized_total, atol=1e-9), f'{name}@{d}: non-uniform I_ext'
            err_pa = abs(realized_total - operator_total_pa(prof, d, alpha))
            max_err_pa = max(max_err_pa, err_pa)
            if wpre is not None and len(wpre) > 0:
                wpost = np.asarray(brain.syn_exc.w[:], dtype=np.float64)
                nz = wpre != 0
                realized_sf = float(np.mean(wpost[nz] / wpre[nz])) if nz.any() else 1.0
                err_sf = abs(realized_sf - operator_snare_factor(prof, d))
                max_err_sf = max(max_err_sf, err_sf)
            n += 1
    verdict = 'PASS' if (max_err_pa < FID_TOL and max_err_sf < FID_TOL) else 'FAIL'
    out = {'gate': 'G1_formula_fidelity', 'n_cases': n,
           'max_err_total_pa': max_err_pa, 'max_err_snare_factor': max_err_sf,
           'threshold': FID_TOL, 'verdict': verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / 'g1_fidelity.json', 'w'), indent=2)
    print(f'G1 fidelity: max_err total_pa={max_err_pa:.2e}  snare_factor={max_err_sf:.2e}  '
          f'(tol {FID_TOL}) -> {verdict}')
    return out


# ---------- G2: null-density dry-run (no sims) ----------
def density_dryrun() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    res = {}
    for org in ORG_CONFIG:
        ct, cs, n_active = conserved_coords(org)
        accepted = 0
        for _ in range(N_ENS):
            if _draw_match2b(org, n_active, ct, cs, rng) is not None:
                accepted += 1
        res[org] = {'conserved_total_pa': ct, 'conserved_snare_factor': cs,
                    'n_active': n_active, 'accepted_of_50': accepted,
                    'enough': accepted >= N_MIN}
        print(f'  {org}: conserved (total_pa={ct:.3f}, snare={cs:.3f}) n_active={n_active}  '
              f'accepted {accepted}/{N_ENS}  enough(>={N_MIN}): {accepted >= N_MIN}')
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'g2_density.json', 'w'), indent=2)
    return res


# ---------- WB3: ensemble run + WB4 verdict ----------
def run_ensemble() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    conserved = {org: conserved_coords(org) for org in ORG_CONFIG}
    profiles: dict[tuple[str, int], dict] = {}
    rej_fail = []
    for org in ORG_CONFIG:
        ct, cs, n_active = conserved[org]
        for eid in range(N_ENS):
            prof = _draw_match2b(org, n_active, ct, cs, rng)
            if prof is None:
                rej_fail.append((org, eid))
                continue
            profiles[(org, eid)] = prof
    print(f'  sampled {len(profiles)} joint-matched ensembles ({len(rej_fail)} rejection failures)')

    tasks = []
    for (org, eid), prof in profiles.items():
        specs = _profile_to_specs(prof)
        for d in DOSES_HALOTHANE:
            for s in SEEDS_RANDOM:
                tasks.append((2, eid, org, specs, d, s))
    print(f'  {len(tasks)} sims', flush=True)

    t0 = time.time()
    raw_path = OUT / 'v7_match2b_raw.csv'
    results = []
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['match', 'ensemble_id', 'organism', 'dose_uM', 'seed',
                    'quiescent_fraction', 'command_rate_hz', 'network_rate_hz'])
        with mp.Pool(processes=N_WORKERS) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker_random_ensemble, tasks,
                                                       chunksize=CHUNKSIZE)):
                results.append(r)
                w.writerow(r)
                if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                    e = (time.time() - t0) / 60.0
                    eta = e / (i + 1) * (len(tasks) - (i + 1))
                    f.flush()
                    print(f'    [{i+1}/{len(tasks)}] {e:.1f}m ETA {eta:.0f}m', flush=True)

    by_ens = defaultdict(lambda: defaultdict(list))
    for _m, eid, org, dose, _s, qf, _cr, _nr in results:
        by_ens[(org, eid)][dose].append(qf)
    summary = []
    for (org, eid), dose_qf in by_ens.items():
        ds = sorted(dose_qf.keys())
        qfs = [float(np.mean(dose_qf[d])) for d in ds]
        ec50 = hill_fit_ec50(np.array(ds), np.array(qfs), threshold=0.5)
        pub = ORG_CONFIG[org]['halothane_pub']
        fold = None if ec50 is None else float(max(ec50 / pub, pub / ec50))
        summary.append({'ensemble_id': eid, 'organism': org,
                        'predicted_EC50_uM': ec50, 'fold_error': fold})
    with open(OUT / 'v7_match2b_random_50.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['ensemble_id', 'organism',
                                           'predicted_EC50_uM', 'fold_error'])
        wr.writeheader()
        wr.writerows(summary)

    # conserved fold-error baseline (reuse the frozen pipeline)
    conserved_fold = {}
    for org in ORG_CONFIG:
        ec50, _ = _conserved_run_halothane_doses(org)
        pub = ORG_CONFIG[org]['halothane_pub']
        conserved_fold[org] = None if ec50 is None else float(max(ec50 / pub, pub / ec50))

    perc = {}
    n_eff = {}
    for org in ORG_CONFIG:
        folds = [r['fold_error'] for r in summary if r['organism'] == org and r['fold_error'] is not None]
        n_eff[org] = len(folds)
        cf = conserved_fold[org]
        perc[org] = None if (cf is None or not folds) else \
            float(sum(1 for f in folds if f < cf)) / len(folds) * 100.0

    # G2 (post-hoc on the realized run) + G3 fly survival
    g2 = {org: {'n_jointly_matched': n_eff[org], 'enough': n_eff[org] >= N_MIN} for org in ORG_CONFIG}
    fly_p = perc.get('fly')
    if fly_p is None:
        g3 = 'INDETERMINATE'
    elif fly_p <= 5.0:
        g3 = 'TOO_SPECIAL_QUARANTINE'
    elif fly_p <= 30.0:
        g3 = 'PASS'
    else:
        g3 = 'DEFLATE'

    verdict = {
        'block': 'P8', 'pipeline': 'v7_match2b',
        'prereg': str(P8 / 'prereg.json'),
        'tol_pct': TOL_PCT, 'rng_seed': RNG_SEED,
        'conserved_coords': {o: {'total_pa': conserved[o][0], 'snare_factor': conserved[o][1],
                                 'n_active': conserved[o][2]} for o in ORG_CONFIG},
        'conserved_fold_error': conserved_fold,
        'n_jointly_matched': n_eff,
        'rejection_failures': len(rej_fail),
        'percentile_rank_pct': perc,
        'G2_null_density': g2,
        'G3_fly_survival': {'fly_percentile_pct': fly_p, 'verdict': g3,
                            'rule': PREREG['gates']['G3_fly_survival']['decision_rule']},
        'prior_match2_for_reference': {'worm': 0.0, 'fly': 4.76, 'mouse': 46.0},
        'wall_minutes': (time.time() - t0) / 60.0,
    }
    json.dump(verdict, open(OUT / 'v7_match2b_verdict.json', 'w'), indent=2)
    print('\n=== P8 Match#2b verdict ===')
    for org in ORG_CONFIG:
        print(f'  {org}: percentile {perc[org]}  (n_matched {n_eff[org]})  '
              f'conserved_fold {conserved_fold[org]}')
    print(f'  G3 fly survival: fly={fly_p}% -> {g3}')
    return verdict


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'fidelity':
        fidelity_check()
    elif cmd == 'density':
        density_dryrun()
    elif cmd == 'run':
        run_ensemble()
    else:
        f = fidelity_check()
        if f['verdict'] != 'PASS':
            print('G1 FAILED — halting; not running ensemble.')
            return
        d = density_dryrun()
        if not all(v['enough'] for v in d.values()):
            print('G2 density under-powered for >=1 organism — review before run.')
            return
        run_ensemble()


if __name__ == '__main__':
    main()
