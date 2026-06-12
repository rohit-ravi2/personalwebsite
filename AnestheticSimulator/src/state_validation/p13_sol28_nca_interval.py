"""p13_sol28_nca_interval — P13-SOL28 nca biophysics-frozen interval magnitude sweep.

LOCAL, load-bearing half of P13 (does NOT wait on Greene SOL27 / ESMFold).

Question: does the nca_block current MAGNITUDE matter to the worm quorum?
The legacy value (40 pA) is a docking-orphaned hand-set scalar. We derive a
biophysically-principled interval [75,120] pA from FROZEN passive constants and
re-simulate the 64 nca-containing worm subsets at the worst-case (lo) edge,
75 pA, to test whether the quorum survives or was resting on the hand-set value.

NON-DESTRUCTIVE / NEW CODE PATH:
  - The frozen operator state_validation.phase_g_state_validator.apply_anesthetic
    is NOT edited.
  - We override ONLY DEFAULT_PER_CLASS_PA_AT_SATURATION['nca_block'] in-process at
    worker start. The 7 non-nca scalars are left untouched and asserted
    byte-identical (G3 leak screen).
  - The Stage-1 pass criterion, frozen alpha, doses, seeds, brain factory, and
    subset-profile builder are REUSED unchanged from v7_subset_search.

Gates (see audits/phase1/P13-SOL28/prereg.json):
  G1 interval-provenance  (FAST)  — interval == closed-form of frozen disk constants.
  G2 quorum-survival      (HEAVY) — at 75 pA, passing nca subsets >= 80% of baseline
                                    AND SNARE-OR-ComplexI universality holds.
  G3 substitution-leak    (FAST)  — 7 non-nca scalars byte-identical across endpoints.

Usage:
  fast gates only:   python p13_sol28_nca_interval.py --fast
  smoke re-sim:      python p13_sol28_nca_interval.py --smoke
  heavy re-sim:      <ml-python> p13_sol28_nca_interval.py --heavy [--nca-pa 75.0]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/rohit/Desktop/website/personalwebsite')
ANESTH = ROOT / 'AnestheticSimulator'
SRC = ANESTH / 'src'
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / 'scripts'))

AUDIT_DIR = ANESTH / 'audits' / 'phase1' / 'P13-SOL28'
ART_DIR = ANESTH / 'artifacts' / 'p13_sol28'

# ---- Frozen disk constants (G1 source of truth = scripts/brain/lif_brain.py) ----
LIF_BRAIN_FILE = ROOT / 'scripts' / 'brain' / 'lif_brain.py'

# ---- Sweep config (REUSED from v7_subset_search, worm only) ----
DOSES_HALOTHANE = [10.0, 30.0, 100.0, 200.0, 350.0, 500.0, 1000.0, 3000.0]
SEEDS = [42, 137, 219, 331, 443]
SIM_DUR_S = 30.0
PASS_FOLD_TOL = 2.0
HALOTHANE_PUB_UM = 340.0
ALPHA_WORM = 0.13
N_WORKERS = 12
CHUNKSIZE = 4

# ---- Frozen non-nca scalar fingerprint (G3) ----
NON_NCA_SHA256 = '54788ddcf622669910f9f6738d8b84167346a3e80f96c1982c26d345388c5478'

# ---- Interval (G1) ----
BLOCK_FRACTION_LO = 0.5
BLOCK_FRACTION_HI = 0.8
LEGACY_NCA_PA = 40.0


# ============================================================
# G1: interval provenance — closed form of frozen disk constants
# ============================================================

def _parse_frozen_constants() -> dict:
    """Byte-parse tau, C_mem, v_rest, v_thr from lif_brain.py keyword defaults."""
    text = LIF_BRAIN_FILE.read_text()
    import re

    def grab(pattern):
        m = re.search(pattern, text)
        if not m:
            raise RuntimeError(f'could not parse {pattern!r} from {LIF_BRAIN_FILE}')
        return float(m.group(1))

    # e.g. "tau     = 10 * ms," / "v_rest  = -25 * mV," / "C_MEM_DEFAULT = 100 * pF"
    tau_ms = grab(r'tau\s*=\s*(-?\d+(?:\.\d+)?)\s*\*\s*ms')
    v_rest_mV = grab(r'v_rest\s*=\s*(-?\d+(?:\.\d+)?)\s*\*\s*mV')
    v_thr_mV = grab(r'v_thr\s*=\s*(-?\d+(?:\.\d+)?)\s*\*\s*mV')
    c_mem_pF = grab(r'C_MEM_DEFAULT\s*=\s*(-?\d+(?:\.\d+)?)\s*\*\s*pF')
    return {'tau_ms': tau_ms, 'C_mem_pF': c_mem_pF,
            'v_rest_mV': v_rest_mV, 'v_thr_mV': v_thr_mV}


def derive_interval() -> dict:
    """Closed-form derivation of [I_lo, I_hi] pA from frozen passive constants."""
    c = _parse_frozen_constants()
    g_leak_nS = c['C_mem_pF'] / c['tau_ms']           # 100/10 = 10 nS
    D_mV = c['v_thr_mV'] - c['v_rest_mV']             # -10 - (-25) = 15 mV
    gD_pA = g_leak_nS * D_mV                          # nS*mV = pA -> 150
    I_lo = BLOCK_FRACTION_LO * gD_pA
    I_hi = BLOCK_FRACTION_HI * gD_pA
    legacy_bf = LEGACY_NCA_PA / gD_pA
    return {
        **c,
        'g_leak_nS': g_leak_nS, 'D_mV': D_mV, 'g_leak_times_D_pA': gD_pA,
        'block_fraction_lo': BLOCK_FRACTION_LO, 'block_fraction_hi': BLOCK_FRACTION_HI,
        'interval_lo_pA': I_lo, 'interval_hi_pA': I_hi,
        'legacy_pA': LEGACY_NCA_PA, 'legacy_block_fraction': legacy_bf,
        'legacy_below_lo': legacy_bf < BLOCK_FRACTION_LO,
    }


def gate_g1() -> dict:
    d = derive_interval()
    checks = {
        'tau_ms==10.0': d['tau_ms'] == 10.0,
        'C_mem_pF==100.0': d['C_mem_pF'] == 100.0,
        'v_rest_mV==-25.0': d['v_rest_mV'] == -25.0,
        'v_thr_mV==-10.0': d['v_thr_mV'] == -10.0,
        'g_leak_nS==10.0': d['g_leak_nS'] == 10.0,
        'D_mV==15.0': d['D_mV'] == 15.0,
        'g_leak*D==150.0': d['g_leak_times_D_pA'] == 150.0,
        'interval_lo==75.0': d['interval_lo_pA'] == 75.0,
        'interval_hi==120.0': d['interval_hi_pA'] == 120.0,
        'legacy_40_below_lo': d['legacy_below_lo'],
        'legacy_outside_interval': not (d['interval_lo_pA'] <= LEGACY_NCA_PA <= d['interval_hi_pA']),
    }
    return {'gate': 'G1_interval_provenance', 'verdict': 'PASS' if all(checks.values()) else 'FAIL',
            'derivation': d, 'checks': checks}


# ============================================================
# G3: substitution-leak screen — 7 non-nca scalars byte-identical
# ============================================================

def _non_nca_sha(scalar_dict) -> str:
    non_nca = {k: v for k, v in scalar_dict.items() if k != 'nca_block'}
    return hashlib.sha256(json.dumps(non_nca, sort_keys=True).encode()).hexdigest()


def set_nca_override(nca_pa: float) -> dict:
    """NEW code path: patch ONLY nca_block in the per-class scalar dict in-process.
    Returns a fingerprint dict for the G3 leak screen. Does NOT touch the operator."""
    from state_validation import phase_g_state_validator as pg
    before_non_nca = _non_nca_sha(pg.DEFAULT_PER_CLASS_PA_AT_SATURATION)
    pg.DEFAULT_PER_CLASS_PA_AT_SATURATION['nca_block'] = float(nca_pa)
    after_non_nca = _non_nca_sha(pg.DEFAULT_PER_CLASS_PA_AT_SATURATION)
    return {
        'nca_block_now': pg.DEFAULT_PER_CLASS_PA_AT_SATURATION['nca_block'],
        'non_nca_sha_before': before_non_nca,
        'non_nca_sha_after': after_non_nca,
        'non_nca_sha_expected': NON_NCA_SHA256,
        'non_nca_unchanged': before_non_nca == after_non_nca == NON_NCA_SHA256,
    }


def gate_g3(nca_pa: float = 75.0) -> dict:
    """Fast: load baseline scalars, apply override, confirm only nca changed."""
    from state_validation import phase_g_state_validator as pg
    baseline_non_nca = _non_nca_sha(pg.DEFAULT_PER_CLASS_PA_AT_SATURATION)
    baseline_nca = pg.DEFAULT_PER_CLASS_PA_AT_SATURATION['nca_block']
    fp = set_nca_override(nca_pa)
    # restore so the fast-gate process leaves no residue
    pg.DEFAULT_PER_CLASS_PA_AT_SATURATION['nca_block'] = baseline_nca
    checks = {
        'baseline_nca==40.0': baseline_nca == LEGACY_NCA_PA,
        'baseline_non_nca_hash_matches': baseline_non_nca == NON_NCA_SHA256,
        'override_set_nca': fp['nca_block_now'] == float(nca_pa),
        'non_nca_byte_identical': fp['non_nca_unchanged'],
    }
    return {'gate': 'G3_substitution_leak_screen', 'nca_pa': nca_pa,
            'verdict': 'PASS' if all(checks.values()) else 'FAIL',
            'fingerprint': fp, 'baseline_nca': baseline_nca, 'checks': checks}


# ============================================================
# Subset enumeration — 64 nca-containing worm subsets
# ============================================================

def nca_worm_subsets() -> list[tuple[str, ...]]:
    """All non-empty worm subsets that contain nca_block (64 of them)."""
    from state_validation.v7_subset_search import enumerate_subsets, ORG_CONFIG
    worm_classes = ORG_CONFIG['worm']['mech_classes']
    return [s for s in enumerate_subsets(worm_classes) if 'nca_block' in s]


# ============================================================
# Heavy worker: re-sim one (subset, dose, seed) at overridden nca
# ============================================================

def _worker_nca_subset(args):
    """args = (subset_csv, dose, seed, nca_pa). Patches nca scalar, runs LIF."""
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(ROOT / 'scripts'))
    subset_csv, dose, seed, nca_pa = args
    # NEW code path: override nca scalar in THIS worker process before any sim.
    fp = set_nca_override(nca_pa)
    if not fp['non_nca_unchanged']:
        raise RuntimeError(f'G3 leak in worker: {fp}')

    from state_validation.phase_g_state_validator import run_single
    from state_validation.v7_subset_search import (
        _organism_runtime, _get_full_profile, _build_subset_profile,
    )
    _, factory, qf_thr, cmd_set = _organism_runtime('worm')
    full_profile = _get_full_profile('worm')
    subset_set = set(subset_csv.split('|'))
    subset_profile = _build_subset_profile(full_profile, subset_set)
    m = run_single(
        anesthetic='halothane', dose_uM=dose, seed=seed,
        sim_duration_s=SIM_DUR_S, profile=subset_profile, mutant=None,
        alpha_calib=ALPHA_WORM, brain_factory=factory,
        quiescent_threshold_hz=qf_thr, command_set=cmd_set,
    )
    return (subset_csv, dose, seed, float(nca_pa),
            float(m['quiescent_fraction']),
            float(m['command_mean_firing_rate_hz']),
            float(m['network_mean_firing_rate_hz']),
            fp['non_nca_sha_after'])


# ============================================================
# Aggregate + G2 verdict
# ============================================================

def _hill_pass(dose_qf: dict) -> tuple[float | None, float | None, bool]:
    import numpy as np
    from state_validation.phase_g_state_validator import hill_fit_ec50
    ds = sorted(dose_qf.keys())
    qfs = [float(np.mean(dose_qf[d])) for d in ds]
    ec50 = hill_fit_ec50(np.array(ds), np.array(qfs), threshold=0.5)
    if ec50 is None:
        return None, None, False
    fold = float(max(ec50 / HALOTHANE_PUB_UM, HALOTHANE_PUB_UM / ec50))
    return ec50, fold, fold <= PASS_FOLD_TOL


def run_heavy(nca_pa: float = 75.0, smoke: bool = False) -> dict:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    subsets = nca_worm_subsets()
    if smoke:
        subsets = subsets[:1]
        doses = [200.0, 1000.0]
        seeds = [42]
    else:
        doses = DOSES_HALOTHANE
        seeds = SEEDS

    tasks = [(  '|'.join(s), d, sd, nca_pa)
             for s in subsets for d in doses for sd in seeds]
    tag = f'{int(round(nca_pa))}pA' + ('_smoke' if smoke else '')
    print(f'P13-SOL28 nca={nca_pa} pA  subsets={len(subsets)} '
          f'doses={len(doses)} seeds={len(seeds)} -> {len(tasks)} sims '
          f'workers={N_WORKERS}', flush=True)

    t0 = time.time()
    results = []
    raw_path = ART_DIR / f'p13_sol28_nca_{tag}_raw.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'dose_uM', 'seed', 'nca_pa',
                    'quiescent_fraction', 'command_rate_hz', 'network_rate_hz',
                    'non_nca_sha'])
        with mp.Pool(processes=N_WORKERS) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker_nca_subset, tasks,
                                                      chunksize=CHUNKSIZE)):
                results.append(r)
                w.writerow(r)
                if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                    el = (time.time() - t0) / 60.0
                    eta = el / (i + 1) * (len(tasks) - (i + 1))
                    f.flush()
                    print(f'    [{i+1}/{len(tasks)}] {el:.1f}m  ETA {eta:.0f}m', flush=True)

    # G3 leak screen across ALL rows: every non_nca_sha must equal frozen fingerprint
    shas = {r[7] for r in results}
    g3_heavy_ok = shas == {NON_NCA_SHA256}

    # Aggregate per subset on the primary endpoint (quiescent_fraction)
    agg = defaultdict(lambda: defaultdict(list))
    for subset_csv, dose, seed, _npa, qf, _cr, _nr, _sha in results:
        agg[subset_csv][dose].append(qf)

    passing = []
    summary = []
    for subset_csv, dose_qf in agg.items():
        ec50, fold, ok = _hill_pass(dose_qf)
        s = subset_csv.split('|')
        summary.append({'subset': subset_csv, 'n_classes': len(s),
                        'predicted_EC50_uM': ec50, 'fold_error': fold,
                        'passes': ok,
                        'has_snare': 'snare_cooperativity' in s,
                        'has_complex_i': 'complex_i_block' in s})
        if ok:
            passing.append(s)

    # Universality: every passing subset has SNARE OR Complex-I
    universality_ok = all(('snare_cooperativity' in s) or ('complex_i_block' in s)
                          for s in passing)
    n_pass = len(passing)
    baseline_n = 15
    threshold = -(-int(0.8 * baseline_n) // 1)  # ceil(0.8*15)=12
    import math
    threshold = math.ceil(0.8 * baseline_n)
    g2_pass = (n_pass >= threshold) and universality_ok and (not smoke)

    summary_path = ART_DIR / f'p13_sol28_nca_{tag}_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        wd = csv.DictWriter(f, fieldnames=['subset', 'n_classes', 'predicted_EC50_uM',
                                           'fold_error', 'passes', 'has_snare',
                                           'has_complex_i'])
        wd.writeheader()
        for row in sorted(summary, key=lambda r: (r['n_classes'], r['subset'])):
            wd.writerow(row)

    verdict = {
        'block': 'P13-SOL28', 'nca_pa': nca_pa, 'smoke': smoke,
        'n_nca_subsets': len(subsets), 'n_sims': len(tasks),
        'wall_minutes': (time.time() - t0) / 60.0,
        'baseline_passing_count': baseline_n,
        'survival_threshold_count': threshold,
        'n_passing_at_nca': n_pass,
        'snare_or_complexi_universality': universality_ok,
        'g2_quorum_survival': 'PASS' if g2_pass else ('SMOKE' if smoke else 'FAIL'),
        'g3_heavy_leak_screen': 'PASS' if g3_heavy_ok else 'FAIL',
        'g3_distinct_non_nca_shas': sorted(shas),
        'raw': str(raw_path), 'summary': str(summary_path),
        'deflation_note': (
            None if g2_pass else
            'G2 FAIL -> nca quorum membership was scalar-dependent; magnitude DOES '
            'matter and NALCN has no Kd (uncalibratable) -> Tier4-adjacent parked residue.'
        ),
    }
    vp = ART_DIR / f'p13_sol28_nca_{tag}_verdict.json'
    vp.write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: verdict[k] for k in
                      ('nca_pa', 'n_passing_at_nca', 'survival_threshold_count',
                       'snare_or_complexi_universality', 'g2_quorum_survival',
                       'g3_heavy_leak_screen', 'wall_minutes')}, indent=2), flush=True)
    return verdict


# ============================================================
# Fast-gate driver
# ============================================================

def run_fast() -> dict:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    g1 = gate_g1()
    g3 = gate_g3(nca_pa=75.0)
    out = {'block': 'P13-SOL28', 'fast_gates': {'G1': g1, 'G3': g3}}
    (AUDIT_DIR / 'fast_gate_results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'G1': g1['verdict'], 'G3': g3['verdict']}, indent=2))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='run FAST gates G1+G3 only')
    ap.add_argument('--smoke', action='store_true', help='tiny re-sim smoke (harness wiring)')
    ap.add_argument('--heavy', action='store_true', help='full 64x8x5 re-sim (ml env)')
    ap.add_argument('--nca-pa', type=float, default=75.0)
    args = ap.parse_args()
    if args.fast:
        run_fast()
    elif args.smoke:
        run_heavy(nca_pa=args.nca_pa, smoke=True)
    elif args.heavy:
        run_heavy(nca_pa=args.nca_pa, smoke=False)
    else:
        run_fast()
