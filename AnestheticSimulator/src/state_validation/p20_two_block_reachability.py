"""p20_two_block_reachability — Genotype x anesthetic two-block reachability partition (WF2 P20).

NON-DESTRUCTIVE: this is a NEW read-only consumer of the frozen operator
(phase_g_state_validator.apply_genotype / apply_anesthetic). It never mutates the operator,
the frozen v7 ensembles, or any frozen artifact. All outputs go to artifacts/p20/.

Two-block composition (verified by code-read, premise of P18 PASS):
  apply_genotype, then apply_anesthetic, are the ONLY state mutations between brain
  construction and brain.run(). They decompose into exactly two genotype blocks acting on
  DISJOINT operator coordinates:

  block-I (I_ext additive, 7 genotypes: gas-1/gas-2/nduf-6/ndus-8/nuo-1/unc-79/unc-80):
      apply_genotype ADDS geno_pa to I_ext[:]; apply_anesthetic ADDS total_pa(dose) to the
      same I_ext[:]. Net = geno_pa + total_pa(dose) => a HORIZONTAL TRANSLATION of the
      volatile dose-response => an EC50 translation. geno_pa is a single uniform scalar.
        complex_i:  ci_pa  = -50.0 * (1 - complex_i_factor) * alpha
        nca:        nca_pa = -30.0 * (1 - nca_leak_factor)  * alpha
        k2p (fly):  k2p_pa = +30.0 * (1 - k2p_baseline_factor) * alpha   (no worm genotypes)

  block-S (synaptic multiplicative, 2 Gao genotypes: goa-1, dgk-1):
      apply_genotype MULTIPLIES syn.w by wsyn_global_factor; apply_anesthetic SEPARATELY
      MULTIPLIES syn.w by snare_factor(dose). Net synaptic gain = wsyn * snare_factor(dose),
      a MULTIPLICATIVE composition on a coordinate block-I never touches.

  The two blocks act on disjoint coordinates (I_ext vs syn.w) => they COMMUTE, and neither
  adds a spatial DOF beyond the operator's two global broadcasts. GATE-A asks whether any
  held-out genotype EC50-ratio band sits OUTSIDE this operator's achievable envelope.

Prereg + gates: audits/phase1/P20/prereg.json  (frozen before the run).

CLI:
  cert     — WB1 analytic rank-certificate + GATE-B fidelity (FAST, no brain.run except 1 build)
  smoke    — 1-genotype x 1-volatile x 2-dose x 1-seed end-to-end LIF smoke
  build    — battery task-graph build + count assertions (FAST, no sim)
  battery  — HEAVY ~1200-sim battery + envelope sweeps + GATE-A/C verdict (ml env, ~2hr)

Env: /home/rohit/miniconda3/envs/ml/bin/python  (brian2) for cert/smoke/battery.
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
sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')

from state_validation.phase_g_state_validator import (  # noqa: E402
    DEFAULT_PER_CLASS_PA_AT_SATURATION,
)

OUT = ANESTH / 'artifacts' / 'p20'
P20 = ANESTH / 'audits' / 'phase1' / 'P20'
PREREG = json.load(open(P20 / 'prereg.json'))

CURRENT_CLASSES = {
    'complex_i_block', 'complex_ii_block', 'nachr_antagonism', 'nca_block',
    'gaba_potentiation', 'glucl_potentiation', 'k2p_potentiation',
}

# Frozen from prereg
BLOCK_I_GENOS = PREREG['two_block_partition']['block_I_Iext_additive']['genotypes']
BLOCK_S_GENOS = PREREG['two_block_partition']['block_S_synaptic_multiplicative']['genotypes']
DEFERRED = PREREG['two_block_partition']['deferred_excluded']['genotypes']
LIT_BANDS = PREREG['held_out_literature_bands']['bands']
VOLATILES = PREREG['protocol']['WB2_live_battery']['volatiles']
DOSES = PREREG['protocol']['WB2_live_battery']['doses_uM']
SEEDS = PREREG['protocol']['WB2_live_battery']['seeds']
GENOS_ACTIVE = BLOCK_I_GENOS + BLOCK_S_GENOS
GENOS_BATTERY = ['WT'] + GENOS_ACTIVE
SIM_DUR_S = PREREG['protocol']['WB2_live_battery']['sim_duration_s']
ALPHA = PREREG['protocol']['WB2_live_battery']['alpha_frozen']
N_WORKERS = PREREG['protocol']['WB2_live_battery']['workers']
CHUNKSIZE = PREREG['protocol']['WB2_live_battery']['chunksize']
FID_TOL_B = PREREG['gates']['P20_GATE_B_gao_multiplicative']['fidelity_tol']
TOO_SPECIAL_PCT = PREREG['gates']['P20_GATE_C_too_special_circularity']['threshold_pct']
MUTANT_CSV = ANESTH / 'data' / 'state_validation' / 'mutant_baseline_perturbations.csv'
PERT_CSV = ANESTH / 'data' / 'state_validation' / 'anesthetic_perturbation_table.csv'


# ---------- operator-faithful coordinate mirrors ----------
def operator_total_pa(profile: dict, dose: float, alpha: float) -> float:
    """Mirror of apply_anesthetic I_ext contribution (lines 308-324)."""
    total = 0.0
    for cls, row in profile.items():
        e = row.engagement(dose)
        if e == 0:
            continue
        if cls in CURRENT_CLASSES:
            total += -DEFAULT_PER_CLASS_PA_AT_SATURATION.get(cls, 0.0) * e
    return total * alpha


def operator_snare_factor(profile: dict, dose: float) -> float:
    """Mirror of apply_anesthetic SNARE multiplier (lines 327-335)."""
    row = profile.get('snare_cooperativity')
    if row is None:
        return 1.0
    e = row.engagement(dose)
    if e <= 0 or row.max_effect_factor is None:
        return 1.0
    return 1.0 + (row.max_effect_factor - 1.0) * e


def operator_geno_pa(mutant, alpha: float) -> float:
    """Mirror of apply_genotype I_ext contribution (block-I). Returns the additive scalar pA."""
    pa = 0.0
    if mutant.complex_i_factor < 1.0:
        pa += -50.0 * (1.0 - mutant.complex_i_factor) * alpha
    if mutant.nca_leak_factor < 1.0:
        pa += -30.0 * (1.0 - mutant.nca_leak_factor) * alpha
    if getattr(mutant, 'k2p_baseline_factor', 1.0) < 1.0:
        pa += +30.0 * (1.0 - mutant.k2p_baseline_factor) * alpha
    return pa


def _load_tables():
    from state_validation.phase_g_state_validator import (
        load_perturbation_table, load_mutant_table,
    )
    profiles = load_perturbation_table(PERT_CSV)   # {anesthetic: {cls: row}}
    mutants = load_mutant_table(MUTANT_CSV)         # {gene: MutantBaseline}
    return profiles, mutants


def _worm_factory():
    from brain.lif_brain import LIFBrain

    def factory(seed):
        class SeededLIF(LIFBrain):
            _brian2_seed = seed
        return SeededLIF(use_per_edge_glu_signs=True)
    return factory


# ---------- WB1 analytic rank-certificate + GATE-B fidelity (FAST) ----------
def cert() -> dict:
    """Operator-immediate certificate: NO integration. One brain build per case, snapshot
    pre/post I_ext and syn.w, assert block-I additive-translation + block-S multiplicative."""
    import brian2
    from state_validation.phase_g_state_validator import apply_genotype, apply_anesthetic
    OUT.mkdir(parents=True, exist_ok=True)
    profiles, mutants = _load_tables()
    factory = _worm_factory()

    test_doses = [100.0, 340.0, 1000.0]
    max_err_blockI = 0.0      # |realized I_ext delta - (geno_pa + total_pa)|
    max_nonuniform_I = 0.0    # spatial non-uniformity of the I_ext write
    max_err_blockS = 0.0      # |realized syn.w ratio - wsyn*snare_factor|
    max_nonuniform_S = 0.0
    min_mult_add_sep = float('inf')  # min |R_mult - R_add| over tested (geno,dose) where snare engaged
    rows = []

    for vol in VOLATILES:
        prof = profiles[vol]
        for gene in GENOS_ACTIVE:
            mut = mutants[gene]
            for d in test_doses:
                brain = factory(0)
                I_pre = np.asarray(brain.neurons.I_ext[:] / brian2.pA, dtype=np.float64).copy()
                w_pre = None
                if getattr(brain, 'syn_exc', None) is not None and len(brain.syn_exc) > 0:
                    w_pre = np.asarray(brain.syn_exc.w[:], dtype=np.float64).copy()
                # frozen operator path, exactly as run_single calls it
                apply_genotype(brain, mut, ALPHA)
                apply_anesthetic(brain, prof, d, ALPHA)
                I_post = np.asarray(brain.neurons.I_ext[:] / brian2.pA, dtype=np.float64)

                dI = I_post - I_pre
                realized_dI = float(np.mean(dI))
                nonuniform_I = float(np.max(np.abs(dI - realized_dI)))
                max_nonuniform_I = max(max_nonuniform_I, nonuniform_I)

                geno_pa = operator_geno_pa(mut, ALPHA)
                tot_pa = operator_total_pa(prof, d, ALPHA)
                predicted_dI = geno_pa + tot_pa
                err_I = abs(realized_dI - predicted_dI)
                max_err_blockI = max(max_err_blockI, err_I)

                # block-S realized synaptic ratio
                realized_sf = 1.0
                if w_pre is not None and len(w_pre) > 0:
                    w_post = np.asarray(brain.syn_exc.w[:], dtype=np.float64)
                    nz = w_pre != 0
                    if nz.any():
                        ratios = w_post[nz] / w_pre[nz]
                        realized_sf = float(np.mean(ratios))
                        nonuniform_S = float(np.max(np.abs(ratios - realized_sf)))
                        max_nonuniform_S = max(max_nonuniform_S, nonuniform_S)

                wsyn = mut.wsyn_global_factor
                snare = operator_snare_factor(prof, d)
                R_mult = wsyn * snare
                R_add = wsyn + (snare - 1.0)
                err_S = abs(realized_sf - R_mult)
                max_err_blockS = max(max_err_blockS, err_S)
                # mult-vs-additive separation is only meaningful for block-S genotypes:
                # block-I genos have wsyn==1 => R_mult==R_add trivially (no synaptic DOF),
                # which is correct, not a GATE-B failure. GATE-B is about the Gao block.
                if gene in BLOCK_S_GENOS and snare != 1.0:
                    min_mult_add_sep = min(min_mult_add_sep, abs(R_mult - R_add))

                rows.append({
                    'volatile': vol, 'gene': gene, 'dose': d, 'block':
                        'I' if gene in BLOCK_I_GENOS else 'S',
                    'geno_pa': geno_pa, 'total_pa': tot_pa,
                    'realized_dI_pA': realized_dI, 'predicted_dI_pA': predicted_dI,
                    'err_blockI_pA': err_I, 'nonuniform_I_pA': nonuniform_I,
                    'wsyn': wsyn, 'snare_factor': snare,
                    'R_mult': R_mult, 'R_add': R_add,
                    'realized_synaptic_ratio': realized_sf, 'err_blockS': err_S,
                    'nonuniform_S': nonuniform_S if w_pre is not None else 0.0,
                })

    # WB1 verdict: additive-translation + multiplicative + spatial uniformity + disjoint coords
    blockI_additive_ok = (max_err_blockI < 1e-9) and (max_nonuniform_I < 1e-9)
    blockS_mult_ok = (max_err_blockS < 1e-9) and (max_nonuniform_S < 1e-9)
    wb1_verdict = 'PASS' if (blockI_additive_ok and blockS_mult_ok) else 'FAIL'

    # GATE-B precondition: multiplicative reproduced + separable from additive null
    gateB_fidelity = max_err_blockS < FID_TOL_B
    gateB_separable = (min_mult_add_sep != float('inf')) and (min_mult_add_sep > 1e-6)
    gateB = 'PASS' if (gateB_fidelity and gateB_separable) else 'FAIL'

    out = {
        'block': 'P20', 'stage': 'WB1_cert + GATE-B-fidelity',
        'n_cases': len(rows),
        'WB1_analytic_certificate': {
            'block_I_max_err_pA': max_err_blockI,
            'block_I_max_nonuniform_pA': max_nonuniform_I,
            'block_I_additive_translation_ok': blockI_additive_ok,
            'block_S_max_err_ratio': max_err_blockS,
            'block_S_max_nonuniform': max_nonuniform_S,
            'block_S_multiplicative_ok': blockS_mult_ok,
            'threshold': 1e-9,
            'verdict': wb1_verdict,
        },
        'GATE_B_gao_multiplicative_precondition': {
            'max_err_synaptic_ratio_vs_mult': max_err_blockS,
            'fidelity_tol': FID_TOL_B,
            'min_mult_minus_add_separation': (None if min_mult_add_sep == float('inf')
                                              else min_mult_add_sep),
            'separable_from_additive_null': gateB_separable,
            'verdict': gateB,
            'note': ('Load-bearing structural half of GATE-B settled here on the realized '
                     'syn.w ratio; the heavy battery only confirms the dose-response shape.'),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / 'p20_cert.json', 'w'), indent=2)
    with open(OUT / 'p20_cert_cases.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f'WB1 block-I additive: max_err={max_err_blockI:.2e} pA  '
          f'nonuniform={max_nonuniform_I:.2e} pA  -> {blockI_additive_ok}')
    print(f'WB1 block-S multiplicative: max_err={max_err_blockS:.2e}  '
          f'nonuniform={max_nonuniform_S:.2e}  -> {blockS_mult_ok}')
    print(f'WB1 verdict: {wb1_verdict}')
    print(f'GATE-B fidelity: max_err={max_err_blockS:.2e} (tol {FID_TOL_B})  '
          f'mult-add sep={min_mult_add_sep:.4f}  -> {gateB}')
    return out


# ---------- battery task graph (FAST build) ----------
def build_battery() -> list[tuple]:
    """Build the (genotype, volatile, dose, seed) task list; assert counts."""
    tasks = []
    for gene in GENOS_BATTERY:
        for vol in VOLATILES:
            for d in DOSES:
                for s in SEEDS:
                    tasks.append((gene, vol, d, s))
    n_expected = len(GENOS_BATTERY) * len(VOLATILES) * len(DOSES) * len(SEEDS)
    assert len(tasks) == n_expected, f'{len(tasks)} != {n_expected}'
    assert len(GENOS_BATTERY) == 10, f'battery genos {len(GENOS_BATTERY)} != 10'
    assert len(GENOS_ACTIVE) == 9, f'active genos {len(GENOS_ACTIVE)} != 9'
    assert len(BLOCK_I_GENOS) == 7 and len(BLOCK_S_GENOS) == 2
    assert all(g not in GENOS_BATTERY for g in DEFERRED), 'deferred genotype leaked into battery'
    n_mutant = len(GENOS_ACTIVE) * len(VOLATILES) * len(DOSES) * len(SEEDS)
    print(f'battery: {len(tasks)} sims  ({len(GENOS_BATTERY)} genos incl WT x '
          f'{len(VOLATILES)} volatiles x {len(DOSES)} doses x {len(SEEDS)} seeds)')
    print(f'  mutant-only rows (roadmap ~1080): {n_mutant}')
    print(f'  block-I (7): {BLOCK_I_GENOS}')
    print(f'  block-S (2): {BLOCK_S_GENOS}')
    print(f'  deferred-excluded (3): {DEFERRED}')
    return tasks


# ---------- worker ----------
def _worker(args):
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src')
    sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')
    from state_validation.phase_g_state_validator import run_single
    gene, vol, dose, seed = args
    profiles, mutants = _load_tables()
    prof = profiles[vol]
    mut = None if gene == 'WT' else mutants[gene]
    factory = _worm_factory()
    m = run_single(
        anesthetic=vol, dose_uM=dose, seed=seed, sim_duration_s=SIM_DUR_S,
        profile=prof, mutant=mut, alpha_calib=ALPHA, brain_factory=factory,
    )
    return (gene, vol, dose, seed, float(m['quiescent_fraction']),
            float(m['command_mean_firing_rate_hz']), float(m['network_mean_firing_rate_hz']))


# ---------- smoke ----------
def smoke() -> dict:
    """1 block-I genotype + 1 block-S genotype x 1 volatile x 2 doses x 1 seed, end-to-end."""
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [('gas-1', 'halothane', 100.0, 42), ('gas-1', 'halothane', 1000.0, 42),
             ('goa-1', 'halothane', 100.0, 42), ('goa-1', 'halothane', 1000.0, 42),
             ('WT', 'halothane', 1000.0, 42)]
    results = [_worker(c) for c in cases]
    rows = []
    for (gene, vol, d, s, qf, cr, nr) in results:
        assert np.isfinite(qf) and 0.0 <= qf <= 1.0, f'bad qf {qf} for {gene}'
        rows.append({'gene': gene, 'volatile': vol, 'dose': d, 'seed': s,
                     'quiescent_fraction': qf, 'command_rate_hz': cr, 'network_rate_hz': nr})
        print(f'  {gene:8s} {vol} d={d:7.1f}  qf={qf:.3f}  cmd_rate={cr:.2f}Hz')
    out = {'block': 'P20', 'stage': 'smoke', 'n_cases': len(rows),
           'all_finite': True, 'rows': rows}
    json.dump(out, open(OUT / 'p20_smoke.json', 'w'), indent=2)
    return out


# ---------- HEAVY battery + envelope + verdict ----------
def run_battery() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = build_battery()
    t0 = time.time()
    raw_path = OUT / 'p20_battery_raw.csv'
    results = []
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gene', 'volatile', 'dose_uM', 'seed',
                    'quiescent_fraction', 'command_rate_hz', 'network_rate_hz'])
        with mp.Pool(processes=N_WORKERS) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=CHUNKSIZE)):
                results.append(r)
                w.writerow(r)
                if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                    e = (time.time() - t0) / 60.0
                    eta = e / (i + 1) * (len(tasks) - (i + 1))
                    f.flush()
                    print(f'    [{i+1}/{len(tasks)}] {e:.1f}m ETA {eta:.0f}m', flush=True)

    from state_validation.phase_g_state_validator import hill_fit_ec50
    agg = defaultdict(lambda: defaultdict(list))
    for gene, vol, dose, seed, qf, _cr, _nr in results:
        agg[(gene, vol)][dose].append(qf)

    ec50 = {}
    for (gene, vol), dose_qf in agg.items():
        ds = sorted(dose_qf.keys())
        qfs = [float(np.mean(dose_qf[d])) for d in ds]
        ec50[(gene, vol)] = hill_fit_ec50(np.array(ds), np.array(qfs), threshold=0.5)

    # per-genotype EC50 ratios vs WT, per volatile + averaged
    ratios = {}
    for gene in GENOS_ACTIVE:
        per_vol = {}
        for vol in VOLATILES:
            wt = ec50.get(('WT', vol))
            g = ec50.get((gene, vol))
            per_vol[vol] = (None if (wt is None or g is None or wt == 0) else float(g / wt))
        valid = [v for v in per_vol.values() if v is not None]
        ratios[gene] = {'per_volatile': per_vol,
                        'mean_ratio': (float(np.mean(valid)) if valid else None)}

    # GATE-A reachability: literature band vs achievable envelope.
    # Envelope = sweep the genotype coordinate (heavy, requires re-sim). We approximate the
    # reachable EC50-ratio envelope by re-simulating the genotype-coordinate grid on halothane.
    envelope = _run_envelope_sweeps()

    gateA = {}
    for gene in GENOS_ACTIVE:
        band = LIT_BANDS[gene]
        block = 'I' if gene in BLOCK_I_GENOS else 'S'
        env = envelope[block]  # (min_ratio, max_ratio) achievable
        reachable = not (band[1] < env['min_ratio'] or band[0] > env['max_ratio'])
        sim_ratio = ratios[gene]['mean_ratio']
        gateA[gene] = {
            'block': block, 'literature_band': band,
            'achievable_envelope': [env['min_ratio'], env['max_ratio']],
            'sim_mean_ratio': sim_ratio,
            'sim_in_band': (None if sim_ratio is None else band[0] <= sim_ratio <= band[1]),
            'structurally_reachable': reachable,
            'ec50_indeterminate': sim_ratio is None,
        }
    n_unreachable = sum(1 for g in gateA.values()
                        if not g['structurally_reachable'] and not g['ec50_indeterminate'])
    gateA_verdict = ('POSITIVE_REAL_POSITIVE_route_to_V2_Tier4' if n_unreachable >= 1
                     else 'NEGATIVE_epistasis_is_bookkeeping_deflation')

    # GATE-C too-special / circularity
    midpoint_errs = {}
    for gene in GENOS_ACTIVE:
        band = LIT_BANDS[gene]
        mid = 0.5 * (band[0] + band[1])
        sim = ratios[gene]['mean_ratio']
        midpoint_errs[gene] = (None if sim is None else abs(sim - mid) / mid * 100.0)
    valid_errs = [e for e in midpoint_errs.values() if e is not None]
    all_under = (len(valid_errs) == len(GENOS_ACTIVE)
                 and all(e < TOO_SPECIAL_PCT for e in valid_errs))
    gateC_verdict = ('CIRCULARITY_FLAG_escalate_P7' if all_under
                     else 'CLEAN_no_circularity_flag')

    # GATE-B (heavy-confirmation half): dose-response shape consistency for Gao genos.
    # Structural half already PASS in cert(); here we report the EC50-ratio direction.
    gateB_dose = {g: ratios[g]['mean_ratio'] for g in BLOCK_S_GENOS}

    verdict = {
        'block': 'P20', 'pipeline': 'p20_two_block_reachability',
        'prereg': str(P20 / 'prereg.json'),
        'n_sims': len(results), 'wall_minutes': (time.time() - t0) / 60.0,
        'ec50_uM': {f'{g}|{v}': ec50.get((g, v)) for g in GENOS_BATTERY for v in VOLATILES},
        'genotype_ratios': ratios,
        'achievable_envelopes': envelope,
        'P20_GATE_A_reachability': {
            'per_genotype': gateA,
            'n_structurally_unreachable': n_unreachable,
            'verdict': gateA_verdict,
            'rule': PREREG['gates']['P20_GATE_A_reachability']['decision'],
        },
        'P20_GATE_B_gao_multiplicative': {
            'structural_half': 'see p20_cert.json (settled FAST)',
            'dose_response_ratios': gateB_dose,
        },
        'P20_GATE_C_too_special': {
            'midpoint_pct_errors': midpoint_errs,
            'all_under_5pct': all_under,
            'verdict': gateC_verdict,
        },
    }
    json.dump(verdict, open(OUT / 'p20_battery_verdict.json', 'w'), indent=2)
    print('\n=== P20 verdict ===')
    print(f'  GATE-A: {n_unreachable} unreachable -> {gateA_verdict}')
    print(f'  GATE-C: all<5%={all_under} -> {gateC_verdict}')
    return verdict


def _run_envelope_sweeps() -> dict:
    """Re-simulate the achievable EC50-ratio envelope per block on halothane (HEAVY).
    block-I: sweep an injected uniform pA shift; block-S: sweep wsyn. Uses a synthetic
    MutantBaseline so the envelope is the OPERATOR's reach, not any single genotype."""
    from state_validation.phase_g_state_validator import (
        run_single, hill_fit_ec50, load_perturbation_table, MutantBaseline,
    )
    profiles = load_perturbation_table(PERT_CSV)
    prof = profiles['halothane']
    factory = _worm_factory()
    wt_ec50 = None

    def ec50_for(mut):
        qfs = []
        for d in DOSES:
            seedvals = []
            for s in SEEDS[:3]:
                m = run_single('halothane', d, s, SIM_DUR_S, prof, mutant=mut,
                               alpha_calib=ALPHA, brain_factory=factory)
                seedvals.append(m['quiescent_fraction'])
            qfs.append(float(np.mean(seedvals)))
        return hill_fit_ec50(np.array(DOSES), np.array(qfs), threshold=0.5)

    wt_ec50 = ec50_for(None)

    # block-I grid: complex_i_factor from 1.0 (no shift) down to 0.0 (max hyperpolarizing)
    blockI_ratios = []
    for cif in np.linspace(1.0, 0.0, 8):
        mut = MutantBaseline('synthI', 'HYPER', float(cif), 1.0, 1.0, 1.0, '', '')
        g = ec50_for(mut)
        if g is not None and wt_ec50:
            blockI_ratios.append(g / wt_ec50)
    # block-S grid: wsyn 1.0 .. 2.5
    blockS_ratios = []
    for wsyn in np.linspace(1.0, 2.5, 8):
        mut = MutantBaseline('synthS', 'RESIST', 1.0, 1.0, float(wsyn), 1.0, '', '')
        g = ec50_for(mut)
        if g is not None and wt_ec50:
            blockS_ratios.append(g / wt_ec50)

    return {
        'wt_ec50_uM': wt_ec50,
        'I': {'min_ratio': float(min(blockI_ratios)) if blockI_ratios else 1.0,
              'max_ratio': float(max(blockI_ratios)) if blockI_ratios else 1.0,
              'grid_ratios': blockI_ratios},
        'S': {'min_ratio': float(min(blockS_ratios)) if blockS_ratios else 1.0,
              'max_ratio': float(max(blockS_ratios)) if blockS_ratios else 1.0,
              'grid_ratios': blockS_ratios},
    }


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'cert'
    if cmd == 'cert':
        cert()
    elif cmd == 'smoke':
        smoke()
    elif cmd == 'build':
        build_battery()
    elif cmd == 'battery':
        run_battery()
    else:
        raise SystemExit(f'unknown cmd {cmd}')


if __name__ == '__main__':
    main()
