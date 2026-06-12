"""P18-B Dynamic dual-input rank certificate (build-only, NO brain.run()).

Empirically certifies whether the V1 perturbation operator is rank-2:
  - current axis: every per-class anesthetic/genotype effect collapses to a single
    scalar total_pa broadcast IDENTICALLY to all N neurons' I_ext (rank-1 in space).
  - synapse axis: SNARE/wsyn effects collapse to a single global multiplicative
    weight scalar (rank-1).

For each case we CONSTRUCT a fresh worm SeededLIFBrain, snapshot neurons.I_ext and
synapse weights, call apply_genotype + apply_anesthetic, snapshot again. NO integration.

Outputs result.json with the frozen gate verdicts.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
import numpy as np

REPO = Path('/mnt/ssd4tb/Desktop/website/personalwebsite')
SRC = REPO / 'AnestheticSimulator' / 'src'
SCRIPTS = REPO / 'scripts'
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

OUTDIR = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/audits/phase0/P18-B')
TAU = 1e-6

from state_validation.phase_g_state_validator import (
    PerturbationRow, MutantBaseline, apply_genotype, apply_anesthetic,
    DEFAULT_PER_CLASS_PA_AT_SATURATION,
)

# The 7 current-axis mechanism classes (route into total_pa -> I_ext broadcast)
CURRENT_CLASSES = [
    'complex_i_block', 'complex_ii_block', 'k2p_potentiation', 'nca_block',
    'gaba_potentiation', 'glucl_potentiation', 'nachr_antagonism',
]
SNARE_CLASS = 'snare_cooperativity'

# Whether each class is "blocking" (max<1) or "potentiating" (max>1) only affects
# the printed factor; for the rank test what matters is that engagement>0 produces
# a nonzero per-class contribution. We pick max_effect_factors consistent with the
# perturbation table convention.
BLOCK_CLASSES = {'complex_i_block', 'complex_ii_block', 'nachr_antagonism', 'nca_block'}


def mk_row(cls: str, engagement: float, max_factor: float | None = None) -> PerturbationRow:
    """Build a PerturbationRow whose engagement(dose=1.0) == `engagement` exactly.

    With hill_n=1 and dose=1: engagement = 1/(1+EC50) => EC50 = (1-e)/e.
    """
    e = float(engagement)
    if e <= 0:
        ec50 = 1e9
    elif e >= 1:
        ec50 = 1e-9
    else:
        ec50 = (1.0 - e) / e
    if max_factor is None:
        max_factor = 0.3 if cls in BLOCK_CLASSES else (0.5 if cls == SNARE_CLASS else 3.0)
    return PerturbationRow(
        anesthetic='synthetic', mechanism_class=cls,
        target_EC50_uM=ec50, max_effect_factor=max_factor, hill_n=1.0,
        source_PMID='AUDIT', evidence_grade='AUDIT',
    )


def build_brain():
    from brain.lif_brain import LIFBrain
    class SeededLIFBrain(LIFBrain):
        _brian2_seed = 1234
    return SeededLIFBrain(use_per_edge_glu_signs=True)


def snapshot_iext(brain) -> np.ndarray:
    import brian2
    return np.asarray(brain.neurons.I_ext[:] / brian2.pA, dtype=np.float64).copy()


def snapshot_syn(brain) -> np.ndarray:
    """Concatenate exc + inh synapse weights into one vector (dimensionless w)."""
    parts = []
    if getattr(brain, 'syn_exc', None) is not None and len(brain.syn_exc) > 0:
        parts.append(np.asarray(brain.syn_exc.w[:], dtype=np.float64).copy())
    if getattr(brain, 'syn_inh', None) is not None and len(brain.syn_inh) > 0:
        parts.append(np.asarray(brain.syn_inh.w[:], dtype=np.float64).copy())
    if not parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(parts)


def build_battery():
    """Return list of cases. Each case is a dict with keys:
       name, kind ('current'|'synapse'|'mixed'|'genotype'),
       profile (dict class->row) or None, mutant (MutantBaseline) or None,
       expected_classes (set), expected_genotype_branch (str or None)
    """
    cases = []

    # --- 7 classes x >=2 engagements (current axis, isolated) ---
    for cls in CURRENT_CLASSES:
        for eng in (0.2, 0.4, 0.6, 0.9):
            cases.append(dict(
                name=f'class[{cls}]_e{eng}', kind='current',
                profile={cls: mk_row(cls, eng)}, mutant=None,
                expected_classes={cls}, expected_genotype_branch=None,
            ))

    # --- SNARE alone x several engagements (synapse axis) ---
    for eng in (0.2, 0.4, 0.6, 0.9):
        cases.append(dict(
            name=f'snare_alone_e{eng}', kind='synapse',
            profile={SNARE_CLASS: mk_row(SNARE_CLASS, eng)}, mutant=None,
            expected_classes={SNARE_CLASS}, expected_genotype_branch=None,
        ))

    # --- mixtures: 3-class, 5-class, 7-class (current axis) at varying engagements ---
    mixes = {
        '3class': CURRENT_CLASSES[:3],
        '5class': CURRENT_CLASSES[:5],
        '7class': CURRENT_CLASSES[:7],
    }
    for mname, classes in mixes.items():
        for eng in (0.3, 0.5, 0.7, 0.85):
            prof = {c: mk_row(c, eng * (0.5 + 0.1 * i)) for i, c in enumerate(classes)}
            cases.append(dict(
                name=f'mix_{mname}_e{eng}', kind='mixed',
                profile=prof, mutant=None,
                expected_classes=set(classes), expected_genotype_branch=None,
            ))

    # --- SNARE co-engaged with current classes ---
    for eng in (0.3, 0.5, 0.7, 0.9):
        prof = {c: mk_row(c, eng) for c in CURRENT_CLASSES[:4]}
        prof[SNARE_CLASS] = mk_row(SNARE_CLASS, eng)
        cases.append(dict(
            name=f'snare_coengaged_e{eng}', kind='mixed',
            profile=prof, mutant=None,
            expected_classes=set(CURRENT_CLASSES[:4]) | {SNARE_CLASS},
            expected_genotype_branch=None,
        ))

    # --- 4 genotype branches (each isolated, plus combos) ---
    # branch1: complex_i_factor<1  (gas-1)
    cases.append(dict(
        name='geno_complex_i', kind='genotype', profile=None,
        mutant=MutantBaseline('gas-1', 'HYPER', 0.40, 1.0, 1.0, 1.0, '', '', 1.0),
        expected_classes=set(), expected_genotype_branch='complex_i_factor<1',
    ))
    # branch2: nca_leak_factor<1  (unc-79)
    cases.append(dict(
        name='geno_nca', kind='genotype', profile=None,
        mutant=MutantBaseline('unc-79', 'HYPER', 1.0, 0.05, 1.0, 1.0, '', '', 1.0),
        expected_classes=set(), expected_genotype_branch='nca_leak_factor<1',
    ))
    # branch3: wsyn_global_factor!=1 (goa-1 >1, Syx1A <1)
    cases.append(dict(
        name='geno_wsyn_up', kind='genotype', profile=None,
        mutant=MutantBaseline('goa-1', 'RESISTANT', 1.0, 1.0, 1.5, 1.0, '', '', 1.0),
        expected_classes=set(), expected_genotype_branch='wsyn_global_factor!=1',
    ))
    cases.append(dict(
        name='geno_wsyn_down', kind='genotype', profile=None,
        mutant=MutantBaseline('Syx1A', 'HYPER', 1.0, 1.0, 0.5, 1.0, '', '', 1.0),
        expected_classes=set(), expected_genotype_branch='wsyn_global_factor!=1',
    ))
    # branch4: k2p_baseline_factor<1 (Sandman, fly)
    cases.append(dict(
        name='geno_k2p', kind='genotype', profile=None,
        mutant=MutantBaseline('Sandman', 'RESISTANT', 1.0, 1.0, 1.0, 1.0, '', '', 0.0),
        expected_classes=set(), expected_genotype_branch='k2p_baseline_factor<1',
    ))
    # combo genotype: complex_i + nca + k2p together (all current-axis genotype writes)
    cases.append(dict(
        name='geno_combo_current', kind='genotype', profile=None,
        mutant=MutantBaseline('combo', 'X', 0.5, 0.5, 1.0, 1.0, '', '', 0.5),
        expected_classes=set(), expected_genotype_branch='complex_i_factor<1+nca_leak_factor<1+k2p_baseline_factor<1',
    ))

    # --- genotype x anesthetic co-engaged (both axes simultaneously) ---
    geno_muts = [
        ('gas-1',  MutantBaseline('gas-1', 'HYPER', 0.40, 1.0, 1.0, 1.0, '', '', 1.0), 'complex_i_factor<1'),
        ('unc-79', MutantBaseline('unc-79', 'HYPER', 1.0, 0.05, 1.0, 1.0, '', '', 1.0), 'nca_leak_factor<1'),
        ('goa-1',  MutantBaseline('goa-1', 'RESISTANT', 1.0, 1.0, 1.5, 1.0, '', '', 1.0), 'wsyn_global_factor!=1'),
        ('Sandman',MutantBaseline('Sandman', 'RESISTANT', 1.0, 1.0, 1.0, 1.0, '', '', 0.0), 'k2p_baseline_factor<1'),
    ]
    for gname, mut, branch in geno_muts:
        for eng in (0.4, 0.8):
            prof = {c: mk_row(c, eng) for c in CURRENT_CLASSES[:3]}
            prof[SNARE_CLASS] = mk_row(SNARE_CLASS, eng)
            cases.append(dict(
                name=f'geno[{gname}]_x_anesth_e{eng}', kind='mixed',
                profile=prof, mutant=mut,
                expected_classes=set(CURRENT_CLASSES[:3]) | {SNARE_CLASS},
                expected_genotype_branch=branch,
            ))

    return cases


def main():
    log = []
    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    P('=== P18-B dynamic rank certificate ===')
    P(f'tau = {TAU}')

    cases = build_battery()
    P(f'battery size = {len(cases)} cases')

    # Coverage trackers
    classes_hit = set()
    genotype_branches_hit = set()
    # load-bearing if-branches inside the operator we want to confirm fired:
    branch_flags = {
        'apply_anesthetic.total_pa_iext_write': False,   # line 323-324
        'apply_anesthetic.snare_syn_scale': False,       # line 327-335
        'apply_genotype.complex_i_iext': False,          # line 260-262
        'apply_genotype.nca_iext': False,                # line 264-267
        'apply_genotype.wsyn_scale': False,              # line 272-277
        'apply_genotype.k2p_iext': False,                # line 281-283
    }

    # Per-case records
    D_rows = []          # I_ext deltas (pA) for nontrivial current cases
    D_names = []
    rho_list = []        # spatial-uniformity residual per current-delta row
    S_rows = []          # synapse ratio rows for nontrivial synapse cases
    S_names = []
    syn_nonuniformity = []
    errors = []

    N_ref = None

    for c in cases:
        try:
            brain = build_brain()
        except Exception as e:
            errors.append(f'{c["name"]}: BUILD FAILED: {e}')
            P('BUILD FAILED', c['name'], e)
            traceback.print_exc()
            continue

        if N_ref is None:
            N_ref = brain.N

        iext_pre = snapshot_iext(brain)
        syn_pre = snapshot_syn(brain)

        # apply genotype then anesthetic (same order as run_single)
        apply_genotype(brain, c['mutant'], alpha_calib=1.0)
        prof = c['profile'] if c['profile'] is not None else {}
        apply_anesthetic(brain, prof, dose_uM=1.0, alpha_calib=1.0)

        iext_post = snapshot_iext(brain)
        syn_post = snapshot_syn(brain)

        d = iext_post - iext_pre            # per-neuron current delta (pA)
        # synapse ratio (guard div-by-zero: only where pre != 0)
        if syn_pre.size > 0:
            nz = syn_pre != 0
            ratio = np.ones_like(syn_pre)
            ratio[nz] = syn_post[nz] / syn_pre[nz]
        else:
            ratio = np.array([])

        # --- coverage bookkeeping ---
        classes_hit |= set(prof.keys())
        if c['expected_genotype_branch']:
            for b in c['expected_genotype_branch'].split('+'):
                genotype_branches_hit.add(b)

        # which operator branches actually fired (detected from observed deltas)
        current_classes_engaged = any(
            k in CURRENT_CLASSES for k in prof.keys()
        )
        if np.any(np.abs(d) > 1e-12):
            # an I_ext write happened
            if current_classes_engaged:
                branch_flags['apply_anesthetic.total_pa_iext_write'] = True
            mut = c['mutant']
            if mut is not None:
                if mut.complex_i_factor < 1.0:
                    branch_flags['apply_genotype.complex_i_iext'] = True
                if mut.nca_leak_factor < 1.0:
                    branch_flags['apply_genotype.nca_iext'] = True
                if getattr(mut, 'k2p_baseline_factor', 1.0) < 1.0:
                    branch_flags['apply_genotype.k2p_iext'] = True
        if ratio.size > 0 and np.any(np.abs(ratio - 1.0) > 1e-12):
            if SNARE_CLASS in prof:
                branch_flags['apply_anesthetic.snare_syn_scale'] = True
            mut = c['mutant']
            if mut is not None and mut.wsyn_global_factor != 1.0:
                branch_flags['apply_genotype.wsyn_scale'] = True

        # --- record nontrivial current-delta rows into D ---
        if np.any(np.abs(d) > 1e-12):
            D_rows.append(d)
            D_names.append(c['name'])
            m = d.mean()
            resid = d - m * np.ones_like(d)
            rho = float(np.linalg.norm(resid) / (np.linalg.norm(d) + 1e-30))
            rho_list.append({'case': c['name'], 'rho': rho,
                             'delta_mean_pA': float(m),
                             'delta_min_pA': float(d.min()),
                             'delta_max_pA': float(d.max())})

        # --- record nontrivial synapse-ratio rows into S ---
        if ratio.size > 0 and np.any(np.abs(ratio - 1.0) > 1e-12):
            S_rows.append(ratio)
            S_names.append(c['name'])
            med = np.median(ratio)
            nonunif = float(np.max(np.abs(ratio - med)))
            syn_nonuniformity.append({'case': c['name'], 'nonuniformity': nonunif,
                                      'ratio_median': float(med),
                                      'ratio_min': float(ratio.min()),
                                      'ratio_max': float(ratio.max())})

    # ===== Build matrices and SVD =====
    D = np.array(D_rows) if D_rows else np.zeros((0, N_ref or 0))
    S = np.array(S_rows) if S_rows else np.zeros((0, 0))

    P(f'N_neurons = {N_ref}')
    P(f'D shape = {D.shape}  (nontrivial current cases)')
    P(f'S shape = {S.shape}  (nontrivial synapse cases)')

    # singular spectra
    if D.size > 0:
        sv_D = np.linalg.svd(D, compute_uv=False)
    else:
        sv_D = np.array([])
    if S.size > 0:
        sv_S = np.linalg.svd(S, compute_uv=False)
    else:
        sv_S = np.array([])

    n_sv_D = int(np.sum(sv_D > TAU))
    n_sv_S = int(np.sum(sv_S > TAU))
    max_rho = max((r['rho'] for r in rho_list), default=0.0)
    max_syn_nonunif = max((s['nonuniformity'] for s in syn_nonuniformity), default=0.0)

    P(f'singular values D (top 6): {np.round(sv_D[:6], 9).tolist()}')
    P(f'  #sv(D) > tau = {n_sv_D}')
    P(f'singular values S (top 6): {np.round(sv_S[:6], 9).tolist()}')
    P(f'  #sv(S) > tau = {n_sv_S}')
    P(f'max rho_k (current spatial non-uniformity) = {max_rho:.3e}')
    P(f'max synapse non-uniformity = {max_syn_nonunif:.3e}')

    # ===== Coverage =====
    classes_hit_current = classes_hit & set(CURRENT_CLASSES)
    n_classes = len(classes_hit_current)
    n_geno = len(genotype_branches_hit & {
        'complex_i_factor<1', 'nca_leak_factor<1',
        'wsyn_global_factor!=1', 'k2p_baseline_factor<1'})
    all_branches_hit = all(branch_flags.values())

    P(f'current classes hit = {n_classes}/7 : {sorted(classes_hit_current)}')
    P(f'genotype branches hit = {n_geno}/4 : {sorted(genotype_branches_hit)}')
    P(f'operator if-branches fired: {branch_flags}')

    # ===== Gates =====
    g1 = (n_sv_D == 1) and (max_rho < TAU)
    g2 = (n_sv_S == 1) and (max_syn_nonunif < TAU)
    g3 = (n_classes == 7) and (n_geno == 4) and all_branches_hit

    def verdict(b): return 'PASS' if b else 'FAIL'

    # G3 voids G1/G2 if incomplete
    overall = 'PASS' if (g1 and g2 and g3) else 'FAIL'
    notes = []
    if not g3:
        notes.append('G3 coverage incomplete => G1/G2 are VOID (indeterminate certificate).')
    if errors:
        notes.append(f'{len(errors)} build errors: {errors}')

    P(f'G1-current-rank = {verdict(g1)}  (n_sv_D=={n_sv_D} need 1; max_rho={max_rho:.2e} need <{TAU})')
    P(f'G2-synapse-rank = {verdict(g2)}  (n_sv_S=={n_sv_S} need 1; max_syn_nonunif={max_syn_nonunif:.2e} need <{TAU})')
    P(f'G3-coverage     = {verdict(g3)}  (classes {n_classes}/7, geno {n_geno}/4, branches {all_branches_hit})')
    P(f'OVERALL = {overall}')

    result = {
        'block_id': 'P18-B',
        'tau': TAU,
        'n_cases': len(cases),
        'N_neurons': N_ref,
        'D_shape': list(D.shape),
        'S_shape': list(S.shape),
        'singular_values_D': [float(x) for x in sv_D],
        'singular_values_S': [float(x) for x in sv_S],
        'n_sv_D_gt_tau': n_sv_D,
        'n_sv_S_gt_tau': n_sv_S,
        'max_rho_k_current': float(max_rho),
        'max_synapse_nonuniformity': float(max_syn_nonunif),
        'coverage': {
            'current_classes_hit': sorted(classes_hit_current),
            'n_current_classes': n_classes,
            'genotype_branches_hit': sorted(genotype_branches_hit),
            'n_genotype_branches': n_geno,
            'operator_if_branches_fired': branch_flags,
            'all_branches_hit': all_branches_hit,
        },
        'per_case_rho': rho_list,
        'per_case_synapse_nonuniformity': syn_nonuniformity,
        'errors': errors,
        'gates': {
            'G1-current-rank': {'verdict': verdict(g1),
                                'n_sv_D': n_sv_D, 'max_rho_k': float(max_rho)},
            'G2-synapse-rank': {'verdict': verdict(g2),
                                'n_sv_S': n_sv_S, 'max_synapse_nonuniformity': float(max_syn_nonunif)},
            'G3-coverage-completeness': {'verdict': verdict(g3),
                                         'n_classes': n_classes, 'n_genotype_branches': n_geno,
                                         'all_branches_hit': all_branches_hit},
        },
        'overall': overall,
        'notes': notes,
    }

    (OUTDIR / 'result.json').write_text(json.dumps(result, indent=2))
    (OUTDIR / 'run.log').write_text('\n'.join(log) + '\n')
    P('wrote result.json and run.log')
    return result


if __name__ == '__main__':
    main()
