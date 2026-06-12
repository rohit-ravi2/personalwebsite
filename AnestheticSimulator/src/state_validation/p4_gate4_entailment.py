"""p4_gate4_entailment — Gate-4 (Eger non-immobilizer) entailment demotion + SNARE falsifier (WF2 P4).

Three CLIs:
  writepath  — G4-WB0: static re-derivation that EXACTLY ONE orthogonal synaptic channel
               (snare_cooperativity) gates a syn.w write in apply_anesthetic. FAST.
  entailment — G4-A: re-analysis on FROZEN v7 Stage3 artifacts. Regress per-(org,subset,compound)
               max_qf on the two operator coordinates (total_pa, snare_gain) at the max_qf dose.
               R2>=0.95 AND zero attrition => Gate-4 entailed by Gates 1+2 => demote sec 8.4. FAST.
  falsifier  — G4-B: build a SNARE-only pseudo-non-immobilizer (total_pa pinned in the non-immobilizer
               band, ONLY SNARE engaged) + a SNARE-lever-sufficiency positive control, run a dose-response.
               HEAVY (~<1hr, ml conda brian2 env). Precondition-gated.

NON-DESTRUCTIVE: reads frozen artifacts only; writes NEW artifacts under artifacts/p4_gate4_entailment/.
Operator coordinate functions are REUSED from v7_match2b (fidelity-gated to 1e-9 in P8) -- single source
of truth; this module never re-implements or mutates the frozen operator.

Prereg + gates: audits/phase1/P4/prereg.json.
Env: pure-Python (numpy) for writepath + entailment; ml conda (brian2) for falsifier.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ANESTH = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator')
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, str(ANESTH.parent / 'scripts'))

P4 = ANESTH / 'audits' / 'phase1' / 'P4'
OUT = ANESTH / 'artifacts' / 'p4_gate4_entailment'
PREREG = json.load(open(P4 / 'prereg.json'))

STAGE3_RAW = ANESTH / 'artifacts' / 'v7_subset_search' / 'v7_stage3_eger_raw.csv'
STAGE3_SUMMARY = ANESTH / 'artifacts' / 'v7_subset_search' / 'v7_stage3_eger.csv'
STAGE3_VERDICT = ANESTH / 'artifacts' / 'v7_subset_search' / 'v7_stage3_verdict.json'
STAGE2_SUMMARY = ANESTH / 'artifacts' / 'v7_subset_search' / 'v7_stage2_isoflurane.csv'

EGER_COMPOUNDS = ['cis_12_dichloroethylene', 'trans_12_dichloroethylene', 'hexafluoroethane']
NON_IMMOBILIZERS = ['trans_12_dichloroethylene', 'hexafluoroethane']


# ============================================================
# G4-WB0 — write-path: exactly one orthogonal synaptic channel = SNARE
# ============================================================
def writepath_check() -> dict:
    src = open(ANESTH / 'src' / 'state_validation' / 'phase_g_state_validator.py').read()
    m = re.search(r'def apply_anesthetic\(.*?\n(.*?)\n# ===== Main', src, re.S)
    if m is None:
        raise RuntimeError('could not isolate apply_anesthetic body')
    body = m.group(1)
    syn_write_lines = [ln.strip() for ln in body.splitlines()
                       if ('syn_exc.w' in ln or 'syn_inh.w' in ln) and '=' in ln]
    # Which mechanism class keys gate those writes (eng-membership + profile[...] indexing)?
    guards = set(re.findall(r"'([a-z_]+)'\s*in eng", body))
    prof_keys = set(re.findall(r"profile\['([a-z_]+)'\]", body))
    syn_channel_keys = sorted(guards | prof_keys)
    n_channels = len(syn_channel_keys)
    verdict = ('PASS' if n_channels == 1 and syn_channel_keys == ['snare_cooperativity']
               else 'FAIL')
    out = {
        'gate': 'G4_WB0_writepath',
        'n_syn_write_lines': len(syn_write_lines),
        'syn_channel_class_keys': syn_channel_keys,
        'n_orthogonal_synaptic_channels': n_channels,
        'verdict': verdict,
        'rule': PREREG['gates']['G4_WB0_writepath']['decision'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / 'g4_wb0_writepath.json', 'w'), indent=2)
    print(f'G4-WB0 write-path: {n_channels} orthogonal synaptic channel(s) = '
          f'{syn_channel_keys} -> {verdict}')
    return out


# ============================================================
# coordinate reconstruction for frozen Stage3 rows
# ============================================================
def _load_full_profiles():
    """Return {organism: {compound: full PerturbationRow profile dict}} from frozen tables."""
    from state_validation.v7_subset_search import _organism_runtime, ORG_CONFIG
    from state_validation.phase_g_state_validator import load_perturbation_table
    profs = {}
    for org in ORG_CONFIG:
        table_path, _, _, _ = _organism_runtime(org)
        all_p = load_perturbation_table(table_path)
        profs[org] = {c: all_p[c] for c in EGER_COMPOUNDS if c in all_p}
    return profs, ORG_CONFIG


def _coords_for_row(full_profile, subset_csv, dose, alpha):
    """(total_pa, snare_gain) for a compound's full profile restricted to subset, at dose.
    Reuses the P8-fidelity-gated operator mirrors (single source of truth)."""
    from state_validation.v7_subset_search import _build_subset_profile
    from state_validation.v7_match2b import operator_total_pa, operator_snare_factor
    subset_set = set(subset_csv.split('|')) if subset_csv else set()
    sub_prof = _build_subset_profile(full_profile, subset_set)
    return (operator_total_pa(sub_prof, dose, alpha),
            operator_snare_factor(sub_prof, dose))


def _ols_r2(X, y):
    """OLS with intercept; return (r2, n, rank_of_predictors, coef)."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rank = int(np.linalg.matrix_rank(X - X.mean(axis=0)))
    return r2, len(y), rank, coef


# ============================================================
# G4-A — entailment regression on FROZEN Stage3
# ============================================================
def entailment_check() -> dict:
    # 1) per-(org,subset,compound) max_qf AND the dose that achieves it, from RAW frozen data
    raw = defaultdict(lambda: defaultdict(list))  # (org,subset,compound) -> dose -> [qf]
    with open(STAGE3_RAW) as f:
        for row in csv.DictReader(f):
            key = (row['organism'], row['subset'], row['compound'])
            raw[key][float(row['dose_uM'])].append(float(row['quiescent_fraction']))

    profs, ORG_CONFIG = _load_full_profiles()

    records = []
    for (org, subset_csv, compound), dose_qf in raw.items():
        if compound not in profs[org]:
            continue
        alpha = ORG_CONFIG[org]['alpha']
        # max over doses of seed-mean qf (matches the frozen Stage3 aggregation)
        best_dose, best_qf = None, -1.0
        for d, qfs in dose_qf.items():
            mq = float(np.mean(qfs))
            if mq > best_qf:
                best_qf, best_dose = mq, d
        tpa, sgain = _coords_for_row(profs[org][compound], subset_csv, best_dose, alpha)
        records.append({'organism': org, 'subset': subset_csv, 'compound': compound,
                        'max_qf': best_qf, 'max_qf_dose_uM': best_dose,
                        'total_pa': tpa, 'snare_gain': sgain,
                        'is_non_immobilizer': compound in NON_IMMOBILIZERS})

    y = np.array([r['max_qf'] for r in records])
    X = np.array([[r['total_pa'], r['snare_gain']] for r in records])

    pooled_r2, n, pred_rank, coef = _ols_r2(X, y)

    # per-organism R2 (only where >2 distinct rows)
    per_org = {}
    for org in ORG_CONFIG:
        idx = [i for i, r in enumerate(records) if r['organism'] == org]
        if len(idx) >= 3:
            r2o, no, ro, _ = _ols_r2(X[idx], y[idx])
            per_org[org] = {'r2': r2o, 'n': no, 'predictor_rank': ro}
        else:
            per_org[org] = {'r2': None, 'n': len(idx), 'predictor_rank': None,
                            'note': 'too few rows for stable per-org R2'}

    # snare degeneracy note
    sgains = X[:, 1]
    snare_degenerate = bool(np.allclose(sgains, sgains[0], atol=1e-9))

    # 2) attrition from frozen verdict: Stage2 passers that fail Stage3
    verdict = json.load(open(STAGE3_VERDICT))
    attrition = 0
    attrition_detail = {}
    for org, v in verdict['per_organism'].items():
        tested = v['n_stage2_passers_tested']
        passed = v['n_stage3_passers']
        a = tested - passed
        attrition += a
        attrition_detail[org] = {'stage2_passers_tested': tested,
                                 'stage3_passers': passed, 'attrition': a}

    r2_min = PREREG['gates']['G4_A_entailment']['thresholds']['r2_min']
    att_max = PREREG['gates']['G4_A_entailment']['thresholds']['attrition_max']

    # LITERAL frozen gate: pooled OLS R2 across all organisms with a single intercept/slope.
    pooled_entailed = (pooled_r2 >= r2_min) and (attrition <= att_max)
    pooled_decision = ('DEMOTE_8.4_entailed' if pooled_entailed
                       else 'RETAIN_8.4_independent')

    # OPERATOR-FAITHFUL reading: alpha + qf-threshold are PER-ORGANISM, so the rank-2 map
    # G(total_pa, snare_gain) is itself per-organism (different intercept/slope per org).
    # The substantive entailment claim is per-organism. Reported alongside, NOT a self-override
    # of the frozen pooled gate.
    org_r2s = [v['r2'] for v in per_org.values() if v['r2'] is not None]
    min_org_r2 = min(org_r2s) if org_r2s else None
    per_org_entailed = (min_org_r2 is not None and min_org_r2 >= r2_min
                        and attrition <= att_max)
    decision = pooled_decision  # frozen gate verdict is the literal pooled rule

    out = {
        'gate': 'G4_A_entailment',
        'n_rows': n,
        'pooled_r2': pooled_r2,
        'predictor_rank_at_maxqf_doses': pred_rank,
        'ols_coef_intercept_totalpa_snaregain': coef.tolist(),
        'per_organism_r2': per_org,
        'snare_gain_degenerate_all_~1': snare_degenerate,
        'total_attrition_stage2_to_stage3': attrition,
        'attrition_detail': attrition_detail,
        'thresholds': {'r2_min': r2_min, 'attrition_max': att_max},
        'verdict_pooled_frozen_gate': pooled_decision,
        'pooled_entailed': pooled_entailed,
        'verdict': decision,
        'operator_faithful_per_organism': {
            'min_per_organism_r2': min_org_r2,
            'per_organism_entailed': per_org_entailed,
            'rationale': ('alpha and qf-threshold are PER-ORGANISM, so the rank-2 map '
                          'G(total_pa,snare_gain) is per-organism; pooling across organisms '
                          'with one intercept/slope is a Simpson-type artifact, NOT evidence '
                          'of residual structure. Min per-org R2 reported as the substantive '
                          'entailment statistic alongside the frozen pooled gate.'),
        },
        'rule': PREREG['gates']['G4_A_entailment']['decision'],
        'records': records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / 'g4_a_entailment.json', 'w'), indent=2)
    # csv for inspection
    with open(OUT / 'g4_a_entailment_rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['organism', 'subset', 'compound', 'max_qf',
                                          'max_qf_dose_uM', 'total_pa', 'snare_gain',
                                          'is_non_immobilizer'])
        w.writeheader()
        w.writerows(records)
    print(f'G4-A entailment: n={n}  pooled R2={pooled_r2:.4f} '
          f'(min {r2_min})  predictor_rank={pred_rank}  attrition={attrition} '
          f'(max {att_max})  snare_degenerate={snare_degenerate}')
    print(f'    FROZEN POOLED GATE -> {pooled_decision}')
    print(f'    OPERATOR-FAITHFUL per-organism: min R2={min_org_r2}  '
          f'entailed={per_org_entailed}')
    for org, v in per_org.items():
        print(f'      {org}: R2={v["r2"]}  n={v["n"]}')
    return out


# ============================================================
# G4-B — SNARE-only pseudo-non-immobilizer falsifier (HEAVY)
# ============================================================
def _non_immobilizer_band() -> dict:
    """Read the FROZEN non-immobilizer total_pa band: the max |total_pa| reached by any
    trans-DCE / hexafluoroethane row at any tested dose. The pseudo-NI must be pinned
    AT OR BELOW this band so it is, by total_pa, indistinguishable from a real non-immobilizer."""
    raw = defaultdict(lambda: defaultdict(list))
    with open(STAGE3_RAW) as f:
        for row in csv.DictReader(f):
            key = (row['organism'], row['subset'], row['compound'])
            raw[key][float(row['dose_uM'])].append(float(row['quiescent_fraction']))
    profs, ORG_CONFIG = _load_full_profiles()
    band = {}
    for org in ORG_CONFIG:
        max_abs = 0.0
        for (o, subset_csv, compound), dose_qf in raw.items():
            if o != org or compound not in NON_IMMOBILIZERS:
                continue
            for d in dose_qf:
                tpa, _ = _coords_for_row(profs[org][compound], subset_csv, d, ORG_CONFIG[org]['alpha'])
                max_abs = max(max_abs, abs(tpa))
        band[org] = max_abs
    return band


def falsifier_run(organism: str = 'worm') -> dict:
    """HEAVY. Build a SNARE-only pseudo-NI + SNARE-sufficiency positive control; run dose-response.

    Profile design:
      - pseudo_NI: ONLY snare_cooperativity engaged (driven hard), all current classes blanked
        -> total_pa == 0 (inside the non-immobilizer band by construction).
      - precondition control: matched total_pa==0 with SNARE OFF (snare_gain==1) vs SNARE ON
        (max engagement). dQF = QF(snare_on) - QF(snare_off) must be >= 0.2 else UNINTERPRETABLE.
    """
    import brian2  # noqa: F401
    from state_validation.v7_subset_search import _organism_runtime, ORG_CONFIG
    from state_validation.phase_g_state_validator import (
        run_single, PerturbationRow, load_perturbation_table,
    )

    band = _non_immobilizer_band()
    table_path, factory, qf_thr, cmd_set = _organism_runtime(organism)
    alpha = ORG_CONFIG[organism]['alpha']
    full = load_perturbation_table(table_path)

    # Borrow a SNARE row that genuinely engages across the Eger dose grid: use halothane's
    # snare_cooperativity (a real, calibrated cooperativity factor) so engagement is non-trivial.
    snare_row = full['halothane'].get('snare_cooperativity')
    if snare_row is None or snare_row.target_EC50_uM is None:
        raise RuntimeError('no engageable snare_cooperativity row available')

    def blank(cls):
        return PerturbationRow('pseudo_NI', cls, None, None, 1.0, snare_row.source_PMID, 'SYNTHETIC_BLANK')

    all_classes = list(__import__('state_validation.phase_g_state_validator', fromlist=['x'])
                       .DEFAULT_PER_CLASS_PA_AT_SATURATION.keys())

    # pseudo-NI: only SNARE live -> total_pa == 0 (in band), snare_gain varies with dose
    pseudo_ni = {c: (snare_row if c == 'snare_cooperativity' else blank(c)) for c in all_classes}
    # control OFF: SNARE blanked too -> total_pa==0, snare_gain==1
    ctrl_off = {c: blank(c) for c in all_classes}

    doses = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]
    seeds = [42, 137, 219, 331, 443]

    def dose_response(profile):
        by_dose = {}
        for d in doses:
            qfs = []
            for s in seeds:
                m = run_single(anesthetic='pseudo_NI', dose_uM=d, seed=s,
                               sim_duration_s=30.0, profile=profile, mutant=None,
                               alpha_calib=alpha, brain_factory=factory,
                               quiescent_threshold_hz=qf_thr, command_set=cmd_set)
                qfs.append(float(m['quiescent_fraction']))
            by_dose[d] = float(np.mean(qfs))
        return by_dose

    print(f'  [G4-B] {organism}: running SNARE-OFF control dose-response...', flush=True)
    qf_off = dose_response(ctrl_off)
    print(f'  [G4-B] {organism}: running SNARE-ONLY pseudo-NI dose-response...', flush=True)
    qf_on = dose_response(pseudo_ni)

    # precondition: max QF lift SNARE-on vs SNARE-off at matched total_pa(==0)
    dqf = max(qf_on[d] - qf_off[d] for d in doses)
    pseudo_ni_max_qf = max(qf_on.values())

    pre_min = PREREG['gates']['G4_B_snare_falsifier']['thresholds']['precondition_dqf_min']
    immob_thr = PREREG['gates']['G4_B_snare_falsifier']['thresholds']['pseudo_ni_immobilization_qf']

    if dqf < pre_min:
        verdict = 'UNINTERPRETABLE_snare_lever_inert'
    elif pseudo_ni_max_qf >= immob_thr:
        verdict = 'OUTCOME_A_snare_alone_immobilizes_DEMOTE'
    else:
        verdict = 'OUTCOME_B_snare_cannot_rescue_RETAIN_total_pa_loadbearing'

    out = {
        'gate': 'G4_B_snare_falsifier', 'organism': organism,
        'non_immobilizer_total_pa_band_abs': band[organism],
        'pseudo_ni_total_pa': 0.0,
        'precondition_max_dQF_snareon_minus_off': dqf,
        'precondition_threshold': pre_min,
        'pseudo_ni_max_qf': pseudo_ni_max_qf,
        'immobilization_threshold': immob_thr,
        'qf_snare_off_by_dose': qf_off,
        'qf_snare_only_by_dose': qf_on,
        'verdict': verdict,
        'rule': PREREG['gates']['G4_B_snare_falsifier']['rule'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f'g4_b_falsifier_{organism}.json', 'w'), indent=2)
    print(f'G4-B falsifier [{organism}]: precondition dQF={dqf:.3f} (>= {pre_min}?), '
          f'pseudo-NI max_qf={pseudo_ni_max_qf:.3f} -> {verdict}')
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'fast'
    if cmd == 'writepath':
        writepath_check()
    elif cmd == 'entailment':
        entailment_check()
    elif cmd == 'fast':
        writepath_check()
        entailment_check()
    elif cmd == 'falsifier':
        org = sys.argv[2] if len(sys.argv) > 2 else 'worm'
        falsifier_run(org)
    else:
        print('Usage: p4_gate4_entailment.py {writepath|entailment|fast|falsifier [organism]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
