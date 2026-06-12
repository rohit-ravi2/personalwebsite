"""P11 — Wave-P K_p reference-frame recompute (molecular-layer honesty).

Recompute halothane occupancy at 1xEC50 under three partition models, varying ONLY the partition
factor against the FROZEN Vina Kd / EC50 / Kp. Tests whether the Gate-C.1 30/30 multi-target headline
is robust to the standard-state frame correction (M0, no Kp) or was Kp-inflated. Pure arithmetic.

Prereg: audits/phase1/P11/prereg.json. Reuses artifacts/occupancy/best_pocket_per_target.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ANESTH = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator')
SRC = ANESTH / 'artifacts/occupancy/best_pocket_per_target.csv'
P11 = ANESTH / 'audits/phase1/P11'
OUT = ANESTH / 'artifacts/p11_kp_reframe'

DOSE_MULT = 1.0          # 1x EC50
OCC_THRESH = 0.10
SAT_THRESH = 0.90
MODELS = {'M0_no_Kp': 0.0, 'M1_sqrt_Kp': 0.5, 'M2_full_Kp': 1.0}


def occupancy(conc, kd):
    if kd <= 0:
        return 1.0
    if conc <= 0:
        return 0.0
    return conc / (conc + kd)


def partition_factor(compartment: str, exp: float, kp: float) -> float:
    # mirror the shipped gating: Kp applied ONLY to membrane_embedded targets
    if 'membrane_embedded' in compartment.lower():
        return kp ** exp
    return 1.0


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    halo = [r for r in csv.DictReader(open(SRC)) if r['anesthetic'].lower() == 'halothane']
    assert len(halo) == 30, f'expected 30 halothane pairs, got {len(halo)}'

    per_gene = []
    counts = {m: 0 for m in MODELS}
    sats = {m: 0 for m in MODELS}
    for r in halo:
        kd = float(r['predicted_Kd_uM'])
        ec50 = float(r['ec50_clinical_aqueous_uM'])
        kp = float(r['Kp_oil_water'])
        comp = r['pocket_compartment']
        conc_aq = DOSE_MULT * ec50
        row = {'gene': r['gene'], 'kd_uM': kd, 'ec50_uM': ec50, 'kp': kp, 'compartment': comp}
        for m, exp in MODELS.items():
            occ = occupancy(conc_aq * partition_factor(comp, exp, kp), kd)
            row[f'occ_{m}'] = round(occ, 4)
            if occ > OCC_THRESH:
                counts[m] += 1
            if occ > SAT_THRESH:
                sats[m] += 1
        per_gene.append(row)

    n_M0, n_M2 = counts['M0_no_Kp'], counts['M2_full_Kp']
    spread = n_M2 - n_M0

    # G_P11_1
    if n_M0 == 30 and sats['M0_no_Kp'] == sats['M2_full_Kp']:
        g1 = 'HALT_METHODOLOGICAL'
    elif n_M0 >= 5 and sats['M0_no_Kp'] < sats['M2_full_Kp']:
        g1 = 'PASS'
    elif n_M0 < 5:
        g1 = 'DEFLATE'
    else:
        g1 = 'REVIEW'
    # G_P11_2
    g2 = 'ROBUST' if spread <= 2 else ('FRAGILE' if spread >= 5 else 'INTERMEDIATE')
    # G_P11_3 pseudo-test screen
    clears_M0 = [r['gene'] for r in per_gene if r['occ_M0_no_Kp'] > OCC_THRESH]
    kp_only = [r['gene'] for r in per_gene
               if r['occ_M2_full_Kp'] > OCC_THRESH and r['occ_M0_no_Kp'] <= OCC_THRESH]
    g3 = 'PASS' if clears_M0 else 'M0_TRIVIALLY_EMPTY'

    verdict = {
        'block': 'P11', 'pipeline': 'p11_kp_reframe',
        'prereg': str(P11 / 'prereg.json'),
        'n_halothane_pairs': len(halo),
        'engagement_counts_gt_10pct': counts,
        'saturation_counts_gt_90pct': sats,
        'frame_fragility_spread_M2_minus_M0': spread,
        'gates': {
            'G_P11_1_engagement_survival': g1,
            'G_P11_2_frame_fragility': g2,
            'G_P11_3_pseudo_test_screen': {'verdict': g3,
                                           'n_clears_M0': len(clears_M0),
                                           'n_kp_only_genes': len(kp_only),
                                           'kp_only_genes': kp_only},
        },
        'interpretation': (
            'Multi-target ENGAGEMENT (>10%) count under correct frame M0 = '
            f'{n_M0}/30; shipped M2 = {n_M2}/30. Saturation (>90%): M0={sats["M0_no_Kp"]}, '
            f'M2={sats["M2_full_Kp"]}. The sat_pa ladder is decoupled from Vina, so this is '
            'a molecular-layer honesty finding only; no V1 network flip.'),
        'per_gene': per_gene,
    }
    json.dump(verdict, open(OUT / 'p11_verdict.json', 'w'), indent=2)
    print(f'  engagement >10% : {counts}')
    print(f'  saturation >90% : {sats}')
    print(f'  frame-fragility spread (M2-M0) = {spread} -> G_P11_2 {g2}')
    print(f'  Kp-only genes (clear 10% ONLY via Kp): {len(kp_only)}')
    print(f'  G_P11_1 = {g1} ; G_P11_3 = {g3}')
    return verdict


if __name__ == '__main__':
    run()
