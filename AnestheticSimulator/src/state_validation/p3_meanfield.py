"""P3 — mean-field collapse: 1/K fluctuation-scaling check on the mouse ER substrate.

Tests G_P3_1: per-neuron weighted-in-degree CV^2 = Var_i(s_i)/mean_i(s_i)^2 scales as 1/K
(mean degree) on the actual mouse random graph, so the homogeneous mean-field reduction holds
and the cell-to-cell input deviation vanishes as 1/sqrt(K). Combined with G_P3_2 (P8 mouse
percentile in the median band), this demotes §8.1 mouse-not-special from an empirical null to a
derived corollary. Pure structural computation on build_mouse_random_graph — no brian2, no sims.

Prereg: audits/phase1/P3/prereg.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ANESTH = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator')
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, '/mnt/ssd4tb/Desktop/website/personalwebsite/scripts')

from state_validation.mouse_brain import build_mouse_random_graph  # noqa: E402

P3 = ANESTH / 'audits' / 'phase1' / 'P3'
OUT = ANESTH / 'artifacts' / 'p3_meanfield'
KS = [10, 20, 40, 80, 160]
N = 2000          # K << N so the ER graph stays sparse across the sweep
SEEDS = [1, 2, 3, 4, 5]


def cv2_weighted_indegree(W: np.ndarray) -> float:
    """CV^2 of the per-neuron weighted input magnitude (sum_j |W[i,j]|)."""
    s = np.abs(W).sum(axis=1)
    s = s[s > 0]
    return float(s.var() / s.mean() ** 2)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    logK, logCV2 = [], []
    for K in KS:
        cvs = []
        for sd in SEEDS:
            W, _signs = build_mouse_random_graph(N=N, mean_degree=float(K), graph_seed=sd)
            cv2 = cv2_weighted_indegree(np.asarray(W, dtype=np.float64))
            cvs.append(cv2)
            logK.append(np.log10(K))
            logCV2.append(np.log10(cv2))
        rows.append({'K': K, 'cv2_mean': float(np.mean(cvs)),
                     'cv2_std': float(np.std(cvs)), 'cv2_x_K': float(np.mean(cvs) * K)})
        print(f'  K={K:4d}  CV^2={np.mean(cvs):.5f}  CV^2*K={np.mean(cvs)*K:.3f}')

    slope, intercept = np.polyfit(logK, logCV2, 1)
    # R^2 of the log-log fit
    pred = slope * np.array(logK) + intercept
    ss_res = np.sum((np.array(logCV2) - pred) ** 2)
    ss_tot = np.sum((np.array(logCV2) - np.mean(logCV2)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    g1 = 'PASS' if -1.3 <= slope <= -0.7 else 'FAIL'

    # G_P3_2 from P8 corrected control
    p8 = json.load(open(ANESTH / 'artifacts/v7_match2b/v7_match2b_verdict.json'))
    perc = p8['percentile_rank_pct']
    mouse_p = perc.get('mouse')
    if mouse_p is None:
        g2 = 'INDETERMINATE'
    elif mouse_p <= 10.0:
        g2 = 'CONTRADICTS_THEOREM'
    elif 25.0 <= mouse_p <= 75.0:
        g2 = 'PASS'
    else:
        g2 = 'MARGINAL'

    verdict = {
        'block': 'P3', 'pipeline': 'p3_meanfield',
        'prereg': str(P3 / 'prereg.json'),
        'fluctuation_scaling': {
            'K_grid': KS, 'N': N, 'n_seeds': len(SEEDS),
            'rows': rows,
            'loglog_slope': float(slope), 'loglog_r2': r2,
            'expected_slope': -1.0, 'band': [-1.3, -0.7],
            'G_P3_1_verdict': g1,
        },
        'mouse_median_collapse': {
            'p8_percentiles': perc, 'mouse_percentile_pct': mouse_p,
            'band': [25.0, 75.0], 'G_P3_2_verdict': g2,
            'note': 'worm/fly below-median deviations are the real-connectome residual (P16), not a contradiction.',
        },
        'overall': 'PASS' if (g1 == 'PASS' and g2 == 'PASS') else 'REVIEW',
    }
    json.dump(verdict, open(OUT / 'p3_verdict.json', 'w'), indent=2)
    print(f'\n  log-log slope = {slope:.3f} (R^2 {r2:.3f}); band [-1.3,-0.7] -> G_P3_1 {g1}')
    print(f'  P8 mouse percentile = {mouse_p}% ; band [25,75] -> G_P3_2 {g2}')
    print(f'  P3 overall = {verdict["overall"]}')
    return verdict


if __name__ == '__main__':
    run()
