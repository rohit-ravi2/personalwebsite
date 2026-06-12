"""P16 — Held-out structure->activity exam.

V1 connectome vs STRICT degree-preserving double-edge-swap, scored against
held-out Atanas (2023) named-neuron GCaMP calcium covariance.

NON-DESTRUCTIVE: new module, new strict-shuffle operator (does NOT reuse the
leaky `m2_connectome_permutation.permute_configuration`), new artifacts under
artifacts/p16/. Spontaneous model runs go through the UNMODIFIED
phase_g_state_validator.run_single with an empty profile (total_pa==0), which is
bit-identical to a bare brain.run().

Prereg: audits/phase1/P16/prereg.json (frozen before any heavy run).

Fast preconditions (run via `python p16_structure_activity_exam.py fastgates`):
  F1  strict_double_edge_swap invariance unit test (HARD; BLOCKS the heavy run)
  F2  Atanas parse smoke (h5py partial read; overlap-set size)
  F3  positive-control scaffold dry-run (synthetic, no Brian2)

Heavy run (`python p16_structure_activity_exam.py heavy`) is GATED on F1 PASS and
is NOT launched here; the orchestrator returns the exact command.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ANESTH = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator')
ROOT = Path('/mnt/ssd4tb/Desktop/website/personalwebsite')
CONNECTOME = ROOT / 'scripts/brain/artifacts/connectome.npz'
ATANAS_DIR = ROOT / 'data/external/atanas2023'
ARTIFACTS = ANESTH / 'artifacts/p16'

# Frozen from prereg
TAU_DECAY_S = 1.5
BIN_DT_S = 0.5
RIDGE_EPS = 1e-3


# ======================================================================
# STRICT degree-preserving double-edge-swap (LOAD-BEARING REMEDIATION)
# ======================================================================

def permute_double_edge_swap(W: np.ndarray, rng: np.random.Generator,
                             n_swap_mult: int = 20) -> np.ndarray:
    """STRICT degree-preserving double-edge-swap.

    Preserves EXACTLY (verified by `assert_strict_invariants`):
      (1) per-node in-degree, (2) per-node out-degree,
      (3) the global weight multiset (sorted nonzero |w|, per sign),
      (4) per-sign edge count, (5) no self-loops, (6) no parallel-edge collapse.

    Swaps are performed WITHIN each sign class (so signs never migrate). A swap
    takes two directed edges (a->b) and (c->d) and rewires them to (a->d) and
    (c->b), carrying their WEIGHTS along (so the weight multiset is preserved
    even when |w| varies per edge). The swap is REJECTED if it would create a
    self-loop or a parallel (already-present) edge, or if a==c / b==d.

    SELF-LOOPS (autapses; the connectome has 34) are a degenerate case for a
    degree-preserving swap: rewiring them would change degrees. They are held
    out as a FROZEN invariant set (off-diagonal edges are swapped among
    themselves; the diagonal is copied through unchanged), which preserves
    in-degree, out-degree AND the self-loop set exactly.

    This is the antithesis of the leaky `permute_configuration` (which shuffles
    only post-indices => scrambles in-degree, and uses += => collapses edges).
    """
    out = np.zeros_like(W)
    # copy self-loops (autapses) through UNCHANGED — frozen invariant set
    diag_mask = np.eye(W.shape[0], dtype=bool) & (W != 0)
    out[diag_mask] = W[diag_mask]

    # GLOBAL occupancy set across BOTH signs AND the diagonal. Without this, a
    # swapped +edge and a swapped -edge can independently land on the SAME (i,j)
    # cell; the second array scatter overwrites the first, silently DROPPING an
    # edge (degree + nnz violation). Seeding it with every original edge position
    # (both signs + self-loops) makes every cross-sign collision a rejection.
    occupied = set(zip(*np.where(W != 0)))
    occupied = {(int(i), int(j)) for (i, j) in occupied}

    for sign in (+1, -1):
        sel = (np.sign(W) == sign)
        sel[np.eye(W.shape[0], dtype=bool)] = False  # exclude self-loops from swap pool
        i_idx, j_idx = np.where(sel)
        n_edges = len(i_idx)
        if n_edges < 2:
            # copy through unchanged
            out[i_idx, j_idx] = W[i_idx, j_idx]
            continue
        # mutable edge list: arrays of (pre, post, weight)
        pre = i_idx.copy()
        post = j_idx.copy()
        wts = W[i_idx, j_idx].copy()

        n_attempts = n_swap_mult * n_edges
        for _ in range(n_attempts):
            e1 = rng.integers(0, n_edges)
            e2 = rng.integers(0, n_edges)
            if e1 == e2:
                continue
            a, b = int(pre[e1]), int(post[e1])
            c, d = int(pre[e2]), int(post[e2])
            # new edges would be a->d and c->b
            if a == d or c == b:        # would create self-loop
                continue
            if a == c or b == d:        # degenerate / no-op
                continue
            # reject if target cell is occupied by ANY edge (either sign / diag),
            # except the two cells we are about to vacate.
            if ((a, d) in occupied and (a, d) not in ((a, b), (c, d))) or \
               ((c, b) in occupied and (c, b) not in ((a, b), (c, d))):
                continue
            # accept: rewire post-endpoints, weights ride with the pre-stub.
            occupied.discard((a, b))
            occupied.discard((c, d))
            occupied.add((a, d))
            occupied.add((c, b))
            post[e1] = d
            post[e2] = b
        out[pre, post] = wts
    return out.astype(W.dtype)


def _degrees(W: np.ndarray):
    A = (W != 0)
    out_deg = A.sum(axis=1)   # per-node out-degree
    in_deg = A.sum(axis=0)    # per-node in-degree
    return in_deg, out_deg


def assert_strict_invariants(W_orig: np.ndarray, W_perm: np.ndarray,
                             name: str = "") -> dict:
    """HARD invariance check. Returns a dict of per-invariant booleans + an
    overall PASS. Raises AssertionError on any failure (so it can BLOCK a run)."""
    res = {}
    in0, out0 = _degrees(W_orig)
    in1, out1 = _degrees(W_perm)
    res['in_degree_preserved'] = bool(np.array_equal(in0, in1))
    res['out_degree_preserved'] = bool(np.array_equal(out0, out1))

    # per-sign weight multiset + count
    sign_ok = {}
    count_ok = {}
    for sign in (+1, -1):
        w0 = np.sort(np.abs(W_orig[np.sign(W_orig) == sign]))
        w1 = np.sort(np.abs(W_perm[np.sign(W_perm) == sign]))
        count_ok[sign] = (w0.size == w1.size)
        sign_ok[sign] = bool(w0.size == w1.size and np.allclose(w0, w1, atol=0, rtol=0))
    res['weight_multiset_pos'] = sign_ok[+1]
    res['weight_multiset_neg'] = sign_ok[-1]
    res['edge_count_pos'] = bool(count_ok[+1])
    res['edge_count_neg'] = bool(count_ok[-1])

    # self-loops (autapses) are a FROZEN invariant set: the diagonal must be
    # bit-identical between original and permuted (they are not swappable for a
    # degree-preserving rewire).
    res['self_loops_preserved'] = bool(np.array_equal(np.diag(W_orig), np.diag(W_perm)))
    res['total_nnz_preserved'] = bool(np.count_nonzero(W_orig) == np.count_nonzero(W_perm))
    # must actually randomize (not the identity)
    res['actually_randomized'] = bool(not np.array_equal(W_orig, W_perm))

    res['PASS'] = all(v for k, v in res.items() if k != 'PASS')
    if not res['PASS']:
        raise AssertionError(f"strict-shuffle invariants FAILED for {name}: {res}")
    return res


# ======================================================================
# Observation model: GCaMP kernel + regression + partial correlation
# ======================================================================

def gcamp_convolve(binned: np.ndarray, tau_decay_s: float = TAU_DECAY_S,
                   bin_dt_s: float = BIN_DT_S) -> np.ndarray:
    """Convolve [neurons x bins] spike-count rate with a unit-area single-exp
    GCaMP6-like kernel along the time axis."""
    n_k = max(3, int(np.ceil(5 * tau_decay_s / bin_dt_s)))
    t = np.arange(n_k) * bin_dt_s
    kern = np.exp(-t / tau_decay_s)
    kern = kern / kern.sum()
    out = np.empty_like(binned, dtype=np.float64)
    for r in range(binned.shape[0]):
        out[r] = np.convolve(binned[r], kern, mode='full')[:binned.shape[1]]
    return out


def regress_out(traces: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """OLS-residualize each row (neuron trace) on the nuisance regressors.

    traces: [neurons x T]; nuisance: [T x k] (intercept added automatically).
    """
    T = traces.shape[1]
    X = np.column_stack([np.ones(T)] + [nuisance[:, j] for j in range(nuisance.shape[1])]) \
        if nuisance.size else np.ones((T, 1))
    # least squares projection residual
    beta, *_ = np.linalg.lstsq(X, traces.T, rcond=None)
    resid = traces.T - X @ beta
    return resid.T


def partial_corr_matrix(traces: np.ndarray, ridge_eps: float = RIDGE_EPS) -> np.ndarray:
    """Partial-correlation matrix from [neurons x T] traces via ridge-regularized
    precision-matrix inversion."""
    Z = traces - traces.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = Z / sd
    C = np.cov(Z)
    C = np.atleast_2d(C)
    n = C.shape[0]
    C = C + ridge_eps * np.eye(n)
    P = np.linalg.inv(C)
    d = np.sqrt(np.abs(np.diag(P)))
    d[d == 0] = 1.0
    PC = -P / np.outer(d, d)
    np.fill_diagonal(PC, 1.0)
    return PC


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two 1-D vectors (rank Pearson)."""
    if a.size < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float('nan')


def score_R(model_pc: np.ndarray, data_pc: np.ndarray,
            rate: np.ndarray, in_deg: np.ndarray, out_deg: np.ndarray) -> float:
    """Score = Spearman between off-diagonal model partial-corr and data
    partial-corr, after residualizing BOTH on pairwise nuisances
    (|rate_i-rate_j|, in-degree product, out-degree product). MANDATORY
    rate+in/out-degree control per prereg WB0."""
    n = model_pc.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    m = model_pc[iu, ju]
    d = data_pc[iu, ju]
    # pairwise nuisances
    nui = np.column_stack([
        np.abs(rate[iu] - rate[ju]),
        in_deg[iu] * in_deg[ju],
        out_deg[iu] * out_deg[ju],
    ]).astype(np.float64)
    X = np.column_stack([np.ones(len(m)), nui])
    bm, *_ = np.linalg.lstsq(X, m, rcond=None)
    bd, *_ = np.linalg.lstsq(X, d, rcond=None)
    m_res = m - X @ bm
    d_res = d - X @ bd
    return _spearman(m_res, d_res)


# ======================================================================
# Atanas loader (h5py partial reads ONLY — never the raw imaging volume)
# ======================================================================

CALCIUM_PATH = 'processing/CalciumActivity/SignalCalciumImResponseSeries/data'
CALCIUM_TS = 'processing/CalciumActivity/SignalCalciumImResponseSeries/timestamps'
LABELS_PATH = 'processing/CalciumActivity/NeuronIDs/labels'
VELOCITY_PATH = 'processing/Behavior/velocity/velocity/data'


def load_atanas_named(nwb_path: Path):
    """Stream the PROCESSED named-neuron traces + velocity from one Atanas NWB.

    Returns dict: traces[T x n_named], names[list], velocity[T], timestamps[T].
    Reads ONLY the small processed arrays; the ~25-27GB raw imaging volume is
    never touched.
    """
    import h5py
    with h5py.File(nwb_path, 'r') as f:
        data = f[CALCIUM_PATH][:]              # (T, 134) float64  ~1.7 MB
        ts = f[CALCIUM_TS][:]
        labels = f[LABELS_PATH][:]
        names = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]
        vel = f[VELOCITY_PATH][:] if VELOCITY_PATH in f else None
    named_mask = np.array([bool(n) and n.lower() != 'nan' for n in names])
    return {
        'traces_all': data,                    # (T, 134)
        'names_all': names,
        'named_mask': named_mask,
        'velocity': vel,
        'timestamps': ts,
    }


def clean_neuron_name(n: str) -> str:
    """Strip Atanas annotation suffixes ('AIY?' -> 'AIY', 'AVAL' -> 'AVAL')."""
    return n.replace('?', '').strip()


# ======================================================================
# FAST GATES
# ======================================================================

def fast_gate_F1_strict_shuffle() -> dict:
    """F1 — strict double-edge-swap invariance unit test. HARD; BLOCKS run."""
    d = np.load(CONNECTOME, allow_pickle=True)
    results = {}
    for key in ('W_chem_per_edge', 'W_chem'):
        W = d[key].astype(np.float32)
        rng = np.random.default_rng(12345)
        Wp = permute_double_edge_swap(W, rng, n_swap_mult=20)
        inv = assert_strict_invariants(W, Wp, name=key)
        # extra: report how much it actually moved
        moved = int(np.count_nonzero((W != 0) != (Wp != 0)))
        inv['n_edge_positions_changed'] = moved
        results[key] = inv
    # negative control: prove the OLD leaky permute_configuration FAILS the test
    leaky_report = {}
    try:
        sys.path.insert(0, str(ANESTH / 'src'))
        from state_validation.m2_connectome_permutation import permute_configuration
        W = d['W_chem_per_edge'].astype(np.float32)
        Wl = permute_configuration(W, np.random.default_rng(7))
        in0, out0 = _degrees(W)
        in1, out1 = _degrees(Wl)
        leaky_report = {
            'leaky_in_degree_preserved': bool(np.array_equal(in0, in1)),
            'leaky_out_degree_preserved': bool(np.array_equal(out0, out1)),
            'leaky_nnz_preserved': bool(np.count_nonzero(W) == np.count_nonzero(Wl)),
            'note': 'leaky permute_configuration is EXPECTED to FAIL in-degree and/or nnz preservation; this confirms the strict swap is a genuine remediation.',
        }
    except Exception as e:
        leaky_report = {'error': str(e)}
    overall = all(results[k]['PASS'] for k in results)
    return {'gate': 'F1_strict_shuffle_invariance', 'PASS': overall,
            'per_matrix': results, 'leaky_negative_control': leaky_report}


def fast_gate_F2_atanas_smoke(n_worms: int = 10) -> dict:
    """F2 — Atanas parse smoke + overlap-set sizing via h5py partial reads."""
    d = np.load(CONNECTOME, allow_pickle=True)
    conn_names = set(str(n) for n in d['names'])
    files = sorted(ATANAS_DIR.glob('atanas_worm_*.nwb'))[:n_worms]
    per_worm = []
    for fp in files:
        try:
            rec = load_atanas_named(fp)
            named = [clean_neuron_name(n) for n, m in
                     zip(rec['names_all'], rec['named_mask']) if m]
            overlap = sorted(set(named) & conn_names)
            per_worm.append({
                'file': fp.name,
                'T': int(rec['traces_all'].shape[0]),
                'n_rois': int(rec['traces_all'].shape[1]),
                'n_named': len(named),
                'n_overlap_with_connectome': len(overlap),
                'velocity_present': rec['velocity'] is not None,
                'overlap_sample': overlap[:12],
            })
        except Exception as e:
            per_worm.append({'file': fp.name, 'error': str(e)})
    valid = [w for w in per_worm if 'error' not in w]
    overlaps = [w['n_overlap_with_connectome'] for w in valid]
    passed = bool(valid and min(overlaps) >= 10)
    return {'gate': 'F2_atanas_parse_smoke', 'PASS': passed,
            'n_files_parsed': len(valid),
            'overlap_min': int(min(overlaps)) if overlaps else 0,
            'overlap_max': int(max(overlaps)) if overlaps else 0,
            'overlap_median': float(np.median(overlaps)) if overlaps else 0.0,
            'per_worm': per_worm}


def fast_gate_F3_positive_control_scaffold() -> dict:
    """F3 — positive-control scaffold dry-run on a TINY synthetic covariance
    (no Brian2). Confirms the GCaMP-kernel + regression + partial-correlation +
    Spearman scorer runs end-to-end and that real >> shuffle when structure
    exists. (Full Gate-C with real model covariance runs in the heavy job.)"""
    rng = np.random.default_rng(2026)
    n, T = 30, 600
    # ground-truth structure: a sparse signed coupling matrix
    L = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    pick = rng.random(len(iu[0])) < 0.15
    L[iu[0][pick], iu[1][pick]] = rng.normal(0, 1, pick.sum())
    L = L + L.T
    # latent dynamics: x_{t+1} = 0.6 x_t + L-driven input + noise (binned "rates")
    x = np.zeros((n, T))
    for t in range(1, T):
        x[:, t] = 0.6 * x[:, t-1] + 0.3 * (L @ x[:, t-1]) + rng.normal(0, 1, n)
    rates = np.clip(x - x.min(), 0, None)  # nonneg "spike rate proxy" [n x T]

    # observation pipeline
    ca = gcamp_convolve(rates)
    glob = ca.mean(axis=0, keepdims=True).repeat(n, axis=0)  # global mean as nuisance
    nuis = ca.mean(axis=0)[:, None]
    ca_res = regress_out(ca, nuis)
    data_pc = partial_corr_matrix(ca_res)

    rate_proxy = rates.mean(axis=1)
    in_deg = (L != 0).sum(axis=0).astype(float)
    out_deg = (L != 0).sum(axis=1).astype(float)

    # "real" model = the SAME structure observed via a different noise seed
    x2 = np.zeros((n, T))
    rng2 = np.random.default_rng(99)
    for t in range(1, T):
        x2[:, t] = 0.6 * x2[:, t-1] + 0.3 * (L @ x2[:, t-1]) + rng2.normal(0, 1, n)
    rates2 = np.clip(x2 - x2.min(), 0, None)
    ca2 = gcamp_convolve(rates2)
    model_pc_real = partial_corr_matrix(regress_out(ca2, ca2.mean(axis=0)[:, None]))
    R_real = score_R(model_pc_real, data_pc, rate_proxy, in_deg, out_deg)

    # "shuffle" model = structure destroyed (independent neurons)
    rates_sh = np.clip(rng.normal(0, 1, (n, T)), 0, None)
    ca_sh = gcamp_convolve(rates_sh)
    model_pc_shuf = partial_corr_matrix(regress_out(ca_sh, ca_sh.mean(axis=0)[:, None]))
    R_shuf = score_R(model_pc_shuf, data_pc, rate_proxy, in_deg, out_deg)

    passed = bool(np.isfinite(R_real) and np.isfinite(R_shuf) and R_real > R_shuf)
    return {'gate': 'F3_positive_control_scaffold', 'PASS': passed,
            'R_real_synthetic': round(float(R_real), 4),
            'R_shuffle_synthetic': round(float(R_shuf), 4),
            'note': 'scaffold validity: with ground-truth structure, real >> shuffle. The R>=0.5 acceptance threshold is evaluated in the HEAVY job on real model covariance, not here.'}


def run_fast_gates() -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = {'block_id': 'P16', 'phase': 'fast_preconditions'}
    out['F1'] = fast_gate_F1_strict_shuffle()
    out['F2'] = fast_gate_F2_atanas_smoke()
    out['F3'] = fast_gate_F3_positive_control_scaffold()
    out['ALL_FAST_PASS'] = bool(out['F1']['PASS'] and out['F2']['PASS'] and out['F3']['PASS'])
    out['heavy_run_unblocked'] = bool(out['F1']['PASS'])  # F1 is the hard run-blocker
    with open(ARTIFACTS / 'fast_gate_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fastgates'
    if mode == 'fastgates':
        r = run_fast_gates()
        print(json.dumps(r, indent=2, default=str))
    elif mode == 'heavy':
        raise SystemExit("Heavy run is launched by the orchestrator; see prereg + README. "
                         "Implement the full scoring loop here before launching.")
    else:
        raise SystemExit(f"unknown mode {mode}")
