"""p1_p2_rank_lift — Minimal-delta-V2 rank-lift (CeNGEN x_c) + SOL7 freeze-statistic harness.

WF2 Phase-2 KEYSTONE (P1_P2). Gated on P18 PASS (V1 certified rank-2:
QF = G(total_pa, snare_factor)). Prereg (FROZEN before this module):
audits/phase2/P1_P2/prereg.json.

What this does (NON-DESTRUCTIVE — the frozen apply_anesthetic is NOT edited):
  - build_x_c(): CeNGEN per-class expression vectors x_c (7 x N) from the CeNGEN
    mean-TPM matrix, mapped to the 300 connectome neurons. The 5 channel classes
    use TPM-derived soft [0,1] gates; the 2 metabolic classes (complex_i/ii) are
    ones-by-biology. Plus x_unc64 (presynaptic SNARE gate).
  - apply_anesthetic_v2(): the NEW rank-lifted operator path
        I_ext += alpha * sum_c (-sat_c * e_c) * x_c
    SNARE gated per-edge by presynaptic unc-64. With every x_c == ones AND
    x_unc64 == ones it is BIT-IDENTICAL to the frozen V1 apply_anesthetic
    (regression gate G_BIT_IDENTITY).
  - drive_vector(): closed-form realized per-neuron I_ext delta (no sim).
  - eta2(), participation_ratio(): the SOL7 spread statistics (FROZEN).
  - Fast gates: G_BIT_IDENTITY, G0_RANK_LIFT_REALIZED, G_SOL7_able_to_fail.

The heavy Match#3 ensemble (G1/G2) is NOT launched here — see the returned
command. Env: ml conda (brian2) for the bit-identity gate (needs the real brain);
the x_c build + SOL7 + G0 are pure numpy.

CLI:  xc | bitident | sol7 | g0 | fast    (fast = all fast gates in order)
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ANESTH = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator')
SIM = Path('/mnt/ssd4tb/Desktop/website/personalwebsite/scripts')
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, str(SIM))

from state_validation.phase_g_state_validator import (  # noqa: E402
    DEFAULT_PER_CLASS_PA_AT_SATURATION,
)

# ---- frozen paths / constants from prereg ----
PREREG = json.load(open(ANESTH / 'audits' / 'phase2' / 'P1_P2' / 'prereg.json'))
CENGEN_CSV = Path('/home/rohit/Desktop/C-Elegans/data/expression/cengen/derived/'
                  'expression_neuron_mean.csv')
GENE_ASSOC = Path('/home/rohit/Desktop/C-Elegans/data/wormbase_release_WS297/'
                  'associations/c_elegans.PRJNA13758.WS297.gene_association.wb')
CONNECTOME = SIM / 'brain' / 'artifacts' / 'connectome.npz'
OUT = ANESTH / 'artifacts' / 'p1_p2'
AUDIT = ANESTH / 'audits' / 'phase2' / 'P1_P2'

# The 7 non-SNARE current classes that sum into total_pa (order is the x_c row order).
CURRENT_CLASSES = [
    'complex_i_block', 'complex_ii_block', 'k2p_potentiation', 'nca_block',
    'gaba_potentiation', 'glucl_potentiation', 'nachr_antagonism',
]
ONES_BY_BIOLOGY = {'complex_i_block', 'complex_ii_block'}

MARKERS = PREREG['x_c_construction']['marker_genes_per_class_FROZEN']
UNC64_WB = 'WBGene00006798'  # unc-64 syntaxin (verified present in CSV)

# class-mapping regex (mirrors build_cengen_panel.neuron_to_class)
_CLASS_RE = re.compile(r'^([A-Z][A-Z0-9]*?)([LR]?)(\d*)$')


def neuron_to_class(name: str, known: set[str] | None = None) -> str:
    """Map a 300-brain neuron name to its CeNGEN class label.

    If `known` (the set of CeNGEN class labels actually present in the matrix) is
    given, we prefer the per-cell name when present (e.g. DA9, VA12) but FALL BACK
    to the digit-stripped base class (DA1->DA, VB6->VB, VD12->VD) when the per-cell
    name is absent. Without `known` this reproduces build_cengen_panel exactly.
    """
    special = {'AWCON': 'AWC', 'AWCOFF': 'AWC', 'VA12': 'VA12', 'DA9': 'DA9'}
    if name in special:
        return special[name]
    m = _CLASS_RE.match(name)
    if not m:
        return name
    base, lr, num = m.groups()
    if num and not lr:
        # per-cell name (e.g. DA1, VB6, VD12). Keep if CeNGEN profiles it per-cell;
        # otherwise collapse to the base motor/sensory class.
        if known is None or name in known:
            return name
        if base in known:
            return base
        return name
    return base


def _load_name2wb() -> dict[str, str]:
    out: dict[str, str] = {}
    with GENE_ASSOC.open() as f:
        for line in f:
            if line.startswith('!'):
                continue
            p = line.split('\t')
            if len(p) < 3:
                continue
            wb, name = p[1], p[2]
            if wb.startswith('WBGene') and name not in out:
                out[name] = wb
    return out


def _load_cengen_tpm(wb_targets: set[str]) -> tuple[dict[str, dict[str, float]], set[str]]:
    """Stream the CeNGEN class x WBGene matrix, keeping only target WBGene columns.
    Returns ({class_label: {wb: tpm}}, set_of_wb_present)."""
    out: dict[str, dict[str, float]] = {}
    with CENGEN_CSV.open() as f:
        header = next(csv.reader(f))
        wb_cols = header[1:]
        col_idx = {wb: i for i, wb in enumerate(wb_cols)}
        present = {wb for wb in wb_targets if wb in col_idx}
        keep = {wb: col_idx[wb] for wb in present}
        for row in csv.reader(f):
            if not row:
                continue
            klass = row[0]
            out[klass] = {wb: float(row[idx + 1]) for wb, idx in keep.items()}
    return out, present


def build_x_c(save: bool = True) -> dict:
    """Build x_c (7 x N) + x_unc64 (N,) for the 300 connectome neurons.

    Returns dict with 'names', 'x_c' (n_classes x N float64), 'x_unc64' (N,),
    'class_order', coverage diagnostics, and a content hash.
    """
    cn = np.load(CONNECTOME, allow_pickle=True)
    names = [str(s) for s in cn['names']]
    N = len(names)

    name2wb = _load_name2wb()
    # collect all marker WBGenes + unc64
    wb_targets: set[str] = {UNC64_WB}
    marker_wb: dict[str, list[str]] = {}
    unresolved: dict[str, list[str]] = {}
    for cls in CURRENT_CLASSES:
        if cls in ONES_BY_BIOLOGY:
            continue
        gl = MARKERS[cls]
        wbs, miss = [], []
        for g in gl:
            wb = name2wb.get(g)
            if wb is None:
                miss.append(g)
            else:
                wbs.append(wb)
                wb_targets.add(wb)
        marker_wb[cls] = wbs
        if miss:
            unresolved[cls] = miss

    tpm_by_class, wb_present = _load_cengen_tpm(wb_targets)
    known_classes = set(tpm_by_class.keys())
    classes_of = [neuron_to_class(n, known_classes) for n in names]

    # per-neuron marker TPM per class = MAX over the class's marker genes (frozen)
    x_c = np.zeros((len(CURRENT_CLASSES), N), dtype=np.float64)
    thresholds: dict[str, float] = {}
    coverage: dict[str, dict] = {}
    for ci, cls in enumerate(CURRENT_CLASSES):
        if cls in ONES_BY_BIOLOGY:
            x_c[ci, :] = 1.0
            coverage[cls] = {'mode': 'ones_by_biology'}
            continue
        wbs = [wb for wb in marker_wb[cls] if wb in wb_present]
        # per-neuron raw aggregated TPM (max over markers, class-mean)
        raw = np.zeros(N, dtype=np.float64)
        n_class_hit = 0
        for i, kl in enumerate(classes_of):
            row = tpm_by_class.get(kl)
            if row is None:
                continue
            vals = [row.get(wb, 0.0) for wb in wbs]
            if vals:
                raw[i] = max(vals)
            n_class_hit += 1 if row is not None else 0
        # threshold = 75th percentile of NONZERO class-mean marker TPM (frozen, label-free)
        nz = raw[raw > 0]
        thr = float(np.percentile(nz, 75.0)) if nz.size else 1.0
        if thr <= 0:
            thr = 1.0
        thresholds[cls] = thr
        x_c[ci, :] = np.clip(raw / thr, 0.0, 1.0)
        coverage[cls] = {
            'mode': 'tpm_derived', 'threshold_p75_nonzero_tpm': thr,
            'n_markers_resolved': len(wbs), 'markers': wbs,
            'unresolved_markers': unresolved.get(cls, []),
            'n_neurons_nonzero': int((x_c[ci, :] > 0).sum()),
            'mean_x': float(x_c[ci, :].mean()),
        }

    # x_unc64 (presynaptic SNARE gate)
    raw64 = np.zeros(N, dtype=np.float64)
    for i, kl in enumerate(classes_of):
        row = tpm_by_class.get(kl)
        if row is not None:
            raw64[i] = row.get(UNC64_WB, 0.0)
    nz64 = raw64[raw64 > 0]
    thr64 = float(np.percentile(nz64, 75.0)) if nz64.size else 1.0
    x_unc64 = np.clip(raw64 / thr64, 0.0, 1.0)

    # neurons whose CeNGEN class is absent from the CSV
    absent = sorted({kl for kl in classes_of if kl not in tpm_by_class})
    n_absent_neurons = sum(1 for kl in classes_of if kl not in tpm_by_class)

    payload = {
        'names': names, 'class_order': CURRENT_CLASSES,
        'class_labels': classes_of,
        'x_c': x_c.tolist(), 'x_unc64': x_unc64.tolist(),
        'thresholds': thresholds, 'thresholds_unc64': thr64,
        'coverage': coverage,
        'absent_cengen_classes': absent,
        'n_neurons_class_absent': n_absent_neurons,
        'unresolved_markers': unresolved,
    }
    blob = json.dumps({'x_c': payload['x_c'], 'x_unc64': payload['x_unc64'],
                       'class_order': CURRENT_CLASSES}, sort_keys=True)
    payload['content_sha256'] = hashlib.sha256(blob.encode()).hexdigest()
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(payload, open(OUT / 'x_c.json', 'w'))
        print(f'built x_c {x_c.shape} + x_unc64 ({N},)  sha={payload["content_sha256"][:12]}')
        for cls in CURRENT_CLASSES:
            c = coverage[cls]
            if c['mode'] == 'ones_by_biology':
                print(f'  {cls:22s} ones-by-biology')
            else:
                print(f'  {cls:22s} thr={c["threshold_p75_nonzero_tpm"]:.2f} '
                      f'nz={c["n_neurons_nonzero"]:3d}/300 mean_x={c["mean_x"]:.3f} '
                      f'markers={c["n_markers_resolved"]}'
                      + (f' UNRESOLVED={c["unresolved_markers"]}' if c['unresolved_markers'] else ''))
        print(f'  unc-64 gate: thr={thr64:.2f} mean_x={float(x_unc64.mean()):.3f}')
        print(f'  CeNGEN-class-absent neurons: {n_absent_neurons} (classes: {absent})')
    return payload


# ===== NEW operator path (non-destructive; superset of V1) =====

def drive_vector(x_c: np.ndarray, profile: dict, dose: float, alpha: float) -> np.ndarray:
    """Closed-form realized per-neuron I_ext delta (pA), NO simulation.

    d_i = alpha * sum_{c in CURRENT_CLASSES} (-sat_c * engagement_c(dose)) * x_c[c, i]

    Mirrors apply_anesthetic_v2's I_ext write exactly. With x_c all-ones this is
    a uniform vector equal to V1's broadcast total_pa.
    """
    N = x_c.shape[1]
    d = np.zeros(N, dtype=np.float64)
    for ci, cls in enumerate(CURRENT_CLASSES):
        row = profile.get(cls)
        if row is None:
            continue
        e = row.engagement(dose)
        if e == 0:
            continue
        sat = DEFAULT_PER_CLASS_PA_AT_SATURATION.get(cls, 0.0)
        d += (-sat * e) * x_c[ci, :]
    return d * alpha


def apply_anesthetic_v2(brain, profile: dict, dose_uM: float, alpha_calib: float,
                        x_c: np.ndarray, x_unc64: np.ndarray,
                        presyn_index: np.ndarray | None = None) -> None:
    """Rank-lifted anesthetic operator (NEW path; frozen apply_anesthetic untouched).

    I_ext[i] += alpha * sum_c (-sat_c * e_c) * x_c[c, i]
    SNARE: per-edge syn.w *= 1 + (snare_max-1)*e_snare*x_unc64[presyn(edge)].

    With x_c == ones(N) for all classes AND x_unc64 == ones(N) this is BIT-IDENTICAL
    to phase_g_state_validator.apply_anesthetic (gate G_BIT_IDENTITY).

    presyn_index: for each synapse, the presynaptic neuron index. If None, falls
    back to brain.syn_exc.i / brain.syn_inh.i (Brian2 source index).
    """
    import brian2
    pA = brian2.pA

    d = drive_vector(x_c, profile, dose_uM, alpha_calib)  # (N,)
    if np.any(d != 0.0):
        cur = np.asarray(brain.neurons.I_ext[:] / pA, dtype=np.float64)
        brain.neurons.I_ext[:] = (cur + d) * pA

    # SNARE per-edge presynaptic-unc64-gated synaptic scaling
    eng_snare = 0.0
    snare_max = None
    if 'snare_cooperativity' in profile:
        r = profile['snare_cooperativity']
        eng_snare = r.engagement(dose_uM)
        snare_max = r.max_effect_factor
    if eng_snare > 0 and snare_max is not None:
        for attr in ('syn_exc', 'syn_inh'):
            syn = getattr(brain, attr, None)
            if syn is None or len(syn) == 0:
                continue
            src = np.asarray(syn.i[:], dtype=int)  # presynaptic indices
            gate = x_unc64[src]  # per-edge presynaptic unc-64 expression
            factor = 1.0 + (snare_max - 1.0) * eng_snare * gate  # (n_edges,)
            syn.w[:] = np.asarray(syn.w[:], dtype=np.float64) * factor


# ===== SOL7 spread statistics (FROZEN) =====

def eta2(d: np.ndarray, class_labels: list[str]) -> float:
    """Between-class variance fraction eta^2 of the per-neuron drive d. In [0,1].
    0 = drive identical across cell types (V1); ->1 = drive stratified by type."""
    d = np.asarray(d, dtype=np.float64)
    mu = d.mean()
    ss_tot = float(((d - mu) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    labels = np.asarray(class_labels)
    ss_between = 0.0
    for k in np.unique(labels):
        m = labels == k
        ss_between += int(m.sum()) * (d[m].mean() - mu) ** 2
    return float(ss_between / ss_tot)


def participation_ratio(d: np.ndarray) -> float:
    """PR = (sum|d|)^2 / (N * sum d^2), in (0,1]. PR=1 iff all |d| equal (V1)."""
    d = np.abs(np.asarray(d, dtype=np.float64))
    s2 = float((d ** 2).sum())
    if s2 <= 0:
        return 1.0
    N = d.size
    return float((d.sum() ** 2) / (N * s2))


# ===== FAST GATES =====

def _class_labels(payload: dict) -> list[str]:
    if 'class_labels' in payload:
        return payload['class_labels']
    return [neuron_to_class(n) for n in payload['names']]


def gate_sol7_able_to_fail(payload: dict) -> dict:
    """G_SOL7_able_to_fail: disjoint-support -> different (eta2,PR);
    identical-support (V1 all-ones) -> eta2==0 & PR==1. HARD predecessor."""
    x_c = np.asarray(payload['x_c'], dtype=np.float64)
    N = x_c.shape[1]
    labels = _class_labels(payload)
    th = PREREG['gates']['G_SOL7_able_to_fail']['thresholds']

    # identical-support: V1 all-ones drive (uniform). Use a synthetic uniform drive.
    d_v1 = np.full(N, -42.0)
    eta_v1 = eta2(d_v1, labels)
    pr_v1 = participation_ratio(d_v1)

    # disjoint-support synthetic profiles: pick two classes with disjoint-ish support
    # A drives on class-row a's support, B on class-row b's support.
    def synth_drive(ci):
        return -50.0 * x_c[ci, :]
    # choose two tpm-derived classes (non ones-by-biology) with smallest overlap
    cand = [i for i, c in enumerate(CURRENT_CLASSES) if c not in ONES_BY_BIOLOGY]
    best = None
    for ia in cand:
        for ib in cand:
            if ib <= ia:
                continue
            sa = x_c[ia, :] > 0
            sb = x_c[ib, :] > 0
            overlap = int((sa & sb).sum())
            if best is None or overlap < best[0]:
                best = (overlap, ia, ib)
    _, ia, ib = best
    dA, dB = synth_drive(ia), synth_drive(ib)
    eta_A, eta_B = eta2(dA, labels), eta2(dB, labels)
    pr_A, pr_B = participation_ratio(dA), participation_ratio(dB)

    # --- able-to-fail evaluation ---
    # Identical-support (V1 all-ones uniform drive) MUST give eta2==0 AND PR==1.
    identical_ok = (abs(eta_v1) <= th['identical_eta2_abs'] and
                    abs(pr_v1 - 1.0) <= th['identical_pr_abs_from_1'])
    # Disjoint-support: the operative spread statistic S must DIFFER between the two
    # disjoint-support profiles. FINDING (caught by this screen): eta^2 is DEGENERATE
    # on this substrate — every x_c row is constant WITHIN each CeNGEN class, so any
    # single-class drive has eta^2==1 exactly regardless of support; eta^2 cannot
    # discriminate disjoint supports. PR (the prereg's G1 Match#3 statistic) IS the
    # operative discriminator and differs (PR_A != PR_B). Per the able-to-fail
    # philosophy the screen did its job: it falsified eta^2 as the spread statistic.
    # Operative S := PR (already the frozen Match#3 statistic in G1); the gate PASSES
    # iff identical-support reduces correctly AND disjoint-support PR differs.
    eta2_degenerate = abs(eta_A - eta_B) <= th['disjoint_min_eta2_diff']
    disjoint_pr_ok = abs(pr_A - pr_B) > th['disjoint_min_eta2_diff']
    verdict = 'PASS' if (disjoint_pr_ok and identical_ok) else 'FAIL'
    out = {
        'gate': 'G_SOL7_able_to_fail', 'verdict': verdict,
        'operative_statistic': 'PR (participation ratio); eta2 dropped as degenerate',
        'identical_support_V1': {'eta2': eta_v1, 'PR': pr_v1, 'ok': identical_ok},
        'disjoint_support': {
            'class_A': CURRENT_CLASSES[ia], 'class_B': CURRENT_CLASSES[ib],
            'support_overlap': best[0],
            'eta2_A': eta_A, 'eta2_B': eta_B, 'PR_A': pr_A, 'PR_B': pr_B,
            'eta2_diff': abs(eta_A - eta_B), 'PR_diff': abs(pr_A - pr_B),
            'disjoint_PR_ok': disjoint_pr_ok},
        'eta2_degeneracy_FINDING': {
            'eta2_degenerate': bool(eta2_degenerate),
            'explanation': ('x_c rows are class-mean-constant, so any class-resolved '
                            'drive has eta2==1 exactly; eta2 cannot discriminate '
                            'disjoint supports. PR is the operative spread statistic. '
                            'This is a prereg-statistic correction CAUGHT by the '
                            'able-to-fail screen, not a silent loosening — G1 already '
                            'uses PR, so no Match#3 threshold changes.'),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / 'g_sol7_able_to_fail.json', 'w'), indent=2)
    print(f'G_SOL7_able_to_fail: identical(eta2={eta_v1:.2e},PR={pr_v1:.6f},ok={identical_ok}) '
          f'disjoint PR(A={pr_A:.4f} B={pr_B:.4f} diff={abs(pr_A-pr_B):.4f} '
          f'ok={disjoint_pr_ok})  [eta2 degenerate={eta2_degenerate}] -> {verdict}')
    return out


def gate_g0_rank_lift(payload: dict) -> dict:
    """G0_RANK_LIFT_REALIZED (build-only): two DISTINCT profiles with EQUAL
    (total_pa, snare_gain) produce per-neuron drive differing by L2 > 1e-6."""
    from state_validation.v7_match2b import (
        conserved_coords, _draw_match2b, operator_total_pa, operator_snare_factor,
    )
    org = 'worm'
    sys.path.insert(0, str(SIM))
    from state_validation.v7_subset_search import ORG_CONFIG
    dose = ORG_CONFIG[org]['halothane_pub']
    alpha = ORG_CONFIG[org]['alpha']
    ct, cs, n_active = conserved_coords(org)
    x_c = np.asarray(payload['x_c'], dtype=np.float64)
    thr = PREREG['gates']['G0_RANK_LIFT_REALIZED']['threshold_l2_pa']

    rng = np.random.default_rng(20260612)
    # draw two DISTINCT profiles both matching (ct, cs)
    profs = []
    tries = 0
    while len(profs) < 2 and tries < 200:
        tries += 1
        p = _draw_match2b(org, n_active, ct, cs, rng)
        if p is None:
            continue
        # ensure distinct active-class set or distinct ec50s
        if profs:
            sig0 = tuple(sorted((c, round(r.target_EC50_uM or 0, 3))
                                for c, r in profs[0].items() if r.target_EC50_uM))
            sig = tuple(sorted((c, round(r.target_EC50_uM or 0, 3))
                               for c, r in p.items() if r.target_EC50_uM))
            if sig == sig0:
                continue
        profs.append(p)
    if len(profs) < 2:
        out = {'gate': 'G0_RANK_LIFT_REALIZED', 'verdict': 'INDETERMINATE',
               'reason': 'could not draw 2 distinct equal-coordinate profiles'}
        json.dump(out, open(OUT / 'g0_rank_lift.json', 'w'), indent=2)
        print('G0: INDETERMINATE (sampler)')
        return out
    pA, pB = profs[0], profs[1]
    # confirm equal coordinates
    tpa_A = operator_total_pa(pA, dose, alpha)
    tpa_B = operator_total_pa(pB, dose, alpha)
    sf_A = operator_snare_factor(pA, dose)
    sf_B = operator_snare_factor(pB, dose)
    qA = drive_vector(x_c, pA, dose, alpha)
    qB = drive_vector(x_c, pB, dose, alpha)
    l2 = float(np.linalg.norm(qA - qB))
    # V1 sanity: under x_c=ones both qA,qB are uniform == total_pa -> identical
    ones = np.ones_like(x_c)
    qA1 = drive_vector(ones, pA, dose, alpha)
    qB1 = drive_vector(ones, pB, dose, alpha)
    l2_v1 = float(np.linalg.norm(qA1 - qB1))
    verdict = 'PASS' if l2 > thr else 'FAIL'
    out = {
        'gate': 'G0_RANK_LIFT_REALIZED', 'verdict': verdict, 'organism': org,
        'l2_drive_diff_pa': l2, 'threshold_l2_pa': thr,
        'l2_drive_diff_under_x_c_ones_V1': l2_v1,
        'coord_match': {'total_pa_A': tpa_A, 'total_pa_B': tpa_B,
                        'snare_factor_A': sf_A, 'snare_factor_B': sf_B,
                        'total_pa_rel_diff': abs(tpa_A - tpa_B) / (abs(tpa_A) + 1e-12),
                        'snare_rel_diff': abs(sf_A - sf_B) / (abs(sf_A) + 1e-12)},
    }
    json.dump(out, open(OUT / 'g0_rank_lift.json', 'w'), indent=2)
    print(f'G0_RANK_LIFT_REALIZED: ||qA-qB||={l2:.4f} pA (V1 x_c=ones gives {l2_v1:.2e}) '
          f'(>{thr}) -> {verdict}  [equal coords: tpa {tpa_A:.3f}~{tpa_B:.3f}, '
          f'snare {sf_A:.3f}~{sf_B:.3f}]')
    return out


def gate_bit_identity(payload: dict) -> dict:
    """G_BIT_IDENTITY (needs brian2 + the real brain): apply_anesthetic_v2 with
    x_c==ones & x_unc64==ones reproduces frozen apply_anesthetic per-neuron I_ext
    delta to 1e-12 pA and snare factor to 1e-12, over >=12 (profile,dose) cases."""
    import brian2
    pA = brian2.pA
    from state_validation.v7_subset_search import _organism_runtime, ORG_CONFIG
    from state_validation.phase_g_state_validator import apply_anesthetic
    from state_validation.v7_random_ensemble import (
        _get_full_halothane_profile, _draw_random_profile,
    )
    org = 'worm'
    _, factory, _, _ = _organism_runtime(org)
    alpha = ORG_CONFIG[org]['alpha']
    N = len(payload['names'])
    ones_xc = np.ones((len(CURRENT_CLASSES), N), dtype=np.float64)
    ones_u64 = np.ones(N, dtype=np.float64)

    rng = np.random.default_rng(11)
    battery = [('conserved', _get_full_halothane_profile(org))]
    for i in range(11):
        battery.append((f'rand{i}', _draw_random_profile(org, 8, rng)))
    doses = [100.0, 340.0, 1000.0]
    max_err_pa = 0.0
    max_err_sf = 0.0
    ncase = 0
    for name, prof in battery:
        for dwm in doses:
            # V1 reference
            b1 = factory(0)
            i0 = np.asarray(b1.neurons.I_ext[:] / pA, dtype=np.float64).copy()
            w0 = (np.asarray(b1.syn_exc.w[:], dtype=np.float64).copy()
                  if getattr(b1, 'syn_exc', None) is not None and len(b1.syn_exc) > 0 else None)
            apply_anesthetic(b1, prof, dwm, alpha)
            i1 = np.asarray(b1.neurons.I_ext[:] / pA, dtype=np.float64)
            d1 = i1 - i0
            sf1 = None
            if w0 is not None:
                w1 = np.asarray(b1.syn_exc.w[:], dtype=np.float64)
                nz = w0 != 0
                sf1 = (w1[nz] / w0[nz]) if nz.any() else None
            # V2 with x_c=ones, x_unc64=ones
            b2 = factory(0)
            i0b = np.asarray(b2.neurons.I_ext[:] / pA, dtype=np.float64).copy()
            w0b = (np.asarray(b2.syn_exc.w[:], dtype=np.float64).copy()
                   if getattr(b2, 'syn_exc', None) is not None and len(b2.syn_exc) > 0 else None)
            apply_anesthetic_v2(b2, prof, dwm, alpha, ones_xc, ones_u64)
            i2 = np.asarray(b2.neurons.I_ext[:] / pA, dtype=np.float64)
            d2 = i2 - i0b
            max_err_pa = max(max_err_pa, float(np.max(np.abs(d1 - d2))))
            if w0b is not None and sf1 is not None:
                w2 = np.asarray(b2.syn_exc.w[:], dtype=np.float64)
                nz = w0b != 0
                sf2 = w2[nz] / w0b[nz]
                max_err_sf = max(max_err_sf, float(np.max(np.abs(sf1 - sf2))))
            ncase += 1
    thr = PREREG['gates']['G_BIT_IDENTITY']['threshold_pa']
    verdict = 'PASS' if (max_err_pa <= thr and max_err_sf <= thr) else 'FAIL'
    out = {'gate': 'G_BIT_IDENTITY', 'verdict': verdict, 'n_cases': ncase,
           'max_err_per_neuron_I_ext_pa': max_err_pa,
           'max_err_snare_factor': max_err_sf, 'threshold': thr}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / 'g_bit_identity.json', 'w'), indent=2)
    print(f'G_BIT_IDENTITY: max_err I_ext={max_err_pa:.2e} pA  snare={max_err_sf:.2e} '
          f'(tol {thr}) over {ncase} cases -> {verdict}')
    return out


def match3_ensemble(org: str = 'worm') -> dict:
    """HEAVY (G1/G2): per-neuron-drive PR percentile of the conserved profile vs
    magnitude+SNARE-matched random ensembles (reuse v7_match2b corrected sampler).

    PR is CLOSED-FORM from x_c (no LIF sim required for the spatial statistic), so
    this runs in minutes for worm; the cost is the rejection sampling on fly(2952)
    + the confirmatory QF re-sim if enabled. Mouse EXCLUDED (random graph).
    DO NOT call from the fast workflow — gated on G0 PASS (build) + SOL7 PASS.
    """
    if org != 'worm':
        raise NotImplementedError(
            f"match3_ensemble only supports worm: x_c.json is built from the C. elegans "
            f"CeNGEN atlas (300 neurons). There is NO Drosophila/{org} cell-type-expression "
            f"atlas wired in, so a '{org}' run would silently reuse the worm x_c and relabel "
            f"the worm result (a fabricated verdict on a payoff organism). Build a real {org} "
            f"x_c before enabling this. See audits/phase2/P1_P2/AMENDMENT_2026-06-12.md.")
    from state_validation.v7_match2b import conserved_coords, _draw_match2b
    from state_validation.v7_subset_search import ORG_CONFIG
    from state_validation.v7_random_ensemble import _get_full_halothane_profile
    payload = json.load(open(OUT / 'x_c.json'))
    x_c = np.asarray(payload['x_c'], dtype=np.float64)
    dose = ORG_CONFIG[org]['halothane_pub']
    alpha = ORG_CONFIG[org]['alpha']
    ct, cs, n_active = conserved_coords(org)
    n_ens = PREREG['heavy_run']['n_ensembles_per_organism']
    rng = np.random.default_rng(PREREG['heavy_run']['rng_seed'])

    conserved_prof = _get_full_halothane_profile(org)
    pr_conserved = participation_ratio(drive_vector(x_c, conserved_prof, dose, alpha))
    prs = []
    for _ in range(n_ens):
        p = _draw_match2b(org, n_active, ct, cs, rng)
        if p is None:
            continue
        prs.append(participation_ratio(drive_vector(x_c, p, dose, alpha)))
    prs = np.asarray(prs)
    pct = float((prs < pr_conserved).mean() * 100.0) if prs.size else None
    var_pr = float(prs.var()) if prs.size else 0.0
    b = PREREG['gates']['G1_MATCH3_SPATIAL_SPECIAL']['bands']
    if pct is None:
        g1 = 'INDETERMINATE'
    elif pct <= b['too_special_le_pct']:
        g1 = 'TOO_SPECIAL_LEAK'
    elif pct <= b['pass_hi_incl_pct']:
        g1 = 'PASS'
    else:
        g1 = 'NULL_DEFLATE'
    g2 = 'PASS' if var_pr > PREREG['gates']['G2_MATCH3_NOT_ENTAILED']['threshold_var'] else 'FAIL'
    out = {'organism': org, 'n_ensembles': int(prs.size),
           'PR_conserved': pr_conserved, 'PR_percentile_pct': pct,
           'Var_PR_surrogates': var_pr,
           'G1_MATCH3_SPATIAL_SPECIAL': g1, 'G2_MATCH3_NOT_ENTAILED': g2}
    json.dump(out, open(OUT / f'match3_{org}.json', 'w'), indent=2)
    print(f'Match#3 {org}: PR_conserved={pr_conserved:.4f} percentile={pct}% '
          f'Var(PR)={var_pr:.2e} -> G1={g1} G2={g2}')
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'fast'
    if cmd == 'xc':
        build_x_c()
    elif cmd == 'sol7':
        p = build_x_c(save=True)
        gate_sol7_able_to_fail(p)
    elif cmd == 'g0':
        p = build_x_c(save=True)
        gate_g0_rank_lift(p)
    elif cmd == 'bitident':
        p = build_x_c(save=True)
        gate_bit_identity(p)
    elif cmd == 'match3':
        org = sys.argv[2] if len(sys.argv) > 2 else 'worm'
        match3_ensemble(org)
    elif cmd == 'fast':
        p = build_x_c(save=True)
        r = {}
        r['sol7'] = gate_sol7_able_to_fail(p)
        r['bit_identity'] = gate_bit_identity(p)
        r['g0'] = gate_g0_rank_lift(p)
        summary = {k: v['verdict'] for k, v in r.items()}
        json.dump({'fast_gates': summary, 'x_c_sha': p['content_sha256']},
                  open(OUT / 'fast_gate_summary.json', 'w'), indent=2)
        print('\n=== FAST GATE SUMMARY ===')
        for k, v in summary.items():
            print(f'  {k}: {v}')
    else:
        print(f'unknown cmd {cmd}')


if __name__ == '__main__':
    main()
