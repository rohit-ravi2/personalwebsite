"""
Phase 1 pre-flight — C-15 sign-flip count reconciliation.

Computes the number of "hard sign flips" (exc↔inh) between default and per-edge
sign modes in the connectome, using the same construction logic as `lif_brain.py`.
Reports counts under multiple definitions to reconcile the prior 518 vs 415
discrepancy.

Output: stdout report + `scripts/brain/artifacts/phase1_signflip_reconciliation.json`.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "brain"))

from lif_brain import DEFAULT_SIGN_OVERRIDES, DOCUMENTED_SIGN_EXCEPTIONS

CONNECTOME_PATH = REPO / "scripts" / "brain" / "artifacts" / "connectome.npz"


def load_connectome():
    d = np.load(CONNECTOME_PATH, allow_pickle=True)
    names = [str(n) for n in d["names"]]
    idx = {n: i for i, n in enumerate(names)}
    sign_base = np.array(d["sign"], dtype=np.int8).copy()
    W_chem_raw = d["W_chem_raw"].astype(np.float32)
    W_chem_per_edge = d["W_chem_per_edge"].astype(np.float32)
    return names, idx, sign_base, W_chem_raw, W_chem_per_edge


def apply_sign_overrides(sign_base, idx, overrides):
    s = sign_base.copy()
    for name, new_sign in overrides.items():
        if name in idx:
            s[idx[name]] = new_sign
    return s


def apply_sign_exceptions(W, idx, exceptions, W_chem_raw):
    W = W.copy()
    for (pre, post), new_sign in exceptions.items():
        if pre not in idx or post not in idx:
            continue
        pi, qi = idx[pre], idx[post]
        if W_chem_raw[pi, qi] == 0:
            continue
        magnitude = abs(W[pi, qi])
        cur = W[pi, qi]
        old_sign = +1 if cur > 0 else (-1 if cur < 0 else 0)
        if old_sign != new_sign:
            W[pi, qi] = new_sign * magnitude
    return W


def count_flips(W_a, W_b, mask=None):
    """Hard sign flips: edges where A is + and B is - (or vice versa).

    Returns dict with multiple counting definitions:
      - both_signed:   edges where both A and B are non-zero AND have opposite sign
      - either_signed: edges where at least one is non-zero AND signs disagree
                       (treats 0 as "no sign", so 0 vs +1 is NOT a flip)
      - any_diff:      edges where signs differ at all (including +1 vs 0, -1 vs 0)
      - zeroed_vs_signed: edges signed in one mode and zeroed in the other
    """
    sign_a = np.sign(W_a).astype(np.int8)
    sign_b = np.sign(W_b).astype(np.int8)
    if mask is not None:
        sign_a = np.where(mask, sign_a, 0)
        sign_b = np.where(mask, sign_b, 0)

    both_nonzero = (sign_a != 0) & (sign_b != 0)
    both_signed_flips = ((sign_a == 1) & (sign_b == -1)) | ((sign_a == -1) & (sign_b == 1))
    n_both_signed = int(both_signed_flips.sum())

    either_nonzero = (sign_a != 0) | (sign_b != 0)
    diff = sign_a != sign_b
    n_either_signed_flips = int((either_nonzero & both_signed_flips).sum())  # = n_both_signed
    n_any_diff = int((either_nonzero & diff).sum())
    n_zeroed_vs_signed = int((either_nonzero & ~both_nonzero).sum())

    return {
        "both_signed_opposite": n_both_signed,
        "any_diff_either_signed": n_any_diff,
        "zeroed_in_one_mode": n_zeroed_vs_signed,
        "n_total_chem_edges": int((W_chem_raw_global != 0).sum()),  # filled in main
    }


def edge_count_summary(W, name=""):
    sign = np.sign(W)
    return {
        "name": name,
        "n_nonzero": int((sign != 0).sum()),
        "n_excitatory": int((sign > 0).sum()),
        "n_inhibitory": int((sign < 0).sum()),
        "sum_excitatory_weight": float(W[W > 0].sum()),
        "sum_inhibitory_weight": float(-W[W < 0].sum()),
    }


def main():
    global W_chem_raw_global

    names, idx, sign_base, W_chem_raw, W_chem_per_edge = load_connectome()
    W_chem_raw_global = W_chem_raw

    n_total = int((W_chem_raw != 0).sum())
    print(f"\n{'='*72}")
    print(f"  C-15 sign-flip reconciliation (n={n_total} non-zero chemical edges)")
    print(f"{'='*72}")

    # Build the 4 mode matrices the simulator can use.
    # M1: default (per-presynaptic-neuron NT signs + DEFAULT_SIGN_OVERRIDES)
    sign_default = apply_sign_overrides(sign_base, idx, DEFAULT_SIGN_OVERRIDES)
    W_default_pre_excep = (sign_default[:, None].astype(np.float32) * W_chem_raw)

    # M2 pure: per-edge from CeNGEN (no DOCUMENTED_SIGN_EXCEPTIONS)
    W_peredge_pure = W_chem_per_edge.astype(np.float32).copy()

    # M1 + DOCUMENTED_SIGN_EXCEPTIONS = current production default mode
    W_default_with_excep = apply_sign_exceptions(
        W_default_pre_excep, idx, DOCUMENTED_SIGN_EXCEPTIONS, W_chem_raw
    )
    # M2 + DOCUMENTED_SIGN_EXCEPTIONS = current production per-edge mode
    W_peredge_with_excep = apply_sign_exceptions(
        W_peredge_pure, idx, DOCUMENTED_SIGN_EXCEPTIONS, W_chem_raw
    )

    # Per-mode summaries
    print("\n--- Per-mode edge summaries ---")
    summaries = {}
    for name, W in [
        ("M1 default (no exceptions)", W_default_pre_excep),
        ("M1 default + exceptions [PRODUCTION DEFAULT]", W_default_with_excep),
        ("M2 per-edge pure (no exceptions)", W_peredge_pure),
        ("M2 per-edge + exceptions [PRODUCTION PER-EDGE]", W_peredge_with_excep),
    ]:
        s = edge_count_summary(W, name)
        summaries[name] = s
        print(
            f"  {name}: nonzero={s['n_nonzero']:4d}  "
            f"exc={s['n_excitatory']:4d}  inh={s['n_inhibitory']:4d}  "
            f"Σexc={s['sum_excitatory_weight']:+.0f}  Σinh={s['sum_inhibitory_weight']:.0f}"
        )

    # Pairwise sign-flip counts under multiple counting definitions
    print("\n--- Sign-flip counts: default vs per-edge ---")
    pairs = [
        ("M1 pure vs M2 pure", W_default_pre_excep, W_peredge_pure),
        ("M1 prod vs M2 prod", W_default_with_excep, W_peredge_with_excep),
        ("M1 pure vs M2 prod", W_default_pre_excep, W_peredge_with_excep),
        ("M1 prod vs M2 pure", W_default_with_excep, W_peredge_pure),
    ]
    pair_results = {}
    for label, W_a, W_b in pairs:
        flips = count_flips(W_a, W_b)
        pair_results[label] = flips
        print(
            f"  {label:30s}: hard-flips={flips['both_signed_opposite']:4d}  "
            f"any-sign-diff={flips['any_diff_either_signed']:4d}  "
            f"zeroed-in-one={flips['zeroed_in_one_mode']:4d}"
        )

    # Reconciliation against the 518 vs 415 prior figures
    print("\n--- Prior figures reconciliation ---")
    print("  Session 1 reported: 518 hard sign flips across 5 counting methods.")
    print("  Session 2 reported: 415 across 4 counting methods.")
    print("  Today's measurement (this script):")
    primary_pure = pair_results["M1 pure vs M2 pure"]["both_signed_opposite"]
    primary_prod = pair_results["M1 prod vs M2 prod"]["both_signed_opposite"]
    print(f"    M1-pure vs M2-pure (no exceptions on either side):           {primary_pure}")
    print(f"    M1-prod vs M2-prod (production default both modes):          {primary_prod}")
    if primary_pure == 518:
        print("    -> Session 1's 518 figure ✓ matches M1-pure-vs-M2-pure (pre-exceptions)")
    if primary_prod == 518:
        print("    -> Session 1's 518 figure ✓ matches M1-prod-vs-M2-prod")
    if primary_pure == 415:
        print("    -> Session 2's 415 figure matches M1-pure-vs-M2-pure")
    if primary_prod == 415:
        print("    -> Session 2's 415 figure matches M1-prod-vs-M2-prod")

    # Persist results
    out_path = REPO / "scripts" / "brain" / "artifacts" / "phase1_signflip_reconciliation.json"
    out = {
        "n_total_nonzero_chem_edges": n_total,
        "per_mode_summary": summaries,
        "pairwise_signflip_counts": pair_results,
        "constants": {
            "n_DEFAULT_SIGN_OVERRIDES": len(DEFAULT_SIGN_OVERRIDES),
            "n_DOCUMENTED_SIGN_EXCEPTIONS": len(DOCUMENTED_SIGN_EXCEPTIONS),
        },
        "reconciliation": {
            "session_1_reported": 518,
            "session_2_reported": 415,
            "M1_pure_vs_M2_pure_hard_flips": primary_pure,
            "M1_prod_vs_M2_prod_hard_flips": primary_prod,
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[results written] {out_path}")

    return out


if __name__ == "__main__":
    main()
