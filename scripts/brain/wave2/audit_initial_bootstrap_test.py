"""
Initial-condition bootstrap test for Layer 2 plateau cells.

Question: does the substrate HAVE a stable plateau attractor that the cells
just can't reach from the default v_init (-60 or Nicoletti hyperpolarized
values), or does the substrate have NO plateau and cells initialized at
plateau slide back to hyperpolarized rest?

Method: override v_init per cell class/role from the PUBLISHED_REST table:
  - plateau-command       (AVA, AVB, AVD, AVE, PVC, RMD*)   -> -25 mV
  - plateau-modulatory    (RIM, HSN, NSM, ADF, RIC, ADE, CEP, PDE) -> -45 mV
  - plateau-pharyngeal    (RIP) or pharyngeal-plateau (MC)  -> -55 mV
  - motor-active          (DA, VA, DB, VB, VC)              -> -50 mV
  - all other cells: keep default v_init

Then run Phase A (5s rest, no I_inj) and report what each plateau cell
settles to. Skip Phase B (stim) entirely — the goal is to see whether the
plateau is a stable attractor of the autonomous network dynamics.

Output:
  artifacts/init_bootstrap_test.log     (stdout)
  artifacts/init_bootstrap_test.json    (per-cell snapshot)
"""
from __future__ import annotations

import sys
import json
import shutil
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from layer2.assemble import assemble_layer2_network
from audit_cell_class_fitness import PUBLISHED_REST, cell_class

REST_MS = 5000.0

# Role -> override v_init (mV). Only these roles get overridden; everything
# else keeps the default v_init from spec_l.v_init_mV.
ROLE_VINIT_OVERRIDE = {
    "plateau-command":    -25.0,
    "plateau-modulatory": -45.0,
    "plateau-pharyngeal": -55.0,  # RIP
    "pharyngeal-plateau": -55.0,  # MC
    "motor-active":       -50.0,
}


def loose_plausible(V, K, Na, Cl, Ca_uM):
    return (-110 < V < 50 and 80 < K < 200 and 0.5 < Na < 50
            and 1 < Cl < 30 and 0 < Ca_uM < 100.0)


def strict_plausible(V, K, Na, Cl, Ca_uM):
    return (-110 < V < 50 and 80 < K < 200 and 0.5 < Na < 50
            and 1 < Cl < 30 and 0 < Ca_uM < 1.0)


def snapshot(bundle, t_index: int = -1) -> dict:
    """Pull V + ions at a given time index from the monitor."""
    mon = bundle["monitor"]
    names = bundle["meta"]["cell_names"]
    snap = {}
    for i, name in enumerate(names):
        snap[name] = {
            "V_mV":  float(mon.v[i][t_index] / 1e-3),
            "K_mM":  float(mon.K_in[i][t_index]),
            "Na_mM": float(mon.Na_in[i][t_index]),
            "Cl_mM": float(mon.Cl_in[i][t_index]),
            "Ca_uM": float(mon.Ca_in[i][t_index]) * 1e3,
            "I_syn_pA": float(mon.I_syn[i][t_index] / 1e-12),
            "I_gap_pA": float(mon.I_gap[i][t_index] / 1e-12),
        }
    return snap


def summarize(snap: dict, label: str):
    Vs = np.array([s["V_mV"] for s in snap.values()])
    Cas = np.array([s["Ca_uM"] for s in snap.values()])
    Ks = np.array([s["K_mM"] for s in snap.values()])
    Nas = np.array([s["Na_mM"] for s in snap.values()])
    Cls = np.array([s["Cl_mM"] for s in snap.values()])

    loose = sum(loose_plausible(s["V_mV"], s["K_mM"], s["Na_mM"], s["Cl_mM"], s["Ca_uM"])
                 for s in snap.values())
    strict = sum(strict_plausible(s["V_mV"], s["K_mM"], s["Na_mM"], s["Cl_mM"], s["Ca_uM"])
                 for s in snap.values())
    nan = sum(1 for s in snap.values() if any(np.isnan(v) for v in s.values()))
    print(f"\n=== {label} ===")
    print(f"  Plausibility: loose {loose}/{len(snap)}, strict {strict}/{len(snap)}, NaN {nan}")
    print(f"  V_rest mV:    min {Vs.min():+.1f}, max {Vs.max():+.1f}, "
          f"med {np.median(Vs):+.1f}, mean+/-std {Vs.mean():+.1f}+/-{Vs.std():.1f}")
    print(f"  K_in mM:      min {Ks.min():.1f}, max {Ks.max():.1f}, med {np.median(Ks):.1f}")
    print(f"  Na_in mM:     min {Nas.min():.1f}, max {Nas.max():.1f}, med {np.median(Nas):.1f}")
    print(f"  Cl_in mM:     min {Cls.min():.1f}, max {Cls.max():.1f}, med {np.median(Cls):.1f}")
    print(f"  Ca_in uM:     min {Cas.min():.3f}, max {Cas.max():.3f}, med {np.median(Cas):.3f}")
    print(f"  Cells V > -30 mV (potential 'active'): {int((Vs > -30).sum())}")
    return {"loose": loose, "strict": strict, "nan": nan,
            "V_min": float(Vs.min()), "V_max": float(Vs.max()),
            "V_median": float(np.median(Vs)),
            "Ca_max_uM": float(Cas.max())}


def apply_init_overrides(bundle) -> dict:
    """Override v_init in per_cell_params for cells whose role matches
    a target plateau/modulatory/motor role, then re-apply group.v and
    re-initialize channel state variables (HCN) consistent with new v_init.

    Returns dict mapping cell_name -> (cls, role, v_init_default, v_init_new).
    Only cells whose role appears in ROLE_VINIT_OVERRIDE are recorded.
    """
    from brian2 import mV
    per_cell = bundle["meta"]["per_cell_params"]
    G = bundle["group"]

    overrides = {}
    new_v_init = np.zeros(len(per_cell))

    for i, p in enumerate(per_cell):
        name = p["cell_name"]
        default_v_init = p["v_init_mV"]
        cls = cell_class(name)
        new_v = default_v_init
        if cls is not None:
            pub_V, role, source = PUBLISHED_REST[cls]
            if role in ROLE_VINIT_OVERRIDE:
                new_v = ROLE_VINIT_OVERRIDE[role]
                overrides[name] = {
                    "class": cls, "role": role,
                    "v_init_default": float(default_v_init),
                    "v_init_new": float(new_v),
                    "V_published": float(pub_V),
                }
        new_v_init[i] = new_v
        # Also patch the stored per_cell record so audit downstream can see it
        p["v_init_mV_override"] = float(new_v)

    # Re-apply on the group
    G.v = new_v_init * mV
    # Re-initialize HCN steady-state at the new v_init (matches network_builder
    # logic). Other channel state vars start at activation=0/inactivation=1 and
    # Brian2 will integrate them to steady-state regardless.
    G.m_hcn = 1.0 / (1.0 + np.exp((new_v_init - (-75.0)) / 8.0))

    return overrides


def categorize(v_init, v_settle):
    delta = v_settle - v_init
    if abs(delta) < 10:
        return "A_maintained"
    if delta < -20:
        return "B_slid_back"
    if delta > 20:
        return "C_over_dep"
    # in between: partial slide / partial drift
    if delta < -10:
        return "B_partial_slide"
    return "C_partial_drift"


def main():
    from brian2 import ms

    print("=" * 80)
    print("Initial-condition bootstrap test — plateau cells seeded at published rest")
    print("=" * 80)

    t0 = time.time()
    bundle = assemble_layer2_network(record_indices=None)
    print(f"\n[init_test] Assembly took {time.time()-t0:.1f}s")
    print(f"[init_test] Network: {len(bundle['meta']['cell_names'])} cells, "
          f"{bundle['meta']['n_chem_excitatory']} exc + "
          f"{bundle['meta']['n_chem_inhibitory']} inh chem synapses, "
          f"{bundle['meta']['n_gap']} gap junctions")

    # Apply v_init overrides
    overrides = apply_init_overrides(bundle)
    print(f"\n[init_test] v_init overrides applied to {len(overrides)} cells:")
    by_role = {}
    for name, info in overrides.items():
        by_role.setdefault(info["role"], []).append(name)
    for role, cells in sorted(by_role.items()):
        print(f"  {role:<22s} n={len(cells):>3d}  v_init={ROLE_VINIT_OVERRIDE[role]:+.0f} mV")
        # Show first 5 cell names
        print(f"    e.g. {cells[:5]}")

    # Phase A: rest sim only (no stim)
    print(f"\n[init_test] Phase A: {REST_MS} ms rest (no external input)...")
    t1 = time.time()
    bundle["network"].run(REST_MS * ms)
    print(f"[init_test] Phase A took {time.time()-t1:.1f}s")

    rest_snap = snapshot(bundle, t_index=-1)
    rest_summary = summarize(rest_snap, f"After rest (t={REST_MS} ms)")

    # Per-overridden-cell detail
    print(f"\n=== Per-cell bootstrap report (overridden cells, t=5s) ===")
    print(f"{'name':<10s} {'class':<7s} {'role':<22s} {'v_init':>7s} "
          f"{'V_5s':>7s} {'d_set':>7s} {'V_pub':>7s} {'d_pub':>7s} "
          f"{'Ca_uM':>8s} {'category':<18s}")
    print("-" * 110)

    detail = []
    cat_counts = {}
    for name in sorted(overrides.keys()):
        info = overrides[name]
        s = rest_snap[name]
        v_init = info["v_init_new"]
        v_settle = s["V_mV"]
        v_pub = info["V_published"]
        d_set = v_settle - v_init
        d_pub = v_settle - v_pub
        cat = categorize(v_init, v_settle)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"{name:<10s} {info['class']:<7s} {info['role']:<22s} "
              f"{v_init:>+7.1f} {v_settle:>+7.1f} {d_set:>+7.1f} "
              f"{v_pub:>+7.1f} {d_pub:>+7.1f} {s['Ca_uM']:>8.3f}  {cat}")
        detail.append({
            "name": name, "class": info["class"], "role": info["role"],
            "v_init": v_init, "V_5s": v_settle,
            "delta_settle": d_set, "V_pub": v_pub, "delta_pub": d_pub,
            "Ca_uM": s["Ca_uM"], "K_mM": s["K_mM"], "Na_mM": s["Na_mM"],
            "Cl_mM": s["Cl_mM"],
            "I_syn_pA": s["I_syn_pA"], "I_gap_pA": s["I_gap_pA"],
            "category": cat,
        })

    print(f"\n=== Category counts ({len(overrides)} overridden cells) ===")
    for cat in sorted(cat_counts):
        print(f"  {cat:<22s} n={cat_counts[cat]:>3d}")

    # Per-role aggregate
    print(f"\n=== Per-role aggregate ===")
    by_role_detail = {}
    for d in detail:
        by_role_detail.setdefault(d["role"], []).append(d)
    print(f"{'role':<22s} {'n':>4s} {'v_init':>7s} {'V_5s_med':>9s} "
          f"{'d_set_med':>10s} {'A%':>6s} {'B%':>6s} {'C%':>6s}")
    print("-" * 80)
    for role, cells in sorted(by_role_detail.items()):
        v_settles = [c["V_5s"] for c in cells]
        d_sets = [c["delta_settle"] for c in cells]
        n_A = sum(1 for c in cells if c["category"].startswith("A"))
        n_B = sum(1 for c in cells if c["category"].startswith("B"))
        n_C = sum(1 for c in cells if c["category"].startswith("C"))
        n = len(cells)
        print(f"{role:<22s} {n:>4d} {ROLE_VINIT_OVERRIDE[role]:>+7.1f} "
              f"{np.median(v_settles):>+9.1f} {np.median(d_sets):>+10.1f} "
              f"{100*n_A/n:>5.0f}% {100*n_B/n:>5.0f}% {100*n_C/n:>5.0f}%")

    # Save artifacts
    out = THIS_DIR / "artifacts" / "init_bootstrap_test.json"
    with out.open("w") as f:
        json.dump({
            "rest_summary": rest_summary,
            "rest_snap": rest_snap,
            "overrides": overrides,
            "detail": detail,
            "category_counts": cat_counts,
            "role_vinit_override": ROLE_VINIT_OVERRIDE,
            "meta": {k: v for k, v in bundle["meta"].items()
                     if k != "per_cell_params"},
        }, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    # Also write a layer2_validation.json-compatible artifact so the existing
    # audit_cell_class_fitness.py reads our rest_snap. We overwrite into the
    # init_bootstrap_audit name to avoid clobbering a concurrent run.
    audit_compat = THIS_DIR / "artifacts" / "init_bootstrap_test_audit.json"
    with audit_compat.open("w") as f:
        json.dump({"rest_snap": rest_snap}, f, indent=2, default=str)
    print(f"Saved audit-compatible: {audit_compat}")

    print(f"\nTotal wall clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
