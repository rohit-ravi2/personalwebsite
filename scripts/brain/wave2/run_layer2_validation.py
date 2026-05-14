"""
Layer 2 minimum-viable validation: 5s rest + 5s stim.

Validates the 300-cell connectome network on the Path 2 substrate.

Phase A (0-5s, rest): no external input. Verify substrate stability under
  network input (graded synaptic + gap-junction coupling).

Phase B (5-10s, stim): inject 5 pA into ASEL/ASER/AWCL/AWCR (chemosensory).
  Verify downstream propagation and that the substrate doesn't break.

Reports:
  - Initial vs final V_rest distribution
  - Plausibility count under like-for-like criteria (Ca < 100 μM loose,
    < 1 μM strict)
  - Whether RIB/RIM/RIP/RMD_DV normalize under tonic input
  - Network activity: cells crossing -30 mV during stim phase
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from layer2.assemble import assemble_layer2_network

REST_MS = 5000.0
STIM_MS = 5000.0
STIM_CELLS = ["ASEL", "ASER", "AWCL", "AWCR"]  # chemosensory pair, classic activators
STIM_AMP_pA = 5.0


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
          f"med {np.median(Vs):+.1f}, mean±std {Vs.mean():+.1f}±{Vs.std():.1f}")
    print(f"  K_in mM:      min {Ks.min():.1f}, max {Ks.max():.1f}, med {np.median(Ks):.1f}")
    print(f"  Na_in mM:     min {Nas.min():.1f}, max {Nas.max():.1f}, med {np.median(Nas):.1f}")
    print(f"  Cl_in mM:     min {Cls.min():.1f}, max {Cls.max():.1f}, med {np.median(Cls):.1f}")
    print(f"  Ca_in μM:     min {Cas.min():.3f}, max {Cas.max():.3f}, med {np.median(Cas):.3f}")
    print(f"  Cells V > -30 mV (potential 'active'): {int((Vs > -30).sum())}")
    return {"loose": loose, "strict": strict, "nan": nan,
            "V_min": float(Vs.min()), "V_max": float(Vs.max()),
            "V_median": float(np.median(Vs)),
            "Ca_max_uM": float(Cas.max())}


def main():
    from brian2 import ms, pA

    print("=" * 80)
    print("Layer 2 minimum-viable validation — 5s rest + 5s stim")
    print("=" * 80)

    t0 = time.time()
    bundle = assemble_layer2_network(record_indices=None)  # record all
    print(f"\n[layer2] Assembly took {time.time()-t0:.1f}s")
    print(f"[layer2] Network: {len(bundle['meta']['cell_names'])} cells, "
          f"{bundle['meta']['n_chem_excitatory']} exc + "
          f"{bundle['meta']['n_chem_inhibitory']} inh chem synapses, "
          f"{bundle['meta']['n_gap']} gap junctions")
    if bundle['meta']['unmapped_cells']:
        print(f"[layer2] Unmapped cells (using fallback): "
              f"{bundle['meta']['unmapped_cells'][:5]}... ({len(bundle['meta']['unmapped_cells'])} total)")

    # Phase A: rest sim
    print(f"\n[layer2] Phase A: {REST_MS} ms rest (no external input)...")
    t1 = time.time()
    bundle["network"].run(REST_MS * ms)
    print(f"[layer2] Phase A took {time.time()-t1:.1f}s")
    rest_snap = snapshot(bundle, t_index=-1)
    rest_summary = summarize(rest_snap, f"After rest (t={REST_MS} ms)")

    # Phase B: stim sim
    cell_idx = bundle["meta"]["cell_name_to_idx"]
    stim_indices = [cell_idx[c] for c in STIM_CELLS if c in cell_idx]
    if not stim_indices:
        # Try class-level (no L/R)
        stim_indices = []
        for c in ["ASEL", "ASER", "AWCL", "AWCR", "ASE", "AWC"]:
            if c in cell_idx:
                stim_indices.append(cell_idx[c])
    print(f"\n[layer2] Phase B: {STIM_MS} ms with {STIM_AMP_pA} pA into "
          f"{[bundle['meta']['cell_names'][i] for i in stim_indices]}...")

    # Set I_inj for stim cells. Brian2 I_inj is in amps; in our convention
    # I_inj is SUBTRACTED from I_total, so positive I_inj depolarizes.
    G = bundle["group"]
    I_inj_array = np.zeros(len(bundle["meta"]["cell_names"])) * pA
    for idx in stim_indices:
        I_inj_array[idx] = STIM_AMP_pA * pA
    G.I_inj = I_inj_array

    t2 = time.time()
    bundle["network"].run(STIM_MS * ms)
    print(f"[layer2] Phase B took {time.time()-t2:.1f}s")
    stim_snap = snapshot(bundle, t_index=-1)
    stim_summary = summarize(stim_snap, f"After stim (t={REST_MS+STIM_MS} ms)")

    # Did the 4 strict failures normalize?
    print(f"\n=== Strict-failure cells under network input ===")
    for cell in ["RIB", "RIM", "RIP", "RMD_DV"]:
        # try variants
        for variant in [cell, cell + "L", cell + "R"]:
            if variant in rest_snap:
                rs = rest_snap[variant]
                ss = stim_snap[variant]
                print(f"  {variant:8s} rest:V={rs['V_mV']:+.1f} Ca={rs['Ca_uM']:.3f} "
                      f"I_syn={rs['I_syn_pA']:+.2f} I_gap={rs['I_gap_pA']:+.2f} | "
                      f"stim:V={ss['V_mV']:+.1f} Ca={ss['Ca_uM']:.3f}")

    # Save artifacts
    out = THIS_DIR / "artifacts" / "layer2_validation.json"
    with out.open("w") as f:
        json.dump({
            "rest_summary": rest_summary,
            "stim_summary": stim_summary,
            "rest_snap": rest_snap,
            "stim_snap": stim_snap,
            "meta": {k: v for k, v in bundle["meta"].items()
                     if k != "per_cell_params"},
        }, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print(f"\nTotal wall clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
