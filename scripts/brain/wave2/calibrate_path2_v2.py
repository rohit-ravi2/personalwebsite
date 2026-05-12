"""
Path 2 v2 per-cell-family C_global calibration — Deliverable 4 (Group C).

Calibrates C_global per cell family (AVA, AIY, RIM) against measured
V_rest targets from `docs/v_rest_targets.md`. Per methodology §3.0:

  For each cell family:
    1. With all other parameters fixed (γ from v2 inventory, TPM from
       Phase 3, E_translation = 1.0, full Layer 1 substrate machinery)
    2. Sweep C_global value
    3. Run cell at rest; measure emergent V_rest
    4. Find C_global value at which V_rest lands within target range
    5. Verify rest stability at that C_global

Sweep strategy: order-of-magnitude scan (1e3, 1e4, 1e5, 1e6, 1e7),
then refine bracketing.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import json
import numpy as np

from layer1_cells import (
    AVAL_SPEC, AVAR_SPEC, AIY_SPEC, RIM_SPEC, CellSpec,
    build_layer1_cell,
)
from channels.derived_channel_parameters import (
    get_derived_gbar, set_c_global_family, C_GLOBAL_PER_FAMILY,
)
from dataclasses import replace


# V_rest targets per `docs/v_rest_targets.md`
V_REST_TARGETS = {
    "AVA": {"central": -32.0, "range": (-50.0, -15.0), "anchor_cell": "AVAL"},
    "AIY": {"central": -75.0, "range": (-95.0, -55.0), "anchor_cell": "AIY"},
    "RIM": {"central": -52.0, "range": (-65.0, -40.0), "anchor_cell": "RIM"},
}

# Cell spec lookup
CELL_SPECS = {"AVAL": AVAL_SPEC, "AVAR": AVAR_SPEC, "AIY": AIY_SPEC, "RIM": RIM_SPEC}


def build_path2_cell(cell_name: str) -> dict:
    """Build a Path 2 cell using current per-family C_global values."""
    nicoletti_spec = CELL_SPECS[cell_name]
    # Map channel module name to Path 2 channel name + override gbar
    ch_p2_map = {
        "egl19": "EGL-19", "irk": "IRK", "nca": "NCA",
        "unc103": "UNC-103", "shl1": "SHL-1", "cca1": "CCA-1",
        "unc2": "UNC-2", "egl2": "EGL-2", "kqt1": "KQT-1",
    }
    path2_channels = {}
    for ch_name in nicoletti_spec.channels:
        ch_p2 = ch_p2_map.get(ch_name, ch_name.upper())
        path2_channels[ch_name] = get_derived_gbar(ch_p2, cell_name)
    spec = replace(nicoletti_spec, channels=path2_channels)
    return build_layer1_cell(spec)


def measure_v_rest(cell_name: str, sim_ms: float = 3000.0) -> dict:
    """Build Path 2 cell, run rest sim, return V_rest + ion deltas."""
    from brian2 import ms
    try:
        bundle = build_path2_cell(cell_name)
        bundle["network"].run(sim_ms * ms)
        mon = bundle["monitor"]
        v_rest = float(mon.v[0][-1] / 1e-3)
        final_K = float(mon.K_in[0][-1])
        final_Na = float(mon.Na_in[0][-1])
        final_Ca = float(mon.Ca_in[0][-1])
        initial_K = float(mon.K_in[0][0])
        initial_Na = float(mon.Na_in[0][0])
        return {
            "v_rest": v_rest, "K_in": final_K, "Na_in": final_Na, "Ca_in": final_Ca,
            "K_pct": 100*(final_K/initial_K - 1),
            "Na_pct": 100*(final_Na/initial_Na - 1),
            "status": "ok",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "v_rest": None}


def calibrate_family(family: str, sweep_values: list[float],
                     verbose: bool = True) -> dict:
    """Calibrate C_global for one cell family via sweep."""
    target = V_REST_TARGETS[family]
    anchor_cell = target["anchor_cell"]
    v_min, v_max = target["range"]
    v_central = target["central"]

    if verbose:
        print(f"\n{'='*72}")
        print(f"Calibrating {family} family (anchor cell {anchor_cell})")
        print(f"  V_rest target: central = {v_central} mV, range = [{v_min}, {v_max}]")
        print(f"{'='*72}")

    sweep_results = []
    for cval in sweep_values:
        set_c_global_family(family, cval)
        result = measure_v_rest(anchor_cell, sim_ms=3000)
        if result["status"] == "error":
            if verbose:
                print(f"  C_global = {cval:.2e}: ERROR — {result['error'][:80]}")
            sweep_results.append({"C_global": cval, "v_rest": None,
                                  "in_range": False, "status": "error"})
            continue
        v = result["v_rest"]
        in_range = v_min <= v <= v_max
        distance_from_central = abs(v - v_central)
        if verbose:
            marker = "  ←  IN RANGE" if in_range else ""
            print(f"  C_global = {cval:.2e}: V_rest = {v:+.2f} mV (Δ from central = {distance_from_central:.1f}){marker}")
        sweep_results.append({
            "C_global": cval, "v_rest": v, "in_range": in_range,
            "K_pct": result["K_pct"], "Na_pct": result["Na_pct"],
            "Ca_in_nM": result["Ca_in"] * 1e6,
            "distance_from_central": distance_from_central,
            "status": "ok",
        })

    in_range_results = [r for r in sweep_results if r.get("in_range")]
    if in_range_results:
        # Pick the C_global closest to central
        best = min(in_range_results, key=lambda r: r["distance_from_central"])
        if verbose:
            print(f"\n  ✓ Calibration succeeded: C_global_{family} = {best['C_global']:.4e}")
            print(f"    V_rest = {best['v_rest']:+.2f} mV (target central {v_central})")
        return {"family": family, "calibrated": best["C_global"],
                "v_rest_achieved": best["v_rest"], "in_range": True,
                "sweep_results": sweep_results}
    else:
        if verbose:
            print(f"\n  ✗ Calibration FAILED: no C_global in sweep produced V_rest in [{v_min}, {v_max}]")
            # Pick closest-to-range
            valid_results = [r for r in sweep_results if r.get("status") == "ok"]
            if valid_results:
                closest = min(valid_results, key=lambda r: r["distance_from_central"])
                print(f"    closest: C_global = {closest['C_global']:.2e}, V_rest = {closest['v_rest']:+.2f}")
        return {"family": family, "calibrated": None, "v_rest_achieved": None,
                "in_range": False, "sweep_results": sweep_results}


def main():
    print(f"\n{'#'*72}")
    print(f"# §7.3.5 v2 Group C — Per-cell-family C_global calibration")
    print(f"# Calibrating against measured V_rest per §8.11 measurement-vs-fit audit")
    print(f"{'#'*72}")

    # Order-of-magnitude sweep first for each family
    om_sweep = [1e3, 1e4, 1e5, 1e6, 1e7]

    all_results = {}
    for family in ("AVA", "AIY", "RIM"):
        result = calibrate_family(family, om_sweep)
        all_results[family] = result
        # Reset to v1 default before next family
        set_c_global_family(family, 1.7297e4)

    # Save results
    out_path = THIS_DIR / "artifacts" / "path2_v2_calibration_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*72}")
    print(f"Per-family C_global calibration summary")
    print(f"{'='*72}")
    print(f"{'family':<8} {'C_global_calibrated':<22} {'V_rest':<10} {'in_range':<10}")
    print('-' * 60)
    for family, r in all_results.items():
        if r["calibrated"] is not None:
            print(f"{family:<8} {r['calibrated']:<22.4e} {r['v_rest_achieved']:<+10.2f} ✓")
        else:
            print(f"{family:<8} {'(none in range)':<22} {'—':<10} ✗")

    return all_results


if __name__ == "__main__":
    main()
