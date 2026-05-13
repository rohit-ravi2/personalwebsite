"""
Full 128-CeNGEN-class scalable substrate sweep.

Builds every supported class via path2_scale.build_scalable_spec → layer1
CellSpec → build_layer1_cell, runs short rest sim, reports distribution of
V_rest + ion concentrations + failures.

Per the execute-first directive: don't methodologize, just run and see what
holes surface.
"""
from __future__ import annotations

import sys
import json
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from path2_scale.scalable_builder import build_scalable_spec, to_layer1_cellspec
from path2_scale.cengen_tpm_data import CENGEN_NEURONS
from layer1_cells import build_layer1_cell


SIM_MS = 1500.0  # shorter than 3s for 128-cell speed


def run_one(cengen_class: str) -> dict:
    from brian2 import ms

    try:
        spec_s = build_scalable_spec(cengen_class)
    except Exception as e:
        return {"cell": cengen_class, "status": "spec_failed", "error": f"{type(e).__name__}: {e}"}

    n_ch = len(spec_s.channels)
    if n_ch == 0:
        return {"cell": cengen_class, "status": "no_supported_channels",
                "n_channels": 0, "nicoletti": spec_s.nicoletti_calibrated}

    try:
        spec_l = to_layer1_cellspec(spec_s)
        bundle = build_layer1_cell(spec_l)
        bundle["network"].run(SIM_MS * ms)
    except Exception as e:
        return {"cell": cengen_class, "status": "build_or_sim_failed",
                "error": f"{type(e).__name__}: {e}"[:200],
                "channels": list(spec_s.channels.keys()),
                "n_channels": n_ch, "nicoletti": spec_s.nicoletti_calibrated}

    mon = bundle["monitor"]
    V = float(mon.v[0][-1] / 1e-3)
    V_traj = np.asarray(mon.v[0] / 1e-3)
    ions = {ion: float(getattr(mon, ion)[0][-1]) for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}

    nan = bool(np.isnan(V_traj).any() or any(np.isnan(np.asarray(getattr(mon, ion)[0])).any()
                                              for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")))
    plausible = (
        -120 < V < 60
        and 80 < ions["K_in"] < 200
        and 0.5 < ions["Na_in"] < 50
        and 1 < ions["Cl_in"] < 30
        and 0 < ions["Ca_in"] < 0.1
        and not nan
    )

    return {
        "cell": cengen_class, "status": "ok",
        "nicoletti": spec_s.nicoletti_calibrated,
        "n_channels": n_ch,
        "channels": list(spec_s.channels.keys()),
        "V_rest_mV": V, "V_min_mV": float(np.min(V_traj)), "V_max_mV": float(np.max(V_traj)),
        "K_in_mM": ions["K_in"], "Na_in_mM": ions["Na_in"],
        "Cl_in_mM": ions["Cl_in"], "Ca_in_uM": ions["Ca_in"] * 1e3,
        "plausible": plausible, "nan": nan,
    }


def main():
    print("=" * 78)
    print(f"Path 2 scalable — full 128-class sweep ({SIM_MS} ms rest)")
    print("=" * 78)
    print()

    results = []
    for i, n in enumerate(CENGEN_NEURONS):
        r = run_one(n)
        results.append(r)
        if (i + 1) % 10 == 0 or r.get("status") != "ok":
            tag = "ok " if r.get("status") == "ok" else r.get("status", "?")[:12]
            v_str = f"V={r.get('V_rest_mV', float('nan')):+6.1f}" if "V_rest_mV" in r else "V=  N/A"
            print(f"  [{i+1:3d}/{len(CENGEN_NEURONS)}] {n:<8} {tag:<14} "
                  f"ch={r.get('n_channels', 0):>2} {v_str}")

    print("\n" + "=" * 78)
    print("Distribution")
    print("=" * 78)

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] not in ("ok", "no_supported_channels")]
    empty = [r for r in results if r["status"] == "no_supported_channels"]

    print(f"  Built + simulated: {len(ok):>3}/{len(results)}")
    print(f"  No expressed channels (empty): {len(empty):>3}/{len(results)}")
    print(f"  Errors during build/sim:       {len(failed):>3}/{len(results)}")
    print(f"  Plausible state (V/ion ranges): {sum(1 for r in ok if r['plausible']):>3}/{len(ok)}")

    if empty:
        print(f"\n  Empty (no T2 supported channels): {[r['cell'] for r in empty]}")

    if failed:
        print(f"\n  Failures:")
        err_kinds = {}
        for r in failed:
            err = r.get("error", "?")[:60]
            err_kinds.setdefault(err, []).append(r["cell"])
        for err, cells in err_kinds.items():
            print(f"    {err}")
            print(f"      → {len(cells)} cells: {cells[:8]}{'...' if len(cells)>8 else ''}")

    if ok:
        V_rests = np.array([r["V_rest_mV"] for r in ok])
        Ca = np.array([r["Ca_in_uM"] for r in ok])
        print(f"\n  V_rest distribution:")
        print(f"    min/max:    {V_rests.min():+.1f} / {V_rests.max():+.1f} mV")
        print(f"    mean ± std: {V_rests.mean():+.1f} ± {V_rests.std():.1f} mV")
        print(f"    median:     {np.median(V_rests):+.1f} mV")
        print(f"    quartiles:  {np.percentile(V_rests, 25):+.1f} / "
              f"{np.percentile(V_rests, 75):+.1f} mV")
        bins = [(-120, -80), (-80, -60), (-60, -40), (-40, -20), (-20, 0), (0, 60)]
        for lo, hi in bins:
            n = int(np.sum((V_rests >= lo) & (V_rests < hi)))
            bar = "#" * int(40 * n / len(ok))
            print(f"    [{lo:+4d}, {hi:+4d}): {n:>3} {bar}")

        print(f"\n  Channel count distribution:")
        chs = np.array([r["n_channels"] for r in ok])
        for c in range(int(chs.min()), int(chs.max()) + 1):
            n = int(np.sum(chs == c))
            if n > 0:
                bar = "#" * int(40 * n / len(ok))
                print(f"    {c} channels: {n:>3} {bar}")

        print(f"\n  Ca_in distribution:")
        print(f"    median: {np.median(Ca):.3f} μM, max: {Ca.max():.3f} μM")
        high_ca = sum(1 for x in Ca if x > 1.0)
        print(f"    cells with Ca > 1 μM: {high_ca}/{len(ok)}")

    # save
    out_path = THIS_DIR / "artifacts" / "scalable_128_sweep.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Detailed results: {out_path}")


if __name__ == "__main__":
    main()
