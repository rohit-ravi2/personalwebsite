"""
End-to-end smoke test for path2_scale: take non-Nicoletti cells, build via
scalable builder, convert to layer1 CellSpec, run 3s rest simulation.

Surfaces implementation holes early per "execute first, methodology after."
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from path2_scale.scalable_builder import build_scalable_spec, to_layer1_cellspec
from layer1_cells import build_layer1_cell


TEST_CELLS = ["ASEL", "AWA", "AVA", "AIY", "RIM"]


def run_rest_test(cengen_class: str, sim_ms: float = 3000.0) -> dict:
    """Build cell from CeNGEN + defaults, run rest simulation, return summary."""
    from brian2 import ms

    spec_scalable = build_scalable_spec(cengen_class)
    spec_layer1 = to_layer1_cellspec(spec_scalable)

    if not spec_layer1.channels:
        return {
            "cell": cengen_class,
            "status": "skip_no_supported_channels",
            "n_channels": 0,
        }

    try:
        bundle = build_layer1_cell(spec_layer1)
    except Exception as e:
        return {
            "cell": cengen_class,
            "status": "build_failed",
            "error": f"{type(e).__name__}: {e}",
            "n_channels": len(spec_layer1.channels),
            "channels": list(spec_layer1.channels.keys()),
        }

    try:
        bundle["network"].run(sim_ms * ms)
    except Exception as e:
        return {
            "cell": cengen_class,
            "status": "sim_failed",
            "error": f"{type(e).__name__}: {e}",
            "n_channels": len(spec_layer1.channels),
            "channels": list(spec_layer1.channels.keys()),
        }

    mon = bundle["monitor"]
    V_final = float(mon.v[0][-1] / 1e-3)
    V_traj = np.asarray(mon.v[0] / 1e-3)
    ions = {ion: float(getattr(mon, ion)[0][-1]) for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}

    nan_present = bool(np.isnan(V_traj).any())
    plausible_V = -120.0 < V_final < 60.0
    plausible_K = 80 < ions["K_in"] < 200
    plausible_Na = 0.5 < ions["Na_in"] < 50
    plausible_Cl = 1 < ions["Cl_in"] < 30
    plausible_Ca = 0 < ions["Ca_in"] < 0.1  # < 100 μM transient cap

    return {
        "cell": cengen_class,
        "status": "ok",
        "nicoletti": spec_scalable.nicoletti_calibrated,
        "n_channels": len(spec_layer1.channels),
        "channels": list(spec_layer1.channels.keys()),
        "surf_cm2": spec_scalable.surf_cm2,
        "V_final_mV": V_final,
        "V_min_mV": float(np.min(V_traj)),
        "V_max_mV": float(np.max(V_traj)),
        "K_in_mM": ions["K_in"],
        "Na_in_mM": ions["Na_in"],
        "Cl_in_mM": ions["Cl_in"],
        "Ca_in_uM": ions["Ca_in"] * 1e3,
        "nan_present": nan_present,
        "all_plausible": (plausible_V and plausible_K and plausible_Na
                          and plausible_Cl and plausible_Ca and not nan_present),
    }


def main():
    print("=" * 78)
    print("Path 2 scalable builder — end-to-end smoke test")
    print("=" * 78)
    print()
    print(f"Running 3s rest sim for: {TEST_CELLS}\n")

    results = []
    for cell in TEST_CELLS:
        print(f"\n--- {cell} ---")
        r = run_rest_test(cell)
        results.append(r)
        if r["status"] != "ok":
            print(f"  STATUS: {r['status']}")
            if "error" in r:
                print(f"  ERROR: {r['error']}")
            if "channels" in r:
                print(f"  channels: {r['channels']}")
            continue
        flag = "OK " if r["all_plausible"] else "WARN"
        nic = "[Nic]" if r["nicoletti"] else "[def]"
        print(f"  [{flag}] {nic} channels={r['n_channels']:>2} ({','.join(r['channels'])})")
        print(f"        V_final={r['V_final_mV']:+.2f} mV  "
              f"(min {r['V_min_mV']:+.1f}, max {r['V_max_mV']:+.1f})")
        print(f"        K_in={r['K_in_mM']:.1f}  Na_in={r['Na_in_mM']:.2f}  "
              f"Cl_in={r['Cl_in_mM']:.2f}  Ca_in={r['Ca_in_uM']:.3f} μM")
        if r["nan_present"]:
            print("        !!! NaN in voltage trace")

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    ok = sum(1 for r in results if r["status"] == "ok")
    plausible = sum(1 for r in results if r["status"] == "ok" and r["all_plausible"])
    failed = sum(1 for r in results if r["status"] != "ok")
    print(f"  Built + simulated: {ok}/{len(results)}")
    print(f"  Plausible state:   {plausible}/{len(results)}")
    print(f"  Failed:            {failed}/{len(results)}")
    for r in results:
        if r["status"] != "ok":
            print(f"    {r['cell']:6s}: {r['status']} — {r.get('error','')[:60]}")
        elif not r["all_plausible"]:
            print(f"    {r['cell']:6s}: implausible state (V={r['V_final_mV']:+.1f}, "
                  f"Ca={r['Ca_in_uM']:.3f} μM)")


if __name__ == "__main__":
    main()
