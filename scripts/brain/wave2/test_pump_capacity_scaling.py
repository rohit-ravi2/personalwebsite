"""
Test channel-load-proportional pump scaling.

Predictions:
  - AVE: pump_scale ≈ 2-3× → V→-78, K→123, fixed
  - Other previously-OK cells: pump_scale ≈ 1-2× → minor changes, still OK
  - RIB / RIM: marginal changes (their issues aren't pump-capacity)
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from path2_scale.scalable_builder import build_scalable_spec, to_layer1_cellspec
from path2_scale.pump_capacity_scaling import channel_load_scale, AVAL_CHANNEL_LOAD_Scm2
from layer1_cells import build_layer1_cell

SIM_MS = 1500.0
TARGETS = ["AVE", "HSN", "RIB", "VD_DD", "RIM", "AVA", "AIY", "ASEL", "AWA",
           "I3", "M3", "PQR", "AVL", "MC"]


def run_one(cls):
    from brian2 import ms
    spec_s = build_scalable_spec(cls)
    spec_l = to_layer1_cellspec(spec_s)
    bundle = build_layer1_cell(spec_l)
    bundle["network"].run(SIM_MS * ms)
    mon = bundle["monitor"]
    return {
        "V": float(mon.v[0][-1] / 1e-3),
        "Ca_uM": float(mon.Ca_in[0][-1]) * 1e3,
        "Na": float(mon.Na_in[0][-1]),
        "K": float(mon.K_in[0][-1]),
        "Cl": float(mon.Cl_in[0][-1]),
        "pump_scale": spec_l.pump_NaK_scale,
        "n_channels": len(spec_s.channels),
        "total_gbar": sum(spec_s.channels.values()),
    }


def plausible(r):
    return (-110 < r["V"] < 50 and r["Ca_uM"] < 1.0
            and 80 < r["K"] < 200 and 0.5 < r["Na"] < 50 and 1 < r["Cl"] < 30)


def main():
    print("=" * 120)
    print(f"Pump-capacity scaling test (channel-load-proportional)")
    print(f"AVAL channel load anchor: {AVAL_CHANNEL_LOAD_Scm2:.3e} S/cm²")
    print("=" * 120)
    print(f"\n{'cell':<8} {'nch':>4} {'totgbar':>10} {'p_scale':>8} | "
          f"{'V mV':>8} {'Ca μM':>10} {'Na mM':>7} {'K mM':>7} {'Cl mM':>6} | OK")
    print("-" * 110)
    n_plaus = 0
    for cls in TARGETS:
        try:
            r = run_one(cls)
            ok_b = plausible(r)
            n_plaus += int(ok_b)
            ok = "OK " if ok_b else "WARN"
            print(f"{cls:<8} {r['n_channels']:>4} {r['total_gbar']:>10.3e} "
                  f"{r['pump_scale']:>8.2f} | {r['V']:+8.2f} {r['Ca_uM']:>10.3f} "
                  f"{r['Na']:>7.2f} {r['K']:>7.2f} {r['Cl']:>6.2f} | {ok}")
        except Exception as e:
            print(f"{cls:<8} FAILED: {type(e).__name__}: {e}")
    print(f"\nPlausible: {n_plaus}/{len(TARGETS)}")


if __name__ == "__main__":
    main()
