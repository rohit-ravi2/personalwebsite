"""
Re-test the 4 implausible-Ca cells (HSN, RIB, RIM, VD_DD) with per-cell
pump scaling now extended to all 128 CeNGEN classes. Compare to AVA
(should be roughly unchanged) and AIY (also roughly unchanged).
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from path2_scale.scalable_builder import build_scalable_spec, to_layer1_cellspec
from layer1_cells import build_layer1_cell

TARGETS = ["HSN", "RIB", "VD_DD", "RIM", "AVA", "AIY", "ASEL"]


def run_one(cls, sim_ms=2000.0):
    from brian2 import ms
    spec = build_scalable_spec(cls)
    cs = to_layer1_cellspec(spec)
    bundle = build_layer1_cell(cs)
    bundle["network"].run(sim_ms * ms)
    mon = bundle["monitor"]
    V = float(mon.v[0][-1] / 1e-3)
    Ca = float(mon.Ca_in[0][-1]) * 1e3  # μM
    Na = float(mon.Na_in[0][-1])
    K = float(mon.K_in[0][-1])
    Cl = float(mon.Cl_in[0][-1])
    pump_NaK = float(mon.pump_NaK_I_mAcm2[0][-1])
    Ca_clear = float(mon.ca_clear_I_mAcm2[0][-1])
    return {
        "cell": cls, "V": V, "Ca_uM": Ca, "Na": Na, "K": K, "Cl": Cl,
        "pump_NaK_mAcm2": pump_NaK, "ca_clear_mAcm2": Ca_clear,
    }


def main():
    print("=" * 80)
    print("Pump-scaling fix test — previously implausible cells")
    print("=" * 80)
    print(f"\n{'cell':<8} {'V mV':>8} {'Ca μM':>10} {'Na mM':>7} {'K mM':>7} {'Cl mM':>6} "
          f"{'NaK pump':>10} {'Ca clear':>10}")
    print("-" * 80)
    for cls in TARGETS:
        try:
            r = run_one(cls)
            flag = "OK " if (-110 < r["V"] < 50 and r["Ca_uM"] < 1.0 and 80 < r["K"] < 200) else "WARN"
            print(f"{cls:<8} {r['V']:+8.2f} {r['Ca_uM']:>10.3f} {r['Na']:>7.2f} "
                  f"{r['K']:>7.2f} {r['Cl']:>6.2f} {r['pump_NaK_mAcm2']:>10.3e} "
                  f"{r['ca_clear_mAcm2']:>10.3e}  [{flag}]")
        except Exception as e:
            print(f"{cls:<8} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
