"""
Test EGL-36 Kv3 wiring. Targets:
  - High EGL-36 expressers: I3=422, M3=371, PQR=371, AVL=299, MC=218
  - Previously failing: AVE (egl-36=183) — diagnosis says no help, verify
  - Controls: HSN/VD_DD/AVA/AIY must not regress
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from path2_scale.scalable_builder import build_scalable_spec, to_layer1_cellspec
from layer1_cells import build_layer1_cell
from path2_scale.cengen_tpm_data import CENGEN_T2_TPM

SIM_MS = 1500.0
TARGETS = ["I3", "M3", "PQR", "AVL", "MC", "AVE", "RIB", "RIM",
           "HSN", "VD_DD", "AVA", "AIY"]


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
        "channels": list(spec_s.channels.keys()),
        "egl36_gbar": spec_s.channels.get("egl36", 0.0),
        "egl36_tpm": CENGEN_T2_TPM.get("egl-36", {}).get(cls, 0.0),
        "n_channels": len(spec_s.channels),
    }


def plausible(r):
    return (-110 < r["V"] < 50 and r["Ca_uM"] < 1.0
            and 80 < r["K"] < 200 and 0.5 < r["Na"] < 50 and 1 < r["Cl"] < 30)


def main():
    print("=" * 120)
    print("EGL-36 Kv3 fix test — full channel set wired (14 channels total)")
    print("=" * 120)
    print(f"\n{'cell':<8} {'egl36 TPM':>10} {'egl36 gbar':>12} {'nch':>4} | "
          f"{'V mV':>8} {'Ca μM':>10} {'Na mM':>7} {'K mM':>7} {'Cl mM':>6} | OK")
    print("-" * 110)
    n_plaus = 0
    for cls in TARGETS:
        try:
            r = run_one(cls)
            ok_b = plausible(r)
            n_plaus += int(ok_b)
            ok = "OK " if ok_b else "WARN"
            print(f"{cls:<8} {r['egl36_tpm']:>10.0f} {r['egl36_gbar']:>12.3e} "
                  f"{r['n_channels']:>4} | {r['V']:+8.2f} {r['Ca_uM']:>10.3f} "
                  f"{r['Na']:>7.2f} {r['K']:>7.2f} {r['Cl']:>6.2f} | {ok}")
        except Exception as e:
            print(f"{cls:<8} FAILED: {type(e).__name__}: {e}")
    print(f"\nPlausible: {n_plaus}/{len(TARGETS)}")


if __name__ == "__main__":
    main()
