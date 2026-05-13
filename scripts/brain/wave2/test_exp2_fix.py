"""
Test the EXP-2 NMODL addition + variant-A pump scaling on the 4 failing
cells + controls. Compare V_rest, Ca, ion concentrations before/after.

Note: tests against EXP-2 expression for each cell — the cells with high
exp-2 TPM (HSN=1643, RIB=819, AVE=341) should see the biggest effect.
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
TARGETS = ["HSN", "RIB", "VD_DD", "RIM", "AVE", "AVA", "AIY", "ASEL", "AWA"]


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
        "exp2_gbar": spec_s.channels.get("exp2", 0.0),
        "exp2_tpm": CENGEN_T2_TPM.get("exp-2", {}).get(cls, 0.0),
    }


def plausible(r):
    return (-110 < r["V"] < 50 and r["Ca_uM"] < 1.0
            and 80 < r["K"] < 200 and 0.5 < r["Na"] < 50 and 1 < r["Cl"] < 30)


def main():
    print("=" * 110)
    print("EXP-2 NMODL fix test — variant A pump scaling + EXP-2 wired in")
    print("=" * 110)
    print(f"\n{'cell':<8} {'exp2 TPM':>10} {'exp2 gbar':>12} | "
          f"{'V mV':>8} {'Ca μM':>10} {'Na mM':>7} {'K mM':>7} {'Cl mM':>6} | OK")
    print("-" * 100)
    for cls in TARGETS:
        try:
            r = run_one(cls)
            ok = "OK " if plausible(r) else "WARN"
            print(f"{cls:<8} {r['exp2_tpm']:>10.0f} {r['exp2_gbar']:>12.3e} | "
                  f"{r['V']:+8.2f} {r['Ca_uM']:>10.3f} {r['Na']:>7.2f} "
                  f"{r['K']:>7.2f} {r['Cl']:>6.2f} | {ok}")
        except Exception as e:
            print(f"{cls:<8} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
