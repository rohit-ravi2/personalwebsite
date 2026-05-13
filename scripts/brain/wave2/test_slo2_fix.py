"""
Test SLO-2 Ca-activated K channel addition.

Target: any remaining Ca-runaway cell that expresses slo-2.
Direct mechanism: SLO-2 opens when Ca rises → outward K → V hyperpolarizes
→ closes Ca channels → Ca clears.
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
TARGETS = ["RIB", "AVE", "HSN", "VD_DD", "RIM", "AVA", "AIY", "ASEL", "AWA"]


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
        "slo2_gbar": spec_s.channels.get("slo2", 0.0),
        "slo2_tpm": CENGEN_T2_TPM.get("slo-2", {}).get(cls, 0.0),
    }


def plausible(r):
    return (-110 < r["V"] < 50 and r["Ca_uM"] < 1.0
            and 80 < r["K"] < 200 and 0.5 < r["Na"] < 50 and 1 < r["Cl"] < 30)


def main():
    print("=" * 120)
    print("SLO-2 Ca-K fix test — variant A pump + EXP-2 + SHK-1 + TWK + SLO-2")
    print("=" * 120)
    print(f"\n{'cell':<8} {'slo2 TPM':>9} {'slo2 gbar':>12} | "
          f"{'V mV':>8} {'Ca μM':>10} {'Na mM':>7} {'K mM':>7} {'Cl mM':>6} | OK")
    print("-" * 105)
    n_plaus = 0
    for cls in TARGETS:
        try:
            r = run_one(cls)
            ok_b = plausible(r)
            n_plaus += int(ok_b)
            ok = "OK " if ok_b else "WARN"
            print(f"{cls:<8} {r['slo2_tpm']:>9.0f} {r['slo2_gbar']:>12.3e} | "
                  f"{r['V']:+8.2f} {r['Ca_uM']:>10.3f} "
                  f"{r['Na']:>7.2f} {r['K']:>7.2f} {r['Cl']:>6.2f} | {ok}")
        except Exception as e:
            print(f"{cls:<8} FAILED: {type(e).__name__}: {e}")
    print(f"\nPlausible: {n_plaus}/{len(TARGETS)}")


if __name__ == "__main__":
    main()
