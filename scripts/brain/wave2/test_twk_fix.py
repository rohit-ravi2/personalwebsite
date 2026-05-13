"""
Test TWK K2P leak channel addition.

Target: VD_DD (twk total 423 TPM) — should fix the major Ca runaway.
Secondary: AVE (twk-40=123) — may improve.
Controls: HSN/RIB/AVA/AIY/ASEL/AWA must not regress.
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
TARGETS = ["VD_DD", "AVE", "HSN", "RIB", "RIM", "AVA", "AIY", "ASEL", "AWA"]


def twk_total(cls):
    return sum(CENGEN_T2_TPM.get(g, {}).get(cls, 0.0)
               for g in ("twk-7", "twk-18", "twk-30", "twk-40"))


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
        "twk_gbar": spec_s.channels.get("twk", 0.0),
        "twk_tpm": twk_total(cls),
    }


def plausible(r):
    return (-110 < r["V"] < 50 and r["Ca_uM"] < 1.0
            and 80 < r["K"] < 200 and 0.5 < r["Na"] < 50 and 1 < r["Cl"] < 30)


def main():
    print("=" * 116)
    print("TWK K2P fix test — variant A pump + EXP-2 + SHK-1 + TWK all wired")
    print("=" * 116)
    print(f"\n{'cell':<8} {'twk TPM':>9} {'twk gbar':>12} | "
          f"{'V mV':>8} {'Ca μM':>10} {'Na mM':>7} {'K mM':>7} {'Cl mM':>6} | OK")
    print("-" * 100)
    n_plaus = 0
    for cls in TARGETS:
        try:
            r = run_one(cls)
            ok_b = plausible(r)
            n_plaus += int(ok_b)
            ok = "OK " if ok_b else "WARN"
            print(f"{cls:<8} {r['twk_tpm']:>9.0f} {r['twk_gbar']:>12.3e} | "
                  f"{r['V']:+8.2f} {r['Ca_uM']:>10.3f} "
                  f"{r['Na']:>7.2f} {r['K']:>7.2f} {r['Cl']:>6.2f} | {ok}")
        except Exception as e:
            print(f"{cls:<8} FAILED: {type(e).__name__}: {e}")
    print(f"\nPlausible: {n_plaus}/{len(TARGETS)}")


if __name__ == "__main__":
    main()
