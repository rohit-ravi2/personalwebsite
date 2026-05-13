"""
Compare 3 pump-scaling variants on a representative set of cells.

Variant A: no per-cell pump scaling — all cells use AVAL anchor I_max
           (force pump_cell_name = "AVAL")
Variant B: damped scaling — I_max scaled by sqrt(TPM_cell / TPM_AVA)
Variant C: full per-cell TPM scaling (current default after extend_pump_dicts)

For each cell × variant, build, run 1.5 s rest, record V_rest + ion
concentrations + plausibility.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from dataclasses import replace

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from path2_scale.scalable_builder import build_scalable_spec, to_layer1_cellspec
from path2_scale.pump_scaling import extend_pump_dicts
from layer1_cells import build_layer1_cell, PUMP_ANCHOR_AVAL
from pumps.na_k_atpase import (
    EAT6_TPM_CENGEN_T2, scale_I_max_by_eat6_tpm, apply_na_k_atpase_params,
)
from pumps.ca_clearance import (
    MCA3_TPM_CENGEN_T2, scale_I_max_by_mca3_tpm, apply_ca_clearance_params,
)
from pumps.kcc2_abts1_lumped import (
    KCC2_TPM_CENGEN_T2, ABTS1_TPM_CENGEN_T2,
    scale_I_max_by_kcc2_tpm, scale_I_max_by_abts1_tpm,
    apply_kcc2_params, apply_abts1_params,
)

extend_pump_dicts()

SIM_MS = 1500.0
CELLS = ["HSN", "RIB", "VD_DD", "RIM", "AVE", "AVA", "AIY", "ASEL", "AWA"]


def build_with_variant(cengen_class: str, variant: str):
    """Build a Path 2 cell using one of the three pump-scaling variants."""
    from brian2 import ms
    spec_s = build_scalable_spec(cengen_class)
    spec_l = to_layer1_cellspec(spec_s)

    # build_layer1_cell will use spec_l.pump_cell_name for full TPM scaling.
    # We then OVERRIDE pump I_max values directly post-build to enforce
    # variant A/B/C semantics consistently.
    bundle = build_layer1_cell(spec_l)
    G = bundle["group"]

    pump_key = spec_l.pump_cell_name  # CeNGEN class name or Nicoletti

    if variant == "A":
        # All pumps at AVAL anchor — no scaling
        I_NaK = PUMP_ANCHOR_AVAL["I_NaK_max"]
        I_Ca  = PUMP_ANCHOR_AVAL["I_Ca_clear_max"]
        I_kcc = PUMP_ANCHOR_AVAL["I_kcc2_max"]
        I_abt = PUMP_ANCHOR_AVAL["I_abts1_max"]
    elif variant == "B":
        # Damped: sqrt(TPM ratio)
        def damp(tpm_dict, key, anchor_value):
            r = tpm_dict[key] / tpm_dict["AVAL"]
            return anchor_value * math.sqrt(max(r, 1e-3))
        I_NaK = damp(EAT6_TPM_CENGEN_T2, pump_key, PUMP_ANCHOR_AVAL["I_NaK_max"])
        I_Ca  = damp(MCA3_TPM_CENGEN_T2, pump_key, PUMP_ANCHOR_AVAL["I_Ca_clear_max"])
        I_kcc = damp(KCC2_TPM_CENGEN_T2, pump_key, PUMP_ANCHOR_AVAL["I_kcc2_max"])
        I_abt = damp(ABTS1_TPM_CENGEN_T2, pump_key, PUMP_ANCHOR_AVAL["I_abts1_max"])
    elif variant == "C":
        # Full per-cell TPM scaling (existing behavior)
        I_NaK = scale_I_max_by_eat6_tpm(PUMP_ANCHOR_AVAL["I_NaK_max"], pump_key)
        I_Ca  = scale_I_max_by_mca3_tpm(PUMP_ANCHOR_AVAL["I_Ca_clear_max"], pump_key)
        I_kcc = scale_I_max_by_kcc2_tpm(PUMP_ANCHOR_AVAL["I_kcc2_max"], pump_key)
        I_abt = scale_I_max_by_abts1_tpm(PUMP_ANCHOR_AVAL["I_abts1_max"], pump_key)
    else:
        raise ValueError(variant)

    G.pump_NaK_I_max_mAcm2 = I_NaK
    G.ca_clear_I_max_mAcm2 = I_Ca
    G.kcc2_I_max_mAcm2     = I_kcc
    G.abts1_I_max_mAcm2    = I_abt

    bundle["network"].run(SIM_MS * ms)
    mon = bundle["monitor"]
    V = float(mon.v[0][-1] / 1e-3)
    Ca = float(mon.Ca_in[0][-1]) * 1e3  # μM
    Na = float(mon.Na_in[0][-1])
    K = float(mon.K_in[0][-1])
    Cl = float(mon.Cl_in[0][-1])
    return {
        "V": V, "Ca_uM": Ca, "Na": Na, "K": K, "Cl": Cl,
        "I_NaK": I_NaK, "I_Ca": I_Ca,
    }


def plausible(r):
    return (-110 < r["V"] < 50 and r["Ca_uM"] < 1.0
            and 80 < r["K"] < 200 and 0.5 < r["Na"] < 50 and 1 < r["Cl"] < 30)


def main():
    print("=" * 110)
    print("Pump scaling variants — head-to-head")
    print("=" * 110)
    print("\nA = AVAL anchor (no per-cell scaling)")
    print("B = damped (sqrt of TPM ratio)")
    print("C = full TPM ratio (current default)")
    print()

    results = {}
    for cls in CELLS:
        results[cls] = {}
        for variant in ["A", "B", "C"]:
            try:
                r = build_with_variant(cls, variant)
                results[cls][variant] = r
            except Exception as e:
                results[cls][variant] = {"error": f"{type(e).__name__}: {e}"}

    print(f"{'cell':<8} | {'variant':<8} | {'V mV':>8} | {'Ca μM':>10} | "
          f"{'Na mM':>7} | {'K mM':>7} | {'Cl mM':>6} | OK")
    print("-" * 90)
    for cls in CELLS:
        for variant in ["A", "B", "C"]:
            r = results[cls][variant]
            if "error" in r:
                print(f"{cls:<8} | {variant:<8} | FAILED: {r['error'][:60]}")
                continue
            ok = "OK " if plausible(r) else "WARN"
            print(f"{cls:<8} | {variant:<8} | {r['V']:+8.2f} | {r['Ca_uM']:>10.3f} | "
                  f"{r['Na']:>7.2f} | {r['K']:>7.2f} | {r['Cl']:>6.2f} | {ok}")
        print("-" * 90)

    # Summary: how many cells plausible per variant
    print("\nPlausibility tally:")
    for variant in ["A", "B", "C"]:
        n_ok = sum(1 for cls in CELLS
                   if "error" not in results[cls][variant]
                   and plausible(results[cls][variant]))
        print(f"  variant {variant}: {n_ok}/{len(CELLS)} plausible")


if __name__ == "__main__":
    main()
