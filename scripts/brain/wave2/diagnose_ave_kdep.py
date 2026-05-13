"""
Diagnose AVE K-depletion. Identify which channel/pump is responsible for
K_in collapsing from 140 → 44 mM by selectively zeroing components.

Tests:
  1. Baseline AVE (all on)        — expect K=44, V=-55
  2. TWK off                      — does K recover?
  3. EGL-36 off                   — does K recover?
  4. EXP-2 off                    — does K recover?
  5. KCC-2 off                    — does K recover?
  6. Na/K-ATPase 2x                — does K recover?
  7. Na/K-ATPase 5x                — does K recover?
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from path2_scale.scalable_builder import build_scalable_spec, to_layer1_cellspec
from layer1_cells import build_layer1_cell

SIM_MS = 1500.0


def run_variant(label, channel_overrides=None, pump_overrides=None):
    from brian2 import ms
    spec_s = build_scalable_spec("AVE")
    spec_l = to_layer1_cellspec(spec_s)
    if channel_overrides:
        new_channels = dict(spec_l.channels)
        for ch, v in channel_overrides.items():
            if ch in new_channels:
                new_channels[ch] = v
        spec_l = replace(spec_l, channels=new_channels)
    bundle = build_layer1_cell(spec_l)
    if pump_overrides:
        G = bundle["group"]
        for attr, val in pump_overrides.items():
            setattr(G, attr, val)
    bundle["network"].run(SIM_MS * ms)
    mon = bundle["monitor"]
    V = float(mon.v[0][-1] / 1e-3)
    K = float(mon.K_in[0][-1])
    Na = float(mon.Na_in[0][-1])
    Ca = float(mon.Ca_in[0][-1]) * 1e3
    Cl = float(mon.Cl_in[0][-1])
    pump_NaK = float(mon.pump_NaK_I_mAcm2[0][-1])
    print(f"  {label:<30} V={V:+6.2f}  K={K:6.2f}  Na={Na:6.2f}  Cl={Cl:5.2f}  "
          f"Ca={Ca:7.3f} μM  pump_NaK={pump_NaK:.3e}")
    return {"V": V, "K": K, "Na": Na, "Cl": Cl, "Ca": Ca, "pump_NaK": pump_NaK}


def main():
    print("=" * 100)
    print("AVE K-depletion diagnostic")
    print("=" * 100)
    print(f"\nTargets: K_in stays near 140 mM (normal C. elegans intracellular K)\n")

    print("--- Baseline ---")
    base = run_variant("baseline (all on)")

    print("\n--- Disable individual K channels ---")
    run_variant("twk = 0",   channel_overrides={"twk": 0.0})
    run_variant("egl36 = 0", channel_overrides={"egl36": 0.0})
    run_variant("exp2 = 0",  channel_overrides={"exp2": 0.0})
    run_variant("irk = 0",   channel_overrides={"irk": 0.0})
    run_variant("shl1 = 0",  channel_overrides={"shl1": 0.0})

    print("\n--- Disable KCC-2 (K-Cl co-transporter, K outflux) ---")
    run_variant("KCC-2 I_max = 0",
                pump_overrides={"kcc2_I_max_mAcm2": 0.0})

    print("\n--- Boost Na/K-ATPase ---")
    run_variant("pump_NaK 2x", pump_overrides={
        "pump_NaK_I_max_mAcm2": 2 * 2.3461e-4})
    run_variant("pump_NaK 5x", pump_overrides={
        "pump_NaK_I_max_mAcm2": 5 * 2.3461e-4})
    run_variant("pump_NaK 10x", pump_overrides={
        "pump_NaK_I_max_mAcm2": 10 * 2.3461e-4})

    print("\n--- Combined: pump 5x + twk halved ---")
    run_variant("pump 5x + twk×0.5",
                channel_overrides={"twk": 0.5 * 4.906e-5},
                pump_overrides={"pump_NaK_I_max_mAcm2": 5 * 2.3461e-4})


if __name__ == "__main__":
    main()
