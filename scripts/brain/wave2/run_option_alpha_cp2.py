"""
Wave 2 option α-1 CP2 — validate IRK translation.

Voltage-clamp Layer A: Brian2 [leak + IRK] vs NEURON [leak + irk]
across 11 holds (-80 to +40 mV in ~10-12 mV steps), AVAL geometry as
neutral testbed.

Per-channel acceptance: voltage-feature ≤5% relative + >80% holds clear.

Usage:
    cd ~/Desktop/website/personalwebsite/scripts/brain
    source ~/venvs/wave2-neuron/bin/activate
    python wave2/run_option_alpha_cp2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_phase_c_channels import validate_channel, save_results


def run_irk():
    from channels import irk as channel_mod
    # gbar=0.1 S/cm² matching SHK-1/UNC-103 convention. AVAL's published value
    # is g_irk=0.1 (raw nS) → at surf=1123.84e-8 cm² that's ~8.9e-6 S/cm².
    # For translation correctness we use a higher density to give visible
    # currents at all holds; correctness is independent of gbar magnitude.
    result = validate_channel(
        channel_name="irk",
        neuron_name="irk",
        gbar_Scm2=0.1,
        channel_module=channel_mod,
        description=(
            "IRK: Kir-family inwardly-rectifying K, single m gate, no inactivation.\n"
            "  Activates as V hyperpolarizes (minf has +30 mV shift inside exp).\n"
            "  U-shaped tau (sum of 2 exponentials with opposite slopes).\n"
            "  Used in Nicoletti's AVAL (4-channel) and AVAR (5-channel) cells."
        ),
    )
    save_results(result, "irk")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Wave 2 option α-1 CP2 — IRK translation validation")
    print("=" * 70)
    result = run_irk()
    if result["panel_pass"]:
        print(f"\n[CP2 PASS] IRK translation validated against NEURON reference.")
        print(f"  {result['n_holds_passing']}/{result['n_holds']} holds pass "
              f"({result['fraction_passing']:.1%}).")
    else:
        print(f"\n[CP2 FAIL] IRK translation FAILS validation.")
        print(f"  {result['n_holds_passing']}/{result['n_holds']} holds pass "
              f"({result['fraction_passing']:.1%}).")
        sys.exit(1)
