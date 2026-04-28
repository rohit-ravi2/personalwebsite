"""
Wave 2 option α-1 CP1 — validate UNC-103 translation.

Voltage-clamp Layer A: Brian2 [leak + UNC-103] vs NEURON [leak + unc103]
across 11 holds (-80 to +40 mV in ~10-12 mV steps), AVAL geometry as
neutral testbed.

Per-channel acceptance: voltage-feature ≤5% relative + >80% holds clear
(per established Phase β methodology).

Usage:
    cd ~/Desktop/website/personalwebsite/scripts/brain
    source ~/venvs/wave2-neuron/bin/activate
    python wave2/run_option_alpha_cp1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_phase_c_channels import validate_channel, save_results


def run_unc103():
    from channels import unc103 as channel_mod
    # gbar=0.1 S/cm² (matching SHK-1 convention) — UNC-103 NMODL default is
    # 2.9 S/cm² which is excessive for isolated translation correctness check.
    # Translation correctness is independent of gbar magnitude (linear scaling).
    result = validate_channel(
        channel_name="unc103",
        neuron_name="unc103",
        gbar_Scm2=0.1,
        channel_module=channel_mod,
        description=(
            "UNC-103: voltage-gated K (ERG-family-like), m·h gates, voltage-only.\n"
            "  PRODUCT-form tau (not sum) — inherits Nicoletti pattern shared with EGL-2.\n"
            "  Translation following SHK-1/SHL-1/KQT-3 pattern (no GLOBAL state).\n"
            "  Used in Nicoletti's AVAR cell (5-channel set), NOT AVAL (4-channel)."
        ),
    )
    save_results(result, "unc103")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Wave 2 option α-1 CP1 — UNC-103 translation validation")
    print("=" * 70)
    result = run_unc103()
    if result["panel_pass"]:
        print(f"\n[CP1 PASS] UNC-103 translation validated against NEURON reference.")
        print(f"  {result['n_holds_passing']}/{result['n_holds']} holds pass "
              f"({result['fraction_passing']:.1%}).")
    else:
        print(f"\n[CP1 FAIL] UNC-103 translation FAILS validation.")
        print(f"  {result['n_holds_passing']}/{result['n_holds']} holds pass "
              f"({result['fraction_passing']:.1%}).")
        sys.exit(1)
