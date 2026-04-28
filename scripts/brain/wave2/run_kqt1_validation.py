"""
Wave 2 cellular extension Option B CP1 — KQT-1 voltage-clamp Layer A validation.

Validates the KQT-1 Brian2 translation against Nicoletti's NEURON kqt1.mod
reference under voltage clamp at AVAL geometry (neutral testbed). Tolerance:
current-domain divergence ≤ 0.05 per feature, > 80% of holds clear all
features. Same harness used for SHK-1, SHL-1, NCA, KQT-3.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_phase_c_channels import validate_channel, save_results


def run_kqt1():
    from channels import kqt1 as channel_mod
    # AIY's parameter-vector value for kqt1 is 0.2 nS at the cell level, which
    # rescaled to S/cm² under AIY's surf=65.89e-8 cm² gives ~3.0e-4 S/cm².
    # For translation validation we use a similar small density at AVAL geometry
    # so currents are well-resolved over the [-80, +40] mV sweep without
    # saturating the leak.
    result = validate_channel(
        channel_name="kqt1",
        neuron_name="kqt1",
        gbar_Scm2=0.001,
        channel_module=channel_mod,
        description=(
            "KQT-1: voltage-gated K (KCNQ-family, 2-state m·s gating; distinct "
            "from KQT-3's 4-state mf/ms·s·w model). Slow inactivation s-gate "
            "with double-Boltzmann sinf and very slow stau component (~186 s)."
        ),
    )
    save_results(result, "kqt1")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Wave 2 cellular extension Option B CP1 — KQT-1 translation validation")
    print("=" * 70)
    result = run_kqt1()
    print()
    print("=" * 70)
    print(f"FINAL: panel_pass={result['panel_pass']} "
          f"({result['n_holds_passing']}/{result['n_holds']} holds, "
          f"frac={result['fraction_passing']:.3f})")
    print("=" * 70)
