"""
Phase β run #2 Phase C — runner that validates each non-Ca channel sequentially.

Run as: python -m wave2.run_phase_c
or:     PYTHONPATH=wave2 python wave2/run_phase_c.py [channel_name]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_phase_c_channels import validate_channel, save_results


def run_shk1():
    from channels import shk1 as channel_mod
    # Use Nicoletti's gbar default; AVAL doesn't include shk1, so this is just for translation correctness
    result = validate_channel(
        channel_name="shk1",
        neuron_name="shk1",
        gbar_Scm2=0.1,  # NMODL default
        channel_module=channel_mod,
        description="SHK-1: voltage-gated K (delayed rectifier-like, slow inactivation tau=1400 ms)",
    )
    save_results(result, "shk1")
    return result


def run_shl1():
    from channels import shl1 as channel_mod
    result = validate_channel(
        channel_name="shl1",
        neuron_name="shl1",
        gbar_Scm2=0.1,  # placeholder
        channel_module=channel_mod,
        description="SHL-1: voltage-gated K (Kv4 A-type, fast inactivation)",
    )
    save_results(result, "shl1")
    return result


def run_nca():
    from channels import nca as channel_mod
    result = validate_channel(
        channel_name="nca",
        neuron_name="nca",
        gbar_Scm2=0.0001,  # NCA is a leak; small density
        channel_module=channel_mod,
        description="NCA: NALCN-homolog Na leak channel",
    )
    save_results(result, "nca")
    return result


def run_kqt3():
    from channels import kqt3 as channel_mod
    result = validate_channel(
        channel_name="kqt3",
        neuron_name="kqt3",
        gbar_Scm2=0.001,
        channel_module=channel_mod,
        description="KQT-3: M-current K channel",
    )
    save_results(result, "kqt3")
    return result


CHANNELS = {
    "shk1": run_shk1,
    "shl1": run_shl1,
    "nca": run_nca,
    "kqt3": run_kqt3,
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        which = sys.argv[1]
        if which not in CHANNELS:
            print(f"Unknown channel: {which}. Choices: {list(CHANNELS)}")
            sys.exit(1)
        CHANNELS[which]()
    else:
        for name, runner in CHANNELS.items():
            print(f"\n{'='*70}")
            print(f"Running {name.upper()} validation")
            print('='*70)
            try:
                runner()
            except Exception as e:
                print(f"FAILED for {name}: {e}")
                import traceback; traceback.print_exc()
