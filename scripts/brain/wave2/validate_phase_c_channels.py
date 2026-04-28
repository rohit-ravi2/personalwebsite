"""
Phase β run #2 Phase C — voltage-clamp validation for non-Ca channels.

Validates SHK-1, SHL-1, NCA, KQT-3 against Nicoletti's NEURON references via
voltage-clamp Layer A. Each channel inserted in a [leak + channel] cell at
AVAL geometry, swept across 11 holds (-80 to +40 mV in 10-12 mV steps).

Per-channel acceptance: voltage-feature ≤5% relative + >80% holds clear.

Channel modules expected at wave2/channels/<name>.py with conventions:
  - <NAME>_PARAMS: dict of NMODL parameter defaults
  - <NAME>_EQS: Brian2 equation string
  - <name>_apply_params(group, ...): set parameters on a NeuronGroup
  - <name>_init_states(group, v_mV=-60): initialize state vars to SS

The cell construction uses AVAL geometry/cm/leak as a "neutral" testbed.
This is for translation correctness, not architectural sufficiency.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from neuron_reference import NEURONReference
from voltage_clamp_harness import voltage_clamp_compare_v2


def build_brian2_kchannel_factory(
    channel_module,
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164 * 1e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_channel_Scm2: float = 0.1,
    ek_mV: float = -80.0,
    v_init_mV: float = -60.0,
):
    """Generic factory for K channel + leak cell.

    channel_module : module with EQS, apply_params, init_states.
    The channel must produce ik_<name>_mAcm2 and read v_mV.
    """
    EQS = channel_module.EQS
    apply_params = channel_module.apply_params
    init_states = channel_module.init_states
    chan_name = channel_module.NAME  # e.g. 'shk1'

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, prefs, start_scope,
        )
        start_scope()
        prefs.codegen.target = "cython"

        ik_var = f"ik_{chan_name}_mAcm2"
        cell_eqs = f"""
        v_mV = v / mV : 1
        # K-channel current density (mA/cm²), summed (only one channel here):
        ik_total_mAcm2 = {ik_var} : 1
        # Leak current density (mA/cm², non-specific):
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        # Total membrane current density (NEURON i convention):
        i_total_mAcm2 = i_leak_mAcm2 + ik_total_mAcm2 : 1
        # I_total in pA: × surf (cm²) × 1e9 to go mA/cm² → mA → pA (×1e-3 mA→A ×1e12 A→pA = 1e9):
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA : amp
        # Capacitance dynamics (only non-clamp regime uses this; during clamp v is force-set):
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + EQS

        from brian2 import pA, pF
        G = NeuronGroup(1, cell_eqs, method="euler")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV
        apply_params(G, gbar_Scm2=g_channel_Scm2, ek_mV=ek_mV)
        init_states(G, v_mV=v_init_mV)

        clamp = {"v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            G.v = clamp["v_target_mV"] * mV

        # Record only what's needed
        record_vars = ["v", "I_total", ik_var, "i_leak_mAcm2"]
        # Add gates if present
        for gate in [f"m_{chan_name}", f"h_{chan_name}", f"n_{chan_name}"]:
            if gate in cell_eqs:
                record_vars.append(gate)
        mon = StateMonitor(G, record_vars, record=True)
        net = Network(G, mon, _clamp)

        def set_v(v_mV: float) -> None:
            clamp["v_target_mV"] = float(v_mV)
            G.v = float(v_mV) * mV

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "set_v": set_v,
        }

    return _factory


def build_neuron_kchannel_cell(
    channel_neuron_name: str,
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164 * 1e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_channel_Scm2: float = 0.1,
    ek_mV: float = -80.0,
) -> NEURONReference:
    """NEURON [leak + <channel>] reference cell."""
    custom_spec = {
        "channels": ["leak", channel_neuron_name],
        "params": {
            ("leak", "gbar"): g_leak_Scm2,
            ("leak", "e"): e_leak_mV,
            (channel_neuron_name, "gbar"): g_channel_Scm2,
        },
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": 60.0,
        "ek_mV": ek_mV,
        "v_init_mV": -60.0,
    }
    return NEURONReference("custom", custom_spec=custom_spec)


def validate_channel(channel_name: str, neuron_name: str, gbar_Scm2: float,
                     channel_module, holds=None,
                     description: str = ""):
    """Run voltage-clamp Layer A comparison for a single channel.

    Returns the results dict (saved to artifacts/<name>_validation_results.json).
    """
    if holds is None:
        holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    # AVAL geometry as a neutral testbed
    surf = 1123.84e-8
    cm_uFcm2 = 0.859551
    g_leak_Scm2 = 0.150164 * 1e-9 / surf
    e_leak_mV = -39.0

    print(f"\n=== Phase C: {channel_name.upper()} translation validation ===")
    if description:
        print(description)
    print(f"Cell: AVAL-like [leak + {channel_name}] @ surf={surf:.3e} cm²")
    print(f"  g_leak={g_leak_Scm2:.3e} S/cm²  e_leak={e_leak_mV} mV")
    print(f"  g_{channel_name}={gbar_Scm2:.3e} S/cm²  ek=-80 mV")

    print(f"\nBuilding NEURON reference [leak + {neuron_name}]...")
    nref = build_neuron_kchannel_cell(
        channel_neuron_name=neuron_name,
        surf_cm2=surf, cm_uFcm2=cm_uFcm2,
        g_leak_Scm2=g_leak_Scm2, e_leak_mV=e_leak_mV,
        g_channel_Scm2=gbar_Scm2, ek_mV=-80.0,
    )

    print(f"Building Brian2 factory [leak + {channel_name}]...")
    factory = build_brian2_kchannel_factory(
        channel_module=channel_module,
        surf_cm2=surf, cm_uFcm2=cm_uFcm2,
        g_leak_Scm2=g_leak_Scm2, e_leak_mV=e_leak_mV,
        g_channel_Scm2=gbar_Scm2, ek_mV=-80.0,
    )

    print(f"Running voltage-clamp comparison at {len(holds)} holds...")
    result = voltage_clamp_compare_v2(
        factory, nref, holds,
        duration_ms=200.0,
        dt_ms=0.025,
        settle_window_ms=20.0,
        feature_tolerance=0.05,
        panel_pass_fraction=0.8,
        feature_keys=("peak_I_pA", "ss_I_pA"),
        skip_initial_transient_ms=2.0,
        brian2_prestep_ms=50.0,
        brian2_prestep_mV=-60.0,
    )
    nref.cleanup()

    print(f"Results: panel_pass={result['panel_pass']}  "
          f"holds_passing={result['n_holds_passing']}/{result['n_holds']}  "
          f"frac={result['fraction_passing']:.3f}")

    # Per-hold detail
    print("\nPer-hold detail:")
    for s, e in zip(result["per_step"], result["evaluation"]["per_step_evaluations"]):
        peak_div = e["feature_results"]["peak_I_pA"]["divergence"]
        ss_div = e["feature_results"]["ss_I_pA"]["divergence"]
        peak_pass = e["feature_results"]["peak_I_pA"]["pass"]
        ss_pass = e["feature_results"]["ss_I_pA"]["pass"]
        print(f"  hold={s['hold_mV']:+6.1f} mV  "
              f"brian2_peak={s['brian2_peak_I_pA']:+8.2f} pA  "
              f"nrn_peak={s['ref_peak_I_pA']:+8.2f} pA  "
              f"div={peak_div:.3f}({'P' if peak_pass else 'F'})  "
              f"brian2_ss={s['brian2_ss_I_pA']:+8.2f}  "
              f"nrn_ss={s['ref_ss_I_pA']:+8.2f}  "
              f"div={ss_div:.3f}({'P' if ss_pass else 'F'})")

    return result


def save_results(result, channel_name: str):
    out_path = Path(__file__).parent / "artifacts" / f"{channel_name}_validation_results.json"
    serializable = {
        "panel_pass": result["panel_pass"],
        "n_holds": result["n_holds"],
        "n_holds_passing": result["n_holds_passing"],
        "fraction_passing": result["fraction_passing"],
        "tolerance_metric": result["tolerance_metric"],
        "per_step": [
            {k: (v if not isinstance(v, np.ndarray) else v.tolist())
             for k, v in s.items()
             if k in ("hold_mV", "brian2_peak_I_pA", "ref_peak_I_pA",
                      "brian2_ss_I_pA", "ref_ss_I_pA")}
            for s in result["per_step"]
        ],
        "per_step_evaluations": result["evaluation"]["per_step_evaluations"],
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  → saved {out_path.name}")
