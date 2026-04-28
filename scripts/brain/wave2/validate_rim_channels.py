"""
Wave 2 RIM channel validation — voltage-clamp Layer A for CCA-1, EGL-2, UNC-2.

Generic harness supporting both K-channel (writes ik) and Ca-channel (writes ica)
translations against Nicoletti's NEURON references. Cell construction:
[leak + channel] at AVAL geometry (neutral testbed). Tolerance: current-domain
divergence ≤ 0.05 per feature, > 80% holds pass.

For Ca-using channels: the [leak+channel] cell has a single USEION ca, so NEURON's
ion_style does NOT override seg.eca. We use eca=60 mV here for both NEURON and
Brian2 (matches NEURON's actual runtime in this single-USEION-ca testbed).
F18 only kicks in when the channel is composed into a multi-USEION-ca cell
(RIM proper at CP4-CP6).

Usage:
    python validate_rim_channels.py            # runs all 3
    python validate_rim_channels.py cca1       # runs single channel
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


# ---------------------------------------------------------------------------
# Generic Brian2 [leak + channel] factory (K or Ca channel)
# ---------------------------------------------------------------------------

def build_brian2_channel_factory(
    channel_module,
    channel_kind: str,            # 'k' or 'ca'
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164 * 1e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_channel_Scm2: float = 0.001,
    ek_mV: float = -80.0,
    eca_mV: float = 60.0,
    v_init_mV: float = -60.0,
):
    """Generic factory for [leak + channel] cell.

    channel_module : module with EQS, apply_params, init_states, NAME.
    channel_kind   : 'k' (writes ik_<name>_mAcm2) or 'ca' (writes ica_<name>_mAcm2).
    """
    EQS = channel_module.EQS
    apply_params = channel_module.apply_params
    init_states = channel_module.init_states
    chan_name = channel_module.NAME

    if channel_kind == "k":
        chan_current_var = f"ik_{chan_name}_mAcm2"
    elif channel_kind == "ca":
        chan_current_var = f"ica_{chan_name}_mAcm2"
    else:
        raise ValueError(f"channel_kind must be 'k' or 'ca', got {channel_kind!r}")

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, prefs, start_scope, pA, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        cell_eqs = f"""
        v_mV = v / mV : 1
        # channel current density (mA/cm²):
        i_chan_mAcm2 = {chan_current_var} : 1
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_total_mAcm2 = i_leak_mAcm2 + i_chan_mAcm2 : 1
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA : amp
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + EQS

        G = NeuronGroup(1, cell_eqs, method="rk4")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV

        if channel_kind == "k":
            apply_params(G, gbar_Scm2=g_channel_Scm2, ek_mV=ek_mV)
        else:
            apply_params(G, gbar_Scm2=g_channel_Scm2, eca_mV=eca_mV)
        init_states(G, v_mV=v_init_mV)

        clamp = {"v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            G.v = clamp["v_target_mV"] * mV

        record_vars = ["v", "I_total", chan_current_var, "i_leak_mAcm2"]
        for gate in [f"m_{chan_name}", f"h_{chan_name}"]:
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


def build_neuron_channel_cell(
    channel_neuron_name: str,
    channel_kind: str,
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164 * 1e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_channel_Scm2: float = 0.001,
    ek_mV: float = -80.0,
    eca_mV: float = 60.0,
) -> NEURONReference:
    """NEURON [leak + channel] reference cell."""
    custom_spec = {
        "channels": ["leak", channel_neuron_name],
        "params": {
            ("leak", "gbar"): g_leak_Scm2,
            ("leak", "e"): e_leak_mV,
            (channel_neuron_name, "gbar"): g_channel_Scm2,
        },
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": eca_mV,
        "ek_mV": ek_mV,
        "v_init_mV": -60.0,
    }
    return NEURONReference("custom", custom_spec=custom_spec)


def validate_channel(channel_name: str,
                     neuron_name: str,
                     channel_kind: str,
                     gbar_Scm2: float,
                     channel_module,
                     holds=None,
                     description: str = "") -> dict:
    """Run voltage-clamp Layer A comparison for a single channel."""
    if holds is None:
        holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    surf = 1123.84e-8
    cm_uFcm2 = 0.859551
    g_leak_Scm2 = 0.150164 * 1e-9 / surf
    e_leak_mV = -39.0

    print(f"\n=== Wave 2 RIM: {channel_name.upper()} translation validation "
          f"({channel_kind.upper()}-channel) ===")
    if description:
        print(description)
    print(f"Cell: AVAL-like [leak + {channel_name}] @ surf={surf:.3e} cm²")
    print(f"  g_leak={g_leak_Scm2:.3e} S/cm², e_leak={e_leak_mV} mV")
    print(f"  g_{channel_name}={gbar_Scm2:.3e} S/cm²", end="")
    if channel_kind == "k":
        print(f", ek=-80 mV")
    else:
        print(f", eca=60 mV (single USEION ca; ion_style does not override here)")

    print(f"\nBuilding NEURON reference [leak + {neuron_name}]...")
    nref = build_neuron_channel_cell(
        channel_neuron_name=neuron_name, channel_kind=channel_kind,
        surf_cm2=surf, cm_uFcm2=cm_uFcm2,
        g_leak_Scm2=g_leak_Scm2, e_leak_mV=e_leak_mV,
        g_channel_Scm2=gbar_Scm2,
    )

    print(f"Building Brian2 factory [leak + {channel_name}]...")
    factory = build_brian2_channel_factory(
        channel_module=channel_module, channel_kind=channel_kind,
        surf_cm2=surf, cm_uFcm2=cm_uFcm2,
        g_leak_Scm2=g_leak_Scm2, e_leak_mV=e_leak_mV,
        g_channel_Scm2=gbar_Scm2,
    )

    print(f"Running voltage-clamp at {len(holds)} holds...")
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
    print("\nPer-hold detail:")
    for s, e in zip(result["per_step"], result["evaluation"]["per_step_evaluations"]):
        peak_div = e["feature_results"]["peak_I_pA"]["divergence"]
        ss_div = e["feature_results"]["ss_I_pA"]["divergence"]
        peak_pass = e["feature_results"]["peak_I_pA"]["pass"]
        ss_pass = e["feature_results"]["ss_I_pA"]["pass"]
        print(f"  hold={s['hold_mV']:+6.1f} mV  "
              f"b2_peak={s['brian2_peak_I_pA']:+9.3f}  "
              f"nrn_peak={s['ref_peak_I_pA']:+9.3f}  "
              f"div={peak_div:.3f}({'P' if peak_pass else 'F'})  "
              f"b2_ss={s['brian2_ss_I_pA']:+9.3f}  "
              f"nrn_ss={s['ref_ss_I_pA']:+9.3f}  "
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


def write_status(channel_name: str, result: dict):
    """Write per-CP status JSON atomically."""
    cp_map = {"cca1": "rim_CP1", "egl2": "rim_CP2", "unc2": "rim_CP3"}
    cp = cp_map[channel_name]
    out_path = Path(__file__).parent / "artifacts" / "checkpoints" / f"{cp}_status.json"
    out_path.parent.mkdir(exist_ok=True)
    payload = {
        "checkpoint": cp,
        "channel": channel_name,
        "panel_pass": bool(result["panel_pass"]),
        "n_holds_passing": result["n_holds_passing"],
        "n_holds": result["n_holds"],
        "fraction_passing": float(result["fraction_passing"]),
    }
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)
    print(f"  → wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Per-channel runners
# ---------------------------------------------------------------------------

def run_cca1():
    from channels import cca1 as channel_mod
    # RIM's cca1 g0 = 8.452e-4 S/cm². Use that for realistic operating regime.
    result = validate_channel(
        channel_name="cca1",
        neuron_name="cca1",
        channel_kind="ca",
        gbar_Scm2=8.452e-4,
        channel_module=channel_mod,
        description=(
            "CCA-1: T-type voltage-gated Ca channel. m^2*h gating. "
            "Reads eca, writes ica. Single USEION ca in [leak+cca1] testbed "
            "(no ion_style override; eca=60 mV preserved)."
        ),
    )
    save_results(result, "cca1")
    write_status("cca1", result)
    return result


def run_egl2():
    from channels import egl2 as channel_mod
    result = validate_channel(
        channel_name="egl2",
        neuron_name="egl2",
        channel_kind="k",
        gbar_Scm2=1.412e-4,  # RIM g0
        channel_module=channel_mod,
        description=(
            "EGL-2: voltage-gated K (EAG family). Single state m, "
            "ik = gbar*m*(v-ek). Standard voltage-gated K pattern; "
            "EAG kinetics characterized by very-slow tau and shallow ka."
        ),
    )
    save_results(result, "egl2")
    write_status("egl2", result)
    return result


def run_unc2():
    from channels import unc2 as channel_mod
    result = validate_channel(
        channel_name="unc2",
        neuron_name="unc2",
        channel_kind="ca",
        gbar_Scm2=9.677e-5,  # RIM g0
        channel_module=channel_mod,
        description=(
            "UNC-2: P/Q-type voltage-gated Ca channel. m*h gating. "
            "NMODL has GLOBAL declarations on minf/hinf/mtau/htau/munc2/hunc2 — "
            "harmless single-cell pitfall; treated as per-cell in Brian2."
        ),
    )
    save_results(result, "unc2")
    write_status("unc2", result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    runners = {"cca1": run_cca1, "egl2": run_egl2, "unc2": run_unc2}
    if not args:
        targets = ["cca1", "egl2", "unc2"]
    else:
        targets = args

    overall = {}
    for t in targets:
        if t not in runners:
            print(f"unknown channel: {t}; choices: {list(runners)}")
            sys.exit(2)
        result = runners[t]()
        overall[t] = result["panel_pass"]
        print()

    print("=" * 70)
    print("Wave 2 RIM channel translations summary:")
    for t, ok in overall.items():
        print(f"  {t}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if all(overall.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
