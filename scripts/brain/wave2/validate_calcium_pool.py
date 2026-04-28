"""
Phase β CP1.B.5/6/7 — Ca-pool validation: cadiff, caintra1, and combined.

Strategy
--------
NEURON's cadiff/caintra1 mods read `ica` (computed by another channel
mechanism). We can't inject `ica` directly. So:

  1. Build a NEURON section with cca1 (T-type Ca channel) + the Ca pool.
     Run a voltage-clamp protocol producing a known ica(t) and cai(t)
     trajectory. Record both at fine timestep.
  2. Replay the recorded ica(t) into a Brian2 NeuronGroup with only the
     Ca pool eqs (no channel kinetics). Solve ODE forward with same dt.
     Compare resulting cai_M(t) trajectory against the NEURON-recorded one.
  3. The match quality is a direct test of the eqs-string translation.

Acceptance per spec
-------------------
- Per-feature divergence ≤ 5% relative or absolute (current-domain analog
  applied to cai trajectory; cai-domain peak floor used).
- Per-panel: > 80% of test points within tolerance.

Note on cadiff vs caintra1
--------------------------
Both write `cai`, NEURON forbids multi-writer. We test them independently
in two separate cells/runs. Combined validation places both into the SAME
Brian2 simulation (with one driving cai_M for one channel set, another for
the other) — but that requires careful handling. For this run we test:
- cadiff alone (cell with cca1 + cadiff)
- caintra1 alone (cell with cca1 + caintra1)
- "combined" interpretation: caintra1 driven by an EGL-19 Ca current, since
  EGL-19 + caintra1 + leak is what CP3 actually uses.

For CP1.B.7 "combined Ca-pool" we use the third configuration since it's
what CP3 builds on.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from neuron_reference import NEURONReference, _nicoletti_env
from calcium_pool import (
    cadiff_eqs, caintra1_eqs,
    cadiff_brian2_factory, caintra1_brian2_factory,
)


def neuron_run_capool(
    ca_pool: str,
    pool_kwargs: dict,
    ca_channel: str = "cca1",
    ca_channel_gbar_Scm2: float = 0.7,
    surf_cm2: float = 65.89e-8,
    cm_uFcm2: float = 1.0,
    holding_potentials: list[float] = None,
    duration_ms: float = 200.0,
    dt_ms: float = 0.025,
    eca_mV: float = 60.0,
) -> dict:
    """Run NEURON section with [ca_channel + ca_pool] under voltage clamp;
    record (t, ica(t), cai(t)) for each holding potential.

    The Ca state recorded depends on the pool:
      - cadiff: records `_ref_cai` (cadiff WRITES cai)
      - caintra1: records mech-private `caintra` via `mech._ref_caintra`
        (caintra1 does NOT write cai; tracks state in private variable)

    Returns dict {pool, holds: [{hold_mV, t_ms, ica_mAcm2, cai}, ...]}.
    """
    if holding_potentials is None:
        holding_potentials = [-60.0, -30.0, 0.0, 30.0]

    custom_spec = {
        "channels": [ca_channel, ca_pool],
        "params": {
            (ca_channel, "gbar"): ca_channel_gbar_Scm2,
        },
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": eca_mV,
        "ek_mV": -80.0,
        "v_init_mV": -60.0,
    }
    # Pool-specific params via custom spec
    for k, v in pool_kwargs.items():
        custom_spec["params"][(ca_pool, k)] = v

    nref = NEURONReference("custom", custom_spec=custom_spec)

    # We need to record cai (the variable written by the pool) — neither
    # voltage_clamp nor current_clamp record cai. We do a manual run here.
    with _nicoletti_env():
        h = nref._h
        soma = nref._soma

        stim = h.VClamp(soma(0.5))
        # Simple single-step protocol: prestep at -60 mV, then step at hold.
        prestep_ms = 50.0
        stim.dur[0] = prestep_ms
        stim.dur[1] = duration_ms
        stim.dur[2] = 0.0
        stim.amp[0] = -60.0
        stim.amp[2] = -60.0

        t_vec = h.Vector()
        v_vec = h.Vector()
        ica_vec = h.Vector()
        cai_vec = h.Vector()

        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)
        ica_vec.record(soma(0.5)._ref_ica)
        # Pool-specific Ca state recording
        if ca_pool == "cadiff":
            cai_vec.record(soma(0.5)._ref_cai)
        elif ca_pool == "caintra1":
            mech = getattr(soma(0.5), ca_pool)
            cai_vec.record(mech._ref_caintra)
        else:
            cai_vec.record(soma(0.5)._ref_cai)

        results = []
        for v_hold in holding_potentials:
            stim.amp[1] = float(v_hold)
            h.tstop = prestep_ms + duration_ms
            h.dt = dt_ms
            h.finitialize(-60.0)
            h.run()

            t_full = np.array(t_vec.to_python())
            v_full = np.array(v_vec.to_python())
            ica_full = np.array(ica_vec.to_python())
            cai_full = np.array(cai_vec.to_python())

            # Extract step window
            step_mask = (t_full >= prestep_ms) & (t_full <= prestep_ms + duration_ms)
            t_step = t_full[step_mask] - prestep_ms
            ica_step = ica_full[step_mask]
            cai_step = cai_full[step_mask]
            v_step = v_full[step_mask]

            results.append({
                "hold_mV": float(v_hold),
                "t_ms": t_step.tolist(),
                "v_mV": v_step.tolist(),
                "ica_mAcm2": ica_step.tolist(),
                "cai": cai_step.tolist(),
            })

    nref.cleanup()
    return {
        "pool": ca_pool,
        "ca_channel": ca_channel,
        "holding_potentials": holding_potentials,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "holds": results,
    }


def brian2_replay_capool(pool_factory_fn, ica_trace_mAcm2: np.ndarray,
                         t_trace_ms: np.ndarray, dt_ms: float = 0.025,
                         init_cai_mM: float = 1e-4) -> np.ndarray:
    """Replay a recorded ica(t) trajectory through a Brian2 Ca-pool factory.

    Returns the resulting cai_mM(t) trajectory at the recorded timepoints.

    Implementation: interpolate ica onto Brian2's defaultclock grid, drive
    G.ica_mAcm2 each timestep via network_operation, record cai_mM, return.
    """
    from brian2 import (
        NeuronGroup, StateMonitor, Network, TimedArray, ms,
        prefs, start_scope, defaultclock,
    )
    start_scope()
    prefs.codegen.target = "cython"

    factory = pool_factory_fn()
    bundle = factory()
    G = bundle["group"]
    G.cai_mM = init_cai_mM

    duration_ms = float(t_trace_ms[-1])

    n = max(2, int(duration_ms / dt_ms) + 1)
    t_eval = np.linspace(0, duration_ms, n)
    ica_eval = np.interp(t_eval, t_trace_ms, ica_trace_mAcm2)
    ica_ta = TimedArray(ica_eval, dt=dt_ms * ms)

    from brian2 import network_operation

    @network_operation(dt=dt_ms * ms)
    def _drive():
        G.ica_mAcm2 = float(ica_ta(defaultclock.t))

    mon = bundle["monitor"]
    net = Network(G, mon, _drive)
    defaultclock.dt = dt_ms * ms
    net.run(duration_ms * ms)

    cai_arr = np.asarray(mon.cai_mM[0])
    t_arr = np.asarray(mon.t) * 1e3  # to ms
    cai_at_t = np.interp(t_trace_ms, t_arr, cai_arr)
    return cai_at_t


def compare_capool(neuron_result: dict, pool_factory_fn, dt_ms: float = 0.025) -> dict:
    """Compare NEURON cai vs Brian2 cai across holds. Apply current-domain-
    analog metric to cai trajectory.
    """
    panel_eval = []
    n_holds = len(neuron_result["holds"])

    # Compute cai-peak across all holds for floor-based tolerance
    cai_peaks = [max(np.abs(h["cai"])) for h in neuron_result["holds"]]
    panel_cai_peak = max(cai_peaks) if cai_peaks else 1e-7

    for h in neuron_result["holds"]:
        t_arr = np.array(h["t_ms"])
        ica_arr = np.array(h["ica_mAcm2"])
        cai_neuron = np.array(h["cai"])

        # Brian2 replay
        cai_brian2 = brian2_replay_capool(
            pool_factory_fn, ica_arr, t_arr, dt_ms=dt_ms,
            init_cai_mM=float(cai_neuron[0]),
        )

        # Current-domain divergence per timepoint
        peak_local = max(np.max(np.abs(cai_neuron)), np.max(np.abs(cai_brian2)), panel_cai_peak)
        denom = np.maximum.reduce([np.abs(cai_neuron), np.abs(cai_brian2),
                                    np.full_like(cai_neuron, 0.1 * peak_local)])
        # Avoid zero denom
        denom = np.where(denom == 0, 1e-30, denom)
        div = np.abs(cai_neuron - cai_brian2) / denom
        n_pass = int(np.sum(div <= 0.05))
        n_total = len(div)
        frac_pass = n_pass / n_total if n_total else 0.0

        panel_eval.append({
            "hold_mV": h["hold_mV"],
            "n_timepoints": n_total,
            "n_passing": n_pass,
            "fraction_passing": frac_pass,
            "max_divergence": float(np.max(div)),
            "median_divergence": float(np.median(div)),
            "neuron_cai_peak": float(np.max(np.abs(cai_neuron))),
            "brian2_cai_peak": float(np.max(np.abs(cai_brian2))),
            "neuron_cai_final": float(cai_neuron[-1]),
            "brian2_cai_final": float(cai_brian2[-1]),
            "neuron_ica_peak": float(np.max(np.abs(ica_arr))),
            "step_pass": frac_pass >= 0.8,
        })

    n_holds_passing = sum(1 for e in panel_eval if e["step_pass"])
    fraction_holds_passing = n_holds_passing / n_holds if n_holds else 0.0

    return {
        "panel_pass": fraction_holds_passing >= 0.8,
        "n_holds": n_holds,
        "n_holds_passing": n_holds_passing,
        "fraction_holds_passing": fraction_holds_passing,
        "panel_cai_peak": float(panel_cai_peak),
        "per_hold": panel_eval,
        "tolerance_metric": (
            "Per-timepoint: divergence(b,n,peak) = |b-n| / max(|b|, |n|, 0.1*peak); "
            "step pass: > 80% of timepoints clear divergence ≤ 0.05; "
            "panel pass: > 80% of holds clear step pass."
        ),
    }


# ---------------------------------------------------------------------------
# Drivers (per subcheckpoint)
# ---------------------------------------------------------------------------

def run_cadiff_validation() -> dict:
    """CP1.B.5: validate cadiff against NEURON."""
    print("=== CP1.B.5: cadiff validation ===")
    print("Running NEURON [cca1 + cadiff] vclamp protocol...")
    nrn_result = neuron_run_capool(
        ca_pool="cadiff",
        pool_kwargs={"depth": 0.1, "beta": 1.0},
        ca_channel="cca1",
        ca_channel_gbar_Scm2=0.7,
        surf_cm2=65.89e-8,
        cm_uFcm2=1.0,
        holding_potentials=[-60.0, -30.0, 0.0, 30.0],
        duration_ms=200.0,
        dt_ms=0.025,
    )
    print(f"  NEURON: ran {len(nrn_result['holds'])} holds")
    for h in nrn_result["holds"]:
        ica_pk = max(np.abs(h["ica_mAcm2"]))
        cai_pk = max(np.abs(h["cai"]))
        cai_init = h["cai"][0]
        cai_final = h["cai"][-1]
        print(f"    hold={h['hold_mV']:+6.1f} mV  ica_peak={ica_pk:.3e}  "
              f"cai_init={cai_init:.3e}  cai_peak={cai_pk:.3e}  cai_final={cai_final:.3e}")

    print("Replaying ica into Brian2 cadiff...")
    factory_fn = lambda: cadiff_brian2_factory(
        depth_um=0.1, beta_per_ms=1.0, cai_floor_mM=1e-4, cai_init_mM=1e-4,
    )
    comparison = compare_capool(nrn_result, factory_fn, dt_ms=0.025)
    print(f"  panel_pass={comparison['panel_pass']}  "
          f"holds_passing={comparison['n_holds_passing']}/{comparison['n_holds']}")
    for e in comparison["per_hold"]:
        print(f"    hold={e['hold_mV']:+6.1f} mV  pass_frac={e['fraction_passing']:.2f}  "
              f"max_div={e['max_divergence']:.3f}  "
              f"nrn_final={e['neuron_cai_final']:.3e}  brian2_final={e['brian2_cai_final']:.3e}")
    return comparison


def run_caintra1_validation() -> dict:
    """CP1.B.6: validate caintra1 against NEURON."""
    print("=== CP1.B.6: caintra1 validation ===")
    print("Running NEURON [cca1 + caintra1] vclamp protocol...")
    # Use AIY caintra1 defaults; vol/surf must match for the pool to be sensible
    surf = 65.89e-8
    vol = 7.42e-12
    nrn_result = neuron_run_capool(
        ca_pool="caintra1",
        pool_kwargs={"vol": vol, "surf": surf},
        ca_channel="cca1",
        ca_channel_gbar_Scm2=0.7,
        surf_cm2=surf,
        cm_uFcm2=1.0,
        holding_potentials=[-60.0, -30.0, 0.0, 30.0],
        duration_ms=200.0,
        dt_ms=0.025,
    )
    print(f"  NEURON: ran {len(nrn_result['holds'])} holds")
    for h in nrn_result["holds"]:
        ica_pk = max(np.abs(h["ica_mAcm2"]))
        cai_pk = max(np.abs(h["cai"]))
        cai_init = h["cai"][0]
        cai_final = h["cai"][-1]
        print(f"    hold={h['hold_mV']:+6.1f} mV  ica_peak={ica_pk:.3e}  "
              f"cai_init={cai_init:.3e}  cai_peak={cai_pk:.3e}  cai_final={cai_final:.3e}")

    print("Replaying ica into Brian2 caintra1...")
    factory_fn = lambda: caintra1_brian2_factory(
        vol_cm3=vol, surf_cm2=surf, fca=0.001, tca_ms=50.0, ca_eq_mM=5e-8,
    )
    comparison = compare_capool(nrn_result, factory_fn, dt_ms=0.025)
    print(f"  panel_pass={comparison['panel_pass']}  "
          f"holds_passing={comparison['n_holds_passing']}/{comparison['n_holds']}")
    for e in comparison["per_hold"]:
        print(f"    hold={e['hold_mV']:+6.1f} mV  pass_frac={e['fraction_passing']:.2f}  "
              f"max_div={e['max_divergence']:.3f}  "
              f"nrn_final={e['neuron_cai_final']:.3e}  brian2_final={e['brian2_cai_final']:.3e}")
    return comparison


def run_combined_capool_validation() -> dict:
    """CP1.B.7: combined Ca-pool — caintra1 driven by EGL-19 (CP3 prefab)."""
    print("=== CP1.B.7: combined Ca-pool (caintra1 + EGL-19 driver) ===")
    surf = 1123.84e-8  # AVAL surf
    vol = 129.6e-12
    nrn_result = neuron_run_capool(
        ca_pool="caintra1",
        pool_kwargs={"vol": vol, "surf": surf},
        ca_channel="egl19",
        ca_channel_gbar_Scm2=0.0929,  # AVAL EGL-19 g rescaled
        surf_cm2=surf,
        cm_uFcm2=1.0,
        holding_potentials=[-60.0, -30.0, 0.0, 30.0],
        duration_ms=300.0,
        dt_ms=0.025,
    )
    print(f"  NEURON: ran {len(nrn_result['holds'])} holds")
    for h in nrn_result["holds"]:
        ica_pk = max(np.abs(h["ica_mAcm2"]))
        cai_pk = max(np.abs(h["cai"]))
        cai_init = h["cai"][0]
        cai_final = h["cai"][-1]
        print(f"    hold={h['hold_mV']:+6.1f} mV  ica_peak={ica_pk:.3e}  "
              f"cai_init={cai_init:.3e}  cai_peak={cai_pk:.3e}  cai_final={cai_final:.3e}")

    print("Replaying ica into Brian2 caintra1...")
    factory_fn = lambda: caintra1_brian2_factory(
        vol_cm3=vol, surf_cm2=surf, fca=0.001, tca_ms=50.0, ca_eq_mM=5e-8,
    )
    comparison = compare_capool(nrn_result, factory_fn, dt_ms=0.025)
    print(f"  panel_pass={comparison['panel_pass']}  "
          f"holds_passing={comparison['n_holds_passing']}/{comparison['n_holds']}")
    for e in comparison["per_hold"]:
        print(f"    hold={e['hold_mV']:+6.1f} mV  pass_frac={e['fraction_passing']:.2f}  "
              f"max_div={e['max_divergence']:.3f}  "
              f"nrn_final={e['neuron_cai_final']:.3e}  brian2_final={e['brian2_cai_final']:.3e}")
    return comparison


def main() -> int:
    results = {}

    print("\n" + "="*60)
    cadiff_result = run_cadiff_validation()
    results["cadiff"] = cadiff_result
    print()

    print("="*60)
    caintra1_result = run_caintra1_validation()
    results["caintra1"] = caintra1_result
    print()

    print("="*60)
    combined_result = run_combined_capool_validation()
    results["combined"] = combined_result
    print()

    print("="*60)
    print("SUMMARY")
    for k, v in results.items():
        print(f"  {k:12s}: panel_pass={v['panel_pass']}  "
              f"frac={v['fraction_holds_passing']:.2f}")

    # Save results
    out_path = Path(__file__).parent / "artifacts" / "calcium_pool_validation_results.json"
    # Strip large per-hold trace data before saving
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            "panel_pass": v["panel_pass"],
            "n_holds": v["n_holds"],
            "n_holds_passing": v["n_holds_passing"],
            "fraction_holds_passing": v["fraction_holds_passing"],
            "panel_cai_peak": v["panel_cai_peak"],
            "tolerance_metric": v["tolerance_metric"],
            "per_hold_summary": [
                {kk: vv for kk, vv in e.items()
                 if not isinstance(vv, list)}
                for e in v["per_hold"]
            ],
        }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Wrote: {out_path}")

    overall_pass = all(v["panel_pass"] for v in results.values())
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
