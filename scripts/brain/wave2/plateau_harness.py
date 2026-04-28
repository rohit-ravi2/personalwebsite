#!/usr/bin/env python3
"""Phase α Deliverable 5 — Brian2 current-clamp plateau harness (Gate 2b).

Distinct from voltage_clamp_harness.py because Gate 2b tests *plateau dynamics*
(amplitude, duration, termination on stim release) — not steady-state IV. This
is the architectural-sufficiency probe per Wave 2 architectural plan condition
6 (channels-correct, architecture-insufficient).

The release-dynamics diagnostic is the load-bearing piece: it distinguishes
SLO-1-mediated termination (correct) from leak-τ-dominated collapse
(architectural-insufficiency signature). Phase α has no real imported
channels to test — instead this module ships two synthetic scaffolds:

  1. `passing_scaffold_factory()` — a deliberately-constructed Brian2 cell
     that satisfies Gate 2b targets (plateau 20 mV, 600 ms, SLO-1-style
     termination on release).
  2. `failing_scaffold_factory()` — a leak-only cell whose plateau collapses
     within τ_m on stim release (the architectural-insufficiency signature).

The harness must classify scaffold #1 as pass and #2 as fail. If it
mis-classifies either, the harness is buggy (not the user's fault per
spec).

Usage:
    from plateau_harness import (
        current_clamp_plateau,
        passing_scaffold_factory,
        failing_scaffold_factory,
    )
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def current_clamp_plateau(
    brian2_factory,
    stim_amp_pA: float = 30.0,
    stim_duration_ms: float = 800.0,
    total_duration_ms: float = 2000.0,
    dt_ms: float = 0.025,
    targets: dict | None = None,
    release_test_at_ms: float | None = None,
) -> dict:
    """Run current-clamp plateau protocol; report measurements + diagnostic.

    Args:
        brian2_factory: callable() -> dict with keys:
            'group', 'monitor', 'network', 'inject_pA' (callable taking pA).
            Monitor must record `v` (volt).
        stim_amp_pA: injection amplitude during plateau-driving phase.
        stim_duration_ms: how long stim is on during the main protocol.
        total_duration_ms: total simulation length (post-stim window matters).
        dt_ms: timestep.
        targets: pass-criteria dict, default uses Mellem 2008 AVA values:
            {'amplitude_mV': (15, 25), 'duration_ms': (400, 800),
             'baseline_settle_mV': 5.0}
        release_test_at_ms: if set, also runs a release-test variant where
            stim is removed at this time (should be <= stim_duration_ms);
            the release-dynamics diagnostic is computed from this run.

    Returns:
        dict with keys:
            pass: bool — overall pass on amp + duration targets
            measured: {amplitude_mV, duration_ms, baseline_post_mV}
            release_dynamics: {tau_release_ms, ratio_to_leak_tau,
                              architectural_signature: 'sufficient'|'leak_dominated'}
            v_trace: np.ndarray (downsampled), t_trace: np.ndarray
    """
    from brian2 import ms, mV, pA  # noqa: WPS433

    if targets is None:
        targets = {
            "amplitude_mV": (15.0, 25.0),
            "duration_ms": (400.0, 800.0),
            "baseline_settle_mV": 5.0,
        }

    # ---- Main protocol: full stim_duration injection ----
    bundle = brian2_factory()
    net = bundle["network"]
    from brian2 import defaultclock
    defaultclock.dt = dt_ms * ms

    # Pre-stim baseline
    bundle["inject_pA"](0.0)
    net.run(100 * ms)
    # Plateau drive
    bundle["inject_pA"](stim_amp_pA)
    net.run(stim_duration_ms * ms)
    # Post-stim window
    bundle["inject_pA"](0.0)
    net.run((total_duration_ms - 100.0 - stim_duration_ms) * ms)

    mon = bundle["monitor"]
    v_arr = np.asarray(mon.v[0]) * 1e3  # to mV
    t_arr = np.asarray(mon.t) * 1e3      # to ms

    # ---- Measurements ----
    # Pre-stim baseline = mean over first 100 ms
    pre_idx = t_arr < 100.0
    baseline_pre = float(np.mean(v_arr[pre_idx])) if pre_idx.any() else float(v_arr[0])

    # Plateau amplitude = mean v during stim minus baseline
    plateau_idx = (t_arr >= 100.0 + 50.0) & (t_arr < 100.0 + stim_duration_ms - 50.0)
    plateau_v = float(np.mean(v_arr[plateau_idx])) if plateau_idx.any() else baseline_pre
    amplitude_mV = plateau_v - baseline_pre

    # Plateau duration: time from first crossing baseline+5mV to falling back
    threshold = baseline_pre + 5.0
    above = v_arr > threshold
    if above.any():
        first = int(np.argmax(above))
        # find last contiguous above sample within stim+post window
        last = int(len(above) - 1 - np.argmax(above[::-1]))
        duration_ms = float(t_arr[last] - t_arr[first])
    else:
        duration_ms = 0.0

    # Post-stim settle: mean v in last 200 ms relative to baseline
    post_idx = t_arr > total_duration_ms - 200.0
    baseline_post = float(np.mean(v_arr[post_idx])) if post_idx.any() else float(v_arr[-1])
    settle_offset = abs(baseline_post - baseline_pre)

    measured = {
        "baseline_pre_mV": baseline_pre,
        "plateau_v_mV": plateau_v,
        "amplitude_mV": amplitude_mV,
        "duration_ms": duration_ms,
        "baseline_post_mV": baseline_post,
        "settle_offset_mV": settle_offset,
    }

    amp_lo, amp_hi = targets["amplitude_mV"]
    dur_lo, dur_hi = targets["duration_ms"]
    pass_amp = amp_lo <= amplitude_mV <= amp_hi
    pass_dur = dur_lo <= duration_ms <= dur_hi
    pass_settle = settle_offset <= targets["baseline_settle_mV"]
    overall_pass = bool(pass_amp and pass_dur and pass_settle)

    measured["pass_amp"] = pass_amp
    measured["pass_dur"] = pass_dur
    measured["pass_settle"] = pass_settle

    # ---- Release-dynamics diagnostic ----
    release_t_ms = release_test_at_ms if release_test_at_ms is not None else 300.0
    release = _release_test(brian2_factory, stim_amp_pA, release_t_ms,
                            total_duration_ms=release_t_ms + 700.0,
                            dt_ms=dt_ms)

    return {
        "pass": overall_pass,
        "measured": measured,
        "release_dynamics": release,
        "v_trace_mV": v_arr,
        "t_trace_ms": t_arr,
    }


# ---------------------------------------------------------------------------
# Release-dynamics test
# ---------------------------------------------------------------------------

def _release_test(brian2_factory, stim_amp_pA: float, release_at_ms: float,
                  total_duration_ms: float = 1000.0, dt_ms: float = 0.025) -> dict:
    """Inject stim until release_at_ms, then drop to 0; measure decay τ.

    Compares decay τ against the cell's pure-leak τ_m (extracted from the
    bundle if the factory exposes `tau_m_ms`, else estimated). If decay
    τ matches τ_m closely, plateau termination is leak-dominated
    (architecturally insufficient). If decay τ is substantially shorter
    than τ_m (active termination), architecture is sufficient.
    """
    from brian2 import ms, mV  # noqa: WPS433
    from brian2 import defaultclock

    bundle = brian2_factory()
    net = bundle["network"]
    defaultclock.dt = dt_ms * ms

    bundle["inject_pA"](0.0)
    net.run(50 * ms)
    bundle["inject_pA"](stim_amp_pA)
    net.run(release_at_ms * ms)
    # Release
    bundle["inject_pA"](0.0)
    net.run((total_duration_ms - release_at_ms - 50.0) * ms)

    mon = bundle["monitor"]
    v_arr = np.asarray(mon.v[0]) * 1e3
    t_arr = np.asarray(mon.t) * 1e3

    # Find release index (just after stim drop, t = 50 + release_at_ms)
    release_t = 50.0 + release_at_ms
    rel_mask = t_arr >= release_t
    if not rel_mask.any():
        return {"error": "release window not reached"}
    rel_idx = int(np.argmax(rel_mask))
    v_at_release = float(v_arr[rel_idx])
    # Asymptotic baseline = mean of last 100 ms
    v_baseline = float(np.mean(v_arr[-int(100 / dt_ms):]))

    # Fit single-exponential decay τ from rel_idx forward.
    # v(t) = v_baseline + (v_at_release - v_baseline) * exp(-(t-t0)/tau)
    v_seg = v_arr[rel_idx:]
    t_seg = t_arr[rel_idx:] - release_t
    delta = v_seg - v_baseline
    # Avoid log of zero / negatives by selecting samples with same sign
    if abs(delta[0]) < 0.5:
        return {
            "tau_release_ms": np.nan,
            "v_at_release_mV": v_at_release,
            "v_baseline_mV": v_baseline,
            "architectural_signature": "no_plateau_to_release_from",
        }
    sign0 = np.sign(delta[0])
    valid = sign0 * delta > max(0.05 * abs(delta[0]), 1e-6)
    if valid.sum() < 5:
        return {
            "tau_release_ms": np.nan,
            "v_at_release_mV": v_at_release,
            "v_baseline_mV": v_baseline,
            "architectural_signature": "insufficient_release_data",
        }
    log_delta = np.log(np.abs(delta[valid]))
    coeffs = np.polyfit(t_seg[valid], log_delta, 1)
    tau_release_ms = float(-1.0 / coeffs[0]) if coeffs[0] != 0 else float("inf")

    # Compare to leak τ_m if factory exposes it
    tau_m_ms = float(bundle.get("tau_m_ms", np.nan))
    if not np.isfinite(tau_m_ms):
        # Estimate from the trace: a leak-only cell would decay to baseline
        # with the same τ as its tau_m; if we don't know it, default to
        # a heuristic comparison threshold.
        ratio = float("nan")
        signature = "unknown_tau_m"
    else:
        ratio = tau_release_ms / tau_m_ms if tau_m_ms > 0 else float("inf")
        # Per Wave 2 spec: if release-tau is dominated by leak, ratio ≈ 1.
        # If active termination (SLO-1) accelerates collapse, ratio < 0.5.
        # If no termination machinery and v stays at plateau, ratio >> 1.
        if ratio < 0.6:
            signature = "active_termination"  # architecturally sufficient
        elif ratio < 1.4:
            signature = "leak_dominated"  # architecturally insufficient
        else:
            signature = "no_termination"  # plateau persists

    return {
        "tau_release_ms": tau_release_ms,
        "tau_m_ms": tau_m_ms,
        "ratio": ratio,
        "v_at_release_mV": v_at_release,
        "v_baseline_mV": v_baseline,
        "architectural_signature": signature,
    }


# ---------------------------------------------------------------------------
# Synthetic scaffolds for smoke testing
# ---------------------------------------------------------------------------

def passing_scaffold_factory():
    """A deliberately-constructed cell that satisfies Gate 2b targets.

    Implements a phenomenological scaffold:
      - leak with τ_m = 50 ms, v_rest = -65 mV
      - a fast "Ca-like" depolarizing drive activated by injected current
        (gate s rises quickly when I_inject > 0)
      - a delayed "SLO-1-like" K+ termination current activated by `s`
        with a slow rise τ_K = 250 ms — this gives plateau ~600 ms
      - on stim release, s decays fast (~30 ms) → I_K decays away with
        τ_K, so plateau collapses faster than leak τ_m alone.
    Targets: plateau ~20 mV, duration ~500-700 ms, settle within 5 mV.
    """
    def _factory():
        from brian2 import (  # noqa: WPS433
            NeuronGroup, StateMonitor, Network, ms, mV, nS, pA, amp, second,
            prefs, start_scope,
        )
        start_scope()
        prefs.codegen.target = "cython"

        # Note: s_inf is a smooth saturation of I_inject above ~1 pA.
        # Using `1/(1+exp(-(I-Ihalf)/Ik))` gives 0 for I≤0 and ~1 for I≥5pA,
        # without requiring relational expressions in Brian2 equations.
        eqs = """
        dv/dt = (-(v - v_rest) / tau_m
                 + g_drive * s * (E_drive - v) / C_mem
                 - g_term * w * (v - E_K) / C_mem
                 + I_inject / C_mem) : volt
        ds/dt = (s_inf - s) / tau_s : 1
        s_inf = 1.0 / (1.0 + exp(-(I_inject/pA - 1.0) / 0.5)) : 1
        dw/dt = (w_inf - w) / tau_w : 1
        w_inf = s : 1
        I_inject : amp
        v_rest : volt
        tau_m : second
        tau_s : second
        tau_w : second
        g_drive : siemens
        g_term : siemens
        E_drive : volt
        E_K : volt
        C_mem : farad
        """
        G = NeuronGroup(1, eqs, method="euler")
        G.v_rest = -65 * mV
        G.tau_m = 50 * ms
        G.tau_s = 10 * ms
        G.tau_w = 250 * ms
        G.g_drive = 4 * nS
        G.g_term = 8 * nS
        G.E_drive = 50 * mV
        G.E_K = -90 * mV
        from brian2 import pF
        G.C_mem = 100 * pF
        G.v = -65 * mV
        G.s = 0
        G.w = 0
        G.I_inject = 0 * pA

        mon = StateMonitor(G, ["v"], record=True)
        net = Network(G, mon)

        def inject_pA(amp_pA: float) -> None:
            G.I_inject = amp_pA * pA

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "inject_pA": inject_pA,
            "tau_m_ms": 50.0,
        }

    return _factory


def failing_scaffold_factory():
    """A leak-only cell whose plateau collapses with τ_m on stim release.

    No active termination, no Ca-like dynamics. With sufficient injection
    current, leak alone produces a small steady offset (g · I / leak), but
    NO plateau in the Mellem sense (sustained 20 mV depolarization for
    600 ms). On release, V decays back to v_rest with τ_m = 10 ms.

    The harness should classify this as fail on amplitude OR duration AND
    label release_dynamics as 'leak_dominated'.
    """
    def _factory():
        from brian2 import (  # noqa: WPS433
            NeuronGroup, StateMonitor, Network, ms, mV, nS, pA, amp, second,
            prefs, start_scope, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        eqs = """
        dv/dt = (-(v - v_rest) / tau_m + I_inject / C_mem) : volt
        I_inject : amp
        v_rest : volt
        tau_m : second
        C_mem : farad
        """
        G = NeuronGroup(1, eqs, method="euler")
        G.v_rest = -65 * mV
        G.tau_m = 10 * ms
        G.C_mem = 100 * pF
        G.v = -65 * mV
        G.I_inject = 0 * pA

        mon = StateMonitor(G, ["v"], record=True)
        net = Network(G, mon)

        def inject_pA(amp_pA: float) -> None:
            G.I_inject = amp_pA * pA

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "inject_pA": inject_pA,
            "tau_m_ms": 10.0,
        }

    return _factory


# ---------------------------------------------------------------------------
# Phase β CP1.A.3 — Layer A current-clamp comparison (Brian2 vs NEURON ref)
# ---------------------------------------------------------------------------

def current_clamp_layer_a_compare(
    brian2_factory,
    neuron_reference,
    cell_name: str,
    injection_pa: float = 50.0,
    injection_duration_ms: float = 100.0,
    settle_ms: float = 200.0,
    post_ms: float = 1500.0,
    v_rest_mv: float = -25.0,
    voltage_feature_tolerance_mV: float = 3.0,
    timepoint_pass_fraction: float = 0.8,
    dt_ms: float = 0.025,
) -> dict:
    """Layer A comparison: Brian2 cell vs NEURON reference under same CC protocol.

    Runs the same current-clamp protocol on a Brian2 factory-built cell and a
    NEURONReference, extracts voltage features per timepoint, and computes the
    voltage-feature gate (≤ voltage_feature_tolerance_mV residual at peak +
    plateau, > timepoint_pass_fraction of timepoints clear).

    Timing features (time-to-peak, settling) are reported as warn-only diagnostics.

    Parameters
    ----------
    brian2_factory : callable
        Factory matching plateau_harness convention: returns dict with
        'group', 'monitor', 'network', 'inject_pA' (callable).
    neuron_reference : NEURONReference
        Instance with .current_clamp(...) method.
    cell_name : str
        Label for output (e.g., "AVA_egl19_only").
    injection_pa, injection_duration_ms, settle_ms, post_ms, v_rest_mv :
        Protocol parameters (same on both cells).
    voltage_feature_tolerance_mV : float
        Per-feature pass: |brian2_v - neuron_v| ≤ tol at peak + plateau.
    timepoint_pass_fraction : float
        Per-panel: fraction of timepoints (peak + plateau across the trace
        comparison window) that must clear the tolerance.
    dt_ms : float
        Brian2 timestep.

    Returns
    -------
    dict with keys:
        panel_pass : bool
        cell : str
        protocol : dict
        brian2_features : {peak_V_mV, plateau_V_mV, baseline_pre_mV, ...}
        neuron_features : same shape
        feature_residuals : {peak_V_mV: float, plateau_V_mV: float}
        n_timepoints, n_timepoints_passing, fraction_passing
        timing_diagnostics : {time_to_peak_residual_ms, settling_time_residual_ms}
        warnings : list
    """
    from brian2 import ms

    warnings_ = []

    # ---- Brian2 run ----
    bundle = brian2_factory()
    net = bundle["network"]
    from brian2 import defaultclock
    defaultclock.dt = dt_ms * ms

    # Initialize at v_rest_mv if factory supports it
    if "set_v" in bundle:
        bundle["set_v"](v_rest_mv)

    bundle["inject_pA"](0.0)
    net.run(settle_ms * ms)
    bundle["inject_pA"](injection_pa)
    net.run(injection_duration_ms * ms)
    bundle["inject_pA"](0.0)
    net.run(post_ms * ms)

    mon = bundle["monitor"]
    v_b = np.asarray(mon.v[0]) * 1e3  # mV
    t_b = np.asarray(mon.t) * 1e3      # ms
    t_b_aligned = t_b - settle_ms      # 0 = stim onset

    # NaN/Inf guard
    if not np.all(np.isfinite(v_b)):
        warnings_.append("Brian2 produced non-finite V; replacing with nan_to_num")
        v_b = np.nan_to_num(v_b)

    # ---- NEURON reference run ----
    ref_result = neuron_reference.current_clamp(
        injection_pa=injection_pa,
        injection_duration_ms=injection_duration_ms,
        settle_ms=settle_ms,
        post_ms=post_ms,
        v_rest_mv=v_rest_mv,
    )
    sweep = ref_result["sweeps"][0]
    t_n = np.asarray(sweep["t_ms"])  # already aligned: 0 = stim onset
    v_n = np.asarray(sweep["V_mV"])
    if not np.all(np.isfinite(v_n)):
        warnings_.append("NEURON produced non-finite V")
        v_n = np.nan_to_num(v_n)

    # ---- Feature extraction ----
    def _features(t_arr: np.ndarray, v_arr: np.ndarray) -> dict:
        pre_mask = t_arr < 0
        step_mask = (t_arr >= 0) & (t_arr < injection_duration_ms)
        post_mask = t_arr >= injection_duration_ms

        baseline_pre = float(np.mean(v_arr[pre_mask])) if pre_mask.any() else float(v_arr[0])
        baseline_post = float(np.mean(v_arr[post_mask][-max(1, int(0.2*post_mask.sum())):])) if post_mask.any() else float(v_arr[-1])

        if step_mask.any():
            v_step = v_arr[step_mask]
            t_step = t_arr[step_mask]
            delta = v_step - baseline_pre
            peak_idx = int(np.argmax(np.abs(delta)))
            peak_V_mV = float(v_step[peak_idx])
            time_to_peak_ms = float(t_step[peak_idx])
            n_plat = max(1, int(0.2 * len(v_step)))
            plateau_V_mV = float(np.median(v_step[-n_plat:]))
        else:
            peak_V_mV = baseline_pre
            time_to_peak_ms = 0.0
            plateau_V_mV = baseline_pre

        # Settling time: time post-stim to return within 5 mV of baseline_pre
        if post_mask.any():
            v_post = v_arr[post_mask]
            t_post = t_arr[post_mask]
            within = np.abs(v_post - baseline_pre) <= 5.0
            settling_time_ms = float(t_post[np.argmax(within)] - injection_duration_ms) if within.any() else float(t_post[-1] - injection_duration_ms)
        else:
            settling_time_ms = 0.0

        return {
            "baseline_pre_mV": baseline_pre,
            "peak_V_mV": peak_V_mV,
            "plateau_V_mV": plateau_V_mV,
            "baseline_post_mV": baseline_post,
            "time_to_peak_ms": time_to_peak_ms,
            "settling_time_ms": settling_time_ms,
        }

    brian2_features = _features(t_b_aligned, v_b)
    neuron_features = _features(t_n, v_n)

    # ---- Voltage-feature residuals ----
    feature_residuals = {
        "peak_V_mV": abs(brian2_features["peak_V_mV"] - neuron_features["peak_V_mV"]),
        "plateau_V_mV": abs(brian2_features["plateau_V_mV"] - neuron_features["plateau_V_mV"]),
        "baseline_pre_mV": abs(brian2_features["baseline_pre_mV"] - neuron_features["baseline_pre_mV"]),
    }

    # ---- Per-timepoint pass: interpolate Brian2 onto NEURON's time grid,
    # compute |v_b - v_n| at each timepoint, fraction within tolerance.
    # Restrict to the comparison window: -settle_ms to +injection_duration_ms+post_ms
    common_t_start = max(t_b_aligned[0], t_n[0])
    common_t_end = min(t_b_aligned[-1], t_n[-1])
    common_t_grid = np.linspace(common_t_start, common_t_end, 1000)
    v_b_interp = np.interp(common_t_grid, t_b_aligned, v_b)
    v_n_interp = np.interp(common_t_grid, t_n, v_n)
    residuals_mV = np.abs(v_b_interp - v_n_interp)
    n_timepoints = len(residuals_mV)
    n_passing = int(np.sum(residuals_mV <= voltage_feature_tolerance_mV))
    fraction_passing = n_passing / n_timepoints

    # Feature-level pass: peak + plateau both ≤ tolerance
    feature_pass = (feature_residuals["peak_V_mV"] <= voltage_feature_tolerance_mV
                    and feature_residuals["plateau_V_mV"] <= voltage_feature_tolerance_mV)
    panel_pass = bool(feature_pass and fraction_passing >= timepoint_pass_fraction)

    timing_diagnostics = {
        "time_to_peak_residual_ms": abs(brian2_features["time_to_peak_ms"] - neuron_features["time_to_peak_ms"]),
        "settling_time_residual_ms": abs(brian2_features["settling_time_ms"] - neuron_features["settling_time_ms"]),
    }

    return {
        "panel_pass": panel_pass,
        "cell": cell_name,
        "protocol": {
            "injection_pa": injection_pa,
            "injection_duration_ms": injection_duration_ms,
            "settle_ms": settle_ms,
            "post_ms": post_ms,
            "v_rest_mv": v_rest_mv,
            "voltage_feature_tolerance_mV": voltage_feature_tolerance_mV,
            "timepoint_pass_fraction": timepoint_pass_fraction,
            "dt_ms": dt_ms,
        },
        "brian2_features": brian2_features,
        "neuron_features": neuron_features,
        "feature_residuals": feature_residuals,
        "feature_pass": feature_pass,
        "n_timepoints": n_timepoints,
        "n_timepoints_passing": n_passing,
        "fraction_passing": float(fraction_passing),
        "timing_diagnostics": timing_diagnostics,
        "warnings": warnings_,
        "v_brian2_mV": v_b.tolist(),
        "t_brian2_ms": t_b_aligned.tolist(),
        "v_neuron_mV": v_n.tolist(),
        "t_neuron_ms": t_n.tolist(),
    }


# ---------------------------------------------------------------------------
# Phase β refactor flags
# ---------------------------------------------------------------------------
#
# 1. The plateau-amplitude measurement uses simple mean-during-stim, which
#    is biased low if the plateau plays out over a long onset transient.
#    For Phase γ Gate 2b proper, use median over a window starting after
#    the onset transient settles (~200 ms post stim onset).
#
# 2. The duration measurement uses a fixed +5 mV threshold over baseline.
#    Mellem 2008 likely uses a different operationalization (probably
#    half-max-amplitude crossings). Re-check before Gate 2b lock-in.
#
# 3. Release-tau exponential fit assumes monotone decay; biexponential
#    cases (slow K_Ca + fast leak) will be poorly fit. Switch to
#    biexponential or to time-to-half-decay if biexponential dominance
#    appears in real channel imports.
#
# 4. The architectural_signature classifier uses ratio thresholds (0.6,
#    1.4). These were chosen to cleanly distinguish the synthetic
#    scaffolds — empirical ratio values from real imported channels may
#    suggest different thresholds.
#
# 5. The pass-vs-fail criterion is currently amp ∧ dur ∧ settle. For
#    Gate 2b proper, weight by per-criterion margin not all-or-nothing.
#
# 6. For Phase β, factor synthetic scaffolds into a separate
#    `_synthetic_scaffolds.py` so this file holds only the harness.
