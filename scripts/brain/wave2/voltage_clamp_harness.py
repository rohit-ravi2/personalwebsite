#!/usr/bin/env python3
"""Voltage-clamp validation harness (Gate 2a).

Phase α deliverable 4 + Phase β CP1.A.2 iteration. Compares Brian2 voltage-clamp
output against a reference (analytic or NEURON) using two tolerance modes:

  * **legacy** (Phase α): single 5%-with-1e-9-floor relative tolerance on SS
    current. Inherits the small-denominator pathology v1/v2/v3 hit three times.
    Retained for backward-compat with existing smoke tests.

  * **current_domain** (Phase β CP1.A.2 — recommended): current-domain analog
    of v3's voltage-feature gate. Per-feature divergence formula:

        divergence(a, b, peak) = |a-b| / max(|a|, |b|, 0.1 * peak)

    Per-feature pass: divergence ≤ 0.05.
    Per-panel pass: > 80% of holding potentials clear ALL features.
    Floor (0.1 * peak) prevents the small-denominator pathology when the
    channel is inactive at some holds.

Conventions:
  * Brian2 model is a NeuronGroup (single neuron) whose `v` state can be
    forced via a network_operation each timestep.
  * NEURON reference is either an analytic callable (legacy `(hold,dur,dt) →
    (t,V,I)`) or a `NEURONReference.voltage_clamp(...)` dict consumer.

Usage:
    from voltage_clamp_harness import (
        voltage_clamp_compare,             # legacy interface
        voltage_clamp_compare_v2,          # CP1.A.2 current-domain
        current_domain_divergence,
        leak_brian2_factory,
        leak_analytic_reference,
    )
"""
from __future__ import annotations

import numpy as np

# Brian2 imports are local to the harness functions so this module can be
# imported even before Brian2 is fully configured.


# ---------------------------------------------------------------------------
# Current-domain tolerance metric (Phase β CP1.A.2)
# ---------------------------------------------------------------------------

def current_domain_divergence(a: float, b: float, peak: float) -> float:
    """Compute the v3-analog tolerance metric on currents.

    div(a, b, peak) = |a - b| / max(|a|, |b|, 0.1 * peak)

    The third term in the denominator prevents pathological small-denominator
    blow-ups when both currents are near zero (e.g., a Ca channel held at the
    reversal potential, or below activation threshold). It places an absolute
    floor on the comparison sensitivity at 10% of the panel-wide peak current.

    Symmetric in a, b. Returns 0 if a == b == peak == 0.
    """
    a = float(a)
    b = float(b)
    peak = float(abs(peak))
    denom = max(abs(a), abs(b), 0.1 * peak)
    if denom == 0.0:
        return 0.0
    return abs(a - b) / denom


def evaluate_current_domain_panel(
    per_step: list[dict],
    feature_keys: tuple[str, ...] = ("peak_I_pA", "ss_I_pA"),
    feature_tolerance: float = 0.05,
    panel_pass_fraction: float = 0.8,
) -> dict:
    """Apply the current-domain divergence gate to a list of per-hold features.

    Parameters
    ----------
    per_step : list of dicts
        Each dict must contain `brian2_<key>` and `ref_<key>` for each
        key in feature_keys. (E.g., 'brian2_peak_I_pA' and 'ref_peak_I_pA'.)
    feature_keys : tuple
        Which features to evaluate (default: peak + steady-state currents).
    feature_tolerance : float
        Per-feature pass threshold (default 0.05 — i.e., 5%).
    panel_pass_fraction : float
        Fraction of holds that must clear ALL features (default 0.8).

    Returns
    -------
    dict with keys:
        panel_pass : bool
        n_holds : int
        n_holds_passing : int
        fraction_passing : float
        per_step_evaluations : list of per-hold detail dicts
        per_feature_peak : dict (peak current per feature, used as floor)
    """
    n_holds = len(per_step)
    if n_holds == 0:
        return {
            "panel_pass": False,
            "n_holds": 0,
            "n_holds_passing": 0,
            "fraction_passing": 0.0,
            "per_step_evaluations": [],
            "per_feature_peak": {},
        }

    # Compute per-feature peak (max abs over all holds, both Brian2 and ref).
    per_feature_peak = {}
    for fkey in feature_keys:
        vals = []
        for s in per_step:
            for src in ("brian2_", "ref_"):
                v = s.get(f"{src}{fkey}", 0.0)
                if v is not None and not np.isnan(v):
                    vals.append(abs(float(v)))
        per_feature_peak[fkey] = max(vals) if vals else 0.0

    n_passing = 0
    evaluations = []
    for s in per_step:
        feature_results = {}
        all_pass = True
        for fkey in feature_keys:
            a = s.get(f"brian2_{fkey}", 0.0)
            b = s.get(f"ref_{fkey}", 0.0)
            div = current_domain_divergence(a, b, per_feature_peak[fkey])
            f_pass = div <= feature_tolerance
            feature_results[fkey] = {
                "brian2": float(a) if a is not None else None,
                "ref": float(b) if b is not None else None,
                "divergence": float(div),
                "pass": bool(f_pass),
            }
            if not f_pass:
                all_pass = False
        if all_pass:
            n_passing += 1
        evaluations.append({
            "hold_mV": s.get("hold_mV"),
            "feature_results": feature_results,
            "step_pass": bool(all_pass),
        })

    fraction = n_passing / n_holds
    return {
        "panel_pass": fraction >= panel_pass_fraction,
        "n_holds": n_holds,
        "n_holds_passing": n_passing,
        "fraction_passing": fraction,
        "per_step_evaluations": evaluations,
        "per_feature_peak": per_feature_peak,
        "tolerance_metric": (
            f"divergence(a,b,peak) = |a-b| / max(|a|, |b|, 0.1*peak); "
            f"per-feature pass: divergence ≤ {feature_tolerance}; "
            f"per-panel pass: > {panel_pass_fraction*100:.0f}% of holds clear ALL features"
        ),
    }


def voltage_clamp_compare_v2(
    brian2_factory,
    neuron_reference,
    holding_potentials_mV,
    duration_ms: float = 250.0,
    dt_ms: float = 0.025,
    settle_window_ms: float = 50.0,
    feature_tolerance: float = 0.05,
    panel_pass_fraction: float = 0.8,
    feature_keys: tuple[str, ...] = ("peak_I_pA", "ss_I_pA"),
    skip_initial_transient_ms: float = 2.0,
    brian2_prestep_ms: float = 0.0,
    brian2_prestep_mV: float = -30.0,
) -> dict:
    """CP1.A.2 voltage-clamp comparison with current-domain tolerance metric.

    Differences from `voltage_clamp_compare`:
      * Tolerance metric: current-domain v3-analog with peak floor.
      * Reference: a NEURONReference instance (preferred) OR an analytic
        callable matching the legacy signature. Auto-detected from type.
      * Returns per-feature breakdown including peak + SS currents per hold.
      * Panel-pass criterion: >80% of holds clear all features (not max-only).

    Args:
        brian2_factory: callable() → dict per legacy spec ('group', 'monitor',
            'network', 'set_v').
        neuron_reference: NEURONReference instance OR legacy callable.
        holding_potentials_mV: list of holds.
        duration_ms: simulation duration per hold.
        dt_ms: integration timestep.
        settle_window_ms: window at end of step for SS.
        feature_tolerance, panel_pass_fraction: gate thresholds.
        feature_keys: which features to compare.

    Returns:
        dict with:
            panel_pass : bool
            n_holds, n_holds_passing, fraction_passing
            per_step : list of per-hold {hold_mV, brian2_*, ref_*}
            evaluation : detailed per-feature breakdown
            warnings : list
    """
    from brian2 import ms

    # Determine reference type
    is_neuron_ref = hasattr(neuron_reference, "voltage_clamp")

    # Run NEURON reference once for all holds if it's a NEURONReference instance
    # (single section reuse).
    if is_neuron_ref:
        ref_result = neuron_reference.voltage_clamp(
            holding_potentials=list(holding_potentials_mV),
            duration_ms=duration_ms,
            prestep_ms=brian2_prestep_ms,
            prestep_mV=brian2_prestep_mV,
            tail_ms=0.0,
            dt_ms=dt_ms,
        )
        # NEURONReference's voltage_clamp returns t_ms aligned to step start
        # (already starts at 0 = step onset). We apply the same skip logic
        # below.
        ref_holds = {h["hold_mV"]: h for h in ref_result["holds"]}
    else:
        ref_holds = None

    warnings_ = []
    per_step = []

    for v_hold in holding_potentials_mV:
        # ---- Brian2 run ----
        bundle = brian2_factory()
        net = bundle["network"]
        from brian2 import defaultclock
        defaultclock.dt = dt_ms * ms
        # Optional pre-step
        if brian2_prestep_ms > 0:
            bundle["set_v"](brian2_prestep_mV)
            net.run(brian2_prestep_ms * ms)
        # Main step
        prestep_n_samples = int(brian2_prestep_ms / dt_ms) if brian2_prestep_ms > 0 else 0
        bundle["set_v"](v_hold)
        net.run(duration_ms * ms)
        mon = bundle["monitor"]
        I_arr_full = np.asarray(mon.I_total[0])  # amp
        t_arr_full = np.asarray(mon.t) * 1e3      # ms
        # Trim to step window only (post-prestep)
        I_arr = I_arr_full[prestep_n_samples:]

        # NaN/Inf guard
        if not np.all(np.isfinite(I_arr)):
            warnings_.append(f"Brian2 produced non-finite I at hold {v_hold} mV")
            I_arr = np.nan_to_num(I_arr)

        I_pA = I_arr * 1e12

        # Apply skip_initial_transient_ms to ignore capacitive transient
        skip_n = max(0, int(skip_initial_transient_ms / dt_ms))
        I_pA_post = I_pA[skip_n:]

        # Peak: signed extremum after skip window (matches NEURONReference convention)
        if len(I_pA_post) > 0:
            peak_idx = int(np.argmax(np.abs(I_pA_post)))
            brian2_peak_I_pA = float(I_pA_post[peak_idx])
        else:
            brian2_peak_I_pA = 0.0

        # SS: mean of last settle_window_ms
        n_settle = max(1, int(settle_window_ms / dt_ms))
        brian2_ss_I_pA = float(np.mean(I_pA[-n_settle:]))

        # ---- Reference ----
        if is_neuron_ref:
            href = ref_holds.get(float(v_hold))
            if href is None:
                # Floating-point key lookup fallback
                href = min(ref_holds.values(),
                           key=lambda h: abs(h["hold_mV"] - float(v_hold)))
            # Recompute peak using skip_initial_transient_ms
            t_ref = np.asarray(href["t_ms"])
            i_ref = np.asarray(href["I_total_pA"])
            mask_post = t_ref >= skip_initial_transient_ms
            if mask_post.any():
                i_post = i_ref[mask_post]
                pidx = int(np.argmax(np.abs(i_post)))
                ref_peak_I_pA = float(i_post[pidx])
            else:
                ref_peak_I_pA = href["peak_I_pA"]
            # F15 (run #2): align SS window to Brian2's (last settle_window_ms)
            # — NEURONReference's stored ss_I_pA uses last 20% of step which
            # differs from Brian2's last settle_window_ms. For inactivating
            # channels (e.g., SHL-1) this produces a systematic SS divergence
            # that's a window-difference artifact, not a translation defect.
            ref_dt = float(t_ref[1] - t_ref[0]) if len(t_ref) > 1 else dt_ms
            ref_n_ss = max(1, int(settle_window_ms / ref_dt))
            ref_ss_I_pA = float(np.mean(i_ref[-ref_n_ss:]))
        else:
            ref_t, ref_V, ref_I_pA = neuron_reference(v_hold, duration_ms, dt_ms)
            if not np.all(np.isfinite(ref_I_pA)):
                warnings_.append(f"Reference produced non-finite I at hold {v_hold} mV")
                ref_I_pA = np.nan_to_num(ref_I_pA)
            if len(ref_I_pA) > 0:
                pidx = int(np.argmax(np.abs(ref_I_pA)))
                ref_peak_I_pA = float(ref_I_pA[pidx])
            else:
                ref_peak_I_pA = 0.0
            ref_dt = ref_t[1] - ref_t[0] if len(ref_t) > 1 else dt_ms
            ref_n = max(1, int(settle_window_ms / ref_dt))
            ref_ss_I_pA = float(np.mean(ref_I_pA[-ref_n:]))

        per_step.append({
            "hold_mV": float(v_hold),
            "brian2_peak_I_pA": brian2_peak_I_pA,
            "brian2_ss_I_pA": brian2_ss_I_pA,
            "ref_peak_I_pA": ref_peak_I_pA,
            "ref_ss_I_pA": ref_ss_I_pA,
        })

    eval_result = evaluate_current_domain_panel(
        per_step,
        feature_keys=feature_keys,
        feature_tolerance=feature_tolerance,
        panel_pass_fraction=panel_pass_fraction,
    )

    return {
        "panel_pass": eval_result["panel_pass"],
        "n_holds": eval_result["n_holds"],
        "n_holds_passing": eval_result["n_holds_passing"],
        "fraction_passing": eval_result["fraction_passing"],
        "per_step": per_step,
        "evaluation": eval_result,
        "warnings": warnings_,
        "tolerance_metric": eval_result["tolerance_metric"],
    }


# ---------------------------------------------------------------------------
# Legacy public API (Phase α)
# ---------------------------------------------------------------------------

def voltage_clamp_compare(
    brian2_factory,
    reference,
    holding_potentials_mV,
    duration_ms: float = 200.0,
    dt_ms: float = 0.025,
    settle_window_ms: float = 50.0,
    tolerance: float = 0.05,
) -> dict:
    """Run Brian2 + reference under voltage-clamp; report divergence.

    Args:
        brian2_factory: callable() -> dict with keys:
            'group': Brian2 NeuronGroup (1 neuron) with `v` state;
            'monitor': StateMonitor recording at minimum `v` and a current
                       expression named `I_total` (in amp); the harness
                       converts to pA.
            'network': Brian2 Network containing group + monitor;
            'set_v': callable(v_mV: float) that forces `v` to the holding
                     potential (used to simulate clamp).
        reference: callable(holding_mV, duration_ms, dt_ms)
                   -> (t_ms, V_mV, I_pA) — both arrays of length n_steps.
        holding_potentials_mV: list of holding potentials.
        duration_ms: simulation duration per holding step.
        dt_ms: integration timestep.
        settle_window_ms: window at end of each step to average for SS current.
        tolerance: max-rel-diff threshold for pass/fail.

    Returns:
        dict with keys:
            pass: bool
            max_divergence: float (max relative difference)
            per_step: list of dicts {hold_mV, brian2_I_pA, ref_I_pA, rel_diff}
            warnings: list of str (units/dt mismatches surfaced)
    """
    from brian2 import ms, mV, amp  # noqa: WPS433

    warnings_ = []
    per_step = []

    for v_hold in holding_potentials_mV:
        # ---- Brian2 run ----
        bundle = brian2_factory()
        net = bundle["network"]
        bundle["set_v"](v_hold)
        # In voltage-clamp the harness implements clamp by forcing v at every
        # timestep via a Brian2 network_operation if the factory wires it.
        # Otherwise we infer the SS current from the equation evaluated at
        # the held v (analytic), which the leak factory does directly.
        from brian2 import defaultclock
        defaultclock.dt = dt_ms * ms
        net.run(duration_ms * ms)
        mon = bundle["monitor"]
        # I_total recorded in amp; convert to pA
        I_arr = np.asarray(mon.I_total[0])  # shape: (n_steps,)
        # Take last settle_window_ms samples
        n_settle = max(1, int(settle_window_ms / dt_ms))
        brian2_I_pA = float(np.mean(I_arr[-n_settle:]) * 1e12)

        # ---- Reference run ----
        ref_t, ref_V, ref_I_pA = reference(v_hold, duration_ms, dt_ms)
        ref_settle = max(1, int(settle_window_ms / (ref_t[1] - ref_t[0])))
        ref_I_ss_pA = float(np.mean(ref_I_pA[-ref_settle:]))

        # ---- Compare ----
        scale = max(abs(brian2_I_pA), abs(ref_I_ss_pA), 1e-9)
        rel_diff = abs(brian2_I_pA - ref_I_ss_pA) / scale
        per_step.append({
            "hold_mV": float(v_hold),
            "brian2_I_pA": brian2_I_pA,
            "ref_I_pA": ref_I_ss_pA,
            "rel_diff": rel_diff,
        })

    max_div = max((s["rel_diff"] for s in per_step), default=0.0)
    return {
        "pass": max_div <= tolerance,
        "max_divergence": max_div,
        "per_step": per_step,
        "warnings": warnings_,
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# Reference: leak channel (analytic)
# ---------------------------------------------------------------------------

def leak_analytic_reference(g_leak_nS: float = 1.0, e_leak_mV: float = -70.0):
    """Build an analytic reference function for a leak channel.

    Returns a callable(holding_mV, duration_ms, dt_ms) -> (t, V, I_pA).
    I = g * (V - E_leak); SS is exact analytic value.
    """
    def ref(holding_mV, duration_ms, dt_ms):
        n = max(2, int(duration_ms / dt_ms))
        t = np.linspace(0, duration_ms, n)
        V = np.full(n, holding_mV, dtype=float)
        # I = g * (V - E_leak); g_nS * (mV) = pA. (1 nS × 1 mV = 1 pA)
        I_pA = g_leak_nS * (holding_mV - e_leak_mV)
        return t, V, np.full(n, I_pA, dtype=float)

    return ref


# ---------------------------------------------------------------------------
# Brian2 factory: leak channel under voltage-clamp
# ---------------------------------------------------------------------------

def leak_brian2_factory(g_leak_nS: float = 1.0, e_leak_mV: float = -70.0,
                        v_init_mV: float = -70.0):
    """Build a factory that returns a fresh Brian2 cell for each call.

    The cell's `v` is held constant by a network_operation that resets v
    to a clamp target each timestep. I_total = g_leak * (v - E_leak).
    """
    def _factory():
        from brian2 import (  # noqa: WPS433
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, nS, amp, pA, prefs,
            start_scope,
        )
        start_scope()
        prefs.codegen.target = "cython"

        eqs = """
        dv/dt = (-(v - E_leak) / tau) : volt
        I_total = g_leak * (v - E_leak) : amp
        E_leak : volt
        g_leak : siemens
        tau : second
        """
        G = NeuronGroup(1, eqs, method="euler")
        G.E_leak = e_leak_mV * mV
        G.g_leak = g_leak_nS * nS
        # tau is irrelevant since v is force-clamped, but Brian2 needs it
        G.tau = 10 * ms
        G.v = v_init_mV * mV

        # Holder for clamp target — modified by set_v()
        clamp = {"v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp_v():
            G.v = clamp["v_target_mV"] * mV

        mon = StateMonitor(G, ["v", "I_total"], record=True)
        net = Network(G, mon, _clamp_v)

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


# ---------------------------------------------------------------------------
# Phase β refactor flags (prototype-first hindsight to-do)
# ---------------------------------------------------------------------------
#
# 1. The factory pattern is awkward — every comparison call creates a fresh
#    Brian2 Network, paying scope-init cost per holding step. For Phase β
#    when the actual EGL-19 / SLO-1 channels are translated, prefer a
#    single Network that resets state between holds via a state-restore
#    object rather than re-instantiating.
#
# 2. The clamp implementation (force `v = v_target` each timestep) is fine
#    for steady-state-only checks. For transient capture (e.g., tail
#    currents on step-down), implement Brian2's native LinkedVariable or
#    a dedicated clamp current via a high-conductance virtual electrode.
#
# 3. The reference callable signature `(hold, dur, dt) -> (t, V, I)` is
#    awkward for NEURON — wrapping `h.VClamp` requires section setup and
#    persistent state. For Phase β, define a NEURONReference class that
#    holds the section + mechanism handles and exposes a matching call
#    signature without spinning up a fresh section per hold.
#
# 4. Tolerance is currently a single global; Gate 2a per the architectural
#    plan is per-channel, with relaxed tolerance during transient / strict
#    at steady state. Phase β should split into `ss_tolerance` and
#    `transient_tolerance`.
#
# 5. Currents are read from `I_total` in amp and converted to pA at the
#    boundary. For multi-channel cells, the harness should accept a list
#    of named current variables and report per-channel SS currents for
#    debugging — this is implicit but should be made explicit in API.
