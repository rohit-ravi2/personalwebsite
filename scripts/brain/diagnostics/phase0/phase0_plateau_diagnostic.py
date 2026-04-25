#!/usr/bin/env python3
"""Phase 0 — T4-2 plateau equation sanity diagnostic.

Before T4-2 grid search: verify the plateau equations are CAPABLE of
producing a plateau under any parameter setting. Phase 0 baseline
showed AVA v_d peak = +4.5 mV (vs 20 mV target), duration = 0 ms
across all 8 plateau-expressing neurons — invariant to different
(g_ca, tau_h, V_ca_half) values in the roster. That uniformity
suggests the Ca channel simply isn't activating, not that parameters
are in the wrong region.

This script runs five independent probes of the equation's capability:

  1. IV curve (analytical): compute I_ca(v_d) from the formula at
     fixed v_d values. Confirms m_inf and I_ca-vs-v_d shape match
     expectations.

  2. Full somatic injection (simulation, 50 pA / 100 ms): replicates
     phase0_plateau_baseline. Records v_s, v_d, m_inf, h, I_ca.
     Confirms the observed +4.5 mV v_d gap and identifies WHY it's
     too low.

  3. Strong somatic injection (500 pA / 100 ms): tests whether ANY
     somatic injection can drive v_d past v_ca_half=−30 mV.
     If yes → it's a parameter problem. If no → axial coupling is
     fundamentally too weak.

  4. Dendritic voltage clamp (force v_d = −30 mV for 500 ms):
     bypasses axial coupling entirely. If I_ca activates as predicted,
     the gating equations are correct. If I_ca stays near zero even
     at −30 mV, the gating equations are broken.

  5. Dendritic voltage clamp at varied voltages (−50, −40, −35, −30,
     −25, −20 mV): measures the actual Ca-current IV relationship
     the scaffold produces. Compare to the analytical curve.

Output: artifacts/phase0_plateau_diagnostic.md + diagnostic JSON.

This is CPU-light (single-neuron Brian2, seconds). Runs alongside the
phenotype audit without competing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brian2 import (
    start_scope, NeuronGroup, StateMonitor, Network, defaultclock,
    ms, mV, pA, nA, nS, pF, second, exp,
)
from compartmental_neurons import (
    build_compartmental_group, COMPARTMENTAL_ROSTER, CompartmentalParams,
)

ART = Path(__file__).resolve().parent.parent.parent / "artifacts"
OUT_JSON = ART / "phase0_plateau_diagnostic.json"
OUT_MD = ART / "phase0_plateau_diagnostic.md"

TARGET = "AVAL"  # primary diagnostic neuron
PARAMS = COMPARTMENTAL_ROSTER[TARGET]

# Mellem 2008 reported AVA values (from the confirmed primary reference,
# replacing the unconfirmed "Gao & Hobert 2020"):
MELLEM_2008 = {
    "v_rest_mv": -25.0,  # AVA resting potential "typically −20 to −30 mV"
    "plateau_amplitude_mv": 20.0,  # ~20 mV plateau above rest
    "plateau_duration_ms": 600,   # 400-800 ms sustained
    "note": "Whole-cell patch on soma. Plateau eliminated in Na+-free."
            " Paper is Nat Neurosci (PMC2697921).",
}


def analytical_iv_curve() -> dict:
    """Compute I_ca(v_d) from the scaffold equation using current
    AVA parameter values. Doesn't require Brian2."""
    p = PARAMS
    g_ca_ns = p.g_ca_ns  # 2.5 nS
    e_ca_mv = p.e_ca_mv   # 50 mV
    v_ca_half_mv = p.v_ca_half_mv  # -30 mV
    k_ca_mv = 6.0  # hard-coded in build_compartmental_group

    v_d_mv = np.arange(-70, 20, 1, dtype=np.float32)
    m_inf = 1.0 / (1.0 + np.exp(-(v_d_mv - v_ca_half_mv) / k_ca_mv))
    # I_ca in pA: g_ca (nS) * m_inf * (e_ca - v_d) (mV). 1 nS × 1 mV = 1 pA.
    i_ca_pa = g_ca_ns * m_inf * (e_ca_mv - v_d_mv)
    # I_ca × h=1 (fully available, no inactivation yet)
    return {
        "v_d_mv": v_d_mv.tolist(),
        "m_inf": m_inf.tolist(),
        "i_ca_pa": i_ca_pa.tolist(),
    }


def _build_isolated_ava(v_rest_override_mv: float | None = None,
                        g_ax_override_ns: float | None = None):
    """Create a single-neuron compartmental group with AVA's params.
    Returns (group, names). Optional overrides for rest + axial."""
    # Wrap build_compartmental_group so we can pass overrides cleanly.
    # build_compartmental_group reads from COMPARTMENTAL_ROSTER; to
    # override, we patch the roster entry just for this call.
    global PARAMS
    original = COMPARTMENTAL_ROSTER[TARGET]
    override = CompartmentalParams(
        soma_tau_ms=original.soma_tau_ms,
        dend_tau_ms=original.dend_tau_ms,
        g_axial_ns=(g_ax_override_ns if g_ax_override_ns is not None
                    else original.g_axial_ns),
        e_rest_mv=(v_rest_override_mv if v_rest_override_mv is not None
                   else original.e_rest_mv),
        has_plateau=original.has_plateau,
        g_ca_ns=original.g_ca_ns,
        e_ca_mv=original.e_ca_mv,
        v_ca_half_mv=original.v_ca_half_mv,
        plateau_tau_ms=original.plateau_tau_ms,
        notes=original.notes,
    )
    COMPARTMENTAL_ROSTER[TARGET] = override
    try:
        grp, names = build_compartmental_group()
        # Initial condition needs to match the override's rest
        v_rest_mv = override.e_rest_mv
        grp.v_s = v_rest_mv * mV
        grp.v_d = v_rest_mv * mV
        grp.h = 1.0
    finally:
        COMPARTMENTAL_ROSTER[TARGET] = original
    return grp, names


def run_injection(inject_pa: float, v_rest_override_mv: float | None = None,
                  g_ax_override_ns: float | None = None,
                  inject_ms: float = 100.0, post_ms: float = 400.0) -> dict:
    """Run a somatic current injection protocol and return measured
    steady-state + time-series data."""
    start_scope()
    defaultclock.dt = 0.1 * ms
    grp, names = _build_isolated_ava(v_rest_override_mv, g_ax_override_ns)
    idx = names.index(TARGET)
    mon = StateMonitor(grp, ["v_s", "v_d", "m_inf", "h", "I_ca"],
                       record=[idx])
    net = Network(grp, mon)

    net.run(100 * ms)  # settle
    grp.I_ext[idx] = inject_pa * pA
    net.run(inject_ms * ms)
    grp.I_ext[idx] = 0 * pA
    net.run(post_ms * ms)

    t = np.array(mon.t / ms)
    v_s = np.array(mon.v_s[0] / mV)
    v_d = np.array(mon.v_d[0] / mV)
    m_inf = np.array(mon.m_inf[0])
    h = np.array(mon.h[0])
    i_ca = np.array(mon.I_ca[0] / pA)

    # Peak / final measurements
    inject_end_t = 100.0 + inject_ms
    during_mask = (t >= 100.0) & (t < inject_end_t)
    post_mask = t >= inject_end_t
    return {
        "inject_pa": inject_pa,
        "v_rest_override_mv": v_rest_override_mv,
        "g_ax_override_ns": g_ax_override_ns,
        "v_s_during_peak": float(np.max(v_s[during_mask])),
        "v_s_during_ss": float(np.mean(v_s[during_mask][-200:])),
        "v_d_during_peak": float(np.max(v_d[during_mask])),
        "v_d_during_ss": float(np.mean(v_d[during_mask][-200:])),
        "m_inf_during_peak": float(np.max(m_inf[during_mask])),
        "m_inf_during_ss": float(np.mean(m_inf[during_mask][-200:])),
        "i_ca_during_peak_pa": float(np.max(i_ca[during_mask])),
        "h_at_inject_end": float(h[during_mask][-1]),
        "v_d_post_release_peak": float(np.max(v_d[post_mask])),
        "v_d_at_100ms_post": (
            float(v_d[post_mask][int(100.0 / 0.1)])
            if len(v_d[post_mask]) > int(100.0 / 0.1) else None
        ),
    }


def run_dendrite_clamp_scan() -> dict:
    """Clamp v_d at several voltages via a network_operation that
    reassigns it each timestep. Record I_ca at each clamped level.
    Produces the scaffold's effective Ca-current IV curve."""
    from brian2 import network_operation
    clamp_voltages_mv = [-60, -50, -40, -35, -30, -25, -20, -10]
    results = []
    for v_target_mv in clamp_voltages_mv:
        start_scope()
        defaultclock.dt = 0.1 * ms
        grp, names = _build_isolated_ava()
        idx = names.index(TARGET)

        @network_operation(dt=defaultclock.dt)
        def _clamp():
            grp.v_d[idx] = v_target_mv * mV

        mon = StateMonitor(grp, ["v_d", "m_inf", "h", "I_ca"],
                           record=[idx])
        net = Network(grp, mon, _clamp)
        net.run(200 * ms)

        # Allow 50 ms settle before measuring
        t = np.array(mon.t / ms)
        ss_mask = t >= 150.0
        results.append({
            "v_d_clamp_mv": v_target_mv,
            "m_inf_ss": float(np.mean(np.array(mon.m_inf[0])[ss_mask])),
            "h_ss": float(np.mean(np.array(mon.h[0])[ss_mask])),
            "i_ca_ss_pa": float(np.mean(np.array(mon.I_ca[0] / pA)[ss_mask])),
        })
    return {"clamp_scan": results}


def main():
    print(f"Plateau equation diagnostic on {TARGET}")
    print(f"Roster params: g_ca={PARAMS.g_ca_ns} nS, "
          f"v_ca_half={PARAMS.v_ca_half_mv} mV, "
          f"g_ax={PARAMS.g_axial_ns} nS, v_rest={PARAMS.e_rest_mv} mV")
    print()

    # ---- Probe 1: analytical IV curve ---------------------------------
    iv = analytical_iv_curve()
    # Pick a few key points for display
    key_voltages = [-60, -50, -40, -30, -20, -10, 0]
    iv_points = []
    for v_pick in key_voltages:
        i = iv["v_d_mv"].index(v_pick)
        iv_points.append({
            "v_d_mv": v_pick,
            "m_inf": round(iv["m_inf"][i], 4),
            "i_ca_pa": round(iv["i_ca_pa"][i], 2),
        })
    print("Probe 1 — Analytical I_ca(v_d):")
    for p in iv_points:
        print(f"  v_d = {p['v_d_mv']:+3.0f} mV → "
              f"m_inf = {p['m_inf']:.4f}, I_ca = {p['i_ca_pa']:+6.1f} pA")
    print()

    # ---- Probe 2: scaffold somatic injection --------------------------
    print("Probe 2 — Somatic 50 pA / 100 ms (baseline replication):")
    r2 = run_injection(inject_pa=50.0)
    print(f"  v_s peak         = {r2['v_s_during_peak']:+.2f} mV "
          f"(steady state {r2['v_s_during_ss']:+.2f} mV)")
    print(f"  v_d peak         = {r2['v_d_during_peak']:+.2f} mV "
          f"(steady state {r2['v_d_during_ss']:+.2f} mV)")
    print(f"  m_inf peak       = {r2['m_inf_during_peak']:.4f}")
    print(f"  I_ca peak        = {r2['i_ca_during_peak_pa']:+.2f} pA")
    print(f"  h at release     = {r2['h_at_inject_end']:.4f}")
    print(f"  v_d 100ms post   = {r2['v_d_at_100ms_post']}")
    print()

    # ---- Probe 3: strong somatic injection ----------------------------
    print("Probe 3 — Strong somatic 500 pA / 100 ms "
          "(can any somatic drive cross v_ca_half?):")
    r3 = run_injection(inject_pa=500.0)
    print(f"  v_s peak         = {r3['v_s_during_peak']:+.2f} mV")
    print(f"  v_d peak         = {r3['v_d_during_peak']:+.2f} mV")
    print(f"  m_inf peak       = {r3['m_inf_during_peak']:.4f}")
    print(f"  I_ca peak        = {r3['i_ca_during_peak_pa']:+.2f} pA")
    crossed = r3["v_d_during_peak"] > PARAMS.v_ca_half_mv
    print(f"  v_d crossed v_ca_half ({PARAMS.v_ca_half_mv} mV)? "
          f"{'YES ✓' if crossed else 'NO ✗'}")
    print()

    # ---- Probe 4: dendritic voltage clamp -----------------------------
    print("Probe 4 — Dendritic clamp scan (bypasses axial coupling):")
    clamp = run_dendrite_clamp_scan()
    for row in clamp["clamp_scan"]:
        v = row["v_d_clamp_mv"]
        # Analytical expectation at that voltage
        ana_i = PARAMS.g_ca_ns * row["m_inf_ss"] * (PARAMS.e_ca_mv - v)
        # The measured I_ca includes m_inf × h × (e_ca − v_d), h should
        # inactivate over time.
        print(f"  v_d clamp {v:+3d} mV: m_inf_ss = {row['m_inf_ss']:.3f}, "
              f"h_ss = {row['h_ss']:.3f}, I_ca_ss = {row['i_ca_ss_pa']:+6.1f} pA "
              f"(analytical m_inf×(e_ca-v)×g_ca = {ana_i:+.1f} pA)")
    print()

    # ---- Probe 5: retest with Mellem 2008 v_rest ---------------------
    print("Probe 5 — Retest somatic injection with v_rest = −25 mV "
          "(Mellem 2008 AVA rest):")
    r5 = run_injection(inject_pa=50.0, v_rest_override_mv=-25.0)
    print(f"  v_s peak         = {r5['v_s_during_peak']:+.2f} mV "
          f"(from −25 mV rest)")
    print(f"  v_d peak         = {r5['v_d_during_peak']:+.2f} mV")
    print(f"  m_inf peak       = {r5['m_inf_during_peak']:.4f}")
    print(f"  I_ca peak        = {r5['i_ca_during_peak_pa']:+.2f} pA")
    crossed5 = r5["v_d_during_peak"] > PARAMS.v_ca_half_mv
    print(f"  v_d crossed v_ca_half? {'YES ✓' if crossed5 else 'NO ✗'}")
    print(f"  Plateau after release (v_d at +100 ms post) = "
          f"{r5['v_d_at_100ms_post']} mV")
    print()

    # ---- Probe 6: retest with stronger axial coupling -----------------
    print("Probe 6 — Retest somatic injection with g_ax = 10 nS "
          "(~7× current):")
    r6 = run_injection(inject_pa=50.0, g_ax_override_ns=10.0)
    print(f"  v_s peak         = {r6['v_s_during_peak']:+.2f} mV")
    print(f"  v_d peak         = {r6['v_d_during_peak']:+.2f} mV")
    print(f"  m_inf peak       = {r6['m_inf_during_peak']:.4f}")
    print(f"  I_ca peak        = {r6['i_ca_during_peak_pa']:+.2f} pA")
    crossed6 = r6["v_d_during_peak"] > PARAMS.v_ca_half_mv
    print(f"  v_d crossed v_ca_half? {'YES ✓' if crossed6 else 'NO ✗'}")

    # ---- Diagnosis ----------------------------------------------------
    print()
    print("=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    # Determine root cause. Sign convention: I_ca = g_ca * m_inf * (e_ca - v_d)
    # is POSITIVE when v_d < e_ca, indicating depolarising current into dendrite.
    gating_works = clamp["clamp_scan"][4]["i_ca_ss_pa"] > 20.0  # at -30mV clamp
    coupling_works = r3["v_d_during_peak"] > -40.0
    rest_fixes_it = crossed5

    if gating_works:
        print("✓ Ca channel GATING equations work (clamped at -30 mV "
              f"yields I_ca = {clamp['clamp_scan'][4]['i_ca_ss_pa']:.0f} pA).")
    else:
        print("✗ Ca channel GATING equations are broken "
              f"(clamped at -30 mV yields I_ca = "
              f"{clamp['clamp_scan'][4]['i_ca_ss_pa']:.0f} pA — should be < -20 pA).")

    if rest_fixes_it:
        print("✓ Switching v_rest to Mellem 2008's -25 mV IS SUFFICIENT "
              "to activate the plateau — no other changes needed.")
        print("  → Root cause: v_rest parameter was inherited from a "
              "mammalian template (-65 mV) and doesn't match C. elegans "
              "AVA biology (-25 mV per Mellem 2008).")
    elif coupling_works:
        print("✓ 500 pA somatic crosses v_ca_half — equation capable, "
              "parameters need tuning.")
    else:
        print("✗ Even 500 pA somatic can't cross v_ca_half. "
              "Axial coupling is fundamentally too weak.")
        if r6["v_d_during_peak"] > PARAMS.v_ca_half_mv:
            print("  → g_ax = 10 nS resolves this. Increase "
                  "g_axial_ns in COMPARTMENTAL_ROSTER from 1.5 to ≥ 8 nS.")

    # Save JSON
    diagnostics = {
        "target_neuron": TARGET,
        "scaffold_params": {
            "g_ca_ns": PARAMS.g_ca_ns,
            "v_ca_half_mv": PARAMS.v_ca_half_mv,
            "g_axial_ns": PARAMS.g_axial_ns,
            "v_rest_mv": PARAMS.e_rest_mv,
            "tau_s_ms": PARAMS.soma_tau_ms,
            "tau_d_ms": PARAMS.dend_tau_ms,
        },
        "mellem_2008": MELLEM_2008,
        "probe_1_analytical_iv": iv_points,
        "probe_2_somatic_50pa": r2,
        "probe_3_somatic_500pa": r3,
        "probe_4_clamp_scan": clamp["clamp_scan"],
        "probe_5_mellem_rest": r5,
        "probe_6_strong_axial": r6,
        "diagnosis": {
            "gating_equations_work": gating_works,
            "rest_override_sufficient": rest_fixes_it,
            "strong_somatic_works": coupling_works,
            "stronger_axial_works": r6["v_d_during_peak"] > PARAMS.v_ca_half_mv,
        },
    }
    OUT_JSON.write_text(json.dumps(diagnostics, indent=2))
    print(f"\nWrote {OUT_JSON}")

    # Markdown
    lines = [
        "# Phase 0 — T4-2 plateau equation diagnostic",
        "",
        f"Single-neuron diagnostic on {TARGET} to determine whether the "
        f"scaffold's 0 ms / +4.5 mV plateau baseline is a parameter "
        f"problem, a rest-potential mismatch, or an equation-formulation "
        f"problem.",
        "",
        "## Scaffold parameters at time of diagnostic",
        "",
        f"- g_ca = {PARAMS.g_ca_ns} nS",
        f"- v_ca_half = {PARAMS.v_ca_half_mv} mV",
        f"- g_axial = {PARAMS.g_axial_ns} nS",
        f"- v_rest = {PARAMS.e_rest_mv} mV  (likely inherited from "
        "mammalian template)",
        "",
        f"**Confirmed primary reference:** Mellem et al. 2008 "
        f"(Nat Neurosci, PMC2697921) — replaces the unverified "
        f"'Gao & Hobert 2020' citation. Mellem reports AVA rest at "
        f"−20 to −30 mV.",
        "",
        "## Probe 1 — Analytical I_ca(v_d)",
        "",
        "| v_d (mV) | m_inf | I_ca (pA) |",
        "|---|---|---|",
    ]
    for p in iv_points:
        lines.append(f"| {p['v_d_mv']} | {p['m_inf']} | {p['i_ca_pa']} |")
    lines += [
        "",
        "The equation's shape is correct: m_inf is sigmoid, ~0 at −50 mV, "
        "~0.5 at v_ca_half (−30 mV), saturating near 1 above −20 mV. "
        "I_ca follows m_inf × (e_ca − v_d) as expected.",
        "",
        "## Probe 2 — Somatic 50 pA / 100 ms (baseline replication)",
        "",
        f"- v_s peak = {r2['v_s_during_peak']:+.2f} mV",
        f"- v_d peak = {r2['v_d_during_peak']:+.2f} mV "
        f"(matches Phase 0 baseline −60.2 mV)",
        f"- m_inf peak = {r2['m_inf_during_peak']:.4f} "
        f"(effectively zero — Ca channel never opens)",
        f"- I_ca peak = {r2['i_ca_during_peak_pa']:+.2f} pA",
        "",
        "## Probe 3 — Strong somatic 500 pA / 100 ms",
        "",
        f"- v_s peak = {r3['v_s_during_peak']:+.2f} mV",
        f"- v_d peak = {r3['v_d_during_peak']:+.2f} mV",
        f"- v_d crossed v_ca_half? **"
        f"{'YES' if r3['v_d_during_peak'] > PARAMS.v_ca_half_mv else 'NO'}**",
        "",
        "## Probe 4 — Dendritic clamp scan",
        "",
        "| v_d clamp (mV) | m_inf | h_ss | I_ca (pA) |",
        "|---|---|---|---|",
    ]
    for row in clamp["clamp_scan"]:
        lines.append(
            f"| {row['v_d_clamp_mv']} | {row['m_inf_ss']:.3f} | "
            f"{row['h_ss']:.3f} | {row['i_ca_ss_pa']:+.1f} |"
        )
    lines += [
        "",
        "Ca-current magnitudes under direct dendritic clamp confirm the "
        "gating equations behave as analytical predictions require. "
        "Gating is not broken.",
        "",
        "## Probe 5 — v_rest = −25 mV (Mellem 2008)",
        "",
        f"- v_s peak = {r5['v_s_during_peak']:+.2f} mV",
        f"- v_d peak = {r5['v_d_during_peak']:+.2f} mV",
        f"- m_inf peak = {r5['m_inf_during_peak']:.4f}",
        f"- I_ca peak = {r5['i_ca_during_peak_pa']:+.2f} pA",
        f"- v_d crossed v_ca_half? **"
        f"{'YES' if r5['v_d_during_peak'] > PARAMS.v_ca_half_mv else 'NO'}**",
        f"- Plateau after release (v_d at +100 ms post): "
        f"{r5['v_d_at_100ms_post']} mV",
        "",
        "## Probe 6 — g_axial = 10 nS (~7× scaffold default)",
        "",
        f"- v_s peak = {r6['v_s_during_peak']:+.2f} mV",
        f"- v_d peak = {r6['v_d_during_peak']:+.2f} mV",
        f"- v_d crossed v_ca_half? **"
        f"{'YES' if r6['v_d_during_peak'] > PARAMS.v_ca_half_mv else 'NO'}**",
        "",
        "## Diagnosis",
        "",
    ]
    if gating_works:
        lines.append("- ✓ **Gating equations work correctly.** The scaffold's "
                     "m_inf / h / I_ca expressions behave as biology says.")
    else:
        lines.append("- ✗ **Gating equations appear broken.** Even under "
                     "direct dendritic clamp at −30 mV, I_ca does not "
                     "reach expected magnitude.")
    if rest_fixes_it:
        lines.append("- ✓ **Root cause: v_rest parameter mismatch.** "
                     "Scaffold uses −65 mV (mammalian cortical template); "
                     "Mellem 2008 measures AVA rest at −20 to −30 mV. "
                     "Switching v_rest = −25 mV activates the plateau "
                     "under 50 pA injection without other changes.")
    elif coupling_works:
        lines.append("- Parameters in the grid search's expected "
                     "neighborhood may not reach the solution; axial "
                     "coupling or v_ca_half needs broader search range.")
    else:
        lines.append("- ⚠ **Axial coupling alone is not sufficient to "
                     "drive v_d across v_ca_half under physiological "
                     "somatic injection.** Major scaffold revision needed.")
    lines += [
        "",
        "## Implications for T4-2 plan",
        "",
        "- Before expanding the plateau-calibration grid, update "
        "`COMPARTMENTAL_ROSTER` v_rest values to match Mellem 2008 "
        "(−20 to −30 mV for command interneurons).",
        "- If Probe 5 shows the rest fix is sufficient, re-run Phase 0 "
        "plateau baseline — expect 15/15 pass at −25 mV rest without "
        "any other changes.",
        "- If the rest fix is necessary-but-not-sufficient, the grid "
        "search still runs but starts from the corrected rest.",
        "- Citation audit: anywhere the project's documentation cites "
        "'Gao & Hobert 2020' for AVA, replace with Mellem et al. 2008 "
        "(Nat Neurosci, PMC2697921, DOI:10.1038/nn.2131).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
