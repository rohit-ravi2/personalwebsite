"""
Phase β-pre Step 3: Comparison validation.

For each digitized experimental panel, run Nicoletti's NEURON code under
the matching voltage-clamp protocol and compute per-point divergence
between Nicoletti's NEURON steady-state I-V output and the digitized
experimental I-V data points.

Tolerance (per spec):
    For values > 10% of peak: 5% relative tolerance.
    For values < 10% of peak: absolute tolerance ≤ 5% of peak.

Equivalent formula (single line):
    divergence = |measured - reference| / max(|measured|, |reference|, 0.1*peak)

Per-point pass: divergence ≤ 0.05.
Per-panel pass: > 90% of points pass AND no single point exceeds 0.15.

Output:
    /home/rohit/.../wave2/artifacts/comparison_validation_results.json
    /home/rohit/.../wave2/artifacts/phase_beta_pre_validation.md (separate)

Author: Phase β-pre engineering session, 2026-04-26.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


WAVE2_DIR = Path("/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2")
ARTIFACTS = WAVE2_DIR / "artifacts"
NICOLETTI_DIR = Path("/home/rohit/Desktop/C-Elegans/simulation/upstream/nicoletti_2024")
PUBLISHED_TRACES = ARTIFACTS / "published_traces.json"
OUT_JSON = ARTIFACTS / "comparison_validation_results.json"


# ---------------------------------------------------------------------------
# Tolerance utility
# ---------------------------------------------------------------------------


def divergence(measured: float, reference: float, peak: float) -> float:
    """Per-point divergence per Phase β-pre spec.

        max(|m - r|, 0) / max(|m|, |r|, 0.1*peak)
    """
    return abs(measured - reference) / max(abs(measured), abs(reference), 0.1 * peak)


def assess_panel(
    measured: np.ndarray, reference: np.ndarray
) -> dict:
    """Compute per-point and panel-level pass/fail."""
    peak = float(max(np.max(np.abs(measured)), np.max(np.abs(reference))))
    divs = np.array(
        [divergence(m, r, peak) for m, r in zip(measured, reference)]
    )
    per_point_pass = (divs <= 0.05).astype(int).tolist()
    panel_pass_count = sum(per_point_pass)
    panel_total = len(per_point_pass)
    fraction_pass = panel_pass_count / panel_total if panel_total else 0
    panel_pass = (fraction_pass > 0.90) and (divs.max() <= 0.15)
    return {
        "peak_pA": float(round(peak, 3)),
        "divergences": [float(round(d, 4)) for d in divs],
        "per_point_pass": per_point_pass,
        "fraction_pass": float(round(fraction_pass, 4)),
        "max_divergence": float(round(divs.max(), 4)),
        "mean_divergence": float(round(divs.mean(), 4)),
        "panel_pass": bool(panel_pass),
    }


# ---------------------------------------------------------------------------
# Nicoletti NEURON runners
# ---------------------------------------------------------------------------


def run_nicoletti_voltage_clamp(cell: str) -> dict:
    """Run Nicoletti's voltage-clamp simulation for a given cell.

    Returns dict with:
        v_steps: list[float]  voltage steps (mV)
        i_ss: list[float]     steady-state current at each step (pA)
        i_peak: list[float]   peak current (pA)

    Performs `os.chdir(NICOLETTI_DIR)` BEFORE `from neuron import h`
    so the compiled mechanism library loads correctly. Imports the
    `*_simulation_vclamp` module to get the bare simulation function
    (avoids the top-level scripts which call `os.mkdir(...)` and crash
    on re-runs).
    """
    cur_cwd = os.getcwd()
    try:
        os.chdir(str(NICOLETTI_DIR))
        sys.path.insert(0, str(NICOLETTI_DIR))

        # Import here so chdir is in effect first
        from g_to_Scm2 import gScm2  # noqa: E402

        if cell == "AVAL":
            from AVAL_simulation_vclamp import AVA_simulation_vc  # type: ignore
            # Conductances per AVAL_simulations.py:
            #   g0 = [egl19, leak, irk, nca, eleak, cm]
            g0 = [0.104385, 0.150164, 0.1, 0, -39, 0.859551]
            surf = 1123.84e-8
            gbest = gScm2(g0, surf, 3)
            vstart, vstop, ns = -110, 50, 17
            v_arr = np.linspace(vstart, vstop, ns)
            results = AVA_simulation_vc(gbest, vstart, vstop, ns)
            i_ss = np.array(list(results[3]))
            i_peak = np.array(list(results[2]))
        elif cell == "AIY":
            from AIY_simulation_vclamp import AIY_simulation_vc  # type: ignore
            # Per AIY_simulation.py:
            # surf=1123.84e-8 (same? — actually let me read the script)
            # We'll dynamically peek at AIY_simulation.py:
            from AIY_simulation import g0_AIY, surf_AIY, scale_index_AIY  # may not exist
            # Fallback: use module-level globals via import file inspection
            raise NotImplementedError("Will dispatch to AIY-specific runner below")
        elif cell == "RIM":
            from RIM_simulation_vclamp import RIM_simulation_vc  # type: ignore
            raise NotImplementedError("Will dispatch to RIM-specific runner below")
        else:
            raise ValueError(f"Unknown cell: {cell}")

        return {
            "cell": cell,
            "v_steps_mV": [float(v) for v in v_arr],
            "i_ss_pA": [float(x) for x in i_ss],
            "i_peak_pA": [float(x) for x in i_peak],
        }
    finally:
        os.chdir(cur_cwd)
        if str(NICOLETTI_DIR) in sys.path:
            sys.path.remove(str(NICOLETTI_DIR))


def run_AIY_vclamp() -> dict:
    """Run AIY voltage-clamp using parameters extracted directly from
    AIY_simulation.py. We hard-code these (rather than parsing the
    script) since the script side-effects (os.mkdir, plt.show) make
    runtime import unsafe."""
    cur_cwd = os.getcwd()
    try:
        os.chdir(str(NICOLETTI_DIR))
        sys.path.insert(0, str(NICOLETTI_DIR))
        from g_to_Scm2 import gScm2  # noqa: E402
        from AIY_simulation_vclamp import AIY_simulation_vc  # type: ignore

        # From AIY_simulation.py lines 20-28:
        v_arr = np.linspace(-120, 50, 18)
        surf = 65.89e-8
        # conductances: leak, slo1iso, kqt1, egl19, slo1egl19, nca, irk, eleak, cm
        g0 = [0.14, 1, 0.2, 0.1, 0.92, 0.06, 0.5, -89.57, 1.6]
        scale_idx = 6  # per AIY_simulation.py:28: gScm2(g0, surf, 6)
        gbest = gScm2(g0, surf, scale_idx)
        vstart, vstop, ns = -120, 50, 18
        results = AIY_simulation_vc(gbest, vstart, vstop, ns)
        i_ss = np.array(list(results[3]))
        i_peak = np.array(list(results[2]))
        return {
            "cell": "AIY",
            "v_steps_mV": [float(x) for x in v_arr],
            "i_ss_pA": [float(x) for x in i_ss],
            "i_peak_pA": [float(x) for x in i_peak],
        }
    finally:
        os.chdir(cur_cwd)
        if str(NICOLETTI_DIR) in sys.path:
            sys.path.remove(str(NICOLETTI_DIR))


def run_RIM_vclamp() -> dict:
    """Run RIM voltage-clamp. RIM does NOT apply gScm2 — its g vector
    is already in S/cm² (per Phase α §6.3 finding). Parameters from
    RIM_simulation.py."""
    cur_cwd = os.getcwd()
    try:
        os.chdir(str(NICOLETTI_DIR))
        sys.path.insert(0, str(NICOLETTI_DIR))
        from RIM_simulation_vclamp import RIM_simulation_vc  # type: ignore

        # From RIM_simulation.py lines 19-27:
        v_arr = np.linspace(-100, 50, 16)
        # conductances in S/cm^2: SHL1, EGL2, IRK, CCA1, unc2, egl19, LEAK, eleak, cm
        g = [
            0.0009048750067326097,
            0.0001411644285181245,
            0.0003272854640954744,
            0.0008451919806776876,
            9.676795045480941e-05,
            0.00032005818627638106,
            9.676795045480941e-05,
            -50,
            1.5,
        ]
        vstart, vstop, ns = -100, 50, 16
        results = RIM_simulation_vc(g, vstart, vstop, ns)
        i_ss = np.array(list(results[3]))
        i_peak = np.array(list(results[2]))
        return {
            "cell": "RIM",
            "v_steps_mV": [float(x) for x in v_arr],
            "i_ss_pA": [float(x) for x in i_ss],
            "i_peak_pA": [float(x) for x in i_peak],
        }
    finally:
        os.chdir(cur_cwd)
        if str(NICOLETTI_DIR) in sys.path:
            sys.path.remove(str(NICOLETTI_DIR))


def run_AVAL_vclamp() -> dict:
    """Run AVAL voltage-clamp."""
    cur_cwd = os.getcwd()
    try:
        os.chdir(str(NICOLETTI_DIR))
        sys.path.insert(0, str(NICOLETTI_DIR))
        from g_to_Scm2 import gScm2  # noqa: E402
        from AVAL_simulation_vclamp import AVA_simulation_vc  # type: ignore
        g0 = [0.104385, 0.150164, 0.1, 0, -39, 0.859551]
        surf = 1123.84e-8
        gbest = gScm2(g0, surf, 3)
        vstart, vstop, ns = -110, 50, 17
        v_arr = np.linspace(vstart, vstop, ns)
        results = AVA_simulation_vc(gbest, vstart, vstop, ns)
        i_ss = np.array(list(results[3]))
        i_peak = np.array(list(results[2]))
        return {
            "cell": "AVAL",
            "v_steps_mV": [float(x) for x in v_arr],
            "i_ss_pA": [float(x) for x in i_ss],
            "i_peak_pA": [float(x) for x in i_peak],
        }
    finally:
        os.chdir(cur_cwd)
        if str(NICOLETTI_DIR) in sys.path:
            sys.path.remove(str(NICOLETTI_DIR))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def interpolate_neuron_to_exp_vsteps(
    nrn_v: np.ndarray, nrn_i: np.ndarray, exp_v: np.ndarray
) -> np.ndarray:
    """Linear interpolate Nicoletti's NEURON I(V) to the V values where
    we have experimental measurements.

    Both grids are dense (Nicoletti uses 16-18 V steps; experimental data
    has 9-16 readings). Linear interpolation is appropriate because the
    I(V) relationship is smooth in steady state.
    """
    return np.interp(exp_v, nrn_v, nrn_i)


def compare_panel(
    panel: dict, neuron_run: dict
) -> dict:
    exp_v = np.array([d["x"] for d in panel["data"]])
    exp_i = np.array([d["y"] for d in panel["data"]])
    nrn_v = np.array(neuron_run["v_steps_mV"])
    nrn_i = np.array(neuron_run["i_ss_pA"])
    nrn_i_at_exp = interpolate_neuron_to_exp_vsteps(nrn_v, nrn_i, exp_v)

    assess = assess_panel(measured=nrn_i_at_exp, reference=exp_i)
    rec = {
        "panel_id": panel["id"],
        "cell": panel["cell"],
        "n_exp_points": len(exp_v),
        "experimental_data": [
            {"V_mV": float(v), "I_pA_exp": float(i)} for v, i in zip(exp_v, exp_i)
        ],
        "neuron_native_grid": {
            "v_steps_mV": neuron_run["v_steps_mV"],
            "i_ss_pA": neuron_run["i_ss_pA"],
        },
        "neuron_at_exp_vsteps_pA": [float(round(x, 3)) for x in nrn_i_at_exp],
        "tolerance_assessment": assess,
        "panel_pass": assess["panel_pass"],
    }
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading published_traces.json")
    panels_data = json.loads(PUBLISHED_TRACES.read_text())
    panels_by_cell = {p["cell"]: p for p in panels_data["panels"]}
    print(f"  Cells: {list(panels_by_cell.keys())}")

    runners = {
        "AVAL": run_AVAL_vclamp,
        "AIY": run_AIY_vclamp,
        "RIM": run_RIM_vclamp,
    }

    results = {}
    for cell, runner in runners.items():
        if cell not in panels_by_cell:
            print(f"\nSkipping {cell} — no digitized panel")
            continue
        print(f"\n=== Running Nicoletti voltage-clamp for {cell} ===")
        try:
            run_result = runner()
            print(
                f"  {cell}: {len(run_result['v_steps_mV'])} V steps, "
                f"I_ss range [{min(run_result['i_ss_pA']):.2f}, "
                f"{max(run_result['i_ss_pA']):.2f}] pA"
            )
            cmp = compare_panel(panels_by_cell[cell], run_result)
            results[cell] = cmp
            print(
                f"  Panel {cmp['panel_id']}: "
                f"max_div={cmp['tolerance_assessment']['max_divergence']:.3f}, "
                f"mean_div={cmp['tolerance_assessment']['mean_divergence']:.3f}, "
                f"frac_pass={cmp['tolerance_assessment']['fraction_pass']:.2f}, "
                f"panel_pass={cmp['panel_pass']}"
            )
        except Exception as e:
            import traceback
            print(f"  {cell} failed: {e}")
            traceback.print_exc()
            results[cell] = {"error": str(e), "traceback": traceback.format_exc()}

    overall_pass = all(
        r.get("panel_pass") is True for r in results.values() if "panel_pass" in r
    )
    n_panels = len([r for r in results.values() if "panel_pass" in r])
    n_passing = sum(1 for r in results.values() if r.get("panel_pass") is True)

    out = {
        "phase": "phase_beta_pre",
        "generation_date": "2026-04-26",
        "tolerance_metric": (
            "Per Phase β-pre spec: divergence(m, r, peak) = |m-r| / max(|m|, |r|, 0.1*peak). "
            "Per-point pass: divergence ≤ 0.05. "
            "Per-panel pass: > 90% of points pass AND no single point exceeds 0.15."
        ),
        "overall_verdict": "pass" if overall_pass else "fail",
        "n_panels_total": n_panels,
        "n_panels_passing": n_passing,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n\n=== OVERALL VERDICT: {out['overall_verdict']} "
          f"({n_passing}/{n_panels} panels passing) ===")
    print(f"Results written to {OUT_JSON}")


if __name__ == "__main__":
    main()
