"""
Phase β-pre v3 Layer B comparison driver.

Compares NEURON output against the digitized red/blue Model traces (deliverable 1).

Per the layered comparison decomposition:
  - Layer A: Brian2 = NEURON (Phase β proper)
  - Layer B: NEURON = Nicoletti's published Model figures   ← THIS IS LAYER B (v3)
  - Layer C: Nicoletti's Model = experimental data (Layer C residuals; v1+v2 territory)

Sources:
  - AVAL/AIY/RIM NEURON output: comparison_validation_results_v2.json (already captured)
  - AVAR NEURON output: re-run with avar_unc103_patch.py (this script)
  - Reference traces: nicoletti_model_traces.json (digitized Model curves, deliverable 1)

Tolerance (matches v2's relative-with-floor formula at 5%):
  divergence(measured, reference, peak) =
      |measured - reference| / max(|measured|, |reference|, 0.1*peak)

Per-feature pass: divergence ≤ 0.05.
Per-step pass: ALL features pass.
Per-panel pass: > 80% of steps pass.
Per-cell pass: panel passes.
Overall pass: all 4 cells pass.

Output: layer_b_validation_results.json
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

V2_NEURON_RESULTS = ARTIFACTS / "comparison_validation_results_v2.json"
MODEL_TRACES_V3 = ARTIFACTS / "nicoletti_model_traces.json"
OUT_JSON = ARTIFACTS / "layer_b_validation_results.json"


# Feature keys (must match v2's order)
FEATURE_KEYS = [
    "peak_voltage_mV",
    "plateau_amplitude_mV",
    "plateau_duration_ms",
    "time_to_peak_ms",
    "settling_time_ms",
]


def divergence(measured: float, reference: float, peak: float) -> float:
    if measured is None or reference is None:
        return float("nan")
    denom = max(abs(measured), abs(reference), 0.1 * peak)
    if denom == 0:
        return 0.0
    return abs(measured - reference) / denom


def extract_features_neuron(
    v_trace, t_trace, stim_onset, stim_offset
) -> dict:
    """Same feature extractor as v2's. Replicated here to keep this script
    self-contained."""
    v = np.asarray(v_trace, dtype=float)
    t = np.asarray(t_trace, dtype=float)
    stim_dur = stim_offset - stim_onset

    pre_mask = t < stim_onset
    if pre_mask.any():
        baseline = float(np.median(v[pre_mask]))
    else:
        baseline = float(v[0])

    in_stim = (t >= stim_onset) & (t <= stim_offset)
    if in_stim.sum() < 5:
        return {
            "peak_voltage_mV": None, "plateau_amplitude_mV": None,
            "plateau_duration_ms": None, "time_to_peak_ms": None,
            "settling_time_ms": None, "baseline_mV": round(baseline, 2),
            "n_in_stim_samples": int(in_stim.sum()),
        }
    ts, vs = t[in_stim], v[in_stim]

    if vs.max() - baseline >= baseline - vs.min():
        peak_idx = int(np.argmax(vs))
    else:
        peak_idx = int(np.argmin(vs))
    peak_v = float(vs[peak_idx])
    peak_t = float(ts[peak_idx])

    plateau_start = stim_onset + 0.7 * stim_dur
    plateau_mask = ts >= plateau_start
    plateau_v = float(np.median(vs[plateau_mask])) if plateau_mask.sum() >= 3 \
        else float(np.median(vs[-max(3, len(vs) // 4):]))

    v_range = abs(peak_v - baseline)
    threshold = max(2.0, 0.10 * v_range)
    in_plat = np.abs(vs - plateau_v) < threshold
    if in_plat.any():
        plat_dur = float(ts[in_plat][-1] - ts[in_plat][0])
    else:
        plat_dur = 0.0

    settling_thresh = max(2.0, 0.10 * v_range)
    settled = np.abs(vs - plateau_v) < settling_thresh
    if settled.any():
        first_settled_idx = int(np.where(settled)[0][0])
        settling_t = float(ts[first_settled_idx]) - stim_onset
    else:
        settling_t = float(stim_dur)

    return {
        "peak_voltage_mV": round(peak_v, 2),
        "plateau_amplitude_mV": round(plateau_v, 2),
        "plateau_duration_ms": round(plat_dur, 1),
        "time_to_peak_ms": round(peak_t - stim_onset, 1),
        "settling_time_ms": round(settling_t, 1),
        "baseline_mV": round(baseline, 2),
        "n_in_stim_samples": int(in_stim.sum()),
    }


def get_neuron_features(
    v_traces, t_traces, n_steps, stim_onset, stim_offset
) -> list[dict]:
    """Per-step feature extraction from NEURON output traces."""
    feats = []
    for i in range(n_steps):
        feats.append(extract_features_neuron(
            v_traces[i], t_traces[i], stim_onset, stim_offset
        ))
    return feats


def compute_layer_b_panel(
    *, panel_id, cell, model_panel, neuron_features, neuron_steps_pa,
    stim_onset, stim_offset, neuron_warning=None,
) -> dict:
    """Per-panel Layer B comparison: NEURON output vs digitized Model trace."""

    model_feats = model_panel["extracted_features"]
    steps_pa = model_panel["current_steps_pA"]
    n_steps_total = len(steps_pa)

    # Compute per-feature peak magnitude across both NEURON and Model series
    feature_peaks = {}
    for key in FEATURE_KEYS:
        all_vals = []
        for step_str, val in model_feats[key].items():
            if val is not None:
                all_vals.append(abs(float(val)))
        for nf in neuron_features:
            if nf.get(key) is not None:
                all_vals.append(abs(nf[key]))
        feature_peaks[key] = max(all_vals) if all_vals else 1.0

    per_step = []
    for step_idx, step_pa in enumerate(steps_pa):
        step_str = str(int(step_pa))
        model_step = {k: model_feats[k].get(step_str) for k in FEATURE_KEYS}
        nrn_idx = int(np.argmin(np.abs(np.array(neuron_steps_pa) - step_pa)))
        nrn_step = neuron_features[nrn_idx]

        divs = {}
        for key in FEATURE_KEYS:
            m = model_step.get(key)  # digitized Model value
            r = nrn_step.get(key)  # NEURON code value
            if m is None or r is None:
                divs[key] = {
                    "model_published": m, "neuron_code": r,
                    "divergence": None, "pass": False,
                    "reason": "feature_missing",
                }
                continue
            d = divergence(m, r, feature_peaks[key])
            divs[key] = {
                "model_published": float(m),
                "neuron_code": float(r),
                "divergence": round(d, 4),
                "pass": d <= 0.05,
            }
        all_pass = all(d["pass"] for d in divs.values())
        per_step.append({
            "current_pA": step_pa,
            "model_published_features": model_step,
            "neuron_code_features": {k: nrn_step.get(k) for k in FEATURE_KEYS},
            "feature_divergences": divs,
            "step_pass": all_pass,
        })

    n_passing_steps = sum(1 for s in per_step if s["step_pass"])
    fraction_passing = n_passing_steps / n_steps_total
    panel_pass = fraction_passing > 0.80

    # Voltage-only secondary diagnostic (peak + plateau, excludes timing noise)
    VOLTAGE_FEATS = ["peak_voltage_mV", "plateau_amplitude_mV"]
    n_voltage_pass = sum(
        1 for s in per_step
        if all(s["feature_divergences"][k]["pass"]
               for k in VOLTAGE_FEATS
               if s["feature_divergences"][k].get("divergence") is not None)
    )
    voltage_only_pass_fraction = n_voltage_pass / n_steps_total
    voltage_panel_pass = voltage_only_pass_fraction > 0.80

    # Voltage absolute error (mV, model-published vs neuron-code)
    voltage_abs_errors = []
    for s in per_step:
        for k in VOLTAGE_FEATS:
            fd = s["feature_divergences"][k]
            if fd.get("divergence") is not None:
                voltage_abs_errors.append(abs(fd["model_published"] - fd["neuron_code"]))
    mean_v_err = float(np.mean(voltage_abs_errors)) if voltage_abs_errors else None
    median_v_err = float(np.median(voltage_abs_errors)) if voltage_abs_errors else None
    max_v_err = float(np.max(voltage_abs_errors)) if voltage_abs_errors else None

    return {
        "panel_id": panel_id,
        "cell": cell,
        "n_steps_total": n_steps_total,
        "n_steps_passing": n_passing_steps,
        "fraction_passing": round(fraction_passing, 4),
        "panel_pass": bool(panel_pass),
        "feature_peaks_for_tolerance_floor": {k: round(v, 2) for k, v in feature_peaks.items()},
        "secondary_voltage_only_diagnostic": {
            "n_voltage_pass": n_voltage_pass,
            "voltage_only_fraction_pass": round(voltage_only_pass_fraction, 4),
            "voltage_only_panel_pass": bool(voltage_panel_pass),
            "voltage_abs_error_mV_mean": round(mean_v_err, 2) if mean_v_err is not None else None,
            "voltage_abs_error_mV_median": round(median_v_err, 2) if median_v_err is not None else None,
            "voltage_abs_error_mV_max": round(max_v_err, 2) if max_v_err is not None else None,
        },
        "per_step_comparison": per_step,
        "neuron_warning": neuron_warning,
    }


def main() -> None:
    print("Loading inputs:")
    v2_results = json.loads(V2_NEURON_RESULTS.read_text())
    model_traces = json.loads(MODEL_TRACES_V3.read_text())
    print(f"  v2 NEURON results: {len(v2_results['panel_results'])} panels")
    print(f"  v3 Model traces: {len(model_traces['panels'])} panels")

    # Index Model traces by cell
    model_by_cell = {p["cell"]: p for p in model_traces["panels"]}

    panel_records = []

    # ----- AVAL: reuse v2 NEURON output -----
    aval_v2 = next(p for p in v2_results["panel_results"] if p["cell"] == "AVAL")
    aval_neuron_feats = [
        s["nrn_features"] for s in aval_v2["per_step_comparison"]
    ]
    aval_steps_pa = [s["current_pA"] for s in aval_v2["per_step_comparison"]]
    panel_records.append(compute_layer_b_panel(
        panel_id="layer_b_AVAL",
        cell="AVAL",
        model_panel=model_by_cell["AVAL"],
        neuron_features=aval_neuron_feats,
        neuron_steps_pa=aval_steps_pa,
        stim_onset=0.0, stim_offset=1000.0,
        neuron_warning=None,
    ))

    # ----- AIY: reuse v2 NEURON output -----
    aiy_v2 = next(p for p in v2_results["panel_results"] if p["cell"] == "AIY")
    aiy_neuron_feats = [s["nrn_features"] for s in aiy_v2["per_step_comparison"]]
    aiy_steps_pa = [s["current_pA"] for s in aiy_v2["per_step_comparison"]]
    panel_records.append(compute_layer_b_panel(
        panel_id="layer_b_AIY",
        cell="AIY",
        model_panel=model_by_cell["AIY"],
        neuron_features=aiy_neuron_feats,
        neuron_steps_pa=aiy_steps_pa,
        stim_onset=1000.0, stim_offset=6000.0,
        neuron_warning=None,
    ))

    # ----- RIM: reuse v2 NEURON output -----
    rim_v2 = next(p for p in v2_results["panel_results"] if p["cell"] == "RIM")
    rim_neuron_feats = [s["nrn_features"] for s in rim_v2["per_step_comparison"]]
    rim_steps_pa = [s["current_pA"] for s in rim_v2["per_step_comparison"]]
    panel_records.append(compute_layer_b_panel(
        panel_id="layer_b_RIM",
        cell="RIM",
        model_panel=model_by_cell["RIM"],
        neuron_features=rim_neuron_feats,
        neuron_steps_pa=rim_steps_pa,
        stim_onset=1000.0, stim_offset=6000.0,
        neuron_warning=None,
    ))

    # ----- AVAR: re-run with patch -----
    print("\nRunning AVAR with UNC-103 patch (avar_unc103_patch.py)...")
    sys.path.insert(0, str(WAVE2_DIR))
    from avar_unc103_patch import run_AVAR_iclamp_patched
    avar_run = run_AVAR_iclamp_patched()
    avar_neuron_feats = get_neuron_features(
        v_traces=avar_run["v_traces_mV"],
        t_traces=avar_run["time_ms"],
        n_steps=avar_run["n_steps"],
        stim_onset=avar_run["stim_onset_ms"],
        stim_offset=avar_run["stim_offset_ms"],
    )
    avar_steps_pa = avar_run["current_steps_pA"]
    print(f"  AVAR rest (mean across steps): "
          f"{np.mean([f['baseline_mV'] for f in avar_neuron_feats]):.2f} mV "
          f"(target -25 ± 5 mV)")
    panel_records.append(compute_layer_b_panel(
        panel_id="layer_b_AVAR",
        cell="AVAR",
        model_panel=model_by_cell["AVAR"],
        neuron_features=avar_neuron_feats,
        neuron_steps_pa=avar_steps_pa,
        stim_onset=avar_run["stim_onset_ms"],
        stim_offset=avar_run["stim_offset_ms"],
        neuron_warning="AVAR run uses avar_unc103_patch.py (v3) — UNC-103 inserted "
                       "with gbar from AVAR_simulation.py line 28. Resolves v2's "
                       "+11 mV resting bias from missing-UNC103 fallback.",
    ))

    # Overall verdict
    n_panels = len(panel_records)
    n_pass = sum(1 for p in panel_records if p["panel_pass"])
    overall_pass = n_pass == n_panels
    multi_panel_fail = (n_panels - n_pass) >= 2

    if overall_pass:
        verdict = "pass_layer_b_clean"
    elif multi_panel_fail:
        verdict = "multi_panel_fail_real_layer_b_finding"
    else:
        verdict = "single_panel_fail_borderline"

    out = {
        "phase": "phase_beta_pre_v3",
        "layer": "B (NEURON code = Nicoletti's published Model figures)",
        "generation_date": "2026-04-26",
        "tolerance_metric": (
            "Per-feature: divergence(m, r, peak) = |m-r| / max(|m|, |r|, 0.1*peak); "
            "feature pass: divergence ≤ 0.05. "
            "Per-step: ALL features pass. "
            "Per-panel: > 80% of steps pass. "
            "Per-cell: panel passes."
        ),
        "overall_verdict": verdict,
        "n_panels_total": n_panels,
        "n_panels_passing": n_pass,
        "panel_results": panel_records,
        "neuron_output_provenance": {
            "AVAL": "comparison_validation_results_v2.json (reused)",
            "AIY":  "comparison_validation_results_v2.json (reused)",
            "RIM":  "comparison_validation_results_v2.json (reused)",
            "AVAR": "re-run via avar_unc103_patch.py (v3)",
        },
        "model_trace_provenance": "nicoletti_model_traces.json (deliverable 1)",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n=== LAYER B VERDICT: {verdict}  ({n_pass}/{n_panels} panels pass) ===")
    for p in panel_records:
        print(f"  {p['cell']}: {p['n_steps_passing']}/{p['n_steps_total']} steps pass "
              f"(panel_pass={p['panel_pass']}; "
              f"V-only mean abs err = "
              f"{p['secondary_voltage_only_diagnostic']['voltage_abs_error_mV_mean']} mV)")
    print(f"\nResults: {OUT_JSON}")


if __name__ == "__main__":
    main()
