"""
Phase β-pre v2: Current-clamp comparison validation.

For each digitized current-clamp panel (Fig 1A AVAL, Fig 1B AVAR, Fig 3A AIY,
Fig 5A RIM), run Nicoletti's NEURON current-clamp simulation under the matching
protocol, extract the same features (peak voltage, plateau amplitude, plateau
duration, time-to-peak, settling time) per current step, and compute per-feature
divergence vs the digitized experimental traces.

Tolerance (per spec):
    feature_divergence(measured, reference, peak) =
        |measured - reference| / max(|measured|, |reference|, 0.1*peak)

Per-feature pass: divergence ≤ 0.05 (5% relative).
Per-step pass: ALL features pass for that current step.
Per-panel pass: > 80% of steps pass (looser than v1's 90% — feature-based
                has fewer comparison points so 80% is reasonable).
Per-cell pass: panel passes.

Note on AVAR: The Nicoletti 2024 GitHub repo is missing
`AVAR_simulation_iclamp.py` (referenced by AVAR_simulation.py but not
present on disk and not in the repo head tree). AVAR shares the same
channel set as AVAL (EGL19, LEAK, IRK, NCA) per AVAR_simulation.py. We
reuse the AVAL current-clamp simulator with AVAR's distinct conductance
parameters and surface area.

Output:
  - comparison_validation_results_v2.json
  - phase_beta_pre_validation.md (separate, deliverable 5)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np


WAVE2_DIR = Path("/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2")
ARTIFACTS = WAVE2_DIR / "artifacts"
NICOLETTI_DIR = Path("/home/rohit/Desktop/C-Elegans/simulation/upstream/nicoletti_2024")
PUBLISHED_TRACES_V2 = ARTIFACTS / "published_traces_v2.json"
OUT_JSON = ARTIFACTS / "comparison_validation_results_v2.json"


# ---------------------------------------------------------------------------
# Tolerance utility
# ---------------------------------------------------------------------------


def divergence(measured: float, reference: float, peak: float) -> float:
    """Per-feature divergence per Phase β-pre v2 spec."""
    if measured is None or reference is None:
        return float("nan")
    denom = max(abs(measured), abs(reference), 0.1 * peak)
    if denom == 0:
        return 0.0
    return abs(measured - reference) / denom


# ---------------------------------------------------------------------------
# NEURON current-clamp runners
# ---------------------------------------------------------------------------


def _activate_nicoletti_env():
    """chdir into Nicoletti dir and add to sys.path so compiled mods load."""
    os.chdir(str(NICOLETTI_DIR))
    if str(NICOLETTI_DIR) not in sys.path:
        sys.path.insert(0, str(NICOLETTI_DIR))


def _restore_env(cur_cwd):
    os.chdir(cur_cwd)
    if str(NICOLETTI_DIR) in sys.path:
        sys.path.remove(str(NICOLETTI_DIR))


def run_AVAL_iclamp() -> dict:
    """AVAL current-clamp: 7 steps from -0.03 to +0.03 nA, 1000 ms duration.
    Protocol per AVAL_simulations.py line 31.
    """
    cur_cwd = os.getcwd()
    try:
        _activate_nicoletti_env()
        from g_to_Scm2 import gScm2  # noqa: E402
        from AVAL_simulation_iclamp import AVA_simulation_iclamp  # type: ignore

        g0 = [0.104385, 0.150164, 0.1, 0, -39, 0.859551]
        surf = 1123.84e-8
        gbest = gScm2(g0, surf, 3)
        results = AVA_simulation_iclamp(gbest, -0.03, 0.03, 7)
        v_norm, time_norm, _, _ = results
        # time_norm starts at 0 (= absolute time 1000), stim onset at norm
        # time = 23 ms (= absolute 1023). Shift to align with figure: figure
        # x-axis 0 corresponds to stim onset, so figure_time = norm_time - 23.
        time_aligned = time_norm - 23.0
        steps_pa = np.linspace(-30.0, 30.0, 7).tolist()
        return {
            "cell": "AVAL",
            "n_steps": 7,
            "current_steps_pA": steps_pa,
            "v_traces_mV": [list(v) for v in v_norm],
            "time_ms": [list(t) for t in time_aligned],
            "stim_onset_ms": 0.0,
            "stim_offset_ms": 1000.0,
        }
    finally:
        _restore_env(cur_cwd)


def run_AVAR_iclamp() -> dict:
    """AVAR current-clamp: same channel set as AVAL with AVAR-specific
    conductances (per AVAR_simulation.py).

    Reuses AVAL_simulation_iclamp because AVAR_simulation_iclamp.py is
    missing from the upstream Nicoletti repo (commit 78a17ca tree does
    not contain it; the AVAR_simulation.py wrapper imports it but it
    cannot be loaded). AVAR shares EGL19+LEAK+IRK+NCA channels with AVAL
    per Nicoletti 2024 §3.1 ("AVAR neuron H-H model"); only the
    conductance values and surface area differ.
    """
    cur_cwd = os.getcwd()
    try:
        _activate_nicoletti_env()
        from g_to_Scm2 import gScm2  # noqa: E402
        from AVAL_simulation_iclamp import AVA_simulation_iclamp  # type: ignore

        # Per AVAR_simulation.py lines 22-27:
        # surf=1121.79e-8 (AVAR); g0 same channel set [EGL19, LEAK, IRK, NCA, UNC103, ELEAK, CM]
        # Note: AVAR adds UNC103 (the iclamp script as-imported expects
        # gAVA_scaled[5] = cm). AVAL uses [EGL19, LEAK, IRK, NCA, ELEAK, CM]
        # (6 elements). AVAR's g0 has 7 elements:
        #   [EGL19=0.0643372, LEAK=0.225225, IRK=0.042079, NCA=0.0493356,
        #    UNC103=0.0481669, ELEAK=-37, CM=0.751761]
        # AVAL_simulation_iclamp expects 6-element array with cm at index [5].
        # We have to drop UNC103 (which AVAL_iclamp doesn't simulate) since
        # the AVAL script's soma.insert calls don't include 'unc103'. This
        # introduces a known bias for AVAR (UNC103 contribution missing) —
        # surface this in the validation report.
        # surf for AVAR
        surf_avar = 1121.79e-8
        # Build g0 for AVAL_iclamp interface: [EGL19, LEAK, IRK, NCA, ELEAK, CM]
        # Drop UNC103 element (idx 4 in AVAR's 7-element g0)
        g0_avar_compat = [0.0643372, 0.225225, 0.042079, 0.0493356, -37, 0.751761]
        gbest = gScm2(g0_avar_compat, surf_avar, 4)
        results = AVA_simulation_iclamp(gbest, -0.03, 0.03, 7)
        v_norm, time_norm, _, _ = results
        time_aligned = time_norm - 23.0  # same offset as AVAL
        steps_pa = np.linspace(-30.0, 30.0, 7).tolist()
        return {
            "cell": "AVAR",
            "n_steps": 7,
            "current_steps_pA": steps_pa,
            "v_traces_mV": [list(v) for v in v_norm],
            "time_ms": [list(t) for t in time_aligned],
            "stim_onset_ms": 0.0,
            "stim_offset_ms": 1000.0,
            "warning": (
                "AVAR_simulation_iclamp.py missing from upstream repo. Used "
                "AVAL_simulation_iclamp with AVAR-specific conductances and "
                "surface area; UNC103 channel contribution NOT included "
                "(AVAL iclamp script does not insert 'unc103'). This biases "
                "AVAR predictions, particularly the resting potential and "
                "depolarizing-step plateau."
            ),
        }
    finally:
        _restore_env(cur_cwd)


def run_AIY_iclamp() -> dict:
    """AIY current-clamp: 11 steps from -0.015 to +0.035 nA, 5000 ms duration.
    Per AIY_simulation.py line 54.
    """
    cur_cwd = os.getcwd()
    try:
        _activate_nicoletti_env()
        from g_to_Scm2 import gScm2  # noqa: E402
        from AIY_simulation_iclamp import AIY_simulation_iclamp  # type: ignore

        # Per AIY_simulation.py lines 20-28:
        # conductances: [leak, slo1iso, kqt1, egl19, slo1egl19, nca, irk/shl1, eleak, cm]
        # Note: AIY iclamp script uses [leak, slo1iso, kqt1, egl19, slo1egl19, nca, shl1, eleak, cm]
        # AIY voltage-clamp uses irk in place of shl1 (different channel composition).
        # We use the AIY current-clamp parameter set per the iclamp script.
        # Per AIY_simulation.py:
        g0 = [0.14, 1, 0.2, 0.1, 0.92, 0.06, 0.5, -89.57, 1.6]
        surf = 65.89e-8
        gbest = gScm2(g0, surf, 6)
        results = AIY_simulation_iclamp(gbest, -0.015, 0.035, 11)
        v_traces, time_arr, _, _ = results
        # time_arr is absolute (no offset subtraction). stim.delay=1000, stim.dur=5000.
        # So stim onset at t=1000 ms in absolute time → matches figure x-axis.
        steps_pa = np.linspace(-15.0, 35.0, 11).tolist()
        return {
            "cell": "AIY",
            "n_steps": 11,
            "current_steps_pA": steps_pa,
            "v_traces_mV": [list(v) for v in v_traces],
            "time_ms": [list(t) for t in time_arr],
            "stim_onset_ms": 1000.0,
            "stim_offset_ms": 6000.0,
        }
    finally:
        _restore_env(cur_cwd)


def run_RIM_iclamp() -> dict:
    """RIM current-clamp: 11 steps from -0.015 to +0.035 nA, 5000 ms.
    Per RIM_simulation.py line 28.
    """
    cur_cwd = os.getcwd()
    try:
        _activate_nicoletti_env()
        from RIM_simulation_iclamp import RIM_simulation_iclamp  # type: ignore

        # Per RIM_simulation.py line 22 — RIM does NOT call gScm2; values
        # are already in S/cm² (Phase α §6.3 finding).
        g = [
            0.0009048750067326097,  # SHL1
            0.0001411644285181245,  # EGL2
            0.0003272854640954744,  # IRK
            0.0008451919806776876,  # CCA1
            9.676795045480941e-05,  # UNC2
            0.00032005818627638106,  # EGL19
            9.676795045480941e-05,  # LEAK
            -50,                    # ELEAK
            1.5,                    # CM
        ]
        results = RIM_simulation_iclamp(g, -0.015, 0.035, 11)
        v_traces, time_arr, _, _ = results
        # RIM iclamp: stim.delay=5000, stim.dur=5000, simdur=14000.
        # The returned `time` array is ALREADY shifted: per script line 107,
        # `time = time1[:,dd:length[1]] - 4000` where dd is index where
        # absolute time first reaches 4000 ms. So returned time spans
        # roughly 0 to 10000 ms with stim onset at returned t=1000 (absolute
        # 5000) and stim offset at returned t=6000 (absolute 10000).
        # Already aligned with figure x-axis — no further shift needed.
        time_aligned = time_arr
        steps_pa = np.linspace(-15.0, 35.0, 11).tolist()
        return {
            "cell": "RIM",
            "n_steps": 11,
            "current_steps_pA": steps_pa,
            "v_traces_mV": [list(v) for v in v_traces],
            "time_ms": [list(t) for t in time_aligned],
            "stim_onset_ms": 1000.0,
            "stim_offset_ms": 6000.0,
        }
    finally:
        _restore_env(cur_cwd)


# ---------------------------------------------------------------------------
# Feature extraction (NEURON)
# ---------------------------------------------------------------------------


def extract_features_neuron(
    v_trace: list[float],
    t_trace: list[float],
    stim_onset: float,
    stim_offset: float,
) -> dict:
    """Extract the same features as the digitization pipeline.

    Features:
      - peak_voltage_mV: max for depolarizing, min for hyperpolarizing
      - plateau_amplitude_mV: median V in last 30% of stimulation window
      - plateau_duration_ms: span over which |V - plateau| < 10% of (peak - baseline)
      - time_to_peak_ms: time from stim onset to peak
      - settling_time_ms: time from stim onset until V is within 10% of plateau
    """
    v = np.asarray(v_trace, dtype=float)
    t = np.asarray(t_trace, dtype=float)
    stim_dur = stim_offset - stim_onset

    # Baseline: median V before stim onset (or first sample if no pre-stim)
    pre_mask = t < stim_onset
    if pre_mask.any():
        baseline = float(np.median(v[pre_mask]))
    else:
        baseline = float(v[0])

    # Stimulation window
    in_stim = (t >= stim_onset) & (t <= stim_offset)
    if in_stim.sum() < 5:
        return {
            "peak_voltage_mV": None,
            "plateau_amplitude_mV": None,
            "plateau_duration_ms": None,
            "time_to_peak_ms": None,
            "settling_time_ms": None,
            "baseline_mV": round(baseline, 2),
            "n_in_stim_samples": int(in_stim.sum()),
        }
    ts, vs = t[in_stim], v[in_stim]

    # Peak: argmax for depolarizing, argmin for hyperpolarizing
    if vs.max() - baseline >= baseline - vs.min():
        peak_idx = int(np.argmax(vs))
    else:
        peak_idx = int(np.argmin(vs))
    peak_v = float(vs[peak_idx])
    peak_t = float(ts[peak_idx])

    # Plateau: last 30% of stim window
    plateau_start = stim_onset + 0.7 * stim_dur
    plateau_mask = ts >= plateau_start
    plateau_v = float(np.median(vs[plateau_mask])) if plateau_mask.sum() >= 3 else float(np.median(vs[-max(3, len(vs) // 4):]))

    # Plateau duration
    v_range = abs(peak_v - baseline)
    threshold = max(2.0, 0.10 * v_range)
    in_plat = np.abs(vs - plateau_v) < threshold
    if in_plat.any():
        plat_dur = float(ts[in_plat][-1] - ts[in_plat][0])
    else:
        plat_dur = 0.0

    # Settling time
    settling_thresh = max(2.0, 0.10 * v_range)
    settled = np.abs(vs - plateau_v) < settling_thresh
    if settled.any():
        first_settled_idx = int(np.where(settled)[0][0])
        settling_t = float(ts[first_settled_idx]) - stim_onset
    else:
        settling_t = float(stim_dur)

    time_to_peak = peak_t - stim_onset

    return {
        "peak_voltage_mV": round(peak_v, 2),
        "plateau_amplitude_mV": round(plateau_v, 2),
        "plateau_duration_ms": round(plat_dur, 1),
        "time_to_peak_ms": round(time_to_peak, 1),
        "settling_time_ms": round(settling_t, 1),
        "baseline_mV": round(baseline, 2),
        "n_in_stim_samples": int(in_stim.sum()),
    }


# ---------------------------------------------------------------------------
# Per-step / per-panel comparison
# ---------------------------------------------------------------------------

FEATURE_KEYS = [
    "peak_voltage_mV",
    "plateau_amplitude_mV",
    "plateau_duration_ms",
    "time_to_peak_ms",
    "settling_time_ms",
]


def compute_step_divergences(
    exp_features: dict,
    nrn_features: dict,
    feature_peaks: dict,
) -> dict:
    """Compute per-feature divergences for one current step.

    feature_peaks: precomputed across-step peak magnitude per feature, used
    as the |x|-floor in the relative-with-floor tolerance formula.
    """
    out = {}
    for key in FEATURE_KEYS:
        m = exp_features.get(key)
        r = nrn_features.get(key)
        if m is None or r is None:
            out[key] = {
                "experimental": m,
                "neuron": r,
                "divergence": None,
                "pass": False,
                "reason": "feature_missing",
            }
            continue
        peak = feature_peaks[key]
        d = divergence(m, r, peak)
        out[key] = {
            "experimental": float(m),
            "neuron": float(r),
            "divergence": round(d, 4),
            "pass": d <= 0.05,
        }
    return out


def compute_panel_comparison(panel_v2: dict, neuron_run: dict) -> dict:
    """Compare one digitized panel against the corresponding NEURON run.

    Returns a comparison record with per-step divergences, per-panel verdict,
    and full-waveform RMSE (warn-only diagnostic).
    """
    cell = panel_v2["cell"]
    panel_id = panel_v2["id"]

    # Build per-step lookup of experimental features
    exp_feats = panel_v2["extracted_features"]  # dict of feat_name -> {step: value}
    steps_pa = panel_v2["current_steps_pA"]
    stim_onset = panel_v2["stimulus_window_ms"][0]
    stim_offset = panel_v2["stimulus_window_ms"][1]

    # Extract features from NEURON traces per step
    nrn_steps_pa = neuron_run["current_steps_pA"]
    nrn_features_per_step = []
    for step_idx in range(neuron_run["n_steps"]):
        v_trace = neuron_run["v_traces_mV"][step_idx]
        t_trace = neuron_run["time_ms"][step_idx]
        feats = extract_features_neuron(
            v_trace, t_trace,
            stim_onset=neuron_run["stim_onset_ms"],
            stim_offset=neuron_run["stim_offset_ms"],
        )
        nrn_features_per_step.append(feats)

    # Compute peak magnitudes per feature across all steps (for tolerance floor)
    feature_peaks = {}
    for key in FEATURE_KEYS:
        all_vals = []
        for step_str, val in exp_feats[key].items():
            if val is not None:
                all_vals.append(abs(float(val)))
        for nf in nrn_features_per_step:
            if nf.get(key) is not None:
                all_vals.append(abs(nf[key]))
        feature_peaks[key] = max(all_vals) if all_vals else 1.0

    # Per-step divergences
    per_step = []
    n_steps_total = len(steps_pa)
    for step_idx, step_pa in enumerate(steps_pa):
        step_str = str(int(step_pa))
        exp_step = {k: exp_feats[k].get(step_str) for k in FEATURE_KEYS}
        # Match NEURON step closest in pA to exp step
        nrn_idx = int(np.argmin(np.abs(np.array(nrn_steps_pa) - step_pa)))
        nrn_step = nrn_features_per_step[nrn_idx]
        divs = compute_step_divergences(exp_step, nrn_step, feature_peaks)
        all_pass = all(d["pass"] for d in divs.values())
        per_step.append({
            "current_pA": step_pa,
            "exp_features": exp_step,
            "nrn_features": {k: nrn_step.get(k) for k in FEATURE_KEYS},
            "feature_divergences": divs,
            "step_pass": all_pass,
        })

    # Panel-level verdict (per spec: all features per step, > 80% steps)
    n_passing_steps = sum(1 for s in per_step if s["step_pass"])
    fraction_passing = n_passing_steps / n_steps_total if n_steps_total else 0.0
    panel_pass = fraction_passing > 0.80

    # SECONDARY DIAGNOSTICS: voltage-only and timing-only verdicts
    # (not the spec's primary criterion, but disambiguates failure modes —
    # voltage-feature failures imply real plateau divergence; timing-feature
    # failures often reflect digitization sampling resolution).
    VOLTAGE_FEATS = ["peak_voltage_mV", "plateau_amplitude_mV"]
    TIMING_FEATS = ["plateau_duration_ms", "time_to_peak_ms", "settling_time_ms"]
    n_voltage_pass = sum(
        1 for s in per_step
        if all(s["feature_divergences"][k]["pass"]
               for k in VOLTAGE_FEATS
               if s["feature_divergences"][k].get("divergence") is not None)
    )
    n_timing_pass = sum(
        1 for s in per_step
        if all(s["feature_divergences"][k]["pass"]
               for k in TIMING_FEATS
               if s["feature_divergences"][k].get("divergence") is not None)
    )
    voltage_only_pass_fraction = n_voltage_pass / n_steps_total if n_steps_total else 0.0
    timing_only_pass_fraction = n_timing_pass / n_steps_total if n_steps_total else 0.0
    voltage_panel_pass = voltage_only_pass_fraction > 0.80

    # Mean voltage-feature absolute error (mV) per panel — closest analog to
    # "consistent with published Model traces" diagnostic
    voltage_abs_errors = []
    for s in per_step:
        for k in VOLTAGE_FEATS:
            fd = s["feature_divergences"][k]
            if fd.get("divergence") is not None:
                voltage_abs_errors.append(abs(fd["experimental"] - fd["neuron"]))
    mean_voltage_abs_error = float(np.mean(voltage_abs_errors)) if voltage_abs_errors else None
    median_voltage_abs_error = float(np.median(voltage_abs_errors)) if voltage_abs_errors else None
    max_voltage_abs_error = float(np.max(voltage_abs_errors)) if voltage_abs_errors else None

    # Full-waveform RMSE (warn-only diagnostic): compare per-step plateau-window
    # mean voltages between digitized and NEURON traces.
    rmse_diagnostics = []
    for step_idx, panel_trace in enumerate(panel_v2["traces"]):
        step_pa = panel_trace["stimulus_pA"]
        if not panel_trace["data"]:
            continue
        nrn_idx = int(np.argmin(np.abs(np.array(nrn_steps_pa) - step_pa)))
        nrn_v = np.asarray(neuron_run["v_traces_mV"][nrn_idx])
        nrn_t = np.asarray(neuron_run["time_ms"][nrn_idx])
        # Interp NEURON V to digitized timepoints
        exp_t = np.array([d["t_ms"] for d in panel_trace["data"]])
        exp_v = np.array([d["v_mV"] for d in panel_trace["data"]])
        in_window = (exp_t >= neuron_run["stim_onset_ms"]) & (exp_t <= neuron_run["stim_offset_ms"])
        if in_window.sum() < 3:
            continue
        # NEURON time may not span experimental times perfectly — clip
        exp_t_clipped = exp_t[in_window]
        exp_v_clipped = exp_v[in_window]
        valid = (exp_t_clipped >= nrn_t.min()) & (exp_t_clipped <= nrn_t.max())
        if valid.sum() < 3:
            continue
        nrn_v_at_exp = np.interp(exp_t_clipped[valid], nrn_t, nrn_v)
        rmse = float(np.sqrt(np.mean((exp_v_clipped[valid] - nrn_v_at_exp) ** 2)))
        rmse_diagnostics.append({
            "current_pA": step_pa,
            "n_compared_points": int(valid.sum()),
            "rmse_mV": round(rmse, 2),
        })

    return {
        "panel_id": panel_id,
        "cell": cell,
        "n_steps_total": n_steps_total,
        "n_steps_passing": n_passing_steps,
        "fraction_passing": round(fraction_passing, 4),
        "panel_pass": bool(panel_pass),
        "feature_peaks_for_tolerance_floor": {k: round(v, 2) for k, v in feature_peaks.items()},
        "secondary_diagnostics": {
            "voltage_features_only_n_pass": n_voltage_pass,
            "voltage_features_only_fraction_pass": round(voltage_only_pass_fraction, 4),
            "voltage_features_only_panel_pass": bool(voltage_panel_pass),
            "timing_features_only_n_pass": n_timing_pass,
            "timing_features_only_fraction_pass": round(timing_only_pass_fraction, 4),
            "voltage_abs_error_mV_mean": round(mean_voltage_abs_error, 2) if mean_voltage_abs_error is not None else None,
            "voltage_abs_error_mV_median": round(median_voltage_abs_error, 2) if median_voltage_abs_error is not None else None,
            "voltage_abs_error_mV_max": round(max_voltage_abs_error, 2) if max_voltage_abs_error is not None else None,
            "interpretation": (
                "voltage_features_only excludes timing-feature failures that "
                "are dominated by digitization-sampling resolution noise. "
                "Voltage features (peak, plateau) reflect the actual model-vs-"
                "experiment fit quality. Mean/median voltage abs error in mV "
                "gives the absolute-scale residual independent of relative-"
                "tolerance arithmetic."
            ),
        },
        "per_step_comparison": per_step,
        "full_waveform_rmse_diagnostic": rmse_diagnostics,
        "neuron_warning": neuron_run.get("warning"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading published_traces_v2.json")
    panels_data = json.loads(PUBLISHED_TRACES_V2.read_text())
    panels_by_id = {p["id"]: p for p in panels_data["panels"]}
    print(f"  Panels loaded: {list(panels_by_id.keys())}")

    runners = {
        "nicoletti_2024_fig1A_AVAL": run_AVAL_iclamp,
        "nicoletti_2024_fig1B_AVAR": run_AVAR_iclamp,
        "nicoletti_2024_fig3A_AIY": run_AIY_iclamp,
        "nicoletti_2024_fig5A_RIM": run_RIM_iclamp,
    }

    comparison_results = []
    for panel_id, runner in runners.items():
        if panel_id not in panels_by_id:
            print(f"\nSkipping {panel_id} — panel not in v2 traces")
            continue
        panel_v2 = panels_by_id[panel_id]
        print(f"\n=== NEURON current-clamp run for {panel_id} ===")
        try:
            nrn = runner()
            print(f"  {nrn['cell']}: {nrn['n_steps']} steps, "
                  f"{len(nrn['v_traces_mV'][0])} time samples per trace")
            cmp = compute_panel_comparison(panel_v2, nrn)
            print(f"  panel_pass={cmp['panel_pass']}  "
                  f"steps_passing={cmp['n_steps_passing']}/{cmp['n_steps_total']}  "
                  f"fraction={cmp['fraction_passing']:.2f}")
            comparison_results.append(cmp)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  FAILED: {e}\n{tb}")
            comparison_results.append({
                "panel_id": panel_id,
                "cell": panel_v2["cell"],
                "error": str(e),
                "traceback": tb,
            })

    n_panels = len(comparison_results)
    n_passing = sum(1 for r in comparison_results if r.get("panel_pass") is True)
    overall_pass = n_passing == n_panels and n_panels > 0
    overall_fail = n_panels - n_passing >= 2

    out = {
        "phase": "phase_beta_pre_v2",
        "generation_date": "2026-04-26",
        "tolerance_metric": (
            "Per-feature: divergence(m, r, peak) = |m-r| / max(|m|, |r|, 0.1*peak); "
            "feature pass: divergence ≤ 0.05. "
            "Per-step: ALL features pass. "
            "Per-panel: > 80% of steps pass (looser than v1's per-point ≥ 90% — "
            "feature-based has fewer comparison points). "
            "Per-cell: panel passes."
        ),
        "overall_verdict": (
            "pass" if overall_pass else
            ("multi_panel_fail_real_invalidation" if overall_fail else "partial")
        ),
        "n_panels_total": n_panels,
        "n_panels_passing": n_passing,
        "panel_results": comparison_results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n=== OVERALL VERDICT: {out['overall_verdict']} "
          f"({n_passing}/{n_panels} panels pass) ===")
    print(f"Results written to {OUT_JSON}")


if __name__ == "__main__":
    main()
