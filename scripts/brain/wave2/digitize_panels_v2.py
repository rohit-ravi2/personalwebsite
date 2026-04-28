"""
Phase β-pre v2 digitization driver — fit-target current-clamp panels.

Selected panels (all current-clamp, fit-target per Nicoletti 2024 captions):
  - Fig 1A: AVAL current-clamp, 7 steps -30 to +30 pA × 1000 ms
  - Fig 1B: AVAR current-clamp, 7 steps -30 to +30 pA × 1000 ms
  - Fig 3A: AIY current-clamp, 11 steps -15 to +35 pA × 5000 ms
  - Fig 5A: RIM current-clamp, 11 steps -15 to +35 pA × 5000 ms

Tool hierarchy (per spec):
  1. plotdigitizer — tested unsuitable for batch multi-curve overlays
     (CLI requires per-trace pixel locations for axis calibration; no
     advantage over a custom OpenCV pipeline for our use case).
  2. OpenCV color-mask + per-step centerline extraction — used.
  3. Manual grid-reading — fallback if OpenCV fails on a panel.

Methodology:
- Calibrate axes once per panel using known tick positions identified via
  one-time visual inspection of a panel + gridline overlay.
- For each current step, the experimental trace stabilizes at a distinct
  steady-state voltage during the stimulation window. The traces don't
  overlap each other in voltage space (each step has a unique plateau).
- Extract black-pixel mask (excluding red Model trace, blue Model trace
  for AVAR, and panel chrome).
- Per stimulation timepoint: sample the median voltage of black pixels
  in a vertical strip around that x position, segregating by trace via
  voltage-range clustering anchored to the protocol's expected steady-
  state voltages.

Author: Phase β-pre v2 engineering session, 2026-04-26.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FIGURES_DIR = Path(
    "/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures"
)
OUT_JSON = Path(
    "/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/published_traces_v2.json"
)


@dataclass
class AxisCalibration:
    """Axis calibration for a panel.

    Linear mapping: data_value = slope * pixel + intercept.
    """
    x_slope: float       # ms per pixel
    x_intercept: float   # ms at pixel x=0
    y_slope: float       # mV per pixel
    y_intercept: float   # mV at pixel y=0
    # Plot frame bounds in pixel space (panel-local)
    plot_x_lo: int
    plot_x_hi: int
    plot_y_lo: int       # top of plot area (high voltage)
    plot_y_hi: int       # bottom of plot area (low voltage)
    # Stimulus window in data coordinates
    stim_t_start_ms: float
    stim_t_end_ms: float

    def px_to_t(self, x_px: float) -> float:
        return self.x_slope * x_px + self.x_intercept

    def px_to_v(self, y_px: float) -> float:
        return self.y_slope * y_px + self.y_intercept

    def t_to_px(self, t_ms: float) -> float:
        return (t_ms - self.x_intercept) / self.x_slope

    def v_to_px(self, v_mV: float) -> float:
        return (v_mV - self.y_intercept) / self.y_slope


# ---------------------------------------------------------------------------
# Per-panel calibrations, identified via one-time visual inspection of grid-
# overlaid panels at the actual pixel resolution on disk.
# ---------------------------------------------------------------------------


def _calibrate_two_points(p1_px, p1_data, p2_px, p2_data):
    """Linear calibration from two (pixel, data) anchors. Returns (slope, intercept)
    such that data = slope * px + intercept.
    """
    slope = (p2_data - p1_data) / (p2_px - p1_px)
    intercept = p1_data - slope * p1_px
    return slope, intercept


# Fig 1A AVAL (520x520 v1 panel)
# Tick positions read from /tmp/v2_inspect/nicoletti_2024_fig1A_AVAL_iclamp_grid.png
# Y: 100 mV at y≈145, -150 mV at y≈345 → slope = -250/200 = -1.25 mV/px
# X: 0 ms at x≈230, 1000 ms at x≈470 → slope = 1000/240 = 4.167 ms/px
def cal_fig1A():
    x_slope, x_int = _calibrate_two_points(230, 0.0, 470, 1000.0)
    y_slope, y_int = _calibrate_two_points(145, 100.0, 345, -150.0)
    return AxisCalibration(
        x_slope=x_slope, x_intercept=x_int,
        y_slope=y_slope, y_intercept=y_int,
        plot_x_lo=230, plot_x_hi=475,
        plot_y_lo=120, plot_y_hi=380,
        stim_t_start_ms=0.0, stim_t_end_ms=1000.0,
    )


# Fig 1B AVAR (520x520 v1 panel) — model traces are BLUE, not red
def cal_fig1B():
    x_slope, x_int = _calibrate_two_points(215, 0.0, 455, 1000.0)
    y_slope, y_int = _calibrate_two_points(155, 100.0, 360, -150.0)
    return AxisCalibration(
        x_slope=x_slope, x_intercept=x_int,
        y_slope=y_slope, y_intercept=y_int,
        plot_x_lo=215, plot_x_hi=460,
        plot_y_lo=120, plot_y_hi=390,
        stim_t_start_ms=0.0, stim_t_end_ms=1000.0,
    )


# Fig 3A AIY (1421x1144 v2 panel)
# Tick positions read from /tmp/v2_inspect/fig3A_left_y_ticks.png and bot_x_ticks.png
# Y: 50 mV at y≈215, -150 mV at y≈880 → slope = -200/665 = -0.3008 mV/px
# X: 2000 ms at x≈480, 8000 ms at x≈1115 → slope = 6000/635 = 9.449 ms/px
# Stimulus window per Fig 3 caption: starts at ~1000 ms, ends at ~6000 ms
def cal_fig3A():
    x_slope, x_int = _calibrate_two_points(480, 2000.0, 1115, 8000.0)
    y_slope, y_int = _calibrate_two_points(215, 50.0, 880, -150.0)
    return AxisCalibration(
        x_slope=x_slope, x_intercept=x_int,
        y_slope=y_slope, y_intercept=y_int,
        plot_x_lo=270, plot_x_hi=1115,
        plot_y_lo=180, plot_y_hi=900,
        stim_t_start_ms=1000.0, stim_t_end_ms=6000.0,
    )


# Fig 5A RIM (1460x1126 v2 panel)
# Y: 100 mV at y≈145, -100 mV at y≈695 → slope = -200/550 = -0.3636 mV/px
# X: 0 ms at x≈215 (extrapolated), 6000 ms at x≈905 (last visible tick), 8000 at x≈1075
# Re-reading from fig5A_left_y_ticks.png: 100 at y=145, 0 at y=425, -100 at y=695
# X: peeked from full thumbnail: protocol same 1000ms-6000ms stimulus window
# Will use: 0 ms at x≈225, 8000 ms at x≈1085  (based on equal-spacing of 2000 ms)
def cal_fig5A():
    x_slope, x_int = _calibrate_two_points(225, 0.0, 1085, 8000.0)
    y_slope, y_int = _calibrate_two_points(145, 100.0, 695, -100.0)
    return AxisCalibration(
        x_slope=x_slope, x_intercept=x_int,
        y_slope=y_slope, y_intercept=y_int,
        plot_x_lo=240, plot_x_hi=1085,
        plot_y_lo=110, plot_y_hi=830,
        stim_t_start_ms=1000.0, stim_t_end_ms=6000.0,
    )


# ---------------------------------------------------------------------------
# Trace extraction
# ---------------------------------------------------------------------------


def extract_black_mask(img_bgr, exclude_red=True, exclude_blue=False):
    """Return a binary mask of pixels that are predominantly black (trace ink),
    excluding red model traces (and optionally blue, for Fig 1B AVAR).
    """
    b, g, r = cv2.split(img_bgr)
    # Black: all channels low
    is_black = (r < 90) & (g < 90) & (b < 90)
    # Exclude red: high R, low G/B
    if exclude_red:
        is_red = (r > 150) & (g < 100) & (b < 100)
        is_black = is_black & ~is_red
    if exclude_blue:
        is_blue = (b > 150) & (g < 120) & (r < 120)
        is_black = is_black & ~is_blue
    return is_black.astype(np.uint8)


def sample_traces_per_step(
    img_bgr,
    cal: AxisCalibration,
    expected_steady_state_v: list[float],
    sampling_ms: list[float],
    exclude_blue: bool = False,
    voltage_capture_radius_mV: float = 6.0,
):
    """Per stimulation timepoint, extract the experimental trace voltage value
    nearest to each expected steady-state plateau.

    Returns: dict mapping (step_index, t_ms) → measured_v_mV (float, or None
    if no black pixels found in the capture window).
    """
    mask = extract_black_mask(img_bgr, exclude_red=True, exclude_blue=exclude_blue)
    H, W = mask.shape
    n_steps = len(expected_steady_state_v)
    # Output: per step, a list of (t_ms, measured_v_mV) samples
    traces = {i: [] for i in range(n_steps)}

    for t_ms in sampling_ms:
        x_px = int(round(cal.t_to_px(t_ms)))
        if not (cal.plot_x_lo <= x_px <= cal.plot_x_hi):
            continue
        # Vertical strip ±2 px around x_px
        x_lo = max(cal.plot_x_lo, x_px - 2)
        x_hi = min(cal.plot_x_hi, x_px + 3)
        strip_cols = mask[:, x_lo:x_hi]
        # For each row, count black pixels — rows with any black are candidates
        row_count = strip_cols.sum(axis=1)
        candidate_rows = np.where(row_count > 0)[0]
        if len(candidate_rows) == 0:
            continue
        # Convert candidate rows to voltages
        candidate_v = np.array([cal.px_to_v(y) for y in candidate_rows])
        # For each step's expected steady-state voltage, find candidates within
        # capture_radius_mV and take the one closest to expected (median, robust
        # to outlier black pixels from axis labels or adjacent traces).
        for step_idx, expected_v in enumerate(expected_steady_state_v):
            within = np.abs(candidate_v - expected_v) < voltage_capture_radius_mV
            if not within.any():
                continue
            # Take median voltage of black pixels in window — robust to noise
            v_meas = float(np.median(candidate_v[within]))
            traces[step_idx].append((float(t_ms), v_meas))
    return traces


def extract_features_per_step(
    traces_per_step: dict,
    cal: AxisCalibration,
    expected_steady_state_v: list[float],
):
    """Compute features per step from the extracted (t, V) trajectory.

    Features:
      - peak_voltage_mV: maximum voltage reached during stimulation window
      - plateau_amplitude_mV: median voltage in the last 30% of stimulation
        window (steady-state value)
      - plateau_duration_ms: estimated duration where |V - plateau| < 10% range
      - time_to_peak_ms: time from stimulus onset to peak (relative to stim_t_start)
      - settling_time_ms: time from stimulus onset until V is within 10% of
        final plateau value (relative to stim_t_start)
    """
    features = {
        "peak_voltage_mV": {},
        "plateau_amplitude_mV": {},
        "plateau_duration_ms": {},
        "time_to_peak_ms": {},
        "settling_time_ms": {},
        "n_samples_per_step": {},
    }
    stim_dur = cal.stim_t_end_ms - cal.stim_t_start_ms

    for step_idx in sorted(traces_per_step.keys()):
        samples = traces_per_step[step_idx]
        # filter to stimulation window
        in_stim = [(t, v) for (t, v) in samples
                   if cal.stim_t_start_ms <= t <= cal.stim_t_end_ms]
        n = len(in_stim)
        features["n_samples_per_step"][step_idx] = n
        if n < 5:
            for k in ["peak_voltage_mV", "plateau_amplitude_mV",
                      "plateau_duration_ms", "time_to_peak_ms",
                      "settling_time_ms"]:
                features[k][step_idx] = None
            continue

        ts = np.array([t for t, _ in in_stim])
        vs = np.array([v for _, v in in_stim])
        # Sort by time
        order = np.argsort(ts)
        ts, vs = ts[order], vs[order]

        # Peak: argmax for depolarizing steps, argmin for hyperpolarizing
        baseline = vs[0]
        if vs.max() - baseline >= baseline - vs.min():
            peak_idx = int(np.argmax(vs))
        else:
            peak_idx = int(np.argmin(vs))
        peak_v = float(vs[peak_idx])
        peak_t = float(ts[peak_idx])

        # Plateau: last 30% of stim window
        plateau_start = cal.stim_t_start_ms + 0.7 * stim_dur
        plateau_mask = ts >= plateau_start
        if plateau_mask.sum() >= 3:
            plateau_v = float(np.median(vs[plateau_mask]))
        else:
            plateau_v = float(np.median(vs[-max(3, n // 4):]))

        # Plateau duration: span over which |V - plateau| < 10% of (peak - baseline)
        v_range = abs(peak_v - baseline)
        threshold = max(2.0, 0.10 * v_range)  # min 2 mV
        in_plat = np.abs(vs - plateau_v) < threshold
        if in_plat.any():
            plat_t_start = float(ts[in_plat][0])
            plat_t_end = float(ts[in_plat][-1])
            plat_dur = plat_t_end - plat_t_start
        else:
            plat_dur = 0.0

        # Settling time: time from stim start until V is within 10% of plateau
        settling_thresh = max(2.0, 0.10 * v_range)
        # First time V settles within threshold of plateau and stays there
        settled = np.abs(vs - plateau_v) < settling_thresh
        if settled.any():
            first_settled_idx = int(np.where(settled)[0][0])
            settling_t = float(ts[first_settled_idx]) - cal.stim_t_start_ms
        else:
            settling_t = float(stim_dur)

        time_to_peak = peak_t - cal.stim_t_start_ms

        features["peak_voltage_mV"][step_idx] = round(peak_v, 2)
        features["plateau_amplitude_mV"][step_idx] = round(plateau_v, 2)
        features["plateau_duration_ms"][step_idx] = round(plat_dur, 1)
        features["time_to_peak_ms"][step_idx] = round(time_to_peak, 1)
        features["settling_time_ms"][step_idx] = round(settling_t, 1)

    return features


# ---------------------------------------------------------------------------
# Per-panel digitization config
# ---------------------------------------------------------------------------


# Expected steady-state voltages per current step, read by visual inspection
# of each panel during axis calibration. These are approximate plateau levels;
# the digitization will home in within ±6 mV of these anchors per step.

# Fig 1A AVAL: 7 steps -30 to +30 pA, 10 pA spacing. Plateau voltages
# determined empirically from a black-pixel histogram diagnostic (t=200,500,
# 700 ms): 7 distinct voltage clusters at ~-170, -130, -85, -30, 40, 80, 105 mV.
FIG1A_STEPS_PA = [-30, -20, -10, 0, 10, 20, 30]
FIG1A_PLATEAUS_MV = [-170, -130, -85, -30, 40, 80, 105]

# Fig 1B AVAR: empirically determined plateau voltages from a y-tick-overlay
# inspection of the panel. AVAR has more compressed I-V than AVAL —
# hyperpolarizing steps are smaller in magnitude.
FIG1B_STEPS_PA = [-30, -20, -10, 0, 10, 20, 30]
FIG1B_PLATEAUS_MV = [-110, -80, -55, -25, 30, 65, 90]

# Fig 3A AIY: 11 steps -15 to +35 pA, 5 pA spacing (per Fig 3 caption: 11 steps).
# Plateau voltages from rendered panel: distinct baselines visible from
# ~-130 mV (most hyperpolarized) up through ~+30 mV (most depolarized).
FIG3A_STEPS_PA = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
FIG3A_PLATEAUS_MV = [-130, -110, -75, -55, -25, 0, 15, 22, 26, 30, 32]

# Fig 5A RIM: 11 steps -15 to +35 pA. Plateau voltages from rendered panel:
# baseline ~-50 mV, hyperpolarization down to ~-105 mV, depolarization up to
# ~+70 mV.
FIG5A_STEPS_PA = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
FIG5A_PLATEAUS_MV = [-105, -100, -90, -50, 5, 30, 40, 50, 60, 65, 70]


# ---------------------------------------------------------------------------
# Build output JSON
# ---------------------------------------------------------------------------


def _digitize_one_panel(
    panel_id: str,
    image_filename: str,
    cell: str,
    figure_number: str,
    panel_letter: str,
    cal: AxisCalibration,
    steps_pa: list[float],
    expected_plateaus: list[float],
    exclude_blue: bool,
    protocol_detail: str,
    experimental_origin: str,
    fit_target_quote: str,
    n_time_samples: int = 60,
) -> dict:
    img_path = FIGURES_DIR / image_filename
    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, f"Cannot read {img_path}"

    # Sampling timepoints span pre-stimulus + stimulation + post-stimulus
    pre_ms = max(0.0, cal.stim_t_start_ms - 200.0)
    post_ms = cal.stim_t_end_ms + 200.0
    sampling_ms = np.linspace(pre_ms, post_ms, n_time_samples).tolist()

    traces_per_step = sample_traces_per_step(
        img_bgr=img_bgr,
        cal=cal,
        expected_steady_state_v=expected_plateaus,
        sampling_ms=sampling_ms,
        exclude_blue=exclude_blue,
        voltage_capture_radius_mV=8.0,
    )

    features = extract_features_per_step(traces_per_step, cal, expected_plateaus)

    # Build trace records
    traces_out = []
    for step_idx, current_pa in enumerate(steps_pa):
        samples = traces_per_step[step_idx]
        traces_out.append({
            "stimulus_pA": float(current_pa),
            "expected_steady_state_v_anchor_mV": float(expected_plateaus[step_idx]),
            "data": [{"t_ms": round(t, 1), "v_mV": round(v, 2)} for t, v in samples],
            "n_points": len(samples),
            "tool": "opencv_color_mask_centerline",
        })

    record = {
        "id": panel_id,
        "source": {
            "paper": "Nicoletti M, Chiodo L, Loppini A, Liu Q, Folli V, Ruocco G, Filippi S. 2024. "
                     "Biophysical modeling of the whole-cell dynamics of C. elegans motor "
                     "and interneurons families.",
            "journal": "PLOS ONE 19(3): e0298105",
            "year": 2024,
            "doi": "10.1371/journal.pone.0298105",
            "figure": figure_number,
            "panel": panel_letter,
            "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298105",
        },
        "shows": "experimental_overlay_current_clamp",
        "cell": cell,
        "protocol": "current_clamp",
        "protocol_detail": protocol_detail,
        "fit_target": True,
        "fit_target_quote": fit_target_quote,
        "experimental_data_origin": experimental_origin,
        "x_axis": {"label": "Time", "units": "ms", "scale": "linear"},
        "y_axis": {"label": "Voltage", "units": "mV", "scale": "linear"},
        "n_steps": len(steps_pa),
        "current_steps_pA": [float(s) for s in steps_pa],
        "stimulus_window_ms": [cal.stim_t_start_ms, cal.stim_t_end_ms],
        "axis_calibration": {
            "x_slope_ms_per_px": round(cal.x_slope, 6),
            "x_intercept_ms": round(cal.x_intercept, 4),
            "y_slope_mV_per_px": round(cal.y_slope, 6),
            "y_intercept_mV": round(cal.y_intercept, 4),
            "plot_frame_px": [cal.plot_x_lo, cal.plot_y_lo, cal.plot_x_hi, cal.plot_y_hi],
        },
        "image_filename": image_filename,
        "traces": traces_out,
        "extracted_features": {
            k: {str(int(steps_pa[i])): v for i, v in d.items()}
            for k, d in features.items()
        },
        "digitization_notes": (
            "OpenCV color-mask extraction. For each protocol-aligned timepoint, "
            "black pixels (excluding red Model traces" +
            (" and blue Model traces" if exclude_blue else "") +
            ") within ±2 px of the expected x are gathered; per-step plateau "
            "voltages anchor a voltage-window matching that segregates traces. "
            "Median voltage of pixels within ±8 mV of each step's expected "
            "plateau gives the per-step (t, V) sample. Tool selection: "
            "plotdigitizer was tested but found unsuitable for batch multi-trace "
            "overlays without per-trace pixel anchors. Manual grid-reading was "
            "kept as fallback but not invoked. Reading uncertainty: ~1 mV "
            "(panel y-resolution) + voltage-window assignment ambiguity in "
            "regions where two traces approach (mainly steps 0 and ±5 pA where "
            "baseline dominates)."
        ),
    }
    return record


def main():
    panels = []

    panels.append(_digitize_one_panel(
        panel_id="nicoletti_2024_fig1A_AVAL",
        image_filename="nicoletti_2024_fig1A_AVAL_iclamp.png",
        cell="AVAL",
        figure_number="1",
        panel_letter="A",
        cal=cal_fig1A(),
        steps_pa=FIG1A_STEPS_PA,
        expected_plateaus=FIG1A_PLATEAUS_MV,
        exclude_blue=False,
        protocol_detail="7 current steps from -30 pA to +30 pA, 1000 ms duration "
                        "(Nicoletti 2024 Fig 1 caption).",
        experimental_origin="Liu P, Chen B, Wang Z-W. 2018 (ref [29] in Nicoletti 2024). "
                            "Patch-clamp recordings on AVAL neurons.",
        fit_target_quote="\"The models were fitted on experimental current-clamp data "
                         "obtained from [29], and shown in black in panels A and B.\" "
                         "(Nicoletti 2024, Fig 1 caption)",
    ))

    panels.append(_digitize_one_panel(
        panel_id="nicoletti_2024_fig1B_AVAR",
        image_filename="nicoletti_2024_fig1B_AVAR_iclamp.png",
        cell="AVAR",
        figure_number="1",
        panel_letter="B",
        cal=cal_fig1B(),
        steps_pa=FIG1B_STEPS_PA,
        expected_plateaus=FIG1B_PLATEAUS_MV,
        exclude_blue=True,  # AVAR Model traces are blue, not red
        protocol_detail="7 current steps from -30 pA to +30 pA, 1000 ms duration "
                        "(Nicoletti 2024 Fig 1 caption).",
        experimental_origin="Liu P, Chen B, Wang Z-W. 2018 (ref [29] in Nicoletti 2024). "
                            "Patch-clamp recordings on AVAR neurons.",
        fit_target_quote="\"The models were fitted on experimental current-clamp data "
                         "obtained from [29], and shown in black in panels A and B.\" "
                         "(Nicoletti 2024, Fig 1 caption)",
    ))

    panels.append(_digitize_one_panel(
        panel_id="nicoletti_2024_fig3A_AIY",
        image_filename="nicoletti_2024_fig3A_AIY_iclamp.png",
        cell="AIY",
        figure_number="3",
        panel_letter="A",
        cal=cal_fig3A(),
        steps_pa=FIG3A_STEPS_PA,
        expected_plateaus=FIG3A_PLATEAUS_MV,
        exclude_blue=False,
        protocol_detail="11 current steps from -15 pA to +35 pA, 5000 ms duration "
                        "(Nicoletti 2024 Fig 3 caption).",
        experimental_origin="Liu Q et al. (ref [30] in Nicoletti 2024). "
                            "Patch-clamp recordings on AIY neurons.",
        fit_target_quote="\"The model was fitted on experimental current-clamp data "
                         "obtained from [30] and shown in black in panel A.\" "
                         "(Nicoletti 2024, Fig 3 caption)",
    ))

    panels.append(_digitize_one_panel(
        panel_id="nicoletti_2024_fig5A_RIM",
        image_filename="nicoletti_2024_fig5A_RIM_iclamp.png",
        cell="RIM",
        figure_number="5",
        panel_letter="A",
        cal=cal_fig5A(),
        steps_pa=FIG5A_STEPS_PA,
        expected_plateaus=FIG5A_PLATEAUS_MV,
        exclude_blue=False,
        protocol_detail="11 current steps from -15 pA to +35 pA, 5000 ms duration "
                        "(Nicoletti 2024 Fig 5 caption).",
        experimental_origin="Liu et al. (ref [30] in Nicoletti 2024). "
                            "Patch-clamp recordings on RIM neurons.",
        fit_target_quote="\"The model was fitted on experimental current- and "
                         "voltage-clamp data obtained from [30] and shown in "
                         "black in panels A and B.\" "
                         "(Nicoletti 2024, Fig 5 caption)",
    ))

    out = {
        "format_version": "2.0",
        "phase": "phase_beta_pre_v2",
        "generation_date": "2026-04-26",
        "selection_rationale": (
            "Four current-clamp panels selected as Nicoletti 2024's fit-target "
            "datasets per the paper's own captions. v1 digitized I-V curves "
            "(Fig 1F, 3D, 5D) which Nicoletti's body text discloses are post-"
            "hoc predictions, NOT fit targets — measuring against I-V curves "
            "yielded 39-149% divergences that re-state Nicoletti's own published "
            "I-V discrepancies. v2 measures the actual fit-target metric: "
            "current-clamp time-series traces shown in panels Fig 1A (AVAL), "
            "Fig 1B (AVAR), Fig 3A (AIY), Fig 5A (RIM)."
        ),
        "tool_hierarchy_used": {
            "plotdigitizer": "tested, found unsuitable for batch multi-trace "
                             "overlays (no per-trace anchor mode in CLI)",
            "opencv_color_mask": "primary tool used for all four panels",
            "manual_grid_reading": "fallback, not invoked",
        },
        "panels": panels,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    n_traces = sum(len(p["traces"]) for p in panels)
    n_pts = sum(t["n_points"] for p in panels for t in p["traces"])
    print(f"Wrote {OUT_JSON}")
    print(f"Total panels: {len(panels)}  traces: {n_traces}  data points: {n_pts}")
    for p in panels:
        n_with_data = sum(1 for t in p["traces"] if t["n_points"] >= 5)
        print(f"  {p['id']}: {n_with_data}/{p['n_steps']} steps with ≥5 data points")


if __name__ == "__main__":
    main()
