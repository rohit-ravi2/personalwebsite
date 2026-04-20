#!/usr/bin/env python3
"""Event & target extraction from Atanas-worm scalar behavior features.

Each extractor takes a per-worm .npz (from parse_atanas_all.py) and
returns a named dict of target arrays, each of length T=1600 aligned
to the neural timestamps. Tier thresholds follow the worm literature
where available.

Target types:
  - binary event-onset arrays (0/1 per timestep, sparse)
  - binary sustained-state arrays (0/1 per timestep, dense)
  - continuous scalar arrays (regression)
  - integer categorical arrays (multi-class classification)

Literature refs cited inline.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


# Sampling rate of Atanas neural/behavior stream
SAMPLE_HZ = 1.667
SAMPLE_DT = 1.0 / SAMPLE_HZ  # ~600 ms

# --- Literature thresholds -------------------------------------------------
# Reversal detection: `reversal_events` is Atanas's already-computed 0–1
# soft signal. Threshold at 0.5 for binary.
REVERSAL_THRESH = 0.5

# Omega turn (Gray, Hill, Bargmann 2005): body curvature >= ~1.5 rad
# (body folds such that head ~touches tail). Atanas body_curvature is
# in radians of cumulative body angle. Use 1.5 rad sustained 2+ samples.
OMEGA_CURV_THRESH_RAD = 1.5
OMEGA_SUSTAIN_SAMPLES = 2

# Pirouette (Pierce-Shimomura, Morse, Lockery 1999; Gray, Hill,
# Bargmann 2005): a high-turning-rate bout typically lasting ~10–20 s,
# containing ≥2 reversals. Window = 25 samples (~15 s).
PIROUETTE_REV_COUNT = 2
PIROUETTE_WINDOW_SAMPLES = 25

# Forward-run detection: velocity > 0.05 sustained ≥3 samples (~2 s).
FWD_VEL_THRESH = 0.05
FWD_SUSTAIN_SAMPLES = 3

# Quiescence (lethargus-like; Raizen 2008 — more stringent but we
# don't have long-recording lethargus here, pragmatic definition):
# |velocity| < 0.02 sustained ≥5 samples (~3 s).
QUI_VEL_THRESH = 0.02
QUI_SUSTAIN_SAMPLES = 5

# Roaming vs dwelling (Flavell, Bargmann et al. 2013): roaming = high
# forward speed over extended window with low turn rate; dwelling =
# inverse. Flavell 2013 used 10 s windows of centroid speed+angular
# speed then 2-state HMM. Pragmatic thresholded version:
ROAMING_VEL_THRESH = 0.05
ROAMING_TURN_THRESH = 0.3   # rad/s equivalent
ROAMING_WINDOW_SAMPLES = 50  # ~30 s window (compressed from Flavell's 10-min
                              # session analysis; OK for a 16-min recording)

# Speed burst: velocity exceeds mean + 1.5σ over the session.
BURST_SIGMA = 1.5

# Head-swing reversal: zero-crossing of head_curvature.
# (Direction is sign of head_curv at each timestep.)


def _binarize(x: np.ndarray, thr: float) -> np.ndarray:
    return (x > thr).astype(np.int64)


def _onsets(binary: np.ndarray) -> np.ndarray:
    """Return 1 at t where binary transitions 0→1."""
    out = np.zeros_like(binary)
    diff = np.diff(binary)
    out[1:] = (diff == 1).astype(binary.dtype)
    return out


def _offsets(binary: np.ndarray) -> np.ndarray:
    out = np.zeros_like(binary)
    diff = np.diff(binary)
    out[1:] = (diff == -1).astype(binary.dtype)
    return out


def _sustained(binary: np.ndarray, min_samples: int) -> np.ndarray:
    """For each t, 1 iff binary[t-min_samples+1..t] are all 1."""
    out = np.zeros_like(binary)
    rolling = np.ones(len(binary), dtype=int)
    for k in range(min_samples):
        if k == 0:
            rolling = binary.copy()
        else:
            rolling[k:] = rolling[k:] & binary[:-k]
            rolling[:k] = 0
    return rolling


def _cluster_events(event_onsets: np.ndarray, count: int,
                    window_samples: int) -> np.ndarray:
    """Return 1 at t where at least `count` event_onsets occur in
    [t-window_samples, t]. Used for pirouette detection."""
    T = len(event_onsets)
    out = np.zeros(T, dtype=np.int64)
    for t in range(T):
        lo = max(0, t - window_samples)
        if event_onsets[lo:t + 1].sum() >= count:
            out[t] = 1
    # Pirouette ENTRY = onset of this sustained state
    return _onsets(out.astype(np.int64))


def extract_events(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute all targets from a single worm's parsed data.

    Args:
        data: dict with keys velocity, head_curv, body_curv, ang_vel,
              reversal, pumping; each a (T,) float array.

    Returns:
        Dict {target_name: (T,) array}. Target types indicated by suffix:
          *_onset, *_offset, *_state (binary events/states)
          *_acc, *_rate, *_pc1, velocity, ... (continuous)
          state_5class, rev_mode_3class (multi-class integer)
    """
    v = data["velocity"]
    hc = data["head_curv"]
    bc = data["body_curv"]
    av = data["ang_vel"]
    rv = data["reversal"]
    T = len(v)

    out: dict[str, np.ndarray] = {}

    # --- Tier 1: rare transition events -----------------------------------
    rv_bin = _binarize(rv, REVERSAL_THRESH)
    out["reversal_onset"] = _onsets(rv_bin)
    out["reversal_offset"] = _offsets(rv_bin)
    out["reversal_state"] = rv_bin.copy()  # baseline comparison

    omega_bin = _binarize(np.abs(bc), OMEGA_CURV_THRESH_RAD)
    omega_sust = _sustained(omega_bin, OMEGA_SUSTAIN_SAMPLES)
    out["omega_onset"] = _onsets(omega_sust)

    out["pirouette_entry"] = _cluster_events(
        out["reversal_onset"], PIROUETTE_REV_COUNT, PIROUETTE_WINDOW_SAMPLES
    )

    # Head-swing direction reversals: zero crossings of head_curv
    hc_sign = np.sign(hc)
    hc_flip = np.zeros(T, dtype=np.int64)
    hc_flip[1:] = (hc_sign[1:] != hc_sign[:-1]).astype(np.int64)
    out["headswing_flip"] = hc_flip
    # Dorsal/ventral head bias (continuous-ish)
    out["head_bias"] = hc_sign.astype(np.float32)

    # --- Tier 2: slower state transitions ---------------------------------
    fwd_bin = _sustained(
        _binarize(v, FWD_VEL_THRESH), FWD_SUSTAIN_SAMPLES
    )
    out["forward_run_state"] = fwd_bin
    out["forward_run_onset"] = _onsets(fwd_bin)
    out["forward_run_offset"] = _offsets(fwd_bin)

    qui_bin = _sustained(
        _binarize(QUI_VEL_THRESH - np.abs(v), 0),  # |v| < threshold
        QUI_SUSTAIN_SAMPLES,
    )
    out["quiescence_state"] = qui_bin
    out["quiescence_onset"] = _onsets(qui_bin)

    # Roaming/dwelling: rolling mean |velocity| + angular turn rate
    def _rolling_abs_mean(x, w):
        k = np.ones(w, dtype=np.float32) / w
        return np.convolve(np.abs(x), k, mode="same")
    rolling_v = _rolling_abs_mean(v, ROAMING_WINDOW_SAMPLES)
    rolling_turn = _rolling_abs_mean(av, ROAMING_WINDOW_SAMPLES)
    roaming = ((rolling_v > ROAMING_VEL_THRESH)
               & (rolling_turn < ROAMING_TURN_THRESH)).astype(np.int64)
    out["roaming_state"] = roaming
    out["roaming_onset"] = _onsets(roaming)

    # --- Tier 3: AR-defeat derivatives ------------------------------------
    out["velocity_acc"] = np.gradient(v).astype(np.float32)
    out["head_ang_vel"] = np.gradient(hc).astype(np.float32)
    out["body_curv_rate"] = np.gradient(bc).astype(np.float32)
    out["ang_acc"] = np.gradient(av).astype(np.float32)

    # Speed bursts: velocity > mean + BURST_SIGMA * std
    vm, vs = np.nanmean(v), np.nanstd(v)
    burst = _binarize(v, vm + BURST_SIGMA * vs)
    out["speed_burst_onset"] = _onsets(burst)

    # --- Tier 4: categorical / multi-class targets ------------------------
    # 5-class behavioral state: 0=quiet, 1=forward, 2=reverse, 3=omega,
    # 4=pirouette. Priority order: pirouette > omega > reverse > forward > quiet.
    state = np.zeros(T, dtype=np.int64)
    state[fwd_bin.astype(bool)] = 1
    state[rv_bin.astype(bool)] = 2
    state[omega_sust.astype(bool)] = 3
    # Pirouette marker: sustained after pirouette entry, up to next long run
    pir_state = np.zeros(T, dtype=np.int64)
    entries = np.where(out["pirouette_entry"] == 1)[0]
    # A pirouette "lasts" until 4 s of sustained non-reversal
    i = 0
    while i < len(entries):
        start = entries[i]
        end = start
        for t in range(start, T):
            if rv_bin[t] == 0 and t - start > PIROUETTE_WINDOW_SAMPLES:
                break
            end = t
        pir_state[start:end + 1] = 1
        # Skip entries that fall within this pirouette
        while (i < len(entries) and entries[i] <= end):
            i += 1
    state[pir_state.astype(bool)] = 4
    state[qui_bin.astype(bool)] = 0  # quiescence overrides all
    out["state_5class"] = state

    # Reversal mode (3-class): short (<2s), long (>=2s non-omega), omega-coupled
    rev_mode = np.full(T, -1, dtype=np.int64)  # -1 = not in reversal
    # Find reversal bouts
    in_rev = rv_bin.astype(bool)
    bout_id = 0
    i = 0
    while i < T:
        if in_rev[i]:
            start = i
            while i < T and in_rev[i]:
                i += 1
            end = i
            duration = end - start
            # Check if omega occurred within ±PIROUETTE_WINDOW_SAMPLES
            lo = max(0, start - PIROUETTE_WINDOW_SAMPLES)
            hi = min(T, end + PIROUETTE_WINDOW_SAMPLES)
            has_omega = omega_sust[lo:hi].sum() > 0
            if has_omega:
                rev_mode[start:end] = 2
            elif duration < 4:  # ~2.4 s
                rev_mode[start:end] = 0  # short
            else:
                rev_mode[start:end] = 1  # long
        else:
            i += 1
    out["rev_mode_3class"] = rev_mode

    # Head-swing direction (categorical): +1 dorsal, -1 ventral, sign of
    # head_curv. Already `head_bias`, but return as binary at high-|hc|:
    big_swing = (np.abs(hc) > 0.5).astype(np.int64)
    hs_dir = np.where(big_swing == 1, (hc > 0).astype(np.int64), -1)
    out["headswing_dir"] = hs_dir   # -1=no swing, 0=ventral, 1=dorsal

    return out


# Metadata: which tier each target belongs to + its type
# Types: "event" (sparse binary, predict with classifier+AUC),
#        "state" (dense binary), "continuous", "multiclass"
TARGET_META: dict[str, tuple[int, str]] = {
    # Tier 1 — rare events
    "reversal_onset":      (1, "event"),
    "reversal_offset":     (1, "event"),
    "reversal_state":      (1, "state"),
    "omega_onset":         (1, "event"),
    "pirouette_entry":     (1, "event"),
    "headswing_flip":      (1, "event"),
    "head_bias":           (1, "continuous"),
    # Tier 2 — slower state transitions
    "forward_run_onset":   (2, "event"),
    "forward_run_offset":  (2, "event"),
    "forward_run_state":   (2, "state"),
    "quiescence_onset":    (2, "event"),
    "quiescence_state":    (2, "state"),
    "roaming_onset":       (2, "event"),
    "roaming_state":       (2, "state"),
    # Tier 3 — derivatives (AR-defeat by construction)
    "velocity_acc":        (3, "continuous"),
    "head_ang_vel":        (3, "continuous"),
    "body_curv_rate":      (3, "continuous"),
    "ang_acc":             (3, "continuous"),
    "speed_burst_onset":   (3, "event"),
    # Tier 4 — categorical
    "state_5class":        (4, "multiclass"),
    "rev_mode_3class":     (4, "multiclass"),
    "headswing_dir":       (4, "multiclass"),
}


def load_and_extract(worm_npz: Path) -> tuple[dict, dict]:
    """Load a worm's npz and return (raw_data_dict, target_dict)."""
    a = np.load(worm_npz, allow_pickle=True)
    data = {k: a[k] for k in
            ["neural", "neural_raw", "neuron_ids", "t",
             "velocity", "head_curv", "body_curv", "ang_vel",
             "reversal", "pumping"]
            if k in a.files}
    targets = extract_events(data)
    return data, targets


if __name__ == "__main__":
    # Smoke test: extract from worm_01, report event counts per target
    ART = Path(__file__).resolve().parent / "artifacts"
    data, targets = load_and_extract(ART / "atanas_worm_01.npz")
    print(f"Worm 01: T={len(data['velocity'])} samples, "
          f"{len(targets)} targets\n")

    print(f"{'target':<22} {'tier':>4} {'type':<12} {'count/mean':>14}")
    print("-" * 60)
    for tgt, val in targets.items():
        tier, kind = TARGET_META[tgt]
        if kind == "event":
            desc = f"{int(val.sum())} events"
        elif kind == "state":
            desc = f"{int(val.sum())} on / {len(val)}"
        elif kind == "multiclass":
            uniq = np.unique(val[val >= 0])
            desc = f"{len(uniq)} classes"
        else:
            desc = f"μ={val.mean():+.3f}  σ={val.std():.3f}"
        print(f"{tgt:<22} {tier:>4} {kind:<12} {desc:>14}")
