#!/usr/bin/env python3
"""Phase 2 sub-task 2.3 — Brian2-→-Atanas distribution calibration under M2-pure.

Per-neuron affine calibration that maps Brian2 synthetic calcium under M2-pure
sign mode onto Atanas ΔF/F statistical moments before classifier input.

    ca_calibrated = (ca_brain - μ_brain_m2pure) / σ_brain_m2pure × σ_atanas + μ_atanas

Differs from `calibrate_distribution.py`:
  - Brain runs under M2-pure (use_per_edge_glu_signs=True, sign_exceptions={})
    instead of default mode + DOCUMENTED_SIGN_EXCEPTIONS.
  - 21-neuron A2-balanced readout (legacy 18 + AVAL + AVAR + AVDL) instead of 18.
  - Output written to calibration_m2pure.npz; legacy calibration.npz preserved.

Methodology preserved: 60 s mixed-stimulus baseline run; per-neuron mean/std
pooled across the run; Atanas stats pooled across all 10 worms (with per-worm
neuron presence handled — AVAR pools from 8 worms, AVAL/AVDL from 9, legacy 18
from all 10).
"""
from __future__ import annotations

import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from lif_brain import LIFBrain  # noqa: E402
from sensory_injection import stimulate  # noqa: E402
from phase2_train_classifier import A2_BALANCED_READOUT, A2_COVERAGE  # noqa: E402

ART = THIS_DIR / "artifacts"
OUT = ART / "calibration_m2pure.npz"

BRAIN_SYNC_MS = 50.0
CLASSIFIER_DT_S = 0.6
STEPS_PER_CA = int(round(CLASSIFIER_DT_S / (BRAIN_SYNC_MS / 1000)))

BASELINE_DURATION_S = 60.0
STIMULUS_SCHEDULE = [
    (10.0, "touch_anterior", 1.0),
    (20.0, "osmotic_shock", 1.0),
    (30.0, "food_signal", 1.0),
    (40.0, "bitter_repellent", 1.0),
    (50.0, "odor_attractant_awc", 0.7),
]


def _norm(name: str) -> str:
    name = str(name).strip().rstrip("?")
    m = re.match(r"^([A-Za-z]+)0(\d)$", name)
    return f"{m.group(1)}{m.group(2)}" if m else name


def atanas_stats_per_neuron(neuron_order: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pool ΔF/F from all 10 worms for the readout neurons; return per-neuron
    (mean, std) plus a per-neuron worm-coverage report."""
    per_neuron_values: dict[str, list[np.ndarray]] = {n: [] for n in neuron_order}
    coverage: dict[str, int] = {n: 0 for n in neuron_order}
    for p in sorted(ART.glob("atanas_worm_*.npz")):
        a = np.load(p, allow_pickle=True)
        ids = [_norm(s) for s in a["neuron_ids"]]
        seen: set[str] = set()
        for col, nm in enumerate(ids):
            if nm in per_neuron_values and nm not in seen:
                per_neuron_values[nm].append(a["neural"][:, col])
                coverage[nm] += 1
                seen.add(nm)
    mu = np.zeros(len(neuron_order), dtype=np.float32)
    sd = np.ones(len(neuron_order), dtype=np.float32)
    for i, n in enumerate(neuron_order):
        if per_neuron_values[n]:
            pooled = np.concatenate(per_neuron_values[n])
            mu[i] = float(pooled.mean())
            sd[i] = float(pooled.std() + 1e-6)
    return mu, sd, coverage


def run_brain_baseline_m2pure(neuron_order: list[str]) -> np.ndarray:
    """Run LIFBrain under M2-pure (per-edge + sign_exceptions={}) for
    BASELINE_DURATION_S with the same 5-stim schedule as the legacy calibration.
    Returns (T, N_readout) synthetic calcium time series at CLASSIFIER_DT_S."""
    brain = LIFBrain(
        use_per_edge_glu_signs=True,
        sign_exceptions={},  # M2-pure: no DOCUMENTED_SIGN_EXCEPTIONS
    )
    print(f"  Brain mode: M2-pure ({len(brain.sign_exceptions_applied)} exceptions, "
          f"{len(brain.sign_overrides_applied)} sign overrides applied)")

    readout_idx = []
    missing = []
    for n in neuron_order:
        if n in brain.idx:
            readout_idx.append(brain.idx[n])
        else:
            missing.append(n)
    if missing:
        print(f"  WARNING: readout cells missing from brain.idx: {missing}")
    N = len(readout_idx)

    total_sync_steps = int(BASELINE_DURATION_S * 1000 / BRAIN_SYNC_MS)
    spike_counts_buffer = []
    prev_spike_len = 0
    schedule = list(STIMULUS_SCHEDULE)

    print(f"  running brain baseline for {BASELINE_DURATION_S:.0f} s "
          f"({total_sync_steps} sync steps, {len(STIMULUS_SCHEDULE)} stimuli)...")

    for step in range(total_sync_steps):
        t_s = brain.time_ms() / 1000
        while schedule and schedule[0][0] <= t_s:
            _, preset, intensity = schedule.pop(0)
            stimulate(brain, preset, intensity=intensity)

        brain.run(BRAIN_SYNC_MS)

        all_t = brain.spikes.t[:]
        all_i = brain.spikes.i[:]
        recent = all_i[prev_spike_len:]
        prev_spike_len = len(all_t)
        counts = np.zeros(brain.N, dtype=np.float32)
        if len(recent) > 0:
            np.add.at(counts, recent, 1)
        spike_counts_buffer.append(counts[readout_idx])

        if (step + 1) % 200 == 0:
            print(f"    step {step+1}/{total_sync_steps} (t={t_s:.0f}s)")

    spike_counts_buffer = np.stack(spike_counts_buffer)

    # Downsample spike counts → calcium samples + IIR smoothing matching closed-loop
    n_ca = spike_counts_buffer.shape[0] // STEPS_PER_CA
    ca_raw = spike_counts_buffer[:n_ca * STEPS_PER_CA].reshape(
        n_ca, STEPS_PER_CA, N
    ).mean(axis=1)

    alpha = 1 - math.exp(-CLASSIFIER_DT_S / 0.5)
    ca_smooth = np.zeros_like(ca_raw)
    v = np.zeros(N, dtype=np.float32)
    for t in range(ca_raw.shape[0]):
        v = (1 - alpha) * v + alpha * ca_raw[t]
        ca_smooth[t] = v
    return ca_smooth


def main() -> None:
    neuron_order = list(A2_BALANCED_READOUT)
    print(f"Calibrating {len(neuron_order)} readout neurons (A2-balanced).")
    print(f"  Cells: {neuron_order}")

    print("\n[1] Atanas ΔF/F stats (pooled across all worms with each neuron)...")
    mu_a, sd_a, atanas_cov = atanas_stats_per_neuron(neuron_order)
    print(f"  {'neuron':<8} {'cov':>5} {'μ_atanas':>10} {'σ_atanas':>10}")
    for n, m, s in zip(neuron_order, mu_a, sd_a):
        cov = atanas_cov.get(n, 0)
        print(f"  {n:<8} {cov:>3}/10 {m:>10.4f} {s:>10.4f}")

    print("\n[2] Brian2 M2-pure baseline run...")
    ca_brain = run_brain_baseline_m2pure(neuron_order)
    mu_b = ca_brain.mean(axis=0).astype(np.float32)
    sd_b = ca_brain.std(axis=0).astype(np.float32) + 1e-6
    print(f"\n   Brian2 M2-pure synthetic calcium stats ({ca_brain.shape[0]} samples):")
    print(f"  {'neuron':<8} {'μ_brain':>10} {'σ_brain':>10}")
    for n, m, s in zip(neuron_order, mu_b, sd_b):
        print(f"  {n:<8} {m:>10.4f} {s:>10.4f}")

    print("\n[3] Saving calibration parameters...")
    np.savez_compressed(
        OUT,
        neuron_order=np.array(neuron_order, dtype=object),
        mu_brain=mu_b,
        sd_brain=sd_b,
        mu_atanas=mu_a,
        sd_atanas=sd_a,
        baseline_duration_s=np.float32(BASELINE_DURATION_S),
        atanas_coverage=np.array(
            [atanas_cov.get(n, 0) for n in neuron_order], dtype=np.int32
        ),
        sign_mode=np.array("M2-pure", dtype=object),
    )
    print(f"   wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")

    print("\n[4] Per-neuron calibration preview:")
    print(f"   {'neuron':<8} {'cov':>5} {'scale':>8} {'offset':>8}  Δ vs legacy?")
    legacy = None
    legacy_path = ART / "calibration.npz"
    if legacy_path.exists():
        d_legacy = np.load(legacy_path, allow_pickle=True)
        legacy = {
            str(n): (float(d_legacy["mu_brain"][i]), float(d_legacy["sd_brain"][i]))
            for i, n in enumerate(d_legacy["neuron_order"])
        }
    for i, n in enumerate(neuron_order):
        scale = sd_a[i] / sd_b[i]
        offset = mu_a[i] - scale * mu_b[i]
        cov = atanas_cov.get(n, 0)
        if legacy and n in legacy:
            l_mu, l_sd = legacy[n]
            l_scale = sd_a[i] / l_sd
            d_scale = scale - l_scale
            note = f"   legacy_scale={l_scale:.3f}  Δ={d_scale:+.3f}"
        else:
            note = "   (new in A2-balanced)"
        print(f"   {n:<8} {cov:>3}/10 {scale:>8.3f} {offset:>+8.3f}{note}")


if __name__ == "__main__":
    main()
