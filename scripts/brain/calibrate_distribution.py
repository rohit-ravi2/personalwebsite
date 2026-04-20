#!/usr/bin/env python3
"""Phase 3c v1.5 — distribution calibration between Brian2 synthetic
calcium and Atanas ΔF/F.

The root cause of the v1 demo smell (pirouette over-triggering,
empirical threshold recalibration from harness-native 0.5 → 0.97):
classifiers were trained on real ΔF/F signals, closed-loop feeds
them synthetic calcium derived from Brian2 spike trains. The two
distributions don't match in scale, mean, or variance per neuron.

Fix: per-neuron affine calibration that maps Brian2 synthetic calcium
onto Atanas-ΔF/F statistical moments before classifier input.

    ca_calibrated = (ca_brain - μ_brain) / σ_brain × σ_atanas + μ_atanas

Step 1: compute per-neuron Atanas ΔF/F stats pooled across 10 worms
        (the 18 strict-intersection neurons only).
Step 2: run LIF brain for 300 s with mixed stimuli, collect synthetic
        calcium time series per readout neuron, compute Brian2 stats.
Step 3: save the affine parameters to artifacts/calibration.npz.

The closed-loop env picks up calibration.npz and applies the transform
before feeding the classifier bank.
"""
from __future__ import annotations

import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_brain import LIFBrain  # noqa: E402
from sensory_injection import stimulate  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
OUT = ART / "calibration.npz"

# Match the closed-loop sync cadence (50 ms brain, 600 ms calcium)
BRAIN_SYNC_MS = 50.0
CLASSIFIER_DT_S = 0.6
STEPS_PER_CA = int(round(CLASSIFIER_DT_S / (BRAIN_SYNC_MS / 1000)))

# Stimulus schedule for brain baseline — varied to match the range
# of conditions the closed-loop will encounter.
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


def _load_bank_neuron_order() -> list[str]:
    d = np.load(ART / "classifier_bank.npz", allow_pickle=True)
    return [str(s) for s in d["neuron_order"]]


def atanas_stats_per_neuron(neuron_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pool ΔF/F from all 10 worms for the readout neurons and return
    per-neuron (mean, std). Missing neurons per worm are skipped, not
    imputed.
    """
    per_neuron_values: dict[str, list[np.ndarray]] = {n: [] for n in neuron_order}
    for p in sorted(ART.glob("atanas_worm_*.npz")):
        a = np.load(p, allow_pickle=True)
        ids = [_norm(s) for s in a["neuron_ids"]]
        seen: set[str] = set()
        for col, nm in enumerate(ids):
            if nm in per_neuron_values and nm not in seen:
                per_neuron_values[nm].append(a["neural"][:, col])
                seen.add(nm)
    mu = np.zeros(len(neuron_order), dtype=np.float32)
    sd = np.ones(len(neuron_order), dtype=np.float32)
    for i, n in enumerate(neuron_order):
        if per_neuron_values[n]:
            pooled = np.concatenate(per_neuron_values[n])
            mu[i] = float(pooled.mean())
            sd[i] = float(pooled.std() + 1e-6)
    return mu, sd


def run_brain_baseline(neuron_order: list[str]) -> np.ndarray:
    """Run the LIF brain for BASELINE_DURATION_S with varied stimuli.
    Return (T, N_readout) synthetic calcium time series at CLASSIFIER_DT."""
    brain = LIFBrain()
    readout_idx = [brain.idx[n] for n in neuron_order if n in brain.idx]
    N = len(readout_idx)

    total_sync_steps = int(BASELINE_DURATION_S * 1000 / BRAIN_SYNC_MS)
    spike_counts_buffer = []
    prev_spike_len = 0
    schedule = list(STIMULUS_SCHEDULE)

    print(f"  running brain baseline for {BASELINE_DURATION_S:.0f} s "
          f"({total_sync_steps} sync steps, {len(STIMULUS_SCHEDULE)} stimuli)…")

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

        if (step + 1) % 1000 == 0:
            print(f"    step {step+1}/{total_sync_steps} "
                  f"(t={t_s:.0f}s)")

    spike_counts_buffer = np.stack(spike_counts_buffer)

    # Downsample spike counts into calcium samples (every STEPS_PER_CA ≈ 12
    # sync steps) and apply the same IIR smoothing as the closed loop.
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
    neuron_order = _load_bank_neuron_order()
    print(f"Calibrating {len(neuron_order)} readout neurons: {neuron_order}")

    print("\n[1] Atanas ΔF/F stats (pooled across 10 worms)…")
    mu_a, sd_a = atanas_stats_per_neuron(neuron_order)
    for n, m, s in zip(neuron_order, mu_a, sd_a):
        print(f"   {n:<8} μ={m:.3f}  σ={s:.3f}")

    print("\n[2] Brian2 baseline run…")
    ca_brain = run_brain_baseline(neuron_order)
    mu_b = ca_brain.mean(axis=0).astype(np.float32)
    sd_b = ca_brain.std(axis=0).astype(np.float32) + 1e-6
    print(f"\n   Brian2 synthetic calcium stats ({ca_brain.shape[0]} samples):")
    for n, m, s in zip(neuron_order, mu_b, sd_b):
        print(f"   {n:<8} μ={m:.3f}  σ={s:.3f}")

    print("\n[3] Saving calibration parameters…")
    np.savez_compressed(
        OUT,
        neuron_order=np.array(neuron_order, dtype=object),
        mu_brain=mu_b,
        sd_brain=sd_b,
        mu_atanas=mu_a,
        sd_atanas=sd_a,
        baseline_duration_s=np.float32(BASELINE_DURATION_S),
    )
    print(f"   wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")

    # Quick sanity: what does the calibration transform look like?
    print("\n[4] Per-neuron calibration preview:")
    print(f"   {'neuron':<8} {'scale':>8} {'offset':>8}")
    for i, n in enumerate(neuron_order):
        scale = sd_a[i] / sd_b[i]
        offset = mu_a[i] - scale * mu_b[i]
        print(f"   {n:<8} {scale:>8.3f} {offset:>+8.3f}")


if __name__ == "__main__":
    main()
