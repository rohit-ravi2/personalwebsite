#!/usr/bin/env python3
"""Phase 3c-3 — Closed-loop brain-body environment.

Brings together:
  - LIFBrain (Brian2 300-neuron LIF, connectome-constrained)
  - ClassifierBank (Phase 3b event predictions from neural activity)
  - BehavioralFSM (5-state mode controller)
  - MuJoCo body (Phase 1a wormbody + resistive-force drag)

Sync cadence: 50 ms between brain spike-rate reads. Brain internal dt
is Brian2's default 0.1 ms; MuJoCo runs at 0.5 ms × 100 = 50 ms per
sync step.

Classifier samples every 600 ms (Atanas sampling rate the bank was
trained at). FSM polls using the most recent classifier output.

Proprioceptive feedback: body curvature magnitude → Poisson onto PDE,
PDA, DVA neurons (the classical worm proprioceptors, Li 2006).

Output: a trace dict with brain spikes, body segment positions, FSM
state, event probabilities, and stimulus schedule — exported as JSON
for site playback.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import mujoco
import numpy as np
from brian2 import ms, Hz

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_brain import LIFBrain  # noqa: E402
from neural_classifier_bank import ClassifierBank, spikes_to_calcium  # noqa: E402
from behavioral_fsm import BehavioralFSM, State, STATE_CPG  # noqa: E402
from sensory_injection import stimulate  # noqa: E402
from modulation_layer import ModulationLayer, TABLES as MOD_TABLES  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
MJCF = REPO / "public" / "data" / "wormbody.xml"

# --- Sync cadences -----------------------------------------------------
BRAIN_SYNC_MS = 50.0       # brain-body sync granularity
CLASSIFIER_DT_MS = 600.0   # classifier sample period (match training)
SIM_DT_S = 0.0005          # MuJoCo timestep (0.5 ms)
STEPS_PER_SYNC = int(round((BRAIN_SYNC_MS / 1000.0) / SIM_DT_S))  # 100
RECORD_HZ = 30             # body position recording rate

# --- Proprioception neurons (may or may not be in the brain) ----------
PROPRIO_NEURONS = ["PDEL", "PDER", "PDA", "DVA"]

# --- Drag from Phase 2a ------------------------------------------------
DRAG_PARA = 2.0
DRAG_PERP = 4.0


def apply_resistive_drag(model, data):
    nb = model.nbody
    axes = data.xmat.reshape(nb, 3, 3)[1:, :, 0]
    v_lin = data.cvel[1:, 3:6]
    para_mag = np.sum(v_lin * axes, axis=1, keepdims=True)
    v_para = para_mag * axes
    v_perp = v_lin - v_para
    force = -DRAG_PARA * v_para - DRAG_PERP * v_perp
    data.xfrc_applied[1:, 0:3] = force
    data.xfrc_applied[1:, 3:6] = 0.0


def cpg_ctrl(nu: int, t: float, params: dict) -> np.ndarray:
    """Compute position-actuator targets given CPG params at sim time t."""
    freq = params["freq"]
    wl = params["wavelength"]
    amp = params["amplitude"]
    phase_sign = params["phase_sign"]
    bias = params["turn_bias"]
    ctrl = np.zeros(nu, dtype=np.float64)
    if amp <= 0 or freq <= 0:
        # quiescent — just apply any turn bias as sustained curvature
        return np.clip(np.full(nu, bias * 0.3), -0.5, 0.5)
    for j in range(nu):
        s = (j + 0.5) / nu
        phase = 2 * math.pi * (s / wl - phase_sign * freq * t)
        ctrl[j] = amp * math.sin(phase) + bias * 0.5 * math.sin(math.pi * s)
    return np.clip(ctrl, -0.5, 0.5)


class ClosedLoopEnv:
    """Closed-loop brain-body simulation."""

    def __init__(self, seed: int = 0, enable_modulation: bool = True,
                 ablate: list[str] | None = None):
        np.random.seed(seed)
        self.brain = LIFBrain()
        self.bank = ClassifierBank()
        self.fsm = BehavioralFSM(State.FORWARD)

        # v3 slow neuromodulation layer (Phase 3d-2)
        self.modulation: ModulationLayer | None = None
        if enable_modulation and MOD_TABLES.exists():
            self.modulation = ModulationLayer(self.brain.names)
            self.modulation.attach_to_brain(self.brain)

        # Optional in-silico ablation (Phase 3d-3 perturbation studies)
        self.ablated: list[str] = []
        if ablate:
            self.ablated = self.brain.ablate(ablate)

        # Per-neuron affine distribution calibration (v1.5 fix):
        # maps Brian2 synthetic calcium moments onto the Atanas ΔF/F
        # moments the classifier was trained on. See calibrate_distribution.py.
        cal_path = Path(__file__).resolve().parent / "artifacts" / "calibration.npz"
        if cal_path.exists():
            cal = np.load(cal_path, allow_pickle=True)
            self.cal_mu_brain = cal["mu_brain"].astype(np.float32)
            self.cal_sd_brain = cal["sd_brain"].astype(np.float32)
            self.cal_mu_atanas = cal["mu_atanas"].astype(np.float32)
            self.cal_sd_atanas = cal["sd_atanas"].astype(np.float32)
            self.use_calibration = True
        else:
            self.use_calibration = False

        # MuJoCo body
        self.model = mujoco.MjModel.from_xml_path(MJCF.as_posix())
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu
        self.seg_body_ids = list(range(1, self.model.nbody))

        # Settle body
        for _ in range(200):
            apply_resistive_drag(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Readout index map
        self.readout_idx = [self.brain.idx[n] for n in self.bank.neuron_order
                            if n in self.brain.idx]
        self.readout_present = [n for n in self.bank.neuron_order
                                if n in self.brain.idx]
        # Precompute calcium kernel
        self._dt_ca = self.bank.dt  # 0.6 s
        self._steps_per_ca = int(round(self._dt_ca / (BRAIN_SYNC_MS / 1000)))

        # Rolling buffers
        self.spike_counts_buffer: list[np.ndarray] = []  # one entry per sync
        self.calcium_buffer: list[np.ndarray] = []       # one entry per ca sample
        self.event_probs: dict[str, list[float]] = {
            e: [] for e in self.bank.events
        }
        self.fsm_states: list[int] = []
        self.body_frames: list[dict] = []
        self.stim_log: list[dict] = []
        # Track previous spike-monitor length to get rates per sync window
        self._prev_spike_len = 0

    def _read_spike_rates(self) -> np.ndarray:
        """Return (N_readout,) spike counts in the last BRAIN_SYNC_MS."""
        all_t = self.brain.spikes.t[:]
        all_i = self.brain.spikes.i[:]
        new = slice(self._prev_spike_len, len(all_t))
        self._prev_spike_len = len(all_t)
        recent_i = all_i[new]
        counts = np.zeros(self.brain.N, dtype=np.float32)
        if len(recent_i) > 0:
            np.add.at(counts, recent_i, 1)
        return counts[self.readout_idx]

    def _inject_proprio(self, body_curv_mag: float):
        """DISABLED in v1 — adding new Poisson Synapses every sync step
        causes Brian2 to recompile the network each call (O(N_syncs²)
        slowdown). V2 will use PoissonInput with a shared rate variable.
        For v1 the loop is brain → body but not body → brain."""
        pass

    def stimulate_sensory(self, preset: str, intensity: float = 1.0):
        injected = stimulate(self.brain, preset, intensity=intensity)
        self.stim_log.append({
            "t": self.brain.time_ms() / 1000,
            "preset": preset, "intensity": intensity,
            "neurons": injected,
        })

    def step_sync(self):
        """Advance brain + body by one BRAIN_SYNC_MS step."""
        t_sync_s = self.brain.time_ms() / 1000

        # 1) Advance brain
        self.brain.run(BRAIN_SYNC_MS)

        # 2) Read spike rates over this window
        counts = self._read_spike_rates()
        self.spike_counts_buffer.append(counts)

        # 3) Build calcium if enough buckets accumulated
        if len(self.spike_counts_buffer) % self._steps_per_ca == 0:
            recent = np.stack(self.spike_counts_buffer[-self._steps_per_ca:])
            ca_sample = recent.sum(axis=0) / self._steps_per_ca

            # Incremental single-tap IIR smoothing (τ=0.5 s) — vectorized
            alpha = 1 - math.exp(-self._dt_ca / 0.5)
            prev = (self.calcium_buffer[-1] if self.calcium_buffer
                    else np.zeros_like(ca_sample))
            smoothed = (1 - alpha) * prev + alpha * ca_sample
            self.calcium_buffer.append(smoothed)

            # Apply per-neuron affine calibration to map Brian2 synthetic
            # calcium moments onto the Atanas ΔF/F distribution the
            # classifier was trained on (v1.5 distribution fix).
            ca_hist = np.stack(self.calcium_buffer, axis=0)
            if self.use_calibration:
                # (x - μ_brain)/σ_brain × σ_atanas + μ_atanas, per neuron
                ca_cal = (ca_hist - self.cal_mu_brain) / self.cal_sd_brain
                ca_cal = ca_cal * self.cal_sd_atanas + self.cal_mu_atanas
                # Clip to the observed Atanas ΔF/F range to avoid
                # extrapolation artefacts
                ca_cal = np.clip(ca_cal, 0.0, 4.0).astype(np.float32)
            else:
                # Fallback: local z-score (old v1 behaviour)
                if len(ca_hist) >= 5:
                    mu = ca_hist.mean(axis=0, keepdims=True)
                    sd = ca_hist.std(axis=0, keepdims=True) + 1e-6
                    ca_cal = (ca_hist - mu) / sd
                else:
                    ca_cal = ca_hist

            probs = self.bank.predict_from_calcium(ca_cal[-10:])
            for e in self.bank.events:
                self.event_probs[e].append(float(probs[e][-1]))

        # 4) Get latest event probs for FSM
        latest_probs = {e: (self.event_probs[e][-1]
                            if self.event_probs[e] else 0.0)
                        for e in self.bank.events}

        # 5) Update FSM
        self.fsm.update(t_sync_s, latest_probs)
        self.fsm_states.append(self.fsm.state.value)

        # 6) Drive body with current CPG params
        params = self.fsm.cpg_params()
        for step in range(STEPS_PER_SYNC):
            t_body = self.data.time
            self.data.ctrl[:] = cpg_ctrl(self.nu, t_body, params)
            apply_resistive_drag(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

        # 7) Record body positions (at ~RECORD_HZ rate → every sync)
        positions = [[float(self.data.xpos[b][0]),
                      float(self.data.xpos[b][1])]
                     for b in self.seg_body_ids]
        self.body_frames.append({
            "t": round(self.data.time, 3),
            "positions": positions,
            "state": self.fsm.state.name,
        })

        # 8) Proprioception back to brain
        # Approximate body curvature from segment angles
        seg_xs = np.array([p[0] for p in positions])
        seg_ys = np.array([p[1] for p in positions])
        dxs = np.diff(seg_xs)
        dys = np.diff(seg_ys)
        headings = np.arctan2(dys, dxs)
        curv = np.mean(np.abs(np.diff(headings)))
        self._inject_proprio(curv)

    def run(self, duration_s: float, stim_schedule: list[tuple] = ()):
        """Run the closed loop for duration_s seconds. stim_schedule is
        [(t_s, preset_name, intensity), ...]."""
        n_steps = int(duration_s * 1000 / BRAIN_SYNC_MS)
        schedule = list(stim_schedule)
        for _ in range(n_steps):
            t_now = self.brain.time_ms() / 1000
            # Deliver any stimuli due
            while schedule and schedule[0][0] <= t_now:
                _, preset, intensity = schedule.pop(0)
                self.stimulate_sensory(preset, intensity)
            self.step_sync()

    def export(self, out_path: Path, scenario_name: str):
        """Write a JSON trace for site playback."""
        # Re-center body trace
        p0 = np.array(self.body_frames[0]["positions"])
        c0 = p0.mean(axis=0)
        for fr in self.body_frames:
            fr["positions"] = [
                [p[0] - float(c0[0]), p[1] - float(c0[1])]
                for p in fr["positions"]
            ]

        # Compress brain spike raster: for each sync bucket, record
        # which readout neurons fired
        spike_raster = []
        for i, counts in enumerate(self.spike_counts_buffer):
            t_s = (i + 1) * BRAIN_SYNC_MS / 1000
            active = [j for j, c in enumerate(counts) if c > 0]
            if active:
                spike_raster.append({"t": round(t_s, 3), "n": active})

        payload = {
            "scenario": scenario_name,
            "meta": {
                "brain_sync_ms": BRAIN_SYNC_MS,
                "classifier_dt_ms": CLASSIFIER_DT_MS,
                "num_segments": self.model.nbody - 1,
                "num_frames": len(self.body_frames),
                "duration_s": len(self.body_frames) * BRAIN_SYNC_MS / 1000,
                "readout_neurons": self.readout_present,
                "events_tracked": self.bank.events,
                "states": ["FORWARD", "REVERSE", "OMEGA", "PIROUETTE",
                           "QUIESCENT"],
                "modulation_enabled": self.modulation is not None,
                "modulators": (list(self.modulation.modulators)
                               if self.modulation else []),
                "sources": {
                    "brain": "Shiu et al. 2024 Nature analog (worm Cook 2019 + Loer&Rand 2022 NT)",
                    "body": "Phase 1a/2a MuJoCo, Boyle-Berri-Cohen 2012 CPG",
                    "classifier": "Phase 3b harness (Atanas 2023)",
                    "modulation": "v3 9-modulator peptidergic+monoamine layer from CeNGEN expression",
                    "sync_pattern": "Eon Systems 2026",
                },
            },
            "frames": self.body_frames,
            "raster": spike_raster,
            "event_probs": {
                e: [round(v, 3) for v in self.event_probs[e]]
                for e in self.bank.events
            },
            "fsm_states": self.fsm_states,
            "stim_log": self.stim_log,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, separators=(",", ":")))
        kb = out_path.stat().st_size / 1024
        print(f"wrote {out_path.name}: "
              f"{len(self.body_frames)} body frames, "
              f"{len(spike_raster)} raster entries, {kb:.1f} KB")


if __name__ == "__main__":
    # Smoke test: 10 s spontaneous
    env = ClosedLoopEnv()
    env.run(10.0)
    tally = {}
    for s in env.fsm_states:
        tally[s] = tally.get(s, 0) + 1
    print(f"\nState distribution over 10s: {tally}")
    print(f"Event prob samples: {len(env.event_probs['reversal_onset'])}")
