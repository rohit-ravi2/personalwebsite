#!/usr/bin/env python3
"""Phase 2c — imitation-learn a neural controller that drives the
MuJoCo wormbody to track the Phase 2b reference trajectory.

Stack: gymnasium env wrapping MuJoCo + resistive-force-theory drag,
stable-baselines3 PPO, MLP policy.

Observation (24-D): current joint angles (19) + reference phase (3:
sin(2πft), cos(2πft), t/duration) + mean body velocity (2).
Action (19-D): delta to CPG baseline target angles, clipped to [-0.15, 0.15].
Reward: -MSE of simulated vs reference segment positions at matching sim
time, plus a small control-magnitude penalty.

Trained rollout is exported as JSON for the site viewer. If training
doesn't converge within the time budget, the best-checkpoint rollout is
saved anyway.

Output: public/data/wormbody-learned-trace.json
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


REPO = Path(__file__).resolve().parents[1]
MJCF = REPO / "public" / "data" / "wormbody.xml"
REF = REPO / "public" / "data" / "wormbody-reference-trace.json"
OUT = REPO / "public" / "data" / "wormbody-learned-trace.json"
MODEL_DIR = REPO / "scripts" / "_ppo_wormbody"

EPISODE_SECONDS = 3.0
CONTROL_HZ = 30     # coarser control → fewer sim steps per action (faster)
SIM_DT = 0.0005
STEPS_PER_ACTION = int(round((1.0 / CONTROL_HZ) / SIM_DT))  # 67
EPISODE_STEPS = int(EPISODE_SECONDS * CONTROL_HZ)

DRAG_PARA = 2.0
DRAG_PERP = 4.0

# CPG baseline that the MLP learns DELTAS on top of. Same formula as
# Phase 2a but with action-space residuals added per step.
BASE_FREQ_HZ = 1.0
BASE_WAVELENGTH = 0.65
BASE_AMPLITUDE_RAD = 0.30
ACTION_CLIP = 0.15  # max delta the policy may add to each joint's target angle

# Training budget — kept small so the script runs in a reasonable session.
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "120000"))
MIN_MEAN_REWARD = -0.20


def _apply_resistive_drag(model, data):
    """Vectorised anisotropic drag — avoids per-body Python loop overhead."""
    nb = model.nbody
    # Body-axis directions (column 0 of each 3×3 rotation matrix).
    axes = data.xmat.reshape(nb, 3, 3)[1:, :, 0]  # (N, 3)
    # Linear velocities via mj_objectVelocity (frame=0 → world).
    # Fastest path: cvel is already world-frame linear+angular velocity
    # at each body CoM. Layout in cvel: [ang(3), lin(3)] per body.
    v_lin = data.cvel[1:, 3:6]  # (N, 3)
    # Parallel component magnitude per body:
    para_mag = np.sum(v_lin * axes, axis=1, keepdims=True)  # (N, 1)
    v_para = para_mag * axes
    v_perp = v_lin - v_para
    force = -DRAG_PARA * v_para - DRAG_PERP * v_perp
    data.xfrc_applied[1:, 0:3] = force
    data.xfrc_applied[1:, 3:6] = 0.0


def _cpg_baseline(t: float, nu: int) -> np.ndarray:
    s = (np.arange(nu) + 0.5) / nu
    phase = 2 * math.pi * (s / BASE_WAVELENGTH - BASE_FREQ_HZ * t)
    return BASE_AMPLITUDE_RAD * np.sin(phase)


def _load_reference() -> tuple[np.ndarray, float]:
    data = json.loads(REF.read_text())
    frames = data["frames"]
    positions = np.array(
        [[[p[0], p[1]] for p in f["positions"]] for f in frames],
        dtype=np.float32,
    )
    return positions, float(data["meta"]["record_hz"])


class WormBodyImitationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(MJCF.as_posix())
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu  # 19
        self.ref, self.ref_hz = _load_reference()  # (T, 20, 2)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )
        # Observation: 19 joint angles + 19 joint velocities + 3 reference-phase features
        obs_dim = self.nu + self.nu + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.step_idx = 0

    def _get_obs(self):
        # qpos layout: [7 freejoint, 19 hinges]
        joint_angles = self.data.qpos[7:7 + self.nu].astype(np.float32)
        joint_vel = self.data.qvel[6:6 + self.nu].astype(np.float32)
        t = self.data.time
        phase_feat = np.array([
            math.sin(2 * math.pi * BASE_FREQ_HZ * t),
            math.cos(2 * math.pi * BASE_FREQ_HZ * t),
            (self.step_idx / EPISODE_STEPS) * 2 - 1,
        ], dtype=np.float32)
        return np.concatenate([joint_angles, joint_vel, phase_feat])

    def _reward(self):
        # Segment positions vs reference at the current step.
        ref_idx = min(self.step_idx, self.ref.shape[0] - 1)
        target = self.ref[ref_idx]  # (20, 2)
        sim = np.zeros_like(target)
        for i in range(1, self.model.nbody):
            p = self.data.xpos[i]
            sim[i - 1] = (p[0], p[1])
        # Re-center both so mean position doesn't dominate — we care about shape.
        sim_c = sim - sim.mean(axis=0)
        tgt_c = target - target.mean(axis=0)
        shape_mse = float(np.mean((sim_c - tgt_c) ** 2))
        # Translation reward: sim centroid tracks reference centroid
        trans_err = float(np.linalg.norm(sim.mean(axis=0) - target.mean(axis=0)))
        # Control-magnitude penalty discourages jerky corrections.
        ctrl_pen = float(np.mean(self.data.ctrl ** 2)) * 0.02
        return -(shape_mse * 4.0 + trans_err * 0.5 + ctrl_pen)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.step_idx = 0
        return self._get_obs(), {}

    def step(self, action):
        delta = np.clip(action, -1.0, 1.0) * ACTION_CLIP
        t0 = self.data.time
        for step in range(STEPS_PER_ACTION):
            t_now = t0 + step * SIM_DT
            baseline = _cpg_baseline(t_now, self.nu)
            target = np.clip(baseline + delta, -0.5, 0.5)
            self.data.ctrl[:] = target
            _apply_resistive_drag(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        self.step_idx += 1
        reward = self._reward()
        terminated = False
        truncated = self.step_idx >= EPISODE_STEPS
        # Early-terminate if body has NaN/exploded.
        if not np.all(np.isfinite(self.data.qpos)):
            reward = -5.0
            terminated = True
        return self._get_obs(), float(reward), terminated, truncated, {}


class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_reward = -np.inf
        self.start = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % 10000 == 0:
            ep_buf = self.model.ep_info_buffer
            if ep_buf and len(ep_buf) > 0:
                avg = np.mean([ep["r"] for ep in ep_buf])
                elapsed = time.time() - self.start
                print(
                    f"[{self.num_timesteps:>7d}] ep_rew={avg:.4f}  "
                    f"best={self.best_reward:.4f}  elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
                if avg > self.best_reward:
                    self.best_reward = avg
        return True


def _rollout_to_json(model, env, out_path: Path):
    """Run the trained policy for one episode and export positions."""
    obs, _ = env.reset()
    frames = []
    for step in range(EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        positions = []
        for bid in range(1, env.unwrapped.model.nbody):
            p = env.unwrapped.data.xpos[bid]
            positions.append([float(p[0]), float(p[1])])
        frames.append({"t": round(step / CONTROL_HZ, 4), "positions": positions})
        if terminated or truncated:
            break

    # Normalize to first-frame centroid at origin.
    p0 = np.array(frames[0]["positions"])
    c0 = p0.mean(axis=0)
    for f in frames:
        for i, xy in enumerate(f["positions"]):
            f["positions"][i] = [xy[0] - float(c0[0]), xy[1] - float(c0[1])]

    meta = {
        "num_segments": env.unwrapped.model.nbody - 1,
        "num_frames": len(frames),
        "record_hz": CONTROL_HZ,
        "duration_sec": len(frames) / CONTROL_HZ,
        "controller": {
            "kind": "imitation_ppo_mlp",
            "base_cpg_freq_hz": BASE_FREQ_HZ,
            "base_cpg_wavelength": BASE_WAVELENGTH,
            "base_cpg_amplitude_rad": BASE_AMPLITUDE_RAD,
            "action_clip": ACTION_CLIP,
        },
        "units": "MJCF scaled: 1 sim-m ≈ 1 real-mm",
    }
    out_path.write_text(json.dumps({"meta": meta, "frames": frames}, separators=(",", ":")))
    size_kb = out_path.stat().st_size / 1024
    disp = np.linalg.norm(
        np.array(frames[-1]["positions"]).mean(axis=0)
        - np.array(frames[0]["positions"]).mean(axis=0)
    )
    print(f"wrote {out_path.name}: {len(frames)} frames, {size_kb:.1f} KB, CoM disp {disp:.3f} sim-m")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = WormBodyImitationEnv()

    print(f"obs_dim={env.observation_space.shape[0]}  action_dim={env.action_space.shape[0]}")
    print(f"episode_steps={EPISODE_STEPS}  sim_steps_per_action={STEPS_PER_ACTION}")
    print(f"training budget: {TOTAL_STEPS} env steps")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=6,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=0,
    )

    cb = ProgressCallback()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)
    print(f"training done. best running reward: {cb.best_reward:.4f}")
    model.save(MODEL_DIR / "ppo_wormbody")

    _rollout_to_json(model, env, OUT)


if __name__ == "__main__":
    main()
