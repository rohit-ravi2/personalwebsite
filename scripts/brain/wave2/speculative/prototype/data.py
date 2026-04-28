"""
X.1d prototype — training data generator.

Generates voltage trajectories from a reference single-compartment EGL-19+leak
cell using explicit ODE integration in pure NumPy. The trajectories are the
training targets for the GNN prototype.

Why synthetic-NumPy ground truth instead of replaying Brian2 traces:
- Bounded prototype effort: avoid Brian2 venv setup + dt-handling complexities.
- Self-contained: same venv (ds) for data generation and PyTorch training.
- The mechanistic equations are Nicoletti 2024 EGL-19 + a leak channel,
  matching the form already validated in `wave2/channels/egl19.py` (NumPy
  re-derivation; the constants are taken from Nicoletti 2024 directly).
- The prototype's architectural claim (a 2-node mechanistic-anchored GNN
  approximates a 1-node integration with the right axial coupling) is
  testable against synthetic ground truth as well as Brian2-derived ground
  truth.

Equations (Nicoletti 2024 EGL-19, voltage in mV, time in ms):
    m_inf(V) = 1 / (1 + exp(-(V - vhm) / ka))
    h_inf(V) = (alpha / (1 + exp((V - vhh) / ki)) + (1 - alpha))
              * (1 / (1 + exp(-(V - vh_h) / kih)))
    tau_m(V) = (3.7e-3 / (1 + exp(-(V - cm) / am))) + 0.06
    tau_h(V) = c1 * (1 / (1 + exp((V - cv1) / cv2)))
              + c2 * (1 / (1 + exp((V - cv3) / cv4)))
    I_egl19  = gbar * m**2 * h * (V - eca)
    I_leak   = g_leak * (V - e_leak)
    dV/dt    = -(I_egl19 + I_leak + I_ext_per_area) / cm

(See `wave2/channels/egl19.py` for cross-validated parameter values.)

Outputs:
    train_data.npz with keys:
      - V_traces: (n_samples, n_steps) voltage trajectories (mV)
      - I_inputs: (n_samples, n_steps) injection current density (mA/cm^2)
      - times:    (n_steps,) ms
      - meta:     dict of params + train/test split indices
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


# Nicoletti 2024 AVAL EGL-19 parameters (matching wave2/channels/egl19.py)
EGL19_PARAMS = {
    "vhm": -4.4,
    "ka": 7.5,
    "alpha": 1.43,
    "vhh": 14.9,
    "ki": 12.0,
    "vh_h": -10.5,
    "kih": 11.0,
    "cm": -4.8,
    "am": 38.1,
    "c1": 18.1,
    "cv1": 24.9,
    "cv2": 3.2,
    "c2": 30.5,
    "cv3": -1.8,
    "cv4": 6.8,
    "gbar_Scm2": 0.104385e-9 / 1123.84e-8,  # AVAL g0 / surf
    "eca_mV": 60.0,
}

LEAK_PARAMS = {
    "g_leak_Scm2": 0.150164e-9 / 1123.84e-8,  # AVAL g0 / surf
    "e_leak_mV": -39.0,
    "cm_uFcm2": 0.859551,
}


def m_inf(V: np.ndarray, p: dict) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V - p["vhm"]) / p["ka"]))


def h_inf(V: np.ndarray, p: dict) -> np.ndarray:
    a = p["alpha"]
    term1 = a / (1.0 + np.exp((V - p["vhh"]) / p["ki"])) + (1.0 - a)
    term2 = 1.0 / (1.0 + np.exp(-(V - p["vh_h"]) / p["kih"]))
    return term1 * term2


def tau_m(V: np.ndarray, p: dict) -> np.ndarray:
    return 3.7e-3 / (1.0 + np.exp(-(V - p["cm"]) / p["am"])) + 0.06


def tau_h(V: np.ndarray, p: dict) -> np.ndarray:
    a = p["c1"] * (1.0 / (1.0 + np.exp((V - p["cv1"]) / p["cv2"])))
    b = p["c2"] * (1.0 / (1.0 + np.exp((V - p["cv3"]) / p["cv4"])))
    return a + b


def integrate_single_compartment(
    I_inj_density: np.ndarray,  # (n_steps,) mA/cm^2
    dt_ms: float,
    V0_mV: float = -60.0,
) -> np.ndarray:
    """Forward-Euler integration of leak + EGL-19 single compartment."""
    p = EGL19_PARAMS
    lp = LEAK_PARAMS
    n = len(I_inj_density)
    V = np.zeros(n)
    V[0] = V0_mV
    m = m_inf(np.array([V0_mV]), p)[0]
    h = h_inf(np.array([V0_mV]), p)[0]

    for t in range(1, n):
        Vt = V[t - 1]
        m_inf_t = 1.0 / (1.0 + np.exp(-(Vt - p["vhm"]) / p["ka"]))
        h_inf_t = (
            p["alpha"] / (1.0 + np.exp((Vt - p["vhh"]) / p["ki"])) + (1.0 - p["alpha"])
        ) * (1.0 / (1.0 + np.exp(-(Vt - p["vh_h"]) / p["kih"])))
        taum = 3.7e-3 / (1.0 + np.exp(-(Vt - p["cm"]) / p["am"])) + 0.06
        tauh = p["c1"] / (1.0 + np.exp((Vt - p["cv1"]) / p["cv2"])) + p["c2"] / (
            1.0 + np.exp((Vt - p["cv3"]) / p["cv4"])
        )

        m += dt_ms * (m_inf_t - m) / max(taum, 1e-3)
        h += dt_ms * (h_inf_t - h) / max(tauh, 1e-3)

        I_egl19 = p["gbar_Scm2"] * m * m * h * (Vt - p["eca_mV"])  # mA/cm^2 numerical
        I_leak = lp["g_leak_Scm2"] * (Vt - lp["e_leak_mV"])
        I_total = I_egl19 + I_leak - I_inj_density[t]  # injection: positive-out convention
        # Units: mA/cm^2 / (uF/cm^2) = mA/uF = (1e-3 A) / (1e-6 F) = 1e3 V/s = 1 V/ms = 1000 mV/ms
        # So multiply by 1000 to get mV/ms
        dVdt = -I_total / lp["cm_uFcm2"] * 1000.0  # mV/ms
        V[t] = Vt + dt_ms * dVdt

    return V


def generate_dataset(
    n_protocols: int = 32,
    duration_ms: float = 200.0,
    dt_ms: float = 0.1,
    seed: int = 0,
    out_path: str | Path | None = None,
) -> dict:
    """Generate a dataset of injection-current protocols + voltage responses."""
    rng = np.random.default_rng(seed)
    n_steps = int(duration_ms / dt_ms)
    times = np.arange(n_steps) * dt_ms

    V_traces = np.zeros((n_protocols, n_steps))
    I_inputs = np.zeros((n_protocols, n_steps))

    for k in range(n_protocols):
        # Random injection step protocol: amplitude in {-50, ..., 200} pA, start 50 ms, length 50-100 ms
        # Convert pA to mA/cm^2: I_pA / surf_cm2 / 1e9 = mA/cm^2
        # AVA's Rin is ~6.7 GΩ; current must be small to stay subthreshold of EGL-19 runaway
        # (no K channel here, so once EGL-19 activates strongly we get unbounded depolarization).
        # Range tuned for sub-EGL-19-activation to mid-EGL-19 regimes.
        I_pA = rng.uniform(-3.0, 5.0)
        start_ms = rng.uniform(20.0, 80.0)
        length_ms = rng.uniform(30.0, 100.0)
        surf = 1123.84e-8
        # Convert pA to mA/cm^2: I_pA * 1e-12 A * 1e3 (mA/A) / surf_cm2 = pA / surf / 1e9? Let's redo:
        # 1 pA = 1e-9 mA. Density mA/cm^2 = I_pA * 1e-9 / surf_cm2.
        I_density = I_pA * 1e-9 / surf  # mA/cm^2
        mask = (times >= start_ms) & (times < start_ms + length_ms)
        I_inputs[k, mask] = I_density

        V_traces[k] = integrate_single_compartment(I_inputs[k], dt_ms=dt_ms)

    # Train / test split (80/20)
    n_train = int(0.8 * n_protocols)
    idx = rng.permutation(n_protocols)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    out = {
        "V_traces": V_traces,
        "I_inputs": I_inputs,
        "times": times,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_protocols": n_protocols,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})

    return out


if __name__ == "__main__":
    import json
    out_dir = Path(__file__).parent
    data = generate_dataset(
        n_protocols=64,
        duration_ms=200.0,
        dt_ms=0.025,
        seed=42,
        out_path=out_dir / "train_data.npz",
    )
    print(f"Generated {data['n_protocols']} protocols, {len(data['times'])} steps each")
    print(f"V range across all traces: [{data['V_traces'].min():.2f}, {data['V_traces'].max():.2f}] mV")
    print(f"Train idx: {len(data['train_idx'])}, test idx: {len(data['test_idx'])}")
    # Sanity check: V should not blow up
    if np.isnan(data['V_traces']).any():
        print("WARN: NaN in voltage traces")
    else:
        print(f"OK: no NaN in voltage traces")
