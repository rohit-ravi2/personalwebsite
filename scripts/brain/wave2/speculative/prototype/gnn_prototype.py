"""
X.1d prototype — minimal Variant A GNN for cellular dynamics.

Architecture: 2-node mechanistic-anchored GNN.
- Node 0 = "soma": EGL-19 + leak, channel kinetics mechanistic.
- Node 1 = "axon-stub": leak only, smaller surface area.
- Edge: axial coupling (V_0 - V_1) / Ra, learnable Ra.
- All currents into node 0 from injection (matching single-compartment ground truth).

The single-compartment NumPy ground truth (see data.py) is what node 0 must
reproduce. The 2-node GNN's job: with axial Ra learnable, can it reproduce node 0
behavior? (Trivially yes for Ra → ∞; the harder question is if axial coupling is
finite, what happens?)

This is the **sanity-check prototype** scoped per X.1d:
- Tests that the training pipeline works (data → forward → loss → backward → param update).
- Tests that the mechanistic-anchored Variant A architecture composes correctly with
  PyTorch autograd through the unrolled time loop.
- It does **not** attempt Mellem-style architectural validation.

Output: trained model + per-trace V trajectory MAE + summary in results.json.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# EGL-19 + leak parameters (matching data.py / wave2/channels/egl19.py / Nicoletti 2024)
EGL19 = {
    "vhm": -4.4, "ka": 7.5, "alpha": 1.43, "vhh": 14.9, "ki": 12.0,
    "vh_h": -10.5, "kih": 11.0, "cm": -4.8, "am": 38.1,
    "c1": 18.1, "cv1": 24.9, "cv2": 3.2, "c2": 30.5, "cv3": -1.8, "cv4": 6.8,
    "gbar_Scm2": 0.104385e-9 / 1123.84e-8,
    "eca_mV": 60.0,
}
LEAK = {
    "g_leak_Scm2": 0.150164e-9 / 1123.84e-8,
    "e_leak_mV": -39.0,
    "cm_uFcm2": 0.859551,
}


def egl19_dynamics(V, m, h):
    """Vectorized EGL-19 mechanistic update (PyTorch).
    V, m, h: tensors broadcastable, V in mV.
    Returns: m_inf, h_inf, tau_m, tau_h (all tensors), I_egl19 (mA/cm^2).
    """
    p = EGL19
    m_inf = 1.0 / (1.0 + torch.exp(-(V - p["vhm"]) / p["ka"]))
    h_inf = (
        p["alpha"] / (1.0 + torch.exp((V - p["vhh"]) / p["ki"]))
        + (1.0 - p["alpha"])
    ) * (1.0 / (1.0 + torch.exp(-(V - p["vh_h"]) / p["kih"])))
    tau_m = 3.7e-3 / (1.0 + torch.exp(-(V - p["cm"]) / p["am"])) + 0.06
    tau_h = (
        p["c1"] / (1.0 + torch.exp((V - p["cv1"]) / p["cv2"]))
        + p["c2"] / (1.0 + torch.exp((V - p["cv3"]) / p["cv4"]))
    )
    I_egl19 = p["gbar_Scm2"] * m * m * h * (V - p["eca_mV"])  # mA/cm^2
    return m_inf, h_inf, tau_m, tau_h, I_egl19


class TwoNodeGNN(nn.Module):
    """2-node mechanistic-anchored Variant A GNN.

    Trainable parameters:
        - log_axial_g: log of axial conductance (S, single edge).  Initialized to medium coupling.
        - log_gbar_egl19_node0: log of EGL-19 gbar on soma node (initialized to AVAL g0).
        - log_gleak_node1: log of leak on axon node (initialized to leak_AVAL).

    Mechanistic:
        - EGL-19 channel kinetics mechanistic (above).
        - Leak channels mechanistic.
        - Capacitance is per-node, fixed.

    Forward: takes I_inj_density per timestep (applied to node 0 only) and returns
    V_0 trajectory.
    """
    def __init__(
        self,
        n_steps: int,
        dt_ms: float = 0.025,
        cm_uFcm2_node0: float = LEAK["cm_uFcm2"],
        cm_uFcm2_node1: float = LEAK["cm_uFcm2"] * 0.3,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.dt_ms = dt_ms
        self.cm0 = cm_uFcm2_node0
        self.cm1 = cm_uFcm2_node1

        # Initialize learnable params at sensible scales
        # axial conductance: start large enough that the two nodes are nearly tied (degenerate to single-compartment)
        # We will let training find the right value.
        self.log_axial_g = nn.Parameter(torch.tensor(np.log(1e-3), dtype=torch.float32))
        self.log_gbar_egl19 = nn.Parameter(torch.tensor(np.log(EGL19["gbar_Scm2"]), dtype=torch.float32))
        self.log_gleak0 = nn.Parameter(torch.tensor(np.log(LEAK["g_leak_Scm2"]), dtype=torch.float32))
        self.log_gleak1 = nn.Parameter(torch.tensor(np.log(LEAK["g_leak_Scm2"] * 0.5), dtype=torch.float32))

    def forward(self, I_inj_density: torch.Tensor) -> torch.Tensor:
        """I_inj_density: (B, T) tensor of injection density (mA/cm^2) into node 0.
        Returns: V0 trajectory (B, T) in mV.
        """
        B, T = I_inj_density.shape
        device = I_inj_density.device
        # Initial conditions
        V0 = torch.full((B,), -60.0, device=device)
        V1 = torch.full((B,), -60.0, device=device)
        m_inf0, h_inf0, _, _, _ = egl19_dynamics(V0, torch.zeros_like(V0), torch.zeros_like(V0))
        m = m_inf0.clone()
        h = h_inf0.clone()

        V0_traj = torch.zeros((B, T), device=device)
        V0_traj[:, 0] = V0

        gbar_egl19 = torch.exp(self.log_gbar_egl19)
        gleak0 = torch.exp(self.log_gleak0)
        gleak1 = torch.exp(self.log_gleak1)
        axial_g = torch.exp(self.log_axial_g)

        for t in range(1, T):
            # Channel kinetics on node 0 (soma)
            m_inf, h_inf, tau_m, tau_h, _ = egl19_dynamics(V0, m, h)
            m = m + self.dt_ms * (m_inf - m) / torch.clamp(tau_m, min=1e-3)
            h = h + self.dt_ms * (h_inf - h) / torch.clamp(tau_h, min=1e-3)

            # Channel currents on node 0
            I_egl19 = gbar_egl19 * m * m * h * (V0 - EGL19["eca_mV"])
            I_leak0 = gleak0 * (V0 - LEAK["e_leak_mV"])

            # Axial current from node 1 to node 0
            I_axial_01 = axial_g * (V0 - V1)  # current leaving node 0 toward node 1
            # Currents on node 1: leak only + axial from 0
            I_leak1 = gleak1 * (V1 - LEAK["e_leak_mV"])
            I_axial_10 = -I_axial_01  # current arriving at node 1 from node 0 (entering node 1 = sign flip)

            # Total density on node 0: leak + EGL-19 + axial - injection
            I_total_0 = I_egl19 + I_leak0 + I_axial_01 - I_inj_density[:, t]
            # Node 1: leak + (-axial coming in from 0)
            I_total_1 = I_leak1 - I_axial_01  # current entering = from 0; net into V1 dynamics is opposite

            # Update V (units conversion as in data.py: × 1000 for mV/ms)
            dV0 = -I_total_0 / self.cm0 * 1000.0
            dV1 = -I_total_1 / self.cm1 * 1000.0
            V0 = V0 + self.dt_ms * dV0
            V1 = V1 + self.dt_ms * dV1

            V0_traj[:, t] = V0

        return V0_traj


def load_data(path: Path):
    d = np.load(path)
    return {
        "V_traces": torch.tensor(d["V_traces"], dtype=torch.float32),
        "I_inputs": torch.tensor(d["I_inputs"], dtype=torch.float32),
        "times": torch.tensor(d["times"], dtype=torch.float32),
        "train_idx": d["train_idx"],
        "test_idx": d["test_idx"],
    }
