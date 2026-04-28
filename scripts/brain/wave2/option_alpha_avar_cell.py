"""
Wave 2 Stage II AVAR cell builder — Brian2 5-channel AVAR.

Channels: [IRK + LEAK + EGL19 + NCA + UNC103] matching Nicoletti's AVAR.
Per `AVAR_simulation_vclamp.py` lines 38-42 (insertion order) and
`AVAR_simulation.py` line 28 (parameter vector g0):

    g0 = [egl19, leak, irk, nca, unc103, eleak, cm]
       = [0.0643372, 0.225225, 0.042079, 0.0493356, 0.0481669, -37, 0.751761]
    surf = 1121.79e-8 cm² (from neuromorpho AVAR — slightly distinct from AVAL's 1123.84e-8)

Differences from AVAL:
  - UNC-103 added (gbar=0.0481669 nS → ~4.29e-6 S/cm²)
  - NCA non-zero (gbar=0.0493356 nS → ~4.40e-6 S/cm²) where AVAL had 0
  - LEAK higher (0.225225 vs AVAL's 0.150164 — wider passive leak)
  - IRK lower (0.042079 vs AVAL's 0.1)
  - EGL19 lower (0.0643372 vs AVAL's 0.104385)
  - cm = 0.751761 (vs AVAL's 0.859551)
  - eleak = -37 mV (vs AVAL's -39 mV)
  - surf = 1121.79e-8 cm² (vs AVAL's 1123.84e-8)

Citations
---------
  - Nicoletti et al. PLoS ONE 2024, 19(3): e0298105.
  - https://doi.org/10.1371/journal.pone.0298105

Architecture
------------

Identical structural pattern to `option_alpha_ava_cell.build_brian2_aval_4channel`,
extended with UNC-103. UNC-103 is a voltage-gated K channel (no Ca dependence)
and shares the IRK/SHK1/SHL1 pattern (P10 leak-relative scale).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from channels import egl19 as egl19_mod
from channels import nca as nca_mod
from channels import irk as irk_mod
from channels import unc103 as unc103_mod


# ---------------------------------------------------------------------------
# Nicoletti's AVAR canonical parameters
# ---------------------------------------------------------------------------

AVAR_SURF_CM2 = 1121.79e-8   # neuromorpho AVAR surface area
AVAR_CM_UFCM2 = 0.751761
AVAR_E_LEAK_MV = -37.0
AVAR_ECA_MV = 60.0
AVAR_EK_MV = -80.0

# Raw nS from g0 = [egl19, leak, irk, nca, unc103, eleak, cm]
# AVAR_simulation.py line 28
AVAR_G0_NS = {
    "egl19":  0.0643372,
    "leak":   0.225225,
    "irk":    0.042079,
    "nca":    0.0493356,
    "unc103": 0.0481669,
}

# Convert to S/cm² (intensive)
AVAR_G_SCM2 = {k: v * 1e-9 / AVAR_SURF_CM2 for k, v in AVAR_G0_NS.items()}
# Sanity values (informational):
#   egl19  ≈ 5.735e-6 S/cm²
#   leak   ≈ 2.008e-5 S/cm²
#   irk    ≈ 3.751e-6 S/cm²
#   nca    ≈ 4.398e-6 S/cm²
#   unc103 ≈ 4.294e-6 S/cm²


# ---------------------------------------------------------------------------
# Brian2 cell factory
# ---------------------------------------------------------------------------

def build_brian2_avar_5channel(
    surf_cm2: float = AVAR_SURF_CM2,
    cm_uFcm2: float = AVAR_CM_UFCM2,
    g_leak_Scm2: float | None = None,
    e_leak_mV: float = AVAR_E_LEAK_MV,
    g_egl19_Scm2: float | None = None,
    g_irk_Scm2: float | None = None,
    g_nca_Scm2: float | None = None,
    g_unc103_Scm2: float | None = None,
    eca_mV: float = AVAR_ECA_MV,
    ek_mV: float = AVAR_EK_MV,
    v_init_mV: float = -60.0,
    record_components: bool = True,
):
    """Factory for Brian2 5-channel AVAR cell.

    Defaults match Nicoletti's published AVAR parameter vector. Override any
    individual gbar to vary densities. Returns a callable factory matching
    the standard wave2 harness convention (returns dict with 'group',
    'monitor', 'network', 'set_v', 'inject_pA', etc.).

    Parameters mirror `build_brian2_aval_4channel` with addition of g_unc103_Scm2.
    """
    if g_leak_Scm2 is None:
        g_leak_Scm2 = AVAR_G_SCM2["leak"]
    if g_egl19_Scm2 is None:
        g_egl19_Scm2 = AVAR_G_SCM2["egl19"]
    if g_irk_Scm2 is None:
        g_irk_Scm2 = AVAR_G_SCM2["irk"]
    if g_nca_Scm2 is None:
        g_nca_Scm2 = AVAR_G_SCM2["nca"]
    if g_unc103_Scm2 is None:
        g_unc103_Scm2 = AVAR_G_SCM2["unc103"]

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, prefs, start_scope, pA, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        # Cell equations: identical structure to AVAL 4-channel, with UNC-103
        # contributing additional ik via ik_unc103_mAcm2.
        # PHASE δ NOTE: I_inj retained for backwards-compat with VC/CC harness;
        # I_ext alias added for ModulationLayer integration. Mathematically
        # identical (sum into the cell).
        cell_eqs = """
        v_mV = v / mV : 1
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        ik_total_mAcm2 = ik_irk_mAcm2 + ik_unc103_mAcm2 : 1
        i_nca_mAcm2 = ik_nca_mAcm2 : 1
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_total_mAcm2 = i_leak_mAcm2 + i_nca_mAcm2 + ica_mAcm2 + ik_total_mAcm2 : 1
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA - I_inj - I_ext : amp
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        I_inj : amp
        I_ext : amp
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + egl19_mod.EGL19_EQS + nca_mod.NCA_EQS + irk_mod.IRK_EQS + unc103_mod.UNC103_EQS

        G = NeuronGroup(1, cell_eqs, method="rk4")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV
        G.I_inj = 0 * pA
        G.I_ext = 0 * pA

        # Channel parameters + state init
        egl19_mod.egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=eca_mV)
        egl19_mod.egl19_init_states(G, v_mV=v_init_mV)
        irk_mod.irk_apply_params(G, gbar_Scm2=g_irk_Scm2, ek_mV=ek_mV)
        irk_mod.irk_init_states(G, v_mV=v_init_mV)
        nca_mod.nca_apply_params(G, gbar_Scm2=g_nca_Scm2)
        nca_mod.nca_init_states(G, v_mV=v_init_mV)
        unc103_mod.unc103_apply_params(G, gbar_Scm2=g_unc103_Scm2, ek_mV=ek_mV)
        unc103_mod.unc103_init_states(G, v_mV=v_init_mV)

        # Voltage-clamp infrastructure (force-set v each dt when enabled)
        clamp = {"enabled": False, "v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            if clamp["enabled"]:
                G.v = clamp["v_target_mV"] * mV

        record_vars = ["v", "I_total", "I_inj"]
        if record_components:
            record_vars += [
                "ica_egl19_mAcm2", "ik_irk_mAcm2", "ik_nca_mAcm2",
                "ik_unc103_mAcm2", "i_leak_mAcm2", "i_total_mAcm2",
            ]
        mon = StateMonitor(G, record_vars, record=True)
        net = Network(G, mon, _clamp)

        def set_v(v_mV: float) -> None:
            """Enable voltage clamp at v_mV."""
            clamp["enabled"] = True
            clamp["v_target_mV"] = float(v_mV)
            G.v = float(v_mV) * mV

        def disable_clamp() -> None:
            clamp["enabled"] = False

        def inject_pA(amp_pA: float) -> None:
            """Set current-clamp injection amplitude (legacy I_inj path)."""
            G.I_inj = amp_pA * pA

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "set_v": set_v,
            "disable_clamp": disable_clamp,
            "inject_pA": inject_pA,
            "config": {
                "surf_cm2": surf_cm2,
                "cm_uFcm2": cm_uFcm2,
                "g_leak_Scm2": g_leak_Scm2,
                "g_egl19_Scm2": g_egl19_Scm2,
                "g_irk_Scm2": g_irk_Scm2,
                "g_nca_Scm2": g_nca_Scm2,
                "g_unc103_Scm2": g_unc103_Scm2,
                "e_leak_mV": e_leak_mV,
                "eca_mV": eca_mV,
                "ek_mV": ek_mV,
                "v_init_mV": v_init_mV,
                "channels": ["egl19", "leak", "irk", "nca", "unc103"],
            },
        }

    return _factory


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Wave 2 Stage II AVAR — Brian2 5-channel cell smoke test")
    print("=" * 70)

    print("\nNicoletti AVAR parameters:")
    print(f"  surf = {AVAR_SURF_CM2:.3e} cm²")
    print(f"  cm   = {AVAR_CM_UFCM2} μF/cm²")
    print(f"  e_leak = {AVAR_E_LEAK_MV} mV")
    print(f"  Channels: {list(AVAR_G0_NS.keys())}")
    print(f"  g0 (nS): {AVAR_G0_NS}")
    print(f"  g (S/cm²):")
    for k, v in AVAR_G_SCM2.items():
        print(f"    {k}: {v:.3e}")

    print("\nBuilding cell + running 100 ms passive smoke test...")
    factory = build_brian2_avar_5channel()
    bundle = factory()

    from brian2 import ms, mV, defaultclock
    import numpy as np
    defaultclock.dt = 0.025 * ms

    bundle["network"].run(100 * ms)

    mon = bundle["monitor"]
    v = np.asarray(mon.v[0]) * 1e3
    print(f"  Initial V: {v[0]:.2f} mV")
    print(f"  Final V (after 100 ms): {v[-1]:.2f} mV")
    print(f"  V range: [{v.min():.2f}, {v.max():.2f}] mV")

    if hasattr(mon, "i_total_mAcm2"):
        i_tot = np.asarray(mon.i_total_mAcm2[0])
        print(f"  i_total_mAcm2 range: [{i_tot.min():.3e}, {i_tot.max():.3e}] mA/cm²")
    if hasattr(mon, "ica_egl19_mAcm2"):
        ica = np.asarray(mon.ica_egl19_mAcm2[0])
        ik_irk = np.asarray(mon.ik_irk_mAcm2[0])
        ik_unc103 = np.asarray(mon.ik_unc103_mAcm2[0])
        i_leak = np.asarray(mon.i_leak_mAcm2[0])
        print(f"  ica_egl19  (final): {ica[-1]:.3e} mA/cm²")
        print(f"  ik_irk     (final): {ik_irk[-1]:.3e} mA/cm²")
        print(f"  ik_unc103  (final): {ik_unc103[-1]:.3e} mA/cm²")
        print(f"  i_leak     (final): {i_leak[-1]:.3e} mA/cm²")

    print("\n[AVAR smoke test PASS] Cell builds and integrates without error.")
