"""
Wave 2 cellular extension Option B CP2 — Brian2 AIY cell with TRUE 7-channel set.

Channels: [egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1] matching Nicoletti's
actual AIY. Per `AIY_simulation_iclamp.py` lines 28-38 (insertion order) and
`AIY_simulation.py` line 25 (parameter vector g0).

    g0 = [leak, slo1iso, kqt1, egl19, slo1egl19, nca, shl1, eleak, cm]
       = [0.14, 1.0, 0.2, 0.1, 0.92, 0.06, 0.5, -89.57, 1.6]
    surf = 65.89e-8 cm² (from neuromorpho AIYL)
    vol  = 7.42e-12 (informational only; not used in single-compartment model)

**Top-level comment in AIY_simulation.py mislabels position [6] as "irk".**
The actual iclamp/vclamp scripts consume index [6] as `seg.shl1.gbar`. **Code
wins over comment** — this module uses shl1 at index [6] per the executed
code path. No IRK channel in AIY.

Architecture choices
--------------------

1. **No Ca pool:** Nicoletti's AIY doesn't insert cadiff or caintra1. EGL-19
   produces ica that's not consumed by any pool. cai stays at NEURON's default
   (5e-5 mM, P13). slo1iso reads this static cai. slo1egl19 uses its own
   closed-form V-dependent calcium nanodomain (no shared pool needed).

2. **slo1egl19 coupled architecture:** matches Nicoletti's slo1egl19.mod —
   reads egl19's m, h, minf, mtau directly. The Brian2 EQS expects egl19's
   eqs to be in the same NeuronGroup, so they share state.

3. **Geometry/cm/eleak from g0:**
     surf = 65.89e-8 cm² (neuromorpho AIYL)
     cm   = 1.6 μF/cm² (g0[8])
     eleak = -89.57 mV (g0[7]) — note: AIY's eleak is much more negative than
            AVAL's (-39 mV). This is the Nicoletti-published value.

4. **g_to_Scm2 conversion:** Nicoletti's g0 has channel conductances in nS at
   the cell level. Brian2 uses S/cm² (intensive). Conversion: g_Scm2 = g_nS *
   1e-9 / surf. Verified scaled values:
     leak      = 2.125e-4 S/cm²
     slo1iso   = 1.518e-3 S/cm²
     kqt1      = 3.035e-4 S/cm²
     egl19     = 1.518e-4 S/cm²
     slo1egl19 = 1.396e-3 S/cm²
     nca       = 9.106e-5 S/cm²
     shl1      = 7.588e-4 S/cm²

5. **Initial conditions:** v_init = -60 mV (matches Nicoletti's
   `h.finitialize(-60)`). All channel gates SS-initialized at -60 mV.

6. **Integration method:** Brian2 `rk4` for the multi-Ca-channel cell (slo1egl19
   has algebraic ratio constructions with rate constants that benefit from
   higher-order integrators). Same choice as option α AVAL precedent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from channels import egl19 as egl19_mod
from channels import slo1_egl19_coupled as slo1egl19_mod
from channels import nca as nca_mod
from channels import slo1_iso as slo1iso_mod
from channels import kqt1 as kqt1_mod
from channels import shl1 as shl1_mod


# ---------------------------------------------------------------------------
# Nicoletti's AIY canonical parameters
# ---------------------------------------------------------------------------

AIY_SURF_CM2 = 65.89e-8     # neuromorpho AIYL surface area
AIY_CM_UFCM2 = 1.6          # specific capacitance (g0[8])
AIY_E_LEAK_MV = -89.57      # leak reversal (g0[7])
AIY_ECA_MV = 127.59         # F18 finding (see below). Nominal `seg.eca = 60` is
                            # overridden by NEURON's ion_style behavior when
                            # multiple USEION ca mechanisms are inserted. The
                            # actual published-model eca is the Nernst-computed
                            # value at celsius=6.3°C, cai=5e-5 mM, cao=2 mM:
                            #   eca = (RT/zF) ln(cao/cai)
                            #       = 12.04 * ln(2 / 5e-5)
                            #       = 12.04 * 10.597 = 127.58 mV
                            # NEURON observed: 127.590 mV (rounding from
                            # NEURON's internal R, F constants).
AIY_EK_MV = -80.0           # set by `seg.ek = -80` — single USEION k mechanism
                            # per channel doesn't trigger ion_style override
                            # for k since each channel writes ik via separate
                            # SUFFIX. (And our diagnostic verified ik values
                            # match between Brian2 and NEURON.)

# Raw nS from g0: [leak, slo1iso, kqt1, egl19, slo1egl19, nca, shl1]
AIY_G0_NS = {
    "leak":      0.14,
    "slo1iso":   1.0,
    "kqt1":      0.2,
    "egl19":     0.1,
    "slo1egl19": 0.92,
    "nca":       0.06,
    "shl1":      0.5,
}

# Convert to S/cm² (intensive)
AIY_G_SCM2 = {k: v * 1e-9 / AIY_SURF_CM2 for k, v in AIY_G0_NS.items()}


# ---------------------------------------------------------------------------
# Brian2 cell factory
# ---------------------------------------------------------------------------

def build_brian2_aiy_7channel(
    surf_cm2: float = AIY_SURF_CM2,
    cm_uFcm2: float = AIY_CM_UFCM2,
    g_leak_Scm2: float | None = None,
    e_leak_mV: float = AIY_E_LEAK_MV,
    g_egl19_Scm2: float | None = None,
    g_slo1egl19_Scm2: float | None = None,
    g_nca_Scm2: float | None = None,
    g_slo1iso_Scm2: float | None = None,
    g_kqt1_Scm2: float | None = None,
    g_shl1_Scm2: float | None = None,
    eca_mV: float = AIY_ECA_MV,
    ek_mV: float = AIY_EK_MV,
    cai_mM: float = 5e-5,           # NEURON cai0_ca_ion default; AIY has no Ca pool
    v_init_mV: float = -60.0,
    record_components: bool = True,
):
    """Factory for Brian2 7-channel AIY cell.

    Defaults match Nicoletti's published AIY parameter vector. Override any
    individual gbar to vary densities. Returns a callable factory matching
    the standard wave2 harness convention (returns dict with 'group',
    'monitor', 'network', 'set_v', 'inject_pA').
    """
    if g_leak_Scm2 is None:
        g_leak_Scm2 = AIY_G_SCM2["leak"]
    if g_egl19_Scm2 is None:
        g_egl19_Scm2 = AIY_G_SCM2["egl19"]
    if g_slo1egl19_Scm2 is None:
        g_slo1egl19_Scm2 = AIY_G_SCM2["slo1egl19"]
    if g_nca_Scm2 is None:
        g_nca_Scm2 = AIY_G_SCM2["nca"]
    if g_slo1iso_Scm2 is None:
        g_slo1iso_Scm2 = AIY_G_SCM2["slo1iso"]
    if g_kqt1_Scm2 is None:
        g_kqt1_Scm2 = AIY_G_SCM2["kqt1"]
    if g_shl1_Scm2 is None:
        g_shl1_Scm2 = AIY_G_SCM2["shl1"]

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, prefs, start_scope, pA, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        # Cell-level eqs: combine all channel currents.
        # ica_total = ica_egl19 (only Ca channel in AIY)
        # ik_total = ik_slo1iso + ik_slo1egl19 + ik_kqt1 + ik_shl1
        # i_nca = nca's contribution (non-specific, treated separately)
        # i_leak = explicit leak
        cell_eqs = """
        v_mV = v / mV : 1
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        ik_total_mAcm2 = ik_slo1iso_mAcm2 + ik_slo1egl19_mAcm2 + ik_kqt1_mAcm2 + ik_shl1_mAcm2 : 1
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
        """ + (
            egl19_mod.EGL19_EQS
            + slo1egl19_mod.SLO1_EGL19_EQS
            + nca_mod.NCA_EQS
            + slo1iso_mod.SLO1_ISO_EQS
            + kqt1_mod.KQT1_EQS
            + shl1_mod.SHL1_EQS
        )

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

        slo1egl19_mod.slo1egl19_apply_params(G, gbar_Scm2=g_slo1egl19_Scm2, ek_mV=ek_mV,
                                              eca_mV=eca_mV)
        slo1egl19_mod.slo1egl19_init_states(G, v_mV=v_init_mV)

        nca_mod.nca_apply_params(G, gbar_Scm2=g_nca_Scm2)
        nca_mod.nca_init_states(G, v_mV=v_init_mV)

        slo1iso_mod.slo1iso_apply_params(G, gbar_Scm2=g_slo1iso_Scm2, ek_mV=ek_mV,
                                          cai_mM=cai_mM)
        slo1iso_mod.slo1iso_init_states(G, v_mV=v_init_mV)

        kqt1_mod.kqt1_apply_params(G, gbar_Scm2=g_kqt1_Scm2, ek_mV=ek_mV)
        kqt1_mod.kqt1_init_states(G, v_mV=v_init_mV)

        shl1_mod.shl1_apply_params(G, gbar_Scm2=g_shl1_Scm2, ek_mV=ek_mV)
        shl1_mod.shl1_init_states(G, v_mV=v_init_mV)

        # Voltage-clamp infrastructure (force-set v each dt when enabled)
        clamp = {"enabled": False, "v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            if clamp["enabled"]:
                G.v = clamp["v_target_mV"] * mV

        record_vars = ["v", "I_total", "I_inj"]
        if record_components:
            record_vars += [
                "ica_egl19_mAcm2",
                "ik_slo1iso_mAcm2", "ik_slo1egl19_mAcm2",
                "ik_kqt1_mAcm2", "ik_shl1_mAcm2",
                "ik_nca_mAcm2", "i_leak_mAcm2",
                "i_total_mAcm2",
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
            """Set current-clamp injection amplitude."""
            G.I_inj = amp_pA * pA

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "set_v": set_v,
            "disable_clamp": disable_clamp,
            "inject_pA": inject_pA,
            # Diagnostic info:
            "config": {
                "surf_cm2": surf_cm2,
                "cm_uFcm2": cm_uFcm2,
                "g_leak_Scm2": g_leak_Scm2,
                "g_egl19_Scm2": g_egl19_Scm2,
                "g_slo1egl19_Scm2": g_slo1egl19_Scm2,
                "g_nca_Scm2": g_nca_Scm2,
                "g_slo1iso_Scm2": g_slo1iso_Scm2,
                "g_kqt1_Scm2": g_kqt1_Scm2,
                "g_shl1_Scm2": g_shl1_Scm2,
                "e_leak_mV": e_leak_mV,
                "eca_mV": eca_mV,
                "ek_mV": ek_mV,
                "cai_mM": cai_mM,
                "v_init_mV": v_init_mV,
                "channels": ["egl19", "slo1egl19", "nca", "leak", "slo1iso", "kqt1", "shl1"],
            },
        }

    return _factory


# ---------------------------------------------------------------------------
# Smoke test (run directly to verify cell builds + integrates)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Wave 2 cellular extension Option B CP2 — Brian2 AIY 7-channel cell smoke test")
    print("=" * 70)

    print("\nNicoletti AIY parameters:")
    print(f"  surf = {AIY_SURF_CM2:.3e} cm²")
    print(f"  cm   = {AIY_CM_UFCM2} μF/cm²")
    print(f"  e_leak = {AIY_E_LEAK_MV} mV")
    print(f"  Channels (insertion order): egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1")
    print(f"  g0 (nS): {AIY_G0_NS}")
    print(f"  g (S/cm²):")
    for k, v in AIY_G_SCM2.items():
        print(f"    {k:10s}: {v:.3e}")

    print("\nBuilding cell + running 100 ms passive smoke test (no clamp, no inject)...")
    factory = build_brian2_aiy_7channel()
    bundle = factory()

    from brian2 import ms, mV, defaultclock
    import numpy as np
    defaultclock.dt = 0.025 * ms

    # No injection, no clamp — let the cell settle at its natural rest
    bundle["network"].run(100 * ms)

    mon = bundle["monitor"]
    v = np.asarray(mon.v[0]) * 1e3
    t = np.asarray(mon.t) * 1e3
    print(f"  Initial V: {v[0]:.2f} mV")
    print(f"  Final V (after 100 ms): {v[-1]:.2f} mV")
    print(f"  V range: [{v.min():.2f}, {v.max():.2f}] mV")

    if hasattr(mon, "i_total_mAcm2"):
        i_tot = np.asarray(mon.i_total_mAcm2[0])
        print(f"  i_total_mAcm2 range: [{i_tot.min():.3e}, {i_tot.max():.3e}] mA/cm²")
    if hasattr(mon, "ica_egl19_mAcm2"):
        ica = np.asarray(mon.ica_egl19_mAcm2[0])
        ik_slo1iso = np.asarray(mon.ik_slo1iso_mAcm2[0])
        ik_slo1egl19 = np.asarray(mon.ik_slo1egl19_mAcm2[0])
        ik_kqt1 = np.asarray(mon.ik_kqt1_mAcm2[0])
        ik_shl1 = np.asarray(mon.ik_shl1_mAcm2[0])
        ik_nca = np.asarray(mon.ik_nca_mAcm2[0])
        i_leak = np.asarray(mon.i_leak_mAcm2[0])
        print(f"  Per-channel current density at t=100 ms (mA/cm²):")
        print(f"    ica_egl19    = {ica[-1]:+.3e}")
        print(f"    ik_slo1iso   = {ik_slo1iso[-1]:+.3e}")
        print(f"    ik_slo1egl19 = {ik_slo1egl19[-1]:+.3e}")
        print(f"    ik_kqt1      = {ik_kqt1[-1]:+.3e}")
        print(f"    ik_shl1      = {ik_shl1[-1]:+.3e}")
        print(f"    ik_nca       = {ik_nca[-1]:+.3e}")
        print(f"    i_leak       = {i_leak[-1]:+.3e}")

    print("\n[CP2 smoke test PASS] Cell builds and integrates without error.")
