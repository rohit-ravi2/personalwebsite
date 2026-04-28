"""
Wave 2 option α-1 CP3 — Brian2 AVA cell with TRUE 4-channel set.

Channels: [IRK + LEAK + EGL19 + NCA] matching Nicoletti's actual AVAL.
Per `AVAL_simulation_iclamp.py` lines 29-32 (insertion order) and
`AVAL_simulations.py` line 26 (parameter vector g0):

    g0 = [egl19, leak, irk, nca, eleak, cm]
       = [0.104385, 0.150164, 0.1, 0, -39, 0.859551]
    surf = 1123.84e-8 cm²

NCA included with gbar=0 for apples-to-apples fidelity (Nicoletti inserts
it but assigns zero conductance — a no-op numerically but matches her
insertion list).

Distinct from `validate_phase_f_gate2.py`'s 2a Brian2 cell, which has only
3 channels (no IRK). This module restores the missing IRK to give the true
4-channel published AVAL parameterization.

Distinct from `validate_phase_f_gate2.py`'s 2b cell, which uses a 7-channel
"essential set" with non-Nicoletti channels (SLO-1, SHK-1, SHL-1, KQT-3) —
that cell has no biological referent in Nicoletti's actual AVAL.

Architecture choices
--------------------

1. **NCA inclusion with gbar=0 (no-op):** matches Nicoletti's insertion list.
   Keeps the channel structure consistent with future AVA work that may
   non-zero NCA. Numerically identical to omitting NCA.
2. **No Ca pool:** Nicoletti's AVAL doesn't insert cadiff or caintra1. EGL-19
   produces ica that's not consumed by any pool. cai stays at NEURON's default
   (5e-5 mM, P13). This matches AVAL's published behavior — Nicoletti's AVA
   is purely V-dynamics with no internal Ca dynamics tracked.
3. **No SLO-1, SHK-1, SHL-1, KQT-3:** these are NOT in Nicoletti's AVAL.
   Adding them would produce a non-Nicoletti synthetic cell.
4. **Geometry/cm/eleak from AVAL_simulations.py:**
     surf = 1123.84e-8 cm² (from neuromorpho AVAL)
     cm   = 0.859551 μF/cm²
     eleak = -39 mV
5. **g_to_Scm2 conversion:** Nicoletti's g_egl19 = 0.104385 nS at the cell
   level. Brian2 uses S/cm² (intensive). Conversion: g_Scm2 = g_nS * 1e-9 / surf.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from channels import egl19 as egl19_mod
from channels import nca as nca_mod
from channels import irk as irk_mod


# ---------------------------------------------------------------------------
# Nicoletti's AVAL canonical parameters
# ---------------------------------------------------------------------------

AVAL_SURF_CM2 = 1123.84e-8   # neuromorpho AVAL surface area
AVAL_CM_UFCM2 = 0.859551     # specific capacitance
AVAL_E_LEAK_MV = -39.0
AVAL_ECA_MV = 60.0
AVAL_EK_MV = -80.0

# Raw nS from g0 = [egl19, leak, irk, nca, eleak, cm]
AVAL_G0_NS = {
    "egl19": 0.104385,
    "leak":  0.150164,
    "irk":   0.1,
    "nca":   0.0,        # Nicoletti's AVAL has g_nca = 0 (no-op)
}

# Convert to S/cm² (intensive)
AVAL_G_SCM2 = {k: v * 1e-9 / AVAL_SURF_CM2 for k, v in AVAL_G0_NS.items()}
# Sanity values (informational):
#   egl19  ≈ 9.288e-6 S/cm²
#   leak   ≈ 1.336e-5 S/cm²
#   irk    ≈ 8.898e-6 S/cm²
#   nca    = 0


# ---------------------------------------------------------------------------
# Brian2 cell factory
# ---------------------------------------------------------------------------

def build_brian2_aval_4channel(
    surf_cm2: float = AVAL_SURF_CM2,
    cm_uFcm2: float = AVAL_CM_UFCM2,
    g_leak_Scm2: float | None = None,
    e_leak_mV: float = AVAL_E_LEAK_MV,
    g_egl19_Scm2: float | None = None,
    g_irk_Scm2: float | None = None,
    g_nca_Scm2: float | None = None,
    eca_mV: float = AVAL_ECA_MV,
    ek_mV: float = AVAL_EK_MV,
    v_init_mV: float = -60.0,
    record_components: bool = True,
):
    """Factory for Brian2 4-channel AVA cell.

    Defaults match Nicoletti's published AVAL parameter vector. Override any
    individual gbar to vary densities. Returns a callable factory matching
    the standard wave2 harness convention (returns dict with 'group',
    'monitor', 'network', 'set_v', 'inject_pA').

    The factory provides BOTH 'set_v' (for voltage clamp via network_operation
    that resets v at every dt) AND 'inject_pA' (for current clamp). The
    network_operation is created but only does anything when use_clamp is
    True — toggled via the returned 'enable_clamp'/'disable_clamp' helpers.

    Parameters
    ----------
    surf_cm2, cm_uFcm2, e_leak_mV, eca_mV, ek_mV, v_init_mV : float
        AVAL canonical values by default. Override to test sensitivity.
    g_*_Scm2 : float, optional
        Per-channel intensive conductance density. None → use Nicoletti
        default for the channel.
    record_components : bool
        If True, monitor records each channel's per-cm² current density
        plus i_leak; if False, only v + I_total recorded (faster for
        long current-clamp runs).
    """
    if g_leak_Scm2 is None:
        g_leak_Scm2 = AVAL_G_SCM2["leak"]
    if g_egl19_Scm2 is None:
        g_egl19_Scm2 = AVAL_G_SCM2["egl19"]
    if g_irk_Scm2 is None:
        g_irk_Scm2 = AVAL_G_SCM2["irk"]
    if g_nca_Scm2 is None:
        g_nca_Scm2 = AVAL_G_SCM2["nca"]

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, prefs, start_scope, pA, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        # Build cell equations:
        # - v_mV: voltage in mV (used by all channel eqs)
        # - ica from EGL-19 (Ca channel)
        # - ik from IRK
        # - i_nca from NCA (treated as ik_nca_mAcm2 by NCA module convention)
        # - i_leak from explicit leak equation
        # - I_total = sum × surf, with I_inj subtracted (current injected INTO
        #   the cell adds positive dV/dt, so I_total convention here matches
        #   NEURON's where membrane currents oppose injection):
        #     dV/dt = (-I_total_membrane + I_inj) / Cm
        #   But sign-convention-wise we keep i_total_mAcm2 as outward-positive
        #   (matching NEURON's i convention: positive ek-side current = outward).
        cell_eqs = """
        v_mV = v / mV : 1
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        ik_total_mAcm2 = ik_irk_mAcm2 : 1
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
        """ + egl19_mod.EGL19_EQS + nca_mod.NCA_EQS + irk_mod.IRK_EQS

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

        # Voltage-clamp infrastructure (force-set v each dt when enabled)
        clamp = {"enabled": False, "v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            if clamp["enabled"]:
                G.v = clamp["v_target_mV"] * mV

        record_vars = ["v", "I_total", "I_inj"]
        if record_components:
            record_vars += ["ica_egl19_mAcm2", "ik_irk_mAcm2", "ik_nca_mAcm2",
                            "i_leak_mAcm2", "i_total_mAcm2"]
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
                "g_irk_Scm2": g_irk_Scm2,
                "g_nca_Scm2": g_nca_Scm2,
                "e_leak_mV": e_leak_mV,
                "eca_mV": eca_mV,
                "ek_mV": ek_mV,
                "v_init_mV": v_init_mV,
                "channels": ["egl19", "leak", "irk", "nca"],
            },
        }

    return _factory


# ---------------------------------------------------------------------------
# Smoke test (run directly to verify cell builds + integrates)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Wave 2 option α-1 CP3 — Brian2 AVA 4-channel cell smoke test")
    print("=" * 70)

    print("\nNicoletti AVAL parameters:")
    print(f"  surf = {AVAL_SURF_CM2:.3e} cm²")
    print(f"  cm   = {AVAL_CM_UFCM2} μF/cm²")
    print(f"  e_leak = {AVAL_E_LEAK_MV} mV")
    print(f"  Channels: {list(AVAL_G0_NS.keys())}")
    print(f"  g0 (nS): {AVAL_G0_NS}")
    print(f"  g (S/cm²):")
    for k, v in AVAL_G_SCM2.items():
        print(f"    {k}: {v:.3e}")

    print("\nBuilding cell + running 100 ms passive smoke test...")
    factory = build_brian2_aval_4channel()
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
        ik = np.asarray(mon.ik_irk_mAcm2[0])
        i_leak = np.asarray(mon.i_leak_mAcm2[0])
        print(f"  ica_egl19 (final): {ica[-1]:.3e} mA/cm²")
        print(f"  ik_irk (final):    {ik[-1]:.3e} mA/cm²")
        print(f"  i_leak (final):    {i_leak[-1]:.3e} mA/cm²")

    print("\n[CP3 smoke test PASS] Cell builds and integrates without error.")
