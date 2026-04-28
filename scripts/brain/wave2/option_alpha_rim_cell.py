"""
Wave 2 cellular extension RIM CP4 — Brian2 RIM 7-channel cell.

Channels: [shl1, egl2, irk, cca1, unc2, egl19, leak] matching Nicoletti's
actual RIM. Per `RIM_simulation_iclamp.py` lines 31-38 (insertion order)
and `RIM_simulation.py` lines 25-27 (parameter vector g, already in S/cm²).

    g = [shl1, egl2, irk, cca1, unc2, egl19, leak, eleak, cm]
      = [9.049e-4, 1.412e-4, 3.273e-4, 8.452e-4, 9.677e-5, 3.201e-4,
         9.677e-5, -50, 1.5]

Per RIM_simulation.py line 25 ("conductances in S/cm^2"), g is **already
in S/cm²** and is passed directly to RIM_simulation_iclamp without
gScm2() rescaling. We use the same convention here — NO further conversion.

Architecture choices
--------------------

1. **No Ca pool.** Nicoletti's RIM doesn't insert cadiff or caintra1.
   ica from cca1 + unc2 + egl19 is not consumed by any pool. cai stays at
   NEURON's default (5e-5 mM, P13). No channel in RIM reads cai, so the
   absence of a pool is consequence-free.

2. **eca = 60 mV (NOT 127.59).** F18 refinement: RIM's three USEION ca
   channels (cca1, unc2, egl19) all have IDENTICAL declarations
   `USEION ca READ eca WRITE ica`. Symmetric contract → NEURON's
   ion_style does NOT override user-set seg.eca. Empirically verified:
   `seg.eca` after `h.run()` = 60.0000 mV (not 127.59). Distinct from
   AIY where slo1egl19's `USEION ca READ eca` (no WRITE ica) triggered
   the asymmetry → override. See cellular_validation_findings.md F18
   refinement entry.

3. **Geometry/cm/eleak from g0:**
     surf  = 103.34e-8 cm² (neuromorpho RIML)
     cm    = 1.5 μF/cm² (g[8])
     eleak = -50 mV (g[7])

4. **No g_to_Scm2 conversion:** Nicoletti's RIM g vector is intensive (S/cm²)
   already. Passing g0[6] = 9.677e-5 directly. The shorthand `gScm2_index=N`
   is NOT applied for RIM (would double-divide by surf and produce wrong
   conductances).

5. **Initial conditions:** v_init = -60 mV (matches `h.finitialize(-60)`).

6. **Integration method:** Brian2 `rk4` for the multi-Ca-channel cell
   (matches AVAL/AIY precedent).

7. **No GLOBAL state in Brian2:** UNC-2's NMODL GLOBAL declarations
   (minf, hinf, mtau, htau, munc2, hunc2) are functionally per-cell in
   Brian2 by default — see `channels/unc2.py` module docstring for the
   GLOBAL handling decision rationale.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from channels import shl1 as shl1_mod
from channels import egl2 as egl2_mod
from channels import irk as irk_mod
from channels import cca1 as cca1_mod
from channels import unc2 as unc2_mod
from channels import egl19 as egl19_mod


# ---------------------------------------------------------------------------
# Nicoletti's RIM canonical parameters (already in S/cm²)
# ---------------------------------------------------------------------------

RIM_SURF_CM2 = 103.34e-8     # neuromorpho RIML surface area
RIM_CM_UFCM2 = 1.5           # specific capacitance (g[8])
RIM_E_LEAK_MV = -50.0        # leak reversal (g[7])
RIM_ECA_MV = 60.0            # F18 refinement: RIM's symmetric USEION ca
                             # contract preserves user-set eca. Empirically
                             # verified seg.eca = 60.0000 after h.run().
RIM_EK_MV = -80.0

# Raw S/cm² from g (already intensive; no rescale applied)
RIM_G_SCM2 = {
    "shl1":  0.0009048750067326097,
    "egl2":  0.0001411644285181245,
    "irk":   0.0003272854640954744,
    "cca1":  0.0008451919806776876,
    "unc2":  9.676795045480941e-05,
    "egl19": 0.00032005818627638106,
    "leak":  9.676795045480941e-05,
}


# ---------------------------------------------------------------------------
# Brian2 cell factory
# ---------------------------------------------------------------------------

def build_brian2_rim_7channel(
    surf_cm2: float = RIM_SURF_CM2,
    cm_uFcm2: float = RIM_CM_UFCM2,
    g_leak_Scm2: float | None = None,
    e_leak_mV: float = RIM_E_LEAK_MV,
    g_shl1_Scm2: float | None = None,
    g_egl2_Scm2: float | None = None,
    g_irk_Scm2: float | None = None,
    g_cca1_Scm2: float | None = None,
    g_unc2_Scm2: float | None = None,
    g_egl19_Scm2: float | None = None,
    eca_mV: float = RIM_ECA_MV,
    ek_mV: float = RIM_EK_MV,
    cai_mM: float = 5e-5,           # NEURON cai0_ca_ion default; RIM has no Ca pool
    v_init_mV: float = -60.0,
    record_components: bool = True,
):
    """Factory for Brian2 7-channel RIM cell.

    Returns a callable factory matching the standard wave2 harness convention
    (returns dict with 'group', 'monitor', 'network', 'set_v', 'inject_pA').
    """
    if g_leak_Scm2 is None:
        g_leak_Scm2 = RIM_G_SCM2["leak"]
    if g_shl1_Scm2 is None:
        g_shl1_Scm2 = RIM_G_SCM2["shl1"]
    if g_egl2_Scm2 is None:
        g_egl2_Scm2 = RIM_G_SCM2["egl2"]
    if g_irk_Scm2 is None:
        g_irk_Scm2 = RIM_G_SCM2["irk"]
    if g_cca1_Scm2 is None:
        g_cca1_Scm2 = RIM_G_SCM2["cca1"]
    if g_unc2_Scm2 is None:
        g_unc2_Scm2 = RIM_G_SCM2["unc2"]
    if g_egl19_Scm2 is None:
        g_egl19_Scm2 = RIM_G_SCM2["egl19"]

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, prefs, start_scope, pA, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        # Cell-level eqs:
        # ica_total = ica_cca1 + ica_unc2 + ica_egl19 (3 Ca channels)
        # ik_total = ik_shl1 + ik_egl2 + ik_irk
        # i_leak = explicit leak
        cell_eqs = """
        v_mV = v / mV : 1
        ica_mAcm2 = ica_cca1_mAcm2 + ica_unc2_mAcm2 + ica_egl19_mAcm2 : 1
        ik_total_mAcm2 = ik_shl1_mAcm2 + ik_egl2_mAcm2 + ik_irk_mAcm2 : 1
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_total_mAcm2 = i_leak_mAcm2 + ica_mAcm2 + ik_total_mAcm2 : 1
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA - I_inj - I_ext : amp
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        I_inj : amp
        I_ext : amp
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + (
            shl1_mod.SHL1_EQS
            + egl2_mod.EGL2_EQS
            + irk_mod.IRK_EQS
            + cca1_mod.CCA1_EQS
            + unc2_mod.UNC2_EQS
            + egl19_mod.EGL19_EQS
        )

        G = NeuronGroup(1, cell_eqs, method="rk4")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV
        G.I_inj = 0 * pA
        G.I_ext = 0 * pA

        # Channel parameters + state init.
        # All three Ca channels receive eca_mV explicitly per F18 methodology.
        shl1_mod.shl1_apply_params(G, gbar_Scm2=g_shl1_Scm2, ek_mV=ek_mV)
        shl1_mod.shl1_init_states(G, v_mV=v_init_mV)

        egl2_mod.egl2_apply_params(G, gbar_Scm2=g_egl2_Scm2, ek_mV=ek_mV)
        egl2_mod.egl2_init_states(G, v_mV=v_init_mV)

        irk_mod.irk_apply_params(G, gbar_Scm2=g_irk_Scm2, ek_mV=ek_mV)
        irk_mod.irk_init_states(G, v_mV=v_init_mV)

        cca1_mod.cca1_apply_params(G, gbar_Scm2=g_cca1_Scm2, eca_mV=eca_mV)
        cca1_mod.cca1_init_states(G, v_mV=v_init_mV)

        unc2_mod.unc2_apply_params(G, gbar_Scm2=g_unc2_Scm2, eca_mV=eca_mV)
        unc2_mod.unc2_init_states(G, v_mV=v_init_mV)

        egl19_mod.egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=eca_mV)
        egl19_mod.egl19_init_states(G, v_mV=v_init_mV)

        # Voltage-clamp infrastructure
        clamp = {"enabled": False, "v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            if clamp["enabled"]:
                G.v = clamp["v_target_mV"] * mV

        record_vars = ["v", "I_total", "I_inj"]
        if record_components:
            record_vars += [
                "ica_cca1_mAcm2", "ica_unc2_mAcm2", "ica_egl19_mAcm2",
                "ik_shl1_mAcm2", "ik_egl2_mAcm2", "ik_irk_mAcm2",
                "i_leak_mAcm2", "i_total_mAcm2",
            ]
        mon = StateMonitor(G, record_vars, record=True)
        net = Network(G, mon, _clamp)

        def set_v(v_mV: float) -> None:
            clamp["enabled"] = True
            clamp["v_target_mV"] = float(v_mV)
            G.v = float(v_mV) * mV

        def disable_clamp() -> None:
            clamp["enabled"] = False

        def inject_pA(amp_pA: float) -> None:
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
                "g_shl1_Scm2": g_shl1_Scm2,
                "g_egl2_Scm2": g_egl2_Scm2,
                "g_irk_Scm2": g_irk_Scm2,
                "g_cca1_Scm2": g_cca1_Scm2,
                "g_unc2_Scm2": g_unc2_Scm2,
                "g_egl19_Scm2": g_egl19_Scm2,
                "e_leak_mV": e_leak_mV,
                "eca_mV": eca_mV,
                "ek_mV": ek_mV,
                "cai_mM": cai_mM,
                "v_init_mV": v_init_mV,
                "channels": ["shl1", "egl2", "irk", "cca1", "unc2", "egl19", "leak"],
            },
        }

    return _factory


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Wave 2 cellular extension Wave 2/RIM CP4 — Brian2 RIM 7-channel cell smoke test")
    print("=" * 70)

    print("\nNicoletti RIM parameters:")
    print(f"  surf  = {RIM_SURF_CM2:.3e} cm²")
    print(f"  cm    = {RIM_CM_UFCM2} μF/cm²")
    print(f"  eleak = {RIM_E_LEAK_MV} mV")
    print(f"  eca   = {RIM_ECA_MV} mV (F18 refinement: symmetric USEION ca contract)")
    print(f"  ek    = {RIM_EK_MV} mV")
    print(f"  Channels (insertion order): shl1, egl2, irk, cca1, unc2, egl19, leak")
    print(f"  g (S/cm²; already intensive — no gScm2 rescale):")
    for k, v in RIM_G_SCM2.items():
        print(f"    {k:6s}: {v:.4e}")

    print("\nBuilding cell + running 100 ms passive smoke test (no clamp, no inject)...")
    factory = build_brian2_rim_7channel()
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

    if hasattr(mon, "ica_cca1_mAcm2"):
        print(f"\n  Per-channel current density at t=100 ms (mA/cm²):")
        print(f"    ica_cca1   = {np.asarray(mon.ica_cca1_mAcm2[0])[-1]:+.3e}")
        print(f"    ica_unc2   = {np.asarray(mon.ica_unc2_mAcm2[0])[-1]:+.3e}")
        print(f"    ica_egl19  = {np.asarray(mon.ica_egl19_mAcm2[0])[-1]:+.3e}")
        print(f"    ik_shl1    = {np.asarray(mon.ik_shl1_mAcm2[0])[-1]:+.3e}")
        print(f"    ik_egl2    = {np.asarray(mon.ik_egl2_mAcm2[0])[-1]:+.3e}")
        print(f"    ik_irk     = {np.asarray(mon.ik_irk_mAcm2[0])[-1]:+.3e}")
        print(f"    i_leak     = {np.asarray(mon.i_leak_mAcm2[0])[-1]:+.3e}")

    print("\n[CP4 smoke test PASS] Cell builds and integrates without error.")
