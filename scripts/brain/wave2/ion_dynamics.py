"""
scripts/brain/wave2/ion_dynamics.py — Layer 1 foundation: ion state variables,
dynamic Nernst, per-cell geometry.

Layer 1 §7.1 foundation work block (2026-05-12).

Scope: Provides Brian2 equation fragments + per-cell geometry bookkeeping +
Nernst/GHK helpers for the four production-grade Wave 2 cells (AVAL, AVAR,
AIY, RIM). Does NOT include channel currents or pump kinetics — only the
substrate state variables that channels + pumps write to.

Per `docs/layer1_design_decisions.md` v2 (authorized 2026-05-12):
- §2.1  Architecture (a): V and [X] coupled exactly via mass + charge conservation
- §2.4  Phenomenological Ca buffering κ_B = 100
- §2.5  r_eff = 0.5 μm default; sensitivity sweep {0.25, 0.5, 1.0}
- §2.6  Mammalian-default initial conditions with explicit labeling
- §6.5  [Cl]_in = 5 mM (Payne 1997 approximation)

Convention: All Brian2 variables are dimensionless (`: 1`). Currents are
in mA/cm² (outward-positive, NEURON convention). Concentrations are in mM.
Time derivatives carry `* Hz` to provide 1/second units. This matches the
existing Wave 2 cell builder convention (see `option_alpha_ava_cell.py`).

Composition contract:
    The cell builder must inject `ion_iK_total_mAcm2`, `ion_iNa_total_mAcm2`,
    `ion_iCl_total_mAcm2`, `ion_iCa_total_mAcm2` between the state/Nernst
    fragment and the ion-balance fragment. These are total per-ion currents
    summed across all channels + pumps in the cell. See `get_ion_dynamics_eqs`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

F_FARADAY_C_PER_MOL = 96485.0
R_GAS_J_PER_MOL_K = 8.314
T_C_ELEGANS_K = 293.15  # 20 °C — C. elegans standard physiological temperature
RT_OVER_F_mV = R_GAS_J_PER_MOL_K * T_C_ELEGANS_K / F_FARADAY_C_PER_MOL * 1000.0
# RT_OVER_F_mV ≈ 25.26 mV at 20 °C

ION_CHARGE = {"K": +1, "Na": +1, "Cl": -1, "Ca": +2}


# ---------------------------------------------------------------------------
# Default ion concentrations (mM)
# ---------------------------------------------------------------------------

# Mammalian defaults — Layer 1 v1 fallback per §2.6, §6.5 authorizations.
# All labeled "approximation from mammalian neurons; awaiting empirical
# C. elegans-specific refinement" in design doc §3.1.

EXTRACELLULAR_DEFAULT_mM: dict[str, float] = {
    "K":  4.0,
    "Na": 140.0,
    "Cl": 110.0,
    "Ca": 2.0,
}

INTRACELLULAR_DEFAULT_mM: dict[str, float] = {
    "K":  140.0,
    "Na": 10.0,
    "Cl": 5.0,        # Payne 1997 mammalian KCC2-dominant approximation
    "Ca": 5.0e-5,     # 50 nM
}

KAPPA_BUFFER_CA_DEFAULT = 100.0  # Phenomenological Ca buffering factor


# ---------------------------------------------------------------------------
# Cell capacitances + specific Cm (Nicoletti 2024 Table 3 + AVAL Wave 2 spec)
# ---------------------------------------------------------------------------

NICOLETTI_CAPACITANCE_pF: dict[str, float] = {
    "AVAL": 9.66,
    "AVAR": 8.43,
    "AIY":  1.05,
    "RIM":  1.55,
}

SPECIFIC_CM_uFcm2 = 0.86  # AVAL canonical (`option_alpha_ava_cell.py` line 60)

R_EFF_DEFAULT_um = 0.5
R_EFF_SWEEP_um: tuple[float, ...] = (0.25, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Per-cell geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellGeometry:
    """Per-cell membrane area + volume derived from Nicoletti capacitance.

    surface_cm2 = C_m / specific_Cm   (well-constrained)
    volume_L    = surface_cm2 · r_eff_cm / 2   (cylindrical compartment;
                                                r_eff is the free parameter
                                                per §6.1 authorization)
    """
    cell_name: str
    capacitance_pF: float
    specific_cm_uFcm2: float = SPECIFIC_CM_uFcm2
    r_eff_um: float = R_EFF_DEFAULT_um

    @property
    def surface_cm2(self) -> float:
        # C_m [F] / specific_Cm [F/cm²] = area [cm²]
        return (self.capacitance_pF * 1e-12) / (self.specific_cm_uFcm2 * 1e-6)

    @property
    def volume_L(self) -> float:
        """Volume in liters (single-compartment cylindrical approximation)."""
        r_eff_cm = self.r_eff_um * 1e-4
        # vol [cm³] = surf [cm²] · r [cm] / 2; 1 cm³ = 1 mL = 1e-3 L
        return self.surface_cm2 * r_eff_cm / 2.0 * 1e-3

    @property
    def volume_fL(self) -> float:
        return self.volume_L * 1e15


def make_cell_geometries(r_eff_um: float = R_EFF_DEFAULT_um) -> dict[str, CellGeometry]:
    """Build CellGeometry for all four production-grade Wave 2 cells."""
    return {
        name: CellGeometry(
            cell_name=name,
            capacitance_pF=NICOLETTI_CAPACITANCE_pF[name],
            r_eff_um=r_eff_um,
        )
        for name in NICOLETTI_CAPACITANCE_pF
    }


# ---------------------------------------------------------------------------
# Nernst + GHK helpers (used for validation hand-checks)
# ---------------------------------------------------------------------------

def nernst_potential_mV(
    conc_out_mM: float,
    conc_in_mM: float,
    z: int,
    temperature_K: float = T_C_ELEGANS_K,
) -> float:
    """Nernst equilibrium potential (mV).

        E_X = (RT / zF) · ln([X]_out / [X]_in)
    """
    if conc_in_mM <= 0 or conc_out_mM <= 0:
        raise ValueError(f"Non-positive concentration: in={conc_in_mM}, out={conc_out_mM}")
    rt_over_zF_mV = R_GAS_J_PER_MOL_K * temperature_K / (z * F_FARADAY_C_PER_MOL) * 1000.0
    return rt_over_zF_mV * math.log(conc_out_mM / conc_in_mM)


def ghk_resting_potential_mV(
    perm_K: float,
    perm_Na: float,
    perm_Cl: float,
    K_in_mM: float = INTRACELLULAR_DEFAULT_mM["K"],
    K_out_mM: float = EXTRACELLULAR_DEFAULT_mM["K"],
    Na_in_mM: float = INTRACELLULAR_DEFAULT_mM["Na"],
    Na_out_mM: float = EXTRACELLULAR_DEFAULT_mM["Na"],
    Cl_in_mM: float = INTRACELLULAR_DEFAULT_mM["Cl"],
    Cl_out_mM: float = EXTRACELLULAR_DEFAULT_mM["Cl"],
    temperature_K: float = T_C_ELEGANS_K,
) -> float:
    """Goldman-Hodgkin-Katz resting potential prediction (mV).

        V = (RT/F) · ln (
            (P_K [K]_out + P_Na [Na]_out + P_Cl [Cl]_in) /
            (P_K [K]_in  + P_Na [Na]_in  + P_Cl [Cl]_out)
        )
    """
    rt_over_F_mV = R_GAS_J_PER_MOL_K * temperature_K / F_FARADAY_C_PER_MOL * 1000.0
    num = perm_K * K_out_mM + perm_Na * Na_out_mM + perm_Cl * Cl_in_mM
    den = perm_K * K_in_mM + perm_Na * Na_in_mM + perm_Cl * Cl_out_mM
    if num <= 0 or den <= 0:
        raise ValueError("Invalid permeability × concentration combination")
    return rt_over_F_mV * math.log(num / den)


# ---------------------------------------------------------------------------
# Brian2 equation fragments
# ---------------------------------------------------------------------------

def get_state_and_nernst_eqs() -> str:
    """Brian2 equation fragment: ion state variable declarations + dynamic
    Nernst potentials.

    Use as the FIRST piece of a cell's equation string. The cell builder
    must add `ion_iK_total_mAcm2 = ... : 1`, `ion_iNa_total_mAcm2 = ... : 1`,
    `ion_iCl_total_mAcm2 = ... : 1`, `ion_iCa_total_mAcm2 = ... : 1`
    composition expressions before adding `get_ion_balance_eqs()`.

    State variables exposed:
        K_in, Na_in, Cl_in, Ca_in   — intracellular concentrations (mM)
        K_out, Na_out, Cl_out, Ca_out — extracellular reservoir (mM, fixed)
        vol_L, surf_cm2             — per-cell geometry
        kappa_B_Ca                  — Ca buffering factor
        rt_over_F_mV                — RT/F at simulation temperature

    Computed dynamic Nernst potentials (subseqent layers read these instead
    of fixed `ek`, `eca`, etc.):
        E_K_mV, E_Na_mV, E_Cl_mV, E_Ca_mV
    """
    return """
    # ---- Ion concentration state variables (mM, dimensionless bare numbers) ----
    # NOTE: K_in, Na_in, Cl_in, Ca_in are implicitly declared by the
    # dK_in/dt, dNa_in/dt, dCl_in/dt, dCa_in/dt equations in
    # get_ion_balance_eqs(). Do NOT declare them as parameters here —
    # Brian2 rejects duplicate declarations.

    # ---- Extracellular reservoir (fixed in Layer 1 v1; true parameters) ----
    K_out : 1
    Na_out : 1
    Cl_out : 1
    Ca_out : 1

    # ---- Per-cell geometry (fixed at construction) ----
    vol_L : 1
    surf_cm2 : 1

    # ---- Ca buffering factor (free / total) ----
    kappa_B_Ca : 1

    # ---- RT/F at simulation temperature (mV) ----
    rt_over_F_mV : 1

    # ---- Dynamic Nernst potentials (recomputed each dt) ----
    E_K_mV  = rt_over_F_mV * log(K_out  / K_in)        : 1
    E_Na_mV = rt_over_F_mV * log(Na_out / Na_in)       : 1
    E_Cl_mV = -rt_over_F_mV * log(Cl_out / Cl_in)      : 1
    E_Ca_mV = (rt_over_F_mV / 2) * log(Ca_out / Ca_in) : 1
    """


def get_ion_balance_eqs() -> str:
    """Brian2 equation fragment: d[X]/dt ion mass conservation.

    Use as the LAST piece of a cell's equation string. Requires
    `ion_iK_total_mAcm2`, `ion_iNa_total_mAcm2`, `ion_iCl_total_mAcm2`,
    `ion_iCa_total_mAcm2` to be defined earlier in the equation string
    (typically composed from channel + pump currents by the cell builder).

    Mass conservation:
        d[X]_in/dt = -I_X · surf / (z_X · F · vol)

    With I_X in mA/cm², surf in cm², vol in L:
        I_X [mA/cm²] · surf [cm²] = current [mA] = 1e-3 [C/s]
        / (z · F [C/mol]) · 1000 = concentration rate [mM/s]

    Convention: outward-positive current (NEURON). Outward I_K (positive)
    decreases [K]_in. Outward I_Cl (positive, but Cl is anion so z=-1)
    means Cl leaves the cell; the -1/(-1·F) in the equation gives the
    correct sign (positive I_Cl_total → [Cl]_in decreases).

    Calcium gets divided by (1 + κ_B) for phenomenological buffering.
    """
    return """
    # ---- d[X]_in/dt: ion mass conservation ----
    # Factor 1000 converts mol/(L·s) → mM/s. Outward-positive current
    # convention → negative sign in dX/dt.
    dK_in/dt  = (-1000 * ion_iK_total_mAcm2  * surf_cm2 / ( 1 * 96485 * vol_L)) * Hz : 1
    dNa_in/dt = (-1000 * ion_iNa_total_mAcm2 * surf_cm2 / ( 1 * 96485 * vol_L)) * Hz : 1
    dCl_in/dt = (-1000 * ion_iCl_total_mAcm2 * surf_cm2 / (-1 * 96485 * vol_L)) * Hz : 1
    # Calcium: phenomenological buffering factor (1 + κ_B); divide by 2 (z_Ca = 2)
    dCa_in/dt = (-1000 * ion_iCa_total_mAcm2 * surf_cm2 / ( 2 * 96485 * vol_L * (1 + kappa_B_Ca))) * Hz : 1
    """


def get_ion_dynamics_eqs(ion_composition_eqs: str) -> str:
    """Compose the full Layer 1 ion-dynamics fragment.

    `ion_composition_eqs` must declare four variables:
        ion_iK_total_mAcm2  = <sum of all K  currents>  : 1
        ion_iNa_total_mAcm2 = <sum of all Na currents>  : 1
        ion_iCl_total_mAcm2 = <sum of all Cl currents>  : 1
        ion_iCa_total_mAcm2 = <sum of all Ca currents>  : 1

    In Layer 1 v1, channels + pumps haven't been rewired yet, so the cell
    builder uses simple test/placeholder compositions. Layer 2+ refactors
    channel modules to contribute to these accumulators directly.
    """
    return (
        get_state_and_nernst_eqs()
        + "\n# ---- Ion-current composition (provided by cell builder) ----\n"
        + ion_composition_eqs
        + "\n"
        + get_ion_balance_eqs()
    )


# ---------------------------------------------------------------------------
# Initialization helper
# ---------------------------------------------------------------------------

def apply_ion_state(
    group,
    geometry: CellGeometry,
    intracellular_mM: dict[str, float] | None = None,
    extracellular_mM: dict[str, float] | None = None,
    kappa_B_Ca: float = KAPPA_BUFFER_CA_DEFAULT,
    temperature_K: float = T_C_ELEGANS_K,
) -> None:
    """Initialize Layer 1 state on a Brian2 NeuronGroup.

    Sets the per-cell geometry + ion concentrations + buffering + RT/F.
    The group must have the variables declared by `get_state_and_nernst_eqs()`.

    Args:
        group: Brian2 NeuronGroup (or subgroup) with Layer 1 state variables.
        geometry: CellGeometry for this cell.
        intracellular_mM: per-ion {K, Na, Cl, Ca} starting concentrations;
            falls back on `INTRACELLULAR_DEFAULT_mM`.
        extracellular_mM: per-ion {K, Na, Cl, Ca} reservoir values;
            falls back on `EXTRACELLULAR_DEFAULT_mM`.
        kappa_B_Ca: Ca buffering factor (free / total).
        temperature_K: simulation temperature for RT/F computation.
    """
    icc = {**INTRACELLULAR_DEFAULT_mM, **(intracellular_mM or {})}
    ecc = {**EXTRACELLULAR_DEFAULT_mM, **(extracellular_mM or {})}

    group.K_in  = icc["K"]
    group.Na_in = icc["Na"]
    group.Cl_in = icc["Cl"]
    group.Ca_in = icc["Ca"]

    group.K_out  = ecc["K"]
    group.Na_out = ecc["Na"]
    group.Cl_out = ecc["Cl"]
    group.Ca_out = ecc["Ca"]

    group.vol_L = geometry.volume_L
    group.surf_cm2 = geometry.surface_cm2
    group.kappa_B_Ca = kappa_B_Ca
    group.rt_over_F_mV = R_GAS_J_PER_MOL_K * temperature_K / F_FARADAY_C_PER_MOL * 1000.0


# ---------------------------------------------------------------------------
# Smoke test (run directly to verify foundation works)
# ---------------------------------------------------------------------------

def _smoke_geometry() -> None:
    print("=" * 72)
    print("Geometry check — Nicoletti 2024 capacitances + r_eff sensitivity sweep")
    print("=" * 72)
    print(f"\n{'cell':<6} {'C_m (pF)':>10} {'surf (μm²)':>12}", end="")
    for r in R_EFF_SWEEP_um:
        print(f"  {'vol (fL) @ r=' + str(r):>16}", end="")
    print()
    print("-" * 80)

    for r_eff in R_EFF_SWEEP_um:
        geoms = make_cell_geometries(r_eff)
        # Just record results; print in a transposed format below.
    # Print cell-major
    geoms_05 = make_cell_geometries(R_EFF_DEFAULT_um)
    for name in NICOLETTI_CAPACITANCE_pF:
        g05 = geoms_05[name]
        row = f"{name:<6} {g05.capacitance_pF:>10.2f} {g05.surface_cm2 * 1e8:>12.1f}"
        for r in R_EFF_SWEEP_um:
            g = CellGeometry(
                cell_name=name,
                capacitance_pF=NICOLETTI_CAPACITANCE_pF[name],
                r_eff_um=r,
            )
            row += f"  {g.volume_fL:>16.1f}"
        print(row)


def _smoke_nernst() -> None:
    print("\n" + "=" * 72)
    print("Nernst hand-checks at default concentrations (20°C)")
    print("=" * 72)
    icc = INTRACELLULAR_DEFAULT_mM
    ecc = EXTRACELLULAR_DEFAULT_mM
    print(f"\nIntracellular: K={icc['K']}, Na={icc['Na']}, Cl={icc['Cl']}, "
          f"Ca={icc['Ca']*1e6:.0f} nM (= {icc['Ca']} mM)")
    print(f"Extracellular: K={ecc['K']}, Na={ecc['Na']}, Cl={ecc['Cl']}, Ca={ecc['Ca']}")
    print(f"RT/F at 20°C: {RT_OVER_F_mV:.3f} mV\n")
    print(f"{'ion':<4} {'E (mV)':>10}  hand-calc check")
    for ion in ("K", "Na", "Cl", "Ca"):
        z = ION_CHARGE[ion]
        e = nernst_potential_mV(ecc[ion], icc[ion], z)
        # Hand-calc for sanity
        hand = (RT_OVER_F_mV / z) * math.log(ecc[ion] / icc[ion])
        agree = "✓" if abs(e - hand) < 1e-9 else "✗"
        print(f"{ion:<4} {e:>10.3f}  (z={z:+d}, hand={hand:.3f}, agree={agree})")


def _smoke_ghk() -> None:
    print("\n" + "=" * 72)
    print("GHK resting potential prediction")
    print("=" * 72)
    # Typical neuron permeability ratios: P_K : P_Na : P_Cl ~ 1.0 : 0.04 : 0.45
    # (Hodgkin & Katz 1949 squid axon). Adult mammalian neuron ratio varies.
    v_ghk = ghk_resting_potential_mV(perm_K=1.0, perm_Na=0.04, perm_Cl=0.45)
    print(f"\nP_K:P_Na:P_Cl = 1.00 : 0.04 : 0.45 (HH 1949 squid axon ratio)")
    print(f"V_GHK = {v_ghk:.2f} mV  (expected ~-70 mV for typical neuron)")


def _smoke_brian2_conservation() -> None:
    print("\n" + "=" * 72)
    print("Brian2 ion conservation test — minimal cell, zero current, 1s")
    print("=" * 72)
    try:
        from brian2 import (
            NeuronGroup, StateMonitor, Network, defaultclock, prefs,
            ms, start_scope,
        )
    except ImportError:
        print("Brian2 not importable; skipping integration test.")
        return

    start_scope()
    prefs.codegen.target = "cython"
    defaultclock.dt = 0.025 * ms

    # Test composition: every ion current is a parameter set to 0
    test_composition = """
    ion_iK_total_mAcm2  : 1
    ion_iNa_total_mAcm2 : 1
    ion_iCl_total_mAcm2 : 1
    ion_iCa_total_mAcm2 : 1
    """
    eqs = get_ion_dynamics_eqs(test_composition)

    G = NeuronGroup(1, eqs, method="rk4")
    geom = make_cell_geometries(R_EFF_DEFAULT_um)["AVAL"]
    apply_ion_state(G, geom)
    # Set test currents to zero
    G.ion_iK_total_mAcm2 = 0.0
    G.ion_iNa_total_mAcm2 = 0.0
    G.ion_iCl_total_mAcm2 = 0.0
    G.ion_iCa_total_mAcm2 = 0.0

    mon = StateMonitor(G, ["K_in", "Na_in", "Cl_in", "Ca_in",
                            "E_K_mV", "E_Na_mV", "E_Cl_mV", "E_Ca_mV"],
                       record=True)
    net = Network(G, mon)

    print(f"\nAVAL initial: K_in={G.K_in[0]:.3f}, Na_in={G.Na_in[0]:.3f}, "
          f"Cl_in={G.Cl_in[0]:.3f}, Ca_in={G.Ca_in[0]:.6f}")
    print(f"AVAL geom: vol_L={G.vol_L[0]:.3e}, surf_cm2={G.surf_cm2[0]:.3e}")

    net.run(1000 * ms)

    final = {
        "K_in":  mon.K_in[0][-1],
        "Na_in": mon.Na_in[0][-1],
        "Cl_in": mon.Cl_in[0][-1],
        "Ca_in": mon.Ca_in[0][-1],
    }
    initial = {
        "K_in":  mon.K_in[0][0],
        "Na_in": mon.Na_in[0][0],
        "Cl_in": mon.Cl_in[0][0],
        "Ca_in": mon.Ca_in[0][0],
    }
    print(f"\nAfter 1s with zero currents:")
    pass_count = 0
    for ion in ("K_in", "Na_in", "Cl_in", "Ca_in"):
        delta = final[ion] - initial[ion]
        # Tolerance: 1e-9 (numerical precision floor)
        ok = abs(delta) < 1e-9
        pass_count += ok
        print(f"  Δ{ion:<6} = {delta:+.3e}  {'PASS' if ok else 'FAIL'}")

    # Nernst at rest (final state) hand-check
    print(f"\nDynamic Nernst at final state (should match hand-calc):")
    e_k = mon.E_K_mV[0][-1]
    e_k_hand = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["K"],
                                    INTRACELLULAR_DEFAULT_mM["K"], +1)
    print(f"  E_K_mV (Brian2)  = {e_k:.4f}")
    print(f"  E_K_mV (hand)    = {e_k_hand:.4f}")
    print(f"  Agreement: {'PASS' if abs(e_k - e_k_hand) < 0.01 else 'FAIL'}")

    if pass_count == 4 and abs(e_k - e_k_hand) < 0.01:
        print("\n[Conservation test PASS]")
    else:
        print("\n[Conservation test FAIL — investigate]")


def _smoke_brian2_mass_balance() -> None:
    print("\n" + "=" * 72)
    print("Brian2 mass-balance test — fixed outward K current, AVAL, 1s")
    print("=" * 72)
    try:
        from brian2 import (
            NeuronGroup, StateMonitor, Network, defaultclock, prefs,
            ms, start_scope,
        )
    except ImportError:
        print("Brian2 not importable; skipping.")
        return

    start_scope()
    prefs.codegen.target = "cython"
    defaultclock.dt = 0.025 * ms

    test_composition = """
    ion_iK_total_mAcm2  : 1
    ion_iNa_total_mAcm2 : 1
    ion_iCl_total_mAcm2 : 1
    ion_iCa_total_mAcm2 : 1
    """
    eqs = get_ion_dynamics_eqs(test_composition)
    G = NeuronGroup(1, eqs, method="rk4")
    geom = make_cell_geometries(R_EFF_DEFAULT_um)["AVAL"]
    apply_ion_state(G, geom)

    # Inject fixed outward K current: 1e-5 mA/cm² (small, observable)
    i_K_test = 1.0e-5
    G.ion_iK_total_mAcm2 = i_K_test
    G.ion_iNa_total_mAcm2 = 0.0
    G.ion_iCl_total_mAcm2 = 0.0
    G.ion_iCa_total_mAcm2 = 0.0

    # Predicted rate: dK_in/dt = -1000 * i_K * surf / (z·F·vol)
    z_K = +1
    predicted_rate_mM_per_s = (-1000.0 * i_K_test * geom.surface_cm2
                                / (z_K * F_FARADAY_C_PER_MOL * geom.volume_L))

    mon = StateMonitor(G, ["K_in"], record=True)
    net = Network(G, mon)
    print(f"\nAVAL: surf={geom.surface_cm2:.3e} cm², vol={geom.volume_L:.3e} L")
    print(f"i_K_test = {i_K_test:.3e} mA/cm² (outward)")
    print(f"Predicted dK_in/dt = {predicted_rate_mM_per_s:+.4e} mM/s")
    print(f"Predicted Δ[K]_in over 1s = {predicted_rate_mM_per_s:+.4e} mM\n")

    net.run(1000 * ms)
    delta_obs = mon.K_in[0][-1] - mon.K_in[0][0]
    expected_delta = predicted_rate_mM_per_s  # over 1 second
    rel_err = abs(delta_obs - expected_delta) / abs(expected_delta)
    print(f"Observed Δ[K]_in = {delta_obs:+.4e} mM")
    print(f"Expected Δ[K]_in = {expected_delta:+.4e} mM")
    print(f"Relative error   = {rel_err * 100:.4f}%")
    print(f"Result: {'PASS' if rel_err < 0.01 else 'FAIL'} (tolerance 1%)")


def main() -> None:
    print("\n" + "#" * 72)
    print("# Layer 1 §7.1 — ion_dynamics.py foundation smoke test")
    print("# " + " " * 70)
    print("# Per docs/layer1_design_decisions.md v2 (authorized 2026-05-12)")
    print("#" * 72)

    _smoke_geometry()
    _smoke_nernst()
    _smoke_ghk()
    _smoke_brian2_conservation()
    _smoke_brian2_mass_balance()

    print("\n" + "#" * 72)
    print("# Smoke test complete.")
    print("#" * 72 + "\n")


if __name__ == "__main__":
    main()
