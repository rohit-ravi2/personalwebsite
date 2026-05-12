"""
Layer 1 §7.2 v2 — Pump module validation + AVAL anchor calibration.

Per Rohit's 2026-05-12 Path (b) authorization:
- Na/K-ATPase: Hill-form (unchanged); leak ratio fixed g_Na_leak ≈ 0.5 · g_K_leak
- KCC-2: Payne 1997 thermodynamic form (built-in equilibrium)
- ABTS-1: approximate first-order toward Cl target (v1 limitation; v2 adds HCO₃/pH)
- Ca clearance: threshold-MM form (turns off below [Ca]_target)

Validation criteria per user:
- AVAL achieves steady-state with [K]_in stable within ±2% over 5s
- AVAL steady-state [Cl]_in in physiologically reasonable range (3-7 mM);
  ACTUAL value documented, not enforced
- AVAL steady-state [Ca]_in near [Ca]_target
- Cross-cell scaling produces stable rest for AVAR (identical TPM → should match)
- RIM, AIY rest stability are testable cross-cell predictions
- Findings reported, NOT autonomously retuned
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import math
import numpy as np

from ion_dynamics import (
    F_FARADAY_C_PER_MOL, RT_OVER_F_mV, T_C_ELEGANS_K,
    NICOLETTI_CAPACITANCE_pF, SPECIFIC_CM_uFcm2,
    INTRACELLULAR_DEFAULT_mM, EXTRACELLULAR_DEFAULT_mM, KAPPA_BUFFER_CA_DEFAULT,
    make_cell_geometries, R_EFF_DEFAULT_um,
    get_state_and_nernst_eqs, get_ion_balance_eqs,
    apply_ion_state, nernst_potential_mV,
)
from pumps.na_k_atpase import (
    NA_K_ATPASE_EQS, apply_na_k_atpase_params, scale_I_max_by_eat6_tpm,
    K_NA_HALF_mM_DEFAULT, K_K_HALF_mM_DEFAULT, K_ATP_HALF_mM_DEFAULT,
    ATP_BASELINE_mM_DEFAULT, N_HILL_NA_DEFAULT, N_HILL_K_DEFAULT, N_HILL_ATP_DEFAULT,
)
from pumps.ca_clearance import (
    LUMPED_CA_CLEARANCE_EQS, apply_ca_clearance_params,
    scale_I_max_by_mca3_tpm,
    K_CA_HALF_mM_DEFAULT, CA_TARGET_mM_DEFAULT,
)
from pumps.kcc2_abts1_lumped import (
    KCC2_EQS, ABTS1_EQS,
    apply_kcc2_params, apply_abts1_params,
    scale_I_max_by_kcc2_tpm, scale_I_max_by_abts1_tpm,
    CL_TARGET_ABTS_mM_DEFAULT,
)


# =========================================================================
# Placeholder ionic leak conductances — corrected for pump-stoichiometry self-consistency
# =========================================================================

# Per Path (b) Issue 1 analytic derivation:
#   At V_rest such that pump stoichiometry balances both K (2 K in / cycle)
#   and Na (3 Na out / cycle), we need g_Na_leak ≈ 0.5 × g_K_leak.
#   This emerges from:
#       (3/2) · g_K · (V - E_K) = g_Na · (E_Na - V)  at the same V_rest
#   With mammalian defaults and V_rest ≈ -50 mV: g_Na ≈ 0.512 · g_K.
DEFAULT_LEAKS = {
    "g_K_leak_Scm2":  5.0e-6,    # main pump-balancing K efflux
    "g_Na_leak_Scm2": 2.5e-6,    # 0.5 × g_K (corrected from v1's 0.1×)
    "g_Cl_leak_Scm2": 5.0e-7,    # balanced by KCC-2 + ABTS-1
    "g_Ca_leak_Scm2": 1.0e-9,    # very small; threshold pump catches at ~target+small
}

V_INIT_mV = -50.0


# =========================================================================
# Hill kinetics in isolation
# =========================================================================

def hill_f(x: float, K: float, n: float) -> float:
    r = (x / K) ** n
    return r / (1 + r)


def predicted_pump_NaK_current_mAcm2(
    I_max_mAcm2: float,
    Na_in_mM: float = INTRACELLULAR_DEFAULT_mM["Na"],
    K_out_mM: float = EXTRACELLULAR_DEFAULT_mM["K"],
    ATP_mM: float = ATP_BASELINE_mM_DEFAULT,
) -> float:
    return I_max_mAcm2 * (
        hill_f(Na_in_mM, K_NA_HALF_mM_DEFAULT, N_HILL_NA_DEFAULT)
        * hill_f(K_out_mM, K_K_HALF_mM_DEFAULT, N_HILL_K_DEFAULT)
        * hill_f(ATP_mM, K_ATP_HALF_mM_DEFAULT, N_HILL_ATP_DEFAULT)
    )


def test_pump_kinetics_isolation() -> None:
    print("=" * 72)
    print("Test 1 — Pump kinetics in isolation (analytic)")
    print("=" * 72)

    print("\n1a. Na/K-ATPase Hill saturation product at default substrates:")
    f_combined = (
        hill_f(INTRACELLULAR_DEFAULT_mM["Na"], K_NA_HALF_mM_DEFAULT, N_HILL_NA_DEFAULT)
        * hill_f(EXTRACELLULAR_DEFAULT_mM["K"], K_K_HALF_mM_DEFAULT, N_HILL_K_DEFAULT)
        * hill_f(ATP_BASELINE_mM_DEFAULT, K_ATP_HALF_mM_DEFAULT, N_HILL_ATP_DEFAULT)
    )
    print(f"   f_Na · f_K · f_ATP = {f_combined:.4f} (at Na=10, K_out=4, ATP=3 mM)")

    print("\n1b. KCC-2 thermodynamic driving force (bounded form ∈ [-1, +1]):")
    P_in = INTRACELLULAR_DEFAULT_mM["K"] * INTRACELLULAR_DEFAULT_mM["Cl"]
    P_out = EXTRACELLULAR_DEFAULT_mM["K"] * EXTRACELLULAR_DEFAULT_mM["Cl"]
    drive = (P_in - P_out) / (P_in + P_out)
    print(f"   [K]_in·[Cl]_in = {P_in:.1f}, [K]_out·[Cl]_out = {P_out:.1f}")
    print(f"   drive = ({P_in:.1f} - {P_out:.1f}) / ({P_in:.1f} + {P_out:.1f}) = {drive:+.4f}")
    print(f"   At equilibrium ([Cl]_in = {P_out / INTRACELLULAR_DEFAULT_mM['K']:.2f} mM)")
    print(f"   drive = 0 → KCC-2 turns off; reverses if [Cl]_in drops below.")

    print("\n1c. ABTS-1 approximate driving force (at default [Cl]_in = 5 mM):")
    delta_Cl = INTRACELLULAR_DEFAULT_mM["Cl"] - CL_TARGET_ABTS_mM_DEFAULT
    v_norm = delta_Cl / CL_TARGET_ABTS_mM_DEFAULT
    print(f"   ([Cl]_in - target)/target = ({INTRACELLULAR_DEFAULT_mM['Cl']} - {CL_TARGET_ABTS_mM_DEFAULT})/{CL_TARGET_ABTS_mM_DEFAULT} = {v_norm:+.4f}")
    print(f"   At rest, ABTS-1 is at its target → v_ABTS = 0.")

    print("\n1d. Ca clearance threshold (at default [Ca]_in = 50 nM):")
    delta_Ca = INTRACELLULAR_DEFAULT_mM["Ca"] - CA_TARGET_mM_DEFAULT
    print(f"   [Ca]_in - target = {INTRACELLULAR_DEFAULT_mM['Ca']:.2e} - {CA_TARGET_mM_DEFAULT:.2e} = {delta_Ca:+.2e}")
    print(f"   delta ≤ 0 → pump OFF at rest. Activates above target.")


# =========================================================================
# Calibration cell builder
# =========================================================================

def _composition_eqs() -> str:
    """Composition: ionic leaks + pump contributions per ion + membrane V."""
    return """
    # ---- Phenomenological ionic leaks (placeholder for §7.3 channels) ----
    g_K_leak_Scm2  : 1
    g_Na_leak_Scm2 : 1
    g_Cl_leak_Scm2 : 1
    g_Ca_leak_Scm2 : 1

    # ---- Brian2 voltage-to-mV bridge ----
    v_mV = v / mV : 1
    cm_uFcm2 : 1

    # ---- Ionic leak currents (mA/cm², outward-positive) ----
    iK_leak_mAcm2  = g_K_leak_Scm2  * (v_mV - E_K_mV)  : 1
    iNa_leak_mAcm2 = g_Na_leak_Scm2 * (v_mV - E_Na_mV) : 1
    iCl_leak_mAcm2 = g_Cl_leak_Scm2 * (v_mV - E_Cl_mV) : 1
    iCa_leak_mAcm2 = g_Ca_leak_Scm2 * (v_mV - E_Ca_mV) : 1

    # ---- Total per-ion currents = ionic leaks + pump contributions ----
    # KCC-2 contributes to K + Cl; ABTS-1 contributes to Cl only (v1 approximation)
    ion_iK_total_mAcm2  = iK_leak_mAcm2  + pump_NaK_iK_mAcm2  + kcc2_iK_mAcm2 : 1
    ion_iNa_total_mAcm2 = iNa_leak_mAcm2 + pump_NaK_iNa_mAcm2 : 1
    ion_iCl_total_mAcm2 = iCl_leak_mAcm2 + kcc2_iCl_mAcm2 + abts1_iCl_mAcm2 : 1
    ion_iCa_total_mAcm2 = iCa_leak_mAcm2 + ca_clear_iCa_mAcm2 : 1

    # ---- Total membrane current density ----
    # Ionic leaks + electrogenic pumps (Na/K + Ca-clear). KCC-2 + ABTS-1 are
    # electroneutral lumped abstractions and don't contribute to membrane V.
    i_total_mAcm2 = (iK_leak_mAcm2 + iNa_leak_mAcm2 + iCl_leak_mAcm2 + iCa_leak_mAcm2
                     + pump_NaK_I_mAcm2 + ca_clear_I_mAcm2) : 1

    # ---- Membrane voltage equation ----
    I_total = i_total_mAcm2 * surf_cm2 * 1e9 * pA - I_inj : amp
    dv/dt = -I_total / (cm_uFcm2 * surf_cm2 * 1e6 * pF) : volt
    I_inj : amp
    """


def build_calibration_cell(
    cell_name: str,
    I_NaK_max: float,
    I_Ca_clear_max: float,
    I_kcc2_max: float,
    I_abts1_max: float,
    leaks: dict | None = None,
    v_init_mV: float = V_INIT_mV,
    r_eff_um: float = R_EFF_DEFAULT_um,
):
    from brian2 import (
        NeuronGroup, StateMonitor, Network, defaultclock, prefs,
        start_scope, ms, mV, pA, pF,
    )
    start_scope()
    prefs.codegen.target = "cython"
    defaultclock.dt = 0.025 * ms

    leaks = leaks or DEFAULT_LEAKS

    eqs = (
        get_state_and_nernst_eqs()
        + NA_K_ATPASE_EQS
        + LUMPED_CA_CLEARANCE_EQS
        + KCC2_EQS
        + ABTS1_EQS
        + _composition_eqs()
        + get_ion_balance_eqs()
    )

    G = NeuronGroup(1, eqs, method="rk4")

    geom = make_cell_geometries(r_eff_um)[cell_name]
    apply_ion_state(G, geom)

    apply_na_k_atpase_params(G, I_max_mAcm2=I_NaK_max)
    apply_ca_clearance_params(G, I_max_mAcm2=I_Ca_clear_max)
    apply_kcc2_params(G, I_max_mAcm2=I_kcc2_max)
    apply_abts1_params(G, I_max_mAcm2=I_abts1_max)

    G.g_K_leak_Scm2 = leaks["g_K_leak_Scm2"]
    G.g_Na_leak_Scm2 = leaks["g_Na_leak_Scm2"]
    G.g_Cl_leak_Scm2 = leaks["g_Cl_leak_Scm2"]
    G.g_Ca_leak_Scm2 = leaks["g_Ca_leak_Scm2"]
    G.cm_uFcm2 = SPECIFIC_CM_uFcm2
    G.I_inj = 0 * pA
    G.v = v_init_mV * mV

    mon = StateMonitor(G, [
        "v", "K_in", "Na_in", "Cl_in", "Ca_in",
        "E_K_mV", "E_Na_mV", "E_Cl_mV", "E_Ca_mV",
        "pump_NaK_I_mAcm2", "ca_clear_I_mAcm2",
        "kcc2_v_mAcm2", "kcc2_drive", "abts1_v_mAcm2",
        "iK_leak_mAcm2", "iNa_leak_mAcm2", "iCl_leak_mAcm2", "iCa_leak_mAcm2",
        "i_total_mAcm2",
    ], record=True)
    net = Network(G, mon)

    return {"group": G, "monitor": mon, "network": net, "geometry": geom}


# =========================================================================
# AVAL anchor calibration — analytic guesses + 5s simulation
# =========================================================================

def analytic_anchor_guesses(v_mV: float = V_INIT_mV, leaks: dict | None = None) -> dict:
    """Analytic estimates for the four AVAL pump I_max values."""
    leaks = leaks or DEFAULT_LEAKS

    # Nernst at default concentrations
    E_K = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["K"], INTRACELLULAR_DEFAULT_mM["K"], +1)
    E_Na = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["Na"], INTRACELLULAR_DEFAULT_mM["Na"], +1)
    E_Cl = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["Cl"], INTRACELLULAR_DEFAULT_mM["Cl"], -1)
    E_Ca = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["Ca"], INTRACELLULAR_DEFAULT_mM["Ca"], +2)

    # ---- I_NaK_max ----
    # K balance: I_K_leak = -2 × I_NaK
    I_K_leak_mAcm2 = leaks["g_K_leak_Scm2"] * (v_mV - E_K)
    f_Na = hill_f(INTRACELLULAR_DEFAULT_mM["Na"], K_NA_HALF_mM_DEFAULT, N_HILL_NA_DEFAULT)
    f_K = hill_f(EXTRACELLULAR_DEFAULT_mM["K"], K_K_HALF_mM_DEFAULT, N_HILL_K_DEFAULT)
    f_ATP = hill_f(ATP_BASELINE_mM_DEFAULT, K_ATP_HALF_mM_DEFAULT, N_HILL_ATP_DEFAULT)
    I_NaK_max = I_K_leak_mAcm2 / (2.0 * f_Na * f_K * f_ATP)

    # ---- I_kcc2_max ----
    # At rest [Cl]_in = 5 mM (initial; steady-state will emerge), drive ≈ 0.228
    P_in = INTRACELLULAR_DEFAULT_mM["K"] * INTRACELLULAR_DEFAULT_mM["Cl"]
    P_out = EXTRACELLULAR_DEFAULT_mM["K"] * EXTRACELLULAR_DEFAULT_mM["Cl"]
    kcc2_drive_at_rest = (P_in - P_out) / (P_in + P_out)
    # Cl leak influx at V_rest (Cl moves IN because V > E_Cl)
    I_Cl_leak_mAcm2 = leaks["g_Cl_leak_Scm2"] * (v_mV - E_Cl)
    # KCC-2 alone balances this (ABTS-1 = 0 at target)
    I_kcc2_max = I_Cl_leak_mAcm2 / max(kcc2_drive_at_rest, 1e-6)

    # ---- I_abts1_max ----
    # ABTS-1 = 0 at rest (Cl_in = target); choose same order of magnitude as KCC-2
    # so it contributes meaningfully to perturbation recovery
    I_abts1_max = I_kcc2_max

    # ---- I_Ca_clear_max ----
    # Threshold form: at small delta = ε above target, v_Ca ≈ I_max × ε / K_half
    # Want balance with leak inward. Pick I_max so steady-state delta is ~K_half/10 (= 50 nM).
    I_Ca_leak_mAcm2 = leaks["g_Ca_leak_Scm2"] * (v_mV - E_Ca)  # negative (inward)
    # At delta = K_half/10, v_Ca = I_max × (1/10)/(1 + 1/10) = I_max × 0.091
    # Balance: I_max × 0.091 = |I_Ca_leak|
    I_Ca_clear_max = abs(I_Ca_leak_mAcm2) / 0.091

    return {
        "I_NaK_max": I_NaK_max,
        "I_kcc2_max": I_kcc2_max,
        "I_abts1_max": I_abts1_max,
        "I_Ca_clear_max": I_Ca_clear_max,
        "diagnostics": {
            "E_K": E_K, "E_Na": E_Na, "E_Cl": E_Cl, "E_Ca": E_Ca,
            "I_K_leak": I_K_leak_mAcm2, "I_Cl_leak": I_Cl_leak_mAcm2,
            "I_Ca_leak": I_Ca_leak_mAcm2,
            "f_combined": f_Na * f_K * f_ATP,
            "kcc2_drive_at_rest": kcc2_drive_at_rest,
        },
    }


def run_cell_and_report(cell_name: str, params: dict, sim_ms: float = 5000.0,
                        verbose: bool = True) -> dict:
    """Build cell, run for sim_ms, return rest snapshot + deltas."""
    from brian2 import ms
    bundle = build_calibration_cell(
        cell_name=cell_name,
        I_NaK_max=params["I_NaK_max"],
        I_Ca_clear_max=params["I_Ca_clear_max"],
        I_kcc2_max=params["I_kcc2_max"],
        I_abts1_max=params["I_abts1_max"],
    )
    bundle["network"].run(sim_ms * ms)
    mon = bundle["monitor"]

    initial = {ion: float(mon.__getattr__(ion)[0][0]) for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}
    final = {ion: float(mon.__getattr__(ion)[0][-1]) for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}
    V_final_mV = float(mon.v[0][-1] / 1e-3)
    deltas_pct = {ion: 100 * (final[ion] / initial[ion] - 1) for ion in initial}

    diagnostics = {
        "kcc2_drive_final": float(mon.kcc2_drive[0][-1]),
        "kcc2_v_final": float(mon.kcc2_v_mAcm2[0][-1]),
        "abts1_v_final": float(mon.abts1_v_mAcm2[0][-1]),
        "pump_NaK_I_final": float(mon.pump_NaK_I_mAcm2[0][-1]),
        "ca_clear_I_final": float(mon.ca_clear_I_mAcm2[0][-1]),
        "E_K_final": float(mon.E_K_mV[0][-1]),
        "E_Na_final": float(mon.E_Na_mV[0][-1]),
        "E_Cl_final": float(mon.E_Cl_mV[0][-1]),
        "E_Ca_final": float(mon.E_Ca_mV[0][-1]),
    }

    if verbose:
        print(f"\n--- {cell_name} ---")
        print(f"  V_rest = {V_final_mV:.2f} mV")
        print(f"  [K]_in:  {final['K_in']:.4f} mM   (Δ {deltas_pct['K_in']:+.2f}%)")
        print(f"  [Na]_in: {final['Na_in']:.4f} mM   (Δ {deltas_pct['Na_in']:+.2f}%)")
        print(f"  [Cl]_in: {final['Cl_in']:.4f} mM   (Δ {deltas_pct['Cl_in']:+.2f}%)")
        print(f"  [Ca]_in: {final['Ca_in']*1e6:.2f} nM   (Δ {deltas_pct['Ca_in']:+.2f}% from 50 nM target)")
        print(f"  Dynamic Nernst: E_K={diagnostics['E_K_final']:.2f}, E_Na={diagnostics['E_Na_final']:.2f}, "
              f"E_Cl={diagnostics['E_Cl_final']:.2f}, E_Ca={diagnostics['E_Ca_final']:.2f}")
        print(f"  Pumps (final): I_NaK={diagnostics['pump_NaK_I_final']:.3e}, "
              f"v_KCC2={diagnostics['kcc2_v_final']:.3e} (drive={diagnostics['kcc2_drive_final']:+.3f}), "
              f"v_ABTS={diagnostics['abts1_v_final']:.3e}, "
              f"I_Ca_clear={diagnostics['ca_clear_I_final']:.3e}")

    return {
        "cell_name": cell_name,
        "V_rest_mV": V_final_mV,
        "initial": initial,
        "final": final,
        "deltas_pct": deltas_pct,
        "diagnostics": diagnostics,
        "params": params,
    }


def calibrate_aval(verbose: bool = True) -> dict:
    print("\n" + "=" * 72)
    print("Test 2 — AVAL anchor calibration (Path-b corrected forms)")
    print("=" * 72)

    guesses = analytic_anchor_guesses()
    if verbose:
        print(f"\nLeak conductances (corrected per Issue 1):")
        for k, v in DEFAULT_LEAKS.items():
            print(f"  {k:<20} = {v:.2e} S/cm²")
        ratio = DEFAULT_LEAKS["g_Na_leak_Scm2"] / DEFAULT_LEAKS["g_K_leak_Scm2"]
        print(f"  g_Na/g_K ratio = {ratio:.3f} (target ~0.512 for 3:2 stoichiometry at V_rest ≈ -50)")

        print(f"\nAnalytic anchor guesses (mA/cm²):")
        print(f"  I_NaK_max       = {guesses['I_NaK_max']:.4e}")
        print(f"  I_kcc2_max      = {guesses['I_kcc2_max']:.4e}")
        print(f"  I_abts1_max     = {guesses['I_abts1_max']:.4e} (= I_kcc2_max by heuristic)")
        print(f"  I_Ca_clear_max  = {guesses['I_Ca_clear_max']:.4e}")

        d = guesses["diagnostics"]
        print(f"\nDiagnostics:")
        print(f"  E_K  = {d['E_K']:+.2f}   E_Na = {d['E_Na']:+.2f}   E_Cl = {d['E_Cl']:+.2f}   E_Ca = {d['E_Ca']:+.2f}")
        print(f"  I_K_leak  = {d['I_K_leak']:+.3e}   I_Cl_leak = {d['I_Cl_leak']:+.3e}   I_Ca_leak = {d['I_Ca_leak']:+.3e}")
        print(f"  Na/K Hill saturation product f_Na·f_K·f_ATP = {d['f_combined']:.4f}")
        print(f"  KCC-2 drive at rest (Cl_in=5) = {d['kcc2_drive_at_rest']:+.4f}")

    aval_params = {
        "I_NaK_max": guesses["I_NaK_max"],
        "I_kcc2_max": guesses["I_kcc2_max"],
        "I_abts1_max": guesses["I_abts1_max"],
        "I_Ca_clear_max": guesses["I_Ca_clear_max"],
    }
    result = run_cell_and_report("AVAL", aval_params, verbose=verbose)
    return result


def cross_cell_validation(aval_result: dict, verbose: bool = True) -> dict:
    print("\n" + "=" * 72)
    print("Test 3 — Cross-cell validation (per-cell TPM scaling)")
    print("=" * 72)

    aval_params = aval_result["params"]
    results = {}
    for cell in ("AVAR", "AIY", "RIM"):
        scaled = {
            "I_NaK_max": scale_I_max_by_eat6_tpm(aval_params["I_NaK_max"], cell),
            "I_Ca_clear_max": scale_I_max_by_mca3_tpm(aval_params["I_Ca_clear_max"], cell),
            "I_kcc2_max": scale_I_max_by_kcc2_tpm(aval_params["I_kcc2_max"], cell),
            "I_abts1_max": scale_I_max_by_abts1_tpm(aval_params["I_abts1_max"], cell),
        }
        if verbose:
            print(f"\n--- {cell} TPM-scaled params ---")
            print(f"  I_NaK_max:      {scaled['I_NaK_max']:.4e}   "
                  f"(× {scaled['I_NaK_max']/aval_params['I_NaK_max']:.3f} via eat-6 TPM)")
            print(f"  I_Ca_clear_max: {scaled['I_Ca_clear_max']:.4e}   "
                  f"(× {scaled['I_Ca_clear_max']/aval_params['I_Ca_clear_max']:.3f} via mca-3 TPM)")
            print(f"  I_kcc2_max:     {scaled['I_kcc2_max']:.4e}   "
                  f"(× {scaled['I_kcc2_max']/aval_params['I_kcc2_max']:.3f} via kcc-2 TPM)")
            print(f"  I_abts1_max:    {scaled['I_abts1_max']:.4e}   "
                  f"(× {scaled['I_abts1_max']/aval_params['I_abts1_max']:.3f} via abts-1 TPM)")
        results[cell] = run_cell_and_report(cell, scaled, verbose=verbose)
    return results


# =========================================================================
# Summary
# =========================================================================

def summary_table(aval_result: dict, cross_cell: dict) -> None:
    print("\n" + "=" * 72)
    print("Summary — Layer 1 §7.2 v2 validation")
    print("=" * 72)

    all_results = {"AVAL": aval_result, **cross_cell}

    print(f"\n{'cell':<6} {'V_rest':>9} {'[K]_in':>10} {'[Na]_in':>10} {'[Cl]_in':>10} {'[Ca]_in':>14}")
    print(f"{'':6} {'mV':>9} {'mM (Δ%)':>10} {'mM (Δ%)':>10} {'mM (Δ%)':>10} {'nM (Δ%)':>14}")
    print("-" * 72)
    for cell, r in all_results.items():
        d = r["deltas_pct"]
        f = r["final"]
        print(f"{cell:<6} {r['V_rest_mV']:>+8.2f}  "
              f"{f['K_in']:>6.2f} ({d['K_in']:+5.2f}%)  "
              f"{f['Na_in']:>6.2f} ({d['Na_in']:+5.2f}%)  "
              f"{f['Cl_in']:>6.2f} ({d['Cl_in']:+5.2f}%)  "
              f"{f['Ca_in']*1e6:>6.2f} ({d['Ca_in']:+5.2f}%)")

    print(f"\nValidation criteria (per Path (b) authorization):")
    aval_passes = {
        "K_in_2pct": abs(aval_result["deltas_pct"]["K_in"]) < 2.0,
        "Cl_in_range_3_7": 3.0 <= aval_result["final"]["Cl_in"] <= 7.0,
        "Ca_in_near_target": aval_result["final"]["Ca_in"] < 5.0e-4,  # < 500 nM = 10× target
    }
    print(f"  AVAL [K]_in within ±2%:                {'PASS' if aval_passes['K_in_2pct'] else 'FAIL'} ({aval_result['deltas_pct']['K_in']:+.3f}%)")
    print(f"  AVAL [Cl]_in in [3, 7] mM (emergent):  {'PASS' if aval_passes['Cl_in_range_3_7'] else 'FAIL'} ({aval_result['final']['Cl_in']:.3f} mM)")
    print(f"  AVAL [Ca]_in near 50 nM target:        {'PASS' if aval_passes['Ca_in_near_target'] else 'FAIL'} ({aval_result['final']['Ca_in']*1e6:.1f} nM)")

    print(f"\nCross-cell findings:")
    for cell, r in cross_cell.items():
        d = r["deltas_pct"]
        k_pass = abs(d["K_in"]) < 5.0
        cl_pass = 2.0 <= r["final"]["Cl_in"] <= 7.0
        ca_pass = r["final"]["Ca_in"] < 5.0e-4
        marker = "stable" if (k_pass and cl_pass and ca_pass) else "FINDING"
        print(f"  {cell:<6} K_drift={d['K_in']:+6.2f}%  [Cl]_ss={r['final']['Cl_in']:.2f} mM  "
              f"[Ca]_ss={r['final']['Ca_in']*1e6:.1f} nM  → {marker}")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print("\n" + "#" * 72)
    print("# Layer 1 §7.2 v2 — pump module validation (Path (b) thermodynamic forms)")
    print("# Per docs/layer1_design_decisions.md + Rohit's 2026-05-12 authorization")
    print("#" * 72)

    test_pump_kinetics_isolation()
    aval = calibrate_aval(verbose=True)
    cross = cross_cell_validation(aval, verbose=True)
    summary_table(aval, cross)

    print("\n" + "#" * 72)
    print("# Validation complete. Steady-state values documented; findings preserved.")
    print("#" * 72 + "\n")


if __name__ == "__main__":
    main()
