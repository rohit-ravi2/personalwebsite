"""
Layer 1 §7.3 — Per-cell integration: ion_dynamics + pumps + Nicoletti channels.

Composes the four production-grade Wave 2 cells (AVAL, AVAR, RIM, AIY) with:
- Layer 1 §7.1 ion dynamics (per-cell [K, Na, Cl, Ca]_in + dynamic Nernst)
- Layer 1 §7.2 v2 pumps (Hill Na/K + Payne KCC-2 + approximate ABTS-1 +
  threshold-MM Ca clearance)
- Nicoletti channel sets (per cell; bridged to dynamic Nernst via subexpression)
- Per-cell LEAK SPLIT into K + Na components by GHK-derived permeability
  fractions (since Nicoletti's e_leak emerges from K-Na permeability ratio)

Per Rohit's 2026-05-12 §7.3 authorization. Validation criteria:
- ±2% ion stability over 5s
- V_rest in published range per cell
- [Cl]_in in [3, 7] mM physiological range

LEAK split rationale: Nicoletti's non-ionic LEAK with e_leak = -39 mV (AVAL)
emerges from a specific K+Na permeability ratio. We solve for the ratio that
gives the right V_GHK at default concentrations:
    e_leak = f_K · E_K + f_Na · E_Na   (when only K + Na leaks balance)
yielding f_K = (e_leak - E_Na) / (E_K - E_Na) and f_Na = 1 - f_K. The leak
current is then split: iK_leak = f_K · g_leak · (V - E_K), and similarly Na.
This recovers Nicoletti's V-dynamics behavior at rest while assigning the
leak to specific ions for substrate consistency.

NCA channel handling: NCA's NMODL declares `i = gbar · (v - e)` with e = 30 mV
(non-specific). Biologically NCA is primarily Na-permeable (NCA-1/NCA-2 are
sodium leak channels per Phase G arch doc). Layer 1 v1: treat ik_nca_mAcm2 as
Na current contributing to ion_iNa_total. Documented simplification.
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from ion_dynamics import (
    INTRACELLULAR_DEFAULT_mM, EXTRACELLULAR_DEFAULT_mM,
    NICOLETTI_CAPACITANCE_pF, SPECIFIC_CM_uFcm2,
    R_EFF_DEFAULT_um, make_cell_geometries,
    get_state_and_nernst_eqs, get_ion_balance_eqs,
    apply_ion_state, nernst_potential_mV,
)
from pumps.na_k_atpase import (
    NA_K_ATPASE_EQS, apply_na_k_atpase_params, scale_I_max_by_eat6_tpm,
)
from pumps.ca_clearance import (
    LUMPED_CA_CLEARANCE_EQS, apply_ca_clearance_params, scale_I_max_by_mca3_tpm,
)
from pumps.kcc2_abts1_lumped import (
    KCC2_EQS, ABTS1_EQS,
    apply_kcc2_params, apply_abts1_params,
    scale_I_max_by_kcc2_tpm, scale_I_max_by_abts1_tpm,
)

# Channel modules
from channels import egl19 as egl19_mod
from channels import irk as irk_mod
from channels import nca as nca_mod
from channels import unc103 as unc103_mod
from channels import shl1 as shl1_mod
from channels import cca1 as cca1_mod
from channels import unc2 as unc2_mod
from channels import egl2 as egl2_mod
from channels import kqt1 as kqt1_mod


# =========================================================================
# AVAL §7.2 anchor pump values (from validate_layer1_pumps.py v2 output)
# =========================================================================

PUMP_ANCHOR_AVAL = {
    "I_NaK_max": 2.3461e-4,
    "I_kcc2_max": 6.1562e-5,
    "I_abts1_max": 6.1562e-5,
    "I_Ca_clear_max": 2.0202e-6,
}


# =========================================================================
# Helpers
# =========================================================================

def strip_param_decl(eqs: str, *param_names: str) -> str:
    """Strip `<name> : 1` parameter declarations from a Brian2 eqs string.

    Used to bridge channel reversal potentials (egl19_eca, irk_ek, etc.) to
    dynamic Nernst via subexpressions declared in the cell builder.
    """
    result = eqs
    for name in param_names:
        # Match `name : 1` line (allow whitespace + line ending)
        pattern = rf"^\s*{re.escape(name)}\s*:\s*1\s*$"
        result = re.sub(pattern, "", result, flags=re.MULTILINE)
    return result


def ghk_leak_split(e_leak_mV: float, E_K_mV: float, E_Na_mV: float) -> tuple[float, float]:
    """Solve for K + Na permeability fractions that produce the given e_leak.

    Given a non-ionic leak with reversal e_leak, decompose into K and Na
    components such that f_K · E_K + f_Na · E_Na = e_leak at the same Nernst
    values. This is the leak-only limit of GHK at the rest state.

    Returns (f_K, f_Na) with f_K + f_Na = 1.
    """
    if not (E_K_mV < e_leak_mV < E_Na_mV) and not (E_Na_mV < e_leak_mV < E_K_mV):
        # Bracketing condition fails — clamp gracefully
        if abs(e_leak_mV - E_K_mV) < abs(e_leak_mV - E_Na_mV):
            return 1.0, 0.0
        else:
            return 0.0, 1.0
    f_K = (e_leak_mV - E_Na_mV) / (E_K_mV - E_Na_mV)
    f_K = max(0.0, min(1.0, f_K))
    return f_K, 1.0 - f_K


# =========================================================================
# Cell specs (Nicoletti-published parameters)
# =========================================================================

@dataclass
class CellSpec:
    name: str
    e_leak_mV: float
    g_leak_Scm2: float
    cm_uFcm2: float
    surf_cm2: float
    channels: dict[str, float]   # {channel_module_name: g_Scm2}
    v_init_mV: float
    pump_cell_name: str          # CeNGEN/pump key (AVAL, AVAR, AIY, RIM)
    rest_published_mV: tuple[float, float]  # (min, max) acceptable V_rest range


AVAL_SPEC = CellSpec(
    name="AVAL",
    e_leak_mV=-39.0,
    g_leak_Scm2=1.336e-5,
    cm_uFcm2=0.859551,
    surf_cm2=1123.84e-8,
    channels={"egl19": 9.288e-6, "irk": 8.898e-6, "nca": 0.0},
    v_init_mV=-39.0,
    pump_cell_name="AVAL",
    rest_published_mV=(-50.0, -30.0),   # Mellem 2008 "AVA rest typically -20 to -30"; Nicoletti -39
)


AVAR_SPEC = CellSpec(
    name="AVAR",
    e_leak_mV=-37.0,
    g_leak_Scm2=2.008e-5,
    cm_uFcm2=0.751761,
    surf_cm2=1121.79e-8,
    channels={"egl19": 5.735e-6, "irk": 3.751e-6, "nca": 4.398e-6, "unc103": 4.294e-6},
    v_init_mV=-37.0,
    pump_cell_name="AVAR",
    rest_published_mV=(-35.0, -15.0),   # Stage IV measured rest -24 mV
)


RIM_SPEC = CellSpec(
    name="RIM",
    e_leak_mV=-50.0,
    g_leak_Scm2=9.676795e-5,
    cm_uFcm2=1.5,
    surf_cm2=103.34e-8,
    channels={
        "shl1": 9.048750e-4,
        "egl2": 1.411644e-4,
        "irk": 3.272855e-4,
        "cca1": 8.451920e-4,
        "unc2": 9.676795e-5,
        "egl19": 3.200582e-4,
    },
    v_init_mV=-50.0,
    pump_cell_name="RIM",
    rest_published_mV=(-65.0, -40.0),   # Nicoletti RIM range
)


# AIY — simplified channel set (SLO1 family deferred to v2)
AIY_SPEC = CellSpec(
    name="AIY",
    e_leak_mV=-89.57,
    g_leak_Scm2=0.14e-9 / 65.89e-8,
    cm_uFcm2=1.6,
    surf_cm2=65.89e-8,
    channels={
        "egl19": 0.10e-9 / 65.89e-8,
        "kqt1":  0.20e-9 / 65.89e-8,
        "shl1":  0.50e-9 / 65.89e-8,
        "nca":   0.06e-9 / 65.89e-8,
        # slo1iso + slo1egl19 omitted in v1 (require coupled Ca-K state + complex param set)
        # Layer 1 v1 AIY tests substrate integration with simplified channels
    },
    v_init_mV=-89.57,
    pump_cell_name="AIY",
    rest_published_mV=(-95.0, -55.0),   # AIY published rest range (broad)
)


CELL_SPECS = {
    "AVAL": AVAL_SPEC, "AVAR": AVAR_SPEC, "RIM": RIM_SPEC, "AIY": AIY_SPEC,
}


# =========================================================================
# Channel-set composition
# =========================================================================

# Channels by ion identity (current variable → ion_iX_total accumulator)
CHANNEL_K_VARS  = ["ik_irk_mAcm2", "ik_unc103_mAcm2", "ik_shl1_mAcm2",
                   "ik_egl2_mAcm2", "ik_kqt1_mAcm2"]
CHANNEL_CA_VARS = ["ica_egl19_mAcm2", "ica_cca1_mAcm2", "ica_unc2_mAcm2"]
CHANNEL_NA_VARS = ["ik_nca_mAcm2"]   # NCA treated as Na-current (Layer 1 v1)


def _build_channel_set(channels: dict) -> tuple[str, list[str], list[str], list[str]]:
    """Return (eqs_string, present_K_vars, present_Ca_vars, present_Na_vars).

    eqs_string is the concatenation of bridged channel module EQS for the
    cell's channel set.
    """
    name_to_module_var = {
        "egl19":  (egl19_mod, "ica_egl19_mAcm2", "Ca", ("egl19_eca",)),
        "irk":    (irk_mod, "ik_irk_mAcm2", "K", ("irk_ek",)),
        "nca":    (nca_mod, "ik_nca_mAcm2", "Na", ()),   # NCA uses its own e (30 mV); no bridge
        "unc103": (unc103_mod, "ik_unc103_mAcm2", "K", ("unc103_ek",)),
        "shl1":   (shl1_mod, "ik_shl1_mAcm2", "K", ("shl1_ek",)),
        "cca1":   (cca1_mod, "ica_cca1_mAcm2", "Ca", ("cca1_eca",)),
        "unc2":   (unc2_mod, "ica_unc2_mAcm2", "Ca", ("unc2_eca",)),
        "egl2":   (egl2_mod, "ik_egl2_mAcm2", "K", ("egl2_ek",)),
        "kqt1":   (kqt1_mod, "ik_kqt1_mAcm2", "K", ("kqt1_ek",)),
    }

    eqs_parts = []
    present_K, present_Ca, present_Na = [], [], []
    for ch_name in channels:
        if ch_name not in name_to_module_var:
            raise KeyError(f"Unknown channel: {ch_name}")
        mod, var, ion, bridge_params = name_to_module_var[ch_name]
        # Strip channel's reversal-potential parameter decls (will be bridged)
        eqs = mod.EQS if hasattr(mod, "EQS") else getattr(mod, f"{ch_name.upper()}_EQS")
        if bridge_params:
            eqs = strip_param_decl(eqs, *bridge_params)
        eqs_parts.append(eqs)
        if ion == "K":   present_K.append(var)
        elif ion == "Ca": present_Ca.append(var)
        elif ion == "Na": present_Na.append(var)

    return "\n".join(eqs_parts), present_K, present_Ca, present_Na


def _build_cell_composition(spec: CellSpec, present_K: list[str],
                             present_Ca: list[str], present_Na: list[str]) -> str:
    """Build the cell-specific composition equations: leak split + bridges +
    ion accumulators + membrane V equation."""

    # Sum strings for each ion (concatenate present channel currents)
    sum_K  = " + ".join(present_K)  if present_K  else "0"
    sum_Ca = " + ".join(present_Ca) if present_Ca else "0"
    sum_Na = " + ".join(present_Na) if present_Na else "0"

    # Bridge channel reversals to dynamic Nernst (only declare for channels in this cell)
    bridges = []
    if any(v.endswith("egl19_mAcm2") for v in present_Ca): bridges.append("egl19_eca = E_Ca_mV : 1")
    if any(v.endswith("cca1_mAcm2") for v in present_Ca):  bridges.append("cca1_eca = E_Ca_mV : 1")
    if any(v.endswith("unc2_mAcm2") for v in present_Ca):  bridges.append("unc2_eca = E_Ca_mV : 1")
    if any(v.endswith("irk_mAcm2") for v in present_K):    bridges.append("irk_ek = E_K_mV : 1")
    if any(v.endswith("unc103_mAcm2") for v in present_K): bridges.append("unc103_ek = E_K_mV : 1")
    if any(v.endswith("shl1_mAcm2") for v in present_K):   bridges.append("shl1_ek = E_K_mV : 1")
    if any(v.endswith("egl2_mAcm2") for v in present_K):   bridges.append("egl2_ek = E_K_mV : 1")
    if any(v.endswith("kqt1_mAcm2") for v in present_K):   bridges.append("kqt1_ek = E_K_mV : 1")
    bridges_str = "\n".join(bridges)

    return f"""
    # ---- Dynamic-Nernst bridges (subexpressions overriding fixed channel reversals) ----
{bridges_str}

    # ---- Per-cell parameters ----
    g_leak_Scm2 : 1
    f_K_leak    : 1
    f_Na_leak   : 1
    cm_uFcm2    : 1

    # ---- v_mV bridge (Brian2 volt → bare mV for channel eqs) ----
    v_mV = v / mV : 1

    # ---- LEAK split into K + Na components (GHK-derived) ----
    iK_leak_mAcm2  = f_K_leak  * g_leak_Scm2 * (v_mV - E_K_mV)  : 1
    iNa_leak_mAcm2 = f_Na_leak * g_leak_Scm2 * (v_mV - E_Na_mV) : 1
    iLeak_total_mAcm2 = iK_leak_mAcm2 + iNa_leak_mAcm2 : 1

    # ---- Per-ion totals (channel + leak + pump contributions) ----
    ion_iK_total_mAcm2  = iK_leak_mAcm2  + ({sum_K}) + pump_NaK_iK_mAcm2 + kcc2_iK_mAcm2 : 1
    ion_iNa_total_mAcm2 = iNa_leak_mAcm2 + ({sum_Na}) + pump_NaK_iNa_mAcm2 : 1
    ion_iCa_total_mAcm2 = ({sum_Ca}) + ca_clear_iCa_mAcm2 : 1
    ion_iCl_total_mAcm2 = kcc2_iCl_mAcm2 + abts1_iCl_mAcm2 : 1

    # ---- Membrane current density (drives dV/dt) ----
    # All currents that move net charge across the membrane: channels + leaks +
    # electrogenic pumps. KCC-2 + ABTS-1 lumped electroneutral → no contribution.
    i_total_mAcm2 = (iLeak_total_mAcm2 + ({sum_K}) + ({sum_Ca}) + ({sum_Na})
                     + pump_NaK_I_mAcm2 + ca_clear_I_mAcm2) : 1

    I_total = i_total_mAcm2 * surf_cm2 * 1e9 * pA - I_inj : amp
    dv/dt = -I_total / (cm_uFcm2 * surf_cm2 * 1e6 * pF) : volt
    I_inj : amp
    """


# =========================================================================
# Channel parameter application (skips bridged reversals)
# =========================================================================

def _apply_channel_params_no_reversal(group, ch_name: str, g_Scm2: float) -> None:
    """Apply channel parameters via the module's apply_params, but skip the
    reversal potential setting (handled by subexpression bridge)."""
    if ch_name == "nca":
        nca_mod.nca_apply_params(group, gbar_Scm2=g_Scm2)  # NCA's e is not bridged
        return
    # For bridged channels, call apply_params without ek/eca arguments —
    # the function will still try to set them. Workaround: read its name_map,
    # call manually with bridged param skipped.
    mod_map = {
        "egl19":  (egl19_mod, "egl19_apply_params", "eca_mV", "irk_apply_params"),
        "irk":    (irk_mod, "irk_apply_params", "ek_mV", None),
        "unc103": (unc103_mod, "unc103_apply_params", "ek_mV", None),
        "shl1":   (shl1_mod, "shl1_apply_params", "ek_mV", None),
        "cca1":   (cca1_mod, "cca1_apply_params", "eca_mV", None),
        "unc2":   (unc2_mod, "unc2_apply_params", "eca_mV", None),
        "egl2":   (egl2_mod, "egl2_apply_params", "ek_mV", None),
        "kqt1":   (kqt1_mod, "kqt1_apply_params", "ek_mV", None),
    }
    if ch_name not in mod_map:
        raise KeyError(f"Unknown channel {ch_name}")
    mod, fn_name, _, _ = mod_map[ch_name]
    fn = getattr(mod, fn_name)
    # Call apply_params with only gbar — but the function will still try to set
    # the reversal parameter to its default. We must intercept that.
    # Approach: monkey-patch setattr on a wrapper group object? Cleaner: inline
    # the parameter setting per channel by reading the apply_params source.
    # Simplest robust path: try apply_params and catch the AttributeError on
    # the bridged reversal.
    try:
        fn(group, gbar_Scm2=g_Scm2)
    except Exception as e:
        # Brian2 will raise on missing attribute (the bridged eca/ek)
        # Fall back to manual parameter setting via the module's PARAMS dict
        params_key = f"{ch_name.upper()}_PARAMS"
        params = getattr(mod, params_key, None)
        if params is None:
            raise RuntimeError(f"Channel {ch_name} apply_params failed and no {params_key}: {e}")
        # Apply each parameter manually except the bridged reversal
        bridge_attrs = {"egl19": "egl19_eca", "irk": "irk_ek", "unc103": "unc103_ek",
                        "shl1": "shl1_ek", "cca1": "cca1_eca", "unc2": "unc2_eca",
                        "egl2": "egl2_ek", "kqt1": "kqt1_ek"}
        bridge_attr = bridge_attrs.get(ch_name)
        # Update gbar
        gbar_key = f"gbar_{ch_name}_Scm2"
        if gbar_key in params:
            local_params = dict(params)
            local_params[gbar_key] = g_Scm2
        else:
            local_params = dict(params)
        # Heuristic: each Brian2 variable for this channel is `<ch>_<short>`
        # where short = key without "_mV" / "_Scm2" suffix and channel-name prefix
        # We can't easily reverse-engineer the name_map; rely on apply_params
        # working partway through (it sets parameters in order; the bridged
        # reversal is set LAST in the name_map iteration — but order isn't
        # guaranteed). So we need to use a smarter workaround.
        raise RuntimeError(f"Channel {ch_name} apply_params failed: {e}")


# =========================================================================
# Cell builder
# =========================================================================

def build_layer1_cell(spec: CellSpec, r_eff_um: float = R_EFF_DEFAULT_um):
    """Build a Layer 1 §7.3 integrated cell.

    Returns dict with 'group', 'monitor', 'network', 'spec', 'pump_params'.
    """
    from brian2 import (
        NeuronGroup, StateMonitor, Network, defaultclock, prefs,
        start_scope, ms, mV, pA, pF,
    )
    start_scope()
    prefs.codegen.target = "cython"
    defaultclock.dt = 0.025 * ms

    # Build channel-set eqs (with bridged reversals stripped)
    ch_eqs, present_K, present_Ca, present_Na = _build_channel_set(spec.channels)

    # Compose final eqs
    composition = _build_cell_composition(spec, present_K, present_Ca, present_Na)
    eqs = (
        get_state_and_nernst_eqs()
        + NA_K_ATPASE_EQS
        + LUMPED_CA_CLEARANCE_EQS
        + KCC2_EQS
        + ABTS1_EQS
        + ch_eqs
        + composition
        + get_ion_balance_eqs()
    )

    G = NeuronGroup(1, eqs, method="rk4")

    # Geometry (Nicoletti capacitance + r_eff geometry for volume)
    # NOTE: we use Nicoletti's specific surface area (not r_eff-derived) per spec
    geom = make_cell_geometries(r_eff_um)[spec.name]
    apply_ion_state(G, geom)
    # Override geometry with Nicoletti's per-cell surface (more biological)
    G.surf_cm2 = spec.surf_cm2

    # Pump parameters: AVAL anchor scaled by TPM for other cells
    pump_params = {
        "I_NaK_max":      scale_I_max_by_eat6_tpm(PUMP_ANCHOR_AVAL["I_NaK_max"], spec.pump_cell_name),
        "I_kcc2_max":     scale_I_max_by_kcc2_tpm(PUMP_ANCHOR_AVAL["I_kcc2_max"], spec.pump_cell_name),
        "I_abts1_max":    scale_I_max_by_abts1_tpm(PUMP_ANCHOR_AVAL["I_abts1_max"], spec.pump_cell_name),
        "I_Ca_clear_max": scale_I_max_by_mca3_tpm(PUMP_ANCHOR_AVAL["I_Ca_clear_max"], spec.pump_cell_name),
    }
    apply_na_k_atpase_params(G, I_max_mAcm2=pump_params["I_NaK_max"])
    apply_ca_clearance_params(G, I_max_mAcm2=pump_params["I_Ca_clear_max"])
    apply_kcc2_params(G, I_max_mAcm2=pump_params["I_kcc2_max"])
    apply_abts1_params(G, I_max_mAcm2=pump_params["I_abts1_max"])

    # Channel parameters (manually, bypassing apply_params to avoid bridged reversal)
    _apply_channels_manual(G, spec)

    # Per-cell LEAK split
    E_K = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["K"], INTRACELLULAR_DEFAULT_mM["K"], +1)
    E_Na = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["Na"], INTRACELLULAR_DEFAULT_mM["Na"], +1)
    f_K, f_Na = ghk_leak_split(spec.e_leak_mV, E_K, E_Na)
    G.g_leak_Scm2 = spec.g_leak_Scm2
    G.f_K_leak = f_K
    G.f_Na_leak = f_Na
    G.cm_uFcm2 = spec.cm_uFcm2

    # Initial V
    G.I_inj = 0 * pA
    G.v = spec.v_init_mV * mV

    # Initialize channel states at v_init
    _init_channel_states(G, spec)

    mon_vars = ["v", "K_in", "Na_in", "Cl_in", "Ca_in",
                "E_K_mV", "E_Na_mV", "E_Cl_mV", "E_Ca_mV",
                "pump_NaK_I_mAcm2", "ca_clear_I_mAcm2",
                "kcc2_v_mAcm2", "abts1_v_mAcm2",
                "iK_leak_mAcm2", "iNa_leak_mAcm2", "iLeak_total_mAcm2",
                "i_total_mAcm2"]
    # Per-channel currents if present (for diagnostics)
    for var in (CHANNEL_K_VARS + CHANNEL_CA_VARS + CHANNEL_NA_VARS):
        if any(var in pres for pres in (present_K, present_Ca, present_Na)):
            mon_vars.append(var)

    mon = StateMonitor(G, mon_vars, record=True)
    net = Network(G, mon)

    return {
        "group": G, "monitor": mon, "network": net,
        "spec": spec, "pump_params": pump_params,
        "leak_split": (f_K, f_Na),
        "geometry": geom,
    }


# Explicit per-channel name maps (gbar key, bridged-reversal key to SKIP, name pairs)
# Built from reading each channel module's apply_params name_map.

_CHANNEL_APPLIES: dict[str, dict] = {
    "egl19": {
        "params_attr": "EGL19_PARAMS",
        "gbar_key": "gbar_egl19_Scm2",
        "skip_keys": {"eca_mV"},
        "pairs": [
            ("va_egl19", "egl19_va"), ("ka_egl19", "egl19_ka"), ("shift", "egl19_shift"),
            ("p1hegl19", "egl19_p1"), ("p2hegl19", "egl19_p2"), ("p3hegl19", "egl19_p3"),
            ("p4hegl19", "egl19_p4"), ("p5hegl19", "egl19_p5"), ("p6hegl19", "egl19_p6"),
            ("p7hegl19", "egl19_p7"), ("p8hegl19", "egl19_p8"),
            ("pdg1", "egl19_pdg1"), ("pdg2", "egl19_pdg2"), ("pdg3", "egl19_pdg3"),
            ("pdg4", "egl19_pdg4"), ("pdg5", "egl19_pdg5"), ("pdg6", "egl19_pdg6"),
            ("pdg7", "egl19_pdg7"), ("ctm19", "egl19_ctm19"),
            ("pds1", "egl19_pds1"), ("pds2", "egl19_pds2"), ("pds3", "egl19_pds3"),
            ("pds4", "egl19_pds4"), ("pds5", "egl19_pds5"), ("pds6", "egl19_pds6"),
            ("pds7", "egl19_pds7"), ("pds8", "egl19_pds8"), ("pds9", "egl19_pds9"),
            ("pds10", "egl19_pds10"), ("pds11", "egl19_pds11"),
            ("gbar_egl19_Scm2", "egl19_gbar"),
        ],
    },
    "irk": {
        "params_attr": "IRK_PARAMS",
        "gbar_key": "gbar_irk_Scm2",
        "skip_keys": {"ek_mV"},
        "pairs": [
            ("va_kir", "irk_va"), ("ka_kir", "irk_ka"),
            ("p1tmkir", "irk_p1tmkir"), ("p2tmkir", "irk_p2tmkir"),
            ("p3tmkir", "irk_p3tmkir"), ("p4tmkir", "irk_p4tmkir"),
            ("p5tmkir", "irk_p5tmkir"), ("p6tmkir", "irk_p6tmkir"),
            ("gbar_irk_Scm2", "irk_gbar"),
        ],
    },
    "unc103": {
        "params_attr": "UNC103_PARAMS",
        "gbar_key": "gbar_unc103_Scm2",
        "skip_keys": {"ek_mV"},
        "pairs": [
            ("va", "unc103_va"), ("ka", "unc103_ka"),
            ("vi", "unc103_vi"), ("ki", "unc103_ki"),
            ("tm1", "unc103_tm1"), ("tm2", "unc103_tm2"),
            ("tm3", "unc103_tm3"), ("tm4", "unc103_tm4"),
            ("th1", "unc103_th1"), ("th2", "unc103_th2"),
            ("th3", "unc103_th3"), ("th4", "unc103_th4"),
            ("gbar_unc103_Scm2", "unc103_gbar"),
        ],
    },
    "shl1": {
        "params_attr": "SHL1_PARAMS",
        "gbar_key": "gbar_shl1_Scm2",
        "skip_keys": {"ek_mV"},
        "pairs": [
            ("vashal", "shl1_vashal"), ("kashal", "shl1_kashal"),
            ("vishal", "shl1_vishal"), ("kishal", "shl1_kishal"),
            ("shalsfhit", "shl1_shalsfhit"),
            ("ptmshal1", "shl1_ptmshal1"), ("ptmshal2", "shl1_ptmshal2"),
            ("ptmshal3", "shl1_ptmshal3"), ("ptmshal4", "shl1_ptmshal4"),
            ("ptmshal5", "shl1_ptmshal5"), ("ptmshal6", "shl1_ptmshal6"),
            ("pthfshal1", "shl1_pthfshal1"), ("pthfshal2", "shl1_pthfshal2"),
            ("pthfshal3", "shl1_pthfshal3"), ("pthfshal4", "shl1_pthfshal4"),
            ("pthsshal1", "shl1_pthsshal1"), ("pthsshal2", "shl1_pthsshal2"),
            ("pthsshal3", "shl1_pthsshal3"), ("pthsshal4", "shl1_pthsshal4"),
            ("a", "shl1_a"),
            ("gbar_shl1_Scm2", "shl1_gbar"),
        ],
    },
    "cca1": {
        "params_attr": "CCA1_PARAMS",
        "gbar_key": "gbar_cca1_Scm2",
        "skip_keys": {"eca_mV"},
        "pairs": [
            ("va_cca1", "cca1_va"), ("ka_cca1", "cca1_ka"),
            ("sscca1", "cca1_ssm"), ("fcca", "cca1_fcca"),
            ("vi_cca1", "cca1_vi"), ("ki_cca1", "cca1_ki"),
            ("sshcca1", "cca1_ssh"), ("f2cca1", "cca1_f2"),
            ("p1tmcca1", "cca1_p1tm"), ("p2tmcca1", "cca1_p2tm"),
            ("p3tmcca1", "cca1_p3tm"), ("p4tmcca1", "cca1_p4tm"),
            ("stmcca1", "cca1_stm"), ("f3ca", "cca1_f3"),
            ("constmcca1", "cca1_constm"),
            ("p1thcca1", "cca1_p1th"), ("p2thcca1", "cca1_p2th"),
            ("p3thcca1", "cca1_p3th"), ("p4thcca1", "cca1_p4th"),
            ("sthcca1", "cca1_sth"), ("f4ca", "cca1_f4"),
            ("consthcca1", "cca1_consth"),
            ("gbar_cca1_Scm2", "cca1_gbar"),
        ],
    },
    "unc2": {
        "params_attr": "UNC2_PARAMS",
        "gbar_key": "gbar_unc2_Scm2",
        "skip_keys": {"eca_mV"},
        "pairs": [
            ("va_unc2", "unc2_va"), ("ka_unc2", "unc2_ka"),
            ("stm2", "unc2_stm2"),
            ("vi_unc2", "unc2_vi"), ("ki_unc2", "unc2_ki"),
            ("sth2", "unc2_sth2"),
            ("p1tmunc2", "unc2_p1tm"), ("p2tmunc2", "unc2_p2tm"),
            ("p3tmunc2", "unc2_p3tm"), ("p4tmunc2", "unc2_p4tm"),
            ("p5tmunc2", "unc2_p5tm"), ("shiftmunc2", "unc2_shiftm"),
            ("fp3", "unc2_fp3"), ("fp4", "unc2_fp4"),
            ("constmunc2", "unc2_constm"),
            ("p1thunc2", "unc2_p1th"), ("p2thunc2", "unc2_p2th"),
            ("p3thunc2", "unc2_p3th"), ("p4thunc2", "unc2_p4th"),
            ("p5thunc2", "unc2_p5th"), ("p6thunc2", "unc2_p6th"),
            ("shifthunc2", "unc2_shifth"), ("fp5", "unc2_fp5"),
            ("consthunc2", "unc2_consth"),
            ("gbar_unc2_Scm2", "unc2_gbar"),
        ],
    },
    "egl2": {
        "params_attr": "EGL2_PARAMS",
        "gbar_key": "gbar_egl2_Scm2",
        "skip_keys": {"ek_mV"},
        "pairs": [
            ("va_egl2", "egl2_va"), ("ka_egl2", "egl2_ka"),
            ("stmegl2", "egl2_stm"), ("cegl2", "egl2_c"),
            ("p1tmegl2", "egl2_p1tm"), ("p2tmegl2", "egl2_p2tm"),
            ("p3tmegl2", "egl2_p3tm"), ("p4tmegl2", "egl2_p4tm"),
            ("fegl2", "egl2_f"),
            ("gbar_egl2_Scm2", "egl2_gbar"),
        ],
    },
    "kqt1": {
        "params_attr": "KQT1_PARAMS",
        "gbar_key": "gbar_kqt1_Scm2",
        "skip_keys": {"ek_mV"},
        "pairs": [
            ("va", "kqt1_va"), ("ka", "kqt1_ka"),
            ("p1tmkqt1", "kqt1_p1tmkqt1"), ("p2tmkqt1", "kqt1_p2tmkqt1"),
            ("p3tmkqt1", "kqt1_p3tmkqt1"), ("p4tmkqt1", "kqt1_p4tmkqt1"),
            ("s1", "kqt1_s1"), ("s2", "kqt1_s2"), ("s3", "kqt1_s3"),
            ("s4", "kqt1_s4"), ("s5", "kqt1_s5"), ("s6", "kqt1_s6"),
            ("p1tskqt1", "kqt1_p1tskqt1"), ("p2tskqt1", "kqt1_p2tskqt1"),
            ("p3tskqt1", "kqt1_p3tskqt1"), ("p4tskqt1", "kqt1_p4tskqt1"),
            ("gbar_kqt1_Scm2", "kqt1_gbar"),
        ],
    },
}


_CHANNEL_MODULE_MAP = {
    "egl19": egl19_mod, "irk": irk_mod, "nca": nca_mod, "unc103": unc103_mod,
    "shl1": shl1_mod, "cca1": cca1_mod, "unc2": unc2_mod, "egl2": egl2_mod,
    "kqt1": kqt1_mod,
}


def _apply_channels_manual(group, spec: CellSpec) -> None:
    """Set channel parameters explicitly, skipping bridged reversal attributes."""
    for ch_name, g_Scm2 in spec.channels.items():
        if ch_name == "nca":
            nca_mod.nca_apply_params(group, gbar_Scm2=g_Scm2)
            continue
        if ch_name not in _CHANNEL_APPLIES:
            raise NotImplementedError(f"Layer 1 apply not implemented for {ch_name!r}")
        cfg = _CHANNEL_APPLIES[ch_name]
        mod = _CHANNEL_MODULE_MAP[ch_name]
        params = dict(getattr(mod, cfg["params_attr"]))
        params[cfg["gbar_key"]] = g_Scm2
        for src, dst in cfg["pairs"]:
            if src in cfg["skip_keys"]:
                continue
            setattr(group, dst, params[src])


def _init_channel_states(group, spec: CellSpec) -> None:
    """Initialize each channel's gating state variables at v_init."""
    v = spec.v_init_mV
    for ch_name in spec.channels:
        if ch_name == "egl19":
            egl19_mod.egl19_init_states(group, v_mV=v)
        elif ch_name == "irk":
            irk_mod.irk_init_states(group, v_mV=v)
        elif ch_name == "nca":
            nca_mod.nca_init_states(group, v_mV=v)
        elif ch_name == "unc103" and hasattr(unc103_mod, "unc103_init_states"):
            unc103_mod.unc103_init_states(group, v_mV=v)
        elif ch_name == "shl1" and hasattr(shl1_mod, "shl1_init_states"):
            shl1_mod.shl1_init_states(group, v_mV=v)
        elif ch_name == "cca1" and hasattr(cca1_mod, "cca1_init_states"):
            cca1_mod.cca1_init_states(group, v_mV=v)
        elif ch_name == "unc2" and hasattr(unc2_mod, "unc2_init_states"):
            unc2_mod.unc2_init_states(group, v_mV=v)
        elif ch_name == "egl2" and hasattr(egl2_mod, "egl2_init_states"):
            egl2_mod.egl2_init_states(group, v_mV=v)
        elif ch_name == "kqt1" and hasattr(kqt1_mod, "kqt1_init_states"):
            kqt1_mod.kqt1_init_states(group, v_mV=v)
