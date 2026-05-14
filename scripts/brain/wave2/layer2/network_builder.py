"""
Layer 2 network builder — 300-cell C. elegans connectome on the Path 2
scalable substrate.

Architecture: one homogeneous Brian2 NeuronGroup of size 300, with ALL 14
channels in the equations. Per-cell gbar = 0 for channels a cell doesn't
express (via CeNGEN T2). Trades integration cost for compile simplicity —
one cython compile total instead of ~30 (one per unique channel set).

Each individual cell (AVAL, AVAR, etc.) is mapped to its CeNGEN class
(AVA) for substrate parameters, then assigned an index in the group.

Synapses: graded (V-dependent continuous release) per C. elegans biology.
Chemical synapses are excitatory (E=0 mV) or inhibitory (E=-70 mV) based
on neurotransmitter + receptor expression. Gap junctions are Ohmic.

Use:
    builder = build_layer2_network(connectome_path)
    bundle = builder.build()
    bundle["network"].run(5 * second)
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
WAVE2_DIR = THIS_DIR.parent
sys.path.insert(0, str(WAVE2_DIR))

import numpy as np

from ion_dynamics import (
    INTRACELLULAR_DEFAULT_mM, EXTRACELLULAR_DEFAULT_mM,
    R_EFF_DEFAULT_um, CellGeometry,
    get_state_and_nernst_eqs, get_ion_balance_eqs,
    nernst_potential_mV, T_C_ELEGANS_K,
    KAPPA_BUFFER_CA_DEFAULT, R_GAS_J_PER_MOL_K, F_FARADAY_C_PER_MOL,
)
from pumps.na_k_atpase import (
    NA_K_ATPASE_EQS, apply_na_k_atpase_params,
)
from pumps.ca_clearance import (
    LUMPED_CA_CLEARANCE_EQS, apply_ca_clearance_params,
)
from pumps.kcc2_abts1_lumped import (
    KCC2_EQS, ABTS1_EQS,
    apply_kcc2_params, apply_abts1_params,
)
from layer1_cells import (
    PUMP_ANCHOR_AVAL, ghk_leak_split, _CHANNEL_APPLIES, _CHANNEL_MODULE_MAP,
    strip_param_decl,
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
from channels import exp2 as exp2_mod
from channels import shk1 as shk1_mod
from channels import twk as twk_mod
from channels import slo2 as slo2_mod
from channels import egl36 as egl36_mod
from channels import kvs1 as kvs1_mod

from path2_scale.scalable_builder import build_scalable_spec, to_layer1_cellspec
from path2_scale.cengen_tpm_data import CENGEN_NEURONS


# All 14 channels with their (module, current_var, ion, bridged_reversal_params)
ALL_CHANNELS = [
    ("egl19",  egl19_mod,  "ica_egl19_mAcm2",  "Ca", ("egl19_eca",)),
    ("irk",    irk_mod,    "ik_irk_mAcm2",     "K",  ("irk_ek",)),
    ("nca",    nca_mod,    "ik_nca_mAcm2",     "Na", ()),
    ("unc103", unc103_mod, "ik_unc103_mAcm2",  "K",  ("unc103_ek",)),
    ("shl1",   shl1_mod,   "ik_shl1_mAcm2",    "K",  ("shl1_ek",)),
    ("cca1",   cca1_mod,   "ica_cca1_mAcm2",   "Ca", ("cca1_eca",)),
    ("unc2",   unc2_mod,   "ica_unc2_mAcm2",   "Ca", ("unc2_eca",)),
    ("egl2",   egl2_mod,   "ik_egl2_mAcm2",    "K",  ("egl2_ek",)),
    ("kqt1",   kqt1_mod,   "ik_kqt1_mAcm2",    "K",  ("kqt1_ek",)),
    ("exp2",   exp2_mod,   "ik_exp2_mAcm2",    "K",  ("exp2_ek",)),
    ("shk1",   shk1_mod,   "ik_shk1_mAcm2",    "K",  ("shk1_ek",)),
    ("twk",    twk_mod,    "ik_twk_mAcm2",     "K",  ("twk_ek",)),
    ("slo2",   slo2_mod,   "ik_slo2_mAcm2",    "K",  ("slo2_ek",)),
    ("egl36",  egl36_mod,  "ik_egl36_mAcm2",   "K",  ("egl36_ek",)),
    ("kvs1",   kvs1_mod,   "ik_kvs1_mAcm2",    "K",  ("kvs1_ek",)),
]

K_CURRENTS  = [v for _, _, v, ion, _ in ALL_CHANNELS if ion == "K"]
CA_CURRENTS = [v for _, _, v, ion, _ in ALL_CHANNELS if ion == "Ca"]
NA_CURRENTS = [v for _, _, v, ion, _ in ALL_CHANNELS if ion == "Na"]


def build_homogeneous_eqs() -> str:
    """Build one eqs string covering all 14 channels + pumps + ion dynamics +
    membrane V. All channel state variables exist for every cell; per-cell
    gbar=0 disables the unused ones."""
    parts = [get_state_and_nernst_eqs(),
             NA_K_ATPASE_EQS,
             LUMPED_CA_CLEARANCE_EQS,
             KCC2_EQS,
             ABTS1_EQS]

    bridges = []
    for name, mod, cur, ion, bridge_params in ALL_CHANNELS:
        eqs = mod.EQS if hasattr(mod, "EQS") else getattr(mod, f"{name.upper()}_EQS")
        if bridge_params:
            eqs = strip_param_decl(eqs, *bridge_params)
            for bp in bridge_params:
                # bp is "<channel>_eca" or "<channel>_ek"
                if bp.endswith("_ek"):
                    bridges.append(f"{bp} = E_K_mV : 1")
                elif bp.endswith("_eca"):
                    bridges.append(f"{bp} = E_Ca_mV : 1")
        parts.append(eqs)

    bridges_str = "\n".join(bridges)
    sum_K  = " + ".join(K_CURRENTS)
    sum_Ca = " + ".join(CA_CURRENTS)
    sum_Na = " + ".join(NA_CURRENTS)

    composition = f"""
    # ---- Dynamic-Nernst bridges ----
{bridges_str}

    # ---- Per-cell parameters ----
    g_leak_Scm2 : 1
    f_K_leak    : 1
    f_Na_leak   : 1
    cm_uFcm2    : 1

    v_mV = v / mV : 1

    # ---- LEAK split ----
    iK_leak_mAcm2  = f_K_leak  * g_leak_Scm2 * (v_mV - E_K_mV)  : 1
    iNa_leak_mAcm2 = f_Na_leak * g_leak_Scm2 * (v_mV - E_Na_mV) : 1
    iLeak_total_mAcm2 = iK_leak_mAcm2 + iNa_leak_mAcm2 : 1

    # ---- Per-ion totals ----
    ion_iK_total_mAcm2  = iK_leak_mAcm2  + ({sum_K}) + pump_NaK_iK_mAcm2 + kcc2_iK_mAcm2 : 1
    ion_iNa_total_mAcm2 = iNa_leak_mAcm2 + ({sum_Na}) + pump_NaK_iNa_mAcm2 : 1
    ion_iCa_total_mAcm2 = ({sum_Ca}) + ca_clear_iCa_mAcm2 : 1
    ion_iCl_total_mAcm2 = kcc2_iCl_mAcm2 + abts1_iCl_mAcm2 : 1

    # ---- Membrane current density (drives dV/dt) ----
    i_total_mAcm2 = (iLeak_total_mAcm2 + ({sum_K}) + ({sum_Ca}) + ({sum_Na})
                     + pump_NaK_I_mAcm2 + ca_clear_I_mAcm2) : 1

    # I_intrinsic comes from substrate; I_syn + I_gap + I_inj are external
    I_intrinsic = i_total_mAcm2 * surf_cm2 * 1e9 * pA : amp
    I_total = I_intrinsic + I_syn + I_gap - I_inj : amp
    dv/dt = -I_total / (cm_uFcm2 * surf_cm2 * 1e6 * pF) : volt
    I_inj : amp
    I_syn : amp
    I_gap : amp
    """

    parts.append(composition)
    parts.append(get_ion_balance_eqs())
    return "\n".join(parts)


def map_cell_to_class(cell_name: str, class_list: list[str]) -> str | None:
    """Map an individual cell name (e.g., AVAL) to its CeNGEN class (AVA).

    Strategy:
      1. exact name match
      2. strip trailing L/R
      3. strip trailing digits (DA01 → DA, VD1 → VD)
      4. Compound-class lookups (e.g., RMD_DV / RMD_LR splits):
         - RMDDL/RMDDR/RMDVL/RMDVR → RMD_DV
         - RMDL/RMDR              → RMD_LR
         - IL2DL/IL2DR/IL2VL/IL2VR → IL2_DV
         - IL2L/IL2R              → IL2_LR
         - RMED/RMEV               → RME_DV
         - RMEL/RMER               → RME_LR
         - URADL/URADR/URAVL/URAVR → URA (try direct), else URA_DV/URA_LR
         - AWCL/AWCR              → AWC_OFF or AWC_ON (split arbitrarily)
         - VD1-13, DD1-6           → VD_DD
      5. Strip CEP/IL1 spatial suffixes
    """
    if cell_name in class_list:
        return cell_name

    # AWC special case
    if cell_name == "AWCL" and "AWC_OFF" in class_list:
        return "AWC_OFF"
    if cell_name == "AWCR" and "AWC_ON" in class_list:
        return "AWC_ON"

    # VD/DD motor neurons → VD_DD
    if cell_name.startswith("VD") and cell_name[2:].isdigit() and "VD_DD" in class_list:
        return "VD_DD"
    if cell_name.startswith("DD") and cell_name[2:].isdigit() and "VD_DD" in class_list:
        return "VD_DD"

    # RMD_DV vs RMD_LR
    if cell_name.startswith("RMDD") or cell_name.startswith("RMDV"):
        if "RMD_DV" in class_list:
            return "RMD_DV"
    if cell_name in ("RMDL", "RMDR"):
        if "RMD_LR" in class_list:
            return "RMD_LR"

    # IL2_DV vs IL2_LR
    if cell_name.startswith("IL2D") or cell_name.startswith("IL2V"):
        if "IL2_DV" in class_list:
            return "IL2_DV"
    if cell_name in ("IL2L", "IL2R"):
        if "IL2_LR" in class_list:
            return "IL2_LR"

    # RME_DV vs RME_LR
    if cell_name in ("RMED", "RMEV") and "RME_DV" in class_list:
        return "RME_DV"
    if cell_name in ("RMEL", "RMER") and "RME_LR" in class_list:
        return "RME_LR"

    # Strip spatial suffixes (DL, DR, VL, VR, D, V, L, R) for motor classes
    for suffix in ["DL", "DR", "VL", "VR", "DV", "LR"]:
        if cell_name.endswith(suffix):
            base = cell_name[:-len(suffix)]
            if base in class_list:
                return base
            # check compound class
            compound = f"{base}_{suffix}"
            if compound in class_list:
                return compound

    # strip trailing L/R
    if len(cell_name) > 1 and cell_name[-1] in "LR":
        base = cell_name[:-1]
        if base in class_list:
            return base
        # also try with _LR / _DV suffix on the base
        for suff in ("_LR", "_DV"):
            if base + suff in class_list:
                return base + suff

    # strip trailing digits (DA01 → DA, M1 → M1 may be in list, etc.)
    stripped = cell_name.rstrip("0123456789")
    if stripped and stripped in class_list:
        return stripped

    # strip trailing single letter (RMED → RME, etc.)
    if len(cell_name) > 2 and cell_name[:-1] in class_list:
        return cell_name[:-1]

    return None


def build_per_cell_params(connectome_names: list[str]) -> list[dict]:
    """For each individual cell, get its substrate parameters from
    build_scalable_spec(klass).

    Returns list of dicts (one per cell), each containing:
      cell_name, cengen_class, channels (gbar dict), e_leak_mV, v_init_mV,
      g_leak_Scm2, surf_cm2, cm_uFcm2, pump_NaK_scale, pump_cell_name
    """
    cengen_classes = list(CENGEN_NEURONS)
    per_cell = []
    unmapped = []
    for name in connectome_names:
        klass = map_cell_to_class(name, cengen_classes)
        if klass is None:
            unmapped.append(name)
            # Use a generic "any" fallback to keep cell count intact
            klass = cengen_classes[0]  # placeholder; will mark as unmapped
        spec_s = build_scalable_spec(klass, cell_name=name)
        spec_l = to_layer1_cellspec(spec_s)
        per_cell.append({
            "cell_name": name,
            "cengen_class": klass,
            "channels": spec_s.channels,
            "e_leak_mV": spec_l.e_leak_mV,
            "v_init_mV": spec_l.v_init_mV,
            "g_leak_Scm2": spec_l.g_leak_Scm2,
            "surf_cm2": spec_l.surf_cm2,
            "cm_uFcm2": spec_l.cm_uFcm2,
            "cm_pF": spec_s.cm_pF,
            "pump_NaK_scale": spec_l.pump_NaK_scale,
            "pump_cell_name": spec_l.pump_cell_name,
            "unmapped": name in unmapped,
        })
    return per_cell, unmapped


def apply_per_cell_params(group, per_cell_params: list[dict]) -> None:
    """Set Brian2 NeuronGroup attributes per cell (vectorized).

    Strategy: for each channel, collect gbar values across all cells (0 for
    cells that don't express it). Set the channel parameters (kinetics) from
    defaults; set per-cell gbar. Set per-cell substrate params
    (g_leak, e_leak split, cm, surf, pump scale).
    """
    from brian2 import mV, pA, pF
    n = len(per_cell_params)

    # Per-cell substrate parameters
    surf_cm2_arr = np.array([p["surf_cm2"] for p in per_cell_params])
    cm_uFcm2_arr = np.array([p["cm_uFcm2"] for p in per_cell_params])
    g_leak_arr   = np.array([p["g_leak_Scm2"] for p in per_cell_params])
    e_leak_arr   = np.array([p["e_leak_mV"] for p in per_cell_params])
    v_init_arr   = np.array([p["v_init_mV"] for p in per_cell_params])
    pump_scale_arr = np.array([p["pump_NaK_scale"] for p in per_cell_params])

    group.surf_cm2 = surf_cm2_arr
    group.cm_uFcm2 = cm_uFcm2_arr
    group.g_leak_Scm2 = g_leak_arr

    # GHK leak split — vectorized per cell
    E_K = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["K"], INTRACELLULAR_DEFAULT_mM["K"], +1)
    E_Na = nernst_potential_mV(EXTRACELLULAR_DEFAULT_mM["Na"], INTRACELLULAR_DEFAULT_mM["Na"], +1)
    f_K = np.zeros(n)
    f_Na = np.zeros(n)
    for i, e in enumerate(e_leak_arr):
        fk, fna = ghk_leak_split(float(e), E_K, E_Na)
        f_K[i] = fk
        f_Na[i] = fna
    group.f_K_leak = f_K
    group.f_Na_leak = f_Na

    # Initialize ion state
    group.K_in  = INTRACELLULAR_DEFAULT_mM["K"]
    group.Na_in = INTRACELLULAR_DEFAULT_mM["Na"]
    group.Cl_in = INTRACELLULAR_DEFAULT_mM["Cl"]
    group.Ca_in = INTRACELLULAR_DEFAULT_mM["Ca"]
    group.K_out  = EXTRACELLULAR_DEFAULT_mM["K"]
    group.Na_out = EXTRACELLULAR_DEFAULT_mM["Na"]
    group.Cl_out = EXTRACELLULAR_DEFAULT_mM["Cl"]
    group.Ca_out = EXTRACELLULAR_DEFAULT_mM["Ca"]
    group.kappa_B_Ca = KAPPA_BUFFER_CA_DEFAULT
    group.rt_over_F_mV = R_GAS_J_PER_MOL_K * T_C_ELEGANS_K / F_FARADAY_C_PER_MOL * 1000.0
    # Volume per cell (use r_eff_um and surf_cm2)
    r_eff_cm = R_EFF_DEFAULT_um * 1e-4
    group.vol_L = surf_cm2_arr * r_eff_cm / 2.0 * 1e-3

    # Initial voltage per cell
    group.v = v_init_arr * mV
    group.I_inj = 0 * pA
    group.I_syn = 0 * pA
    group.I_gap = 0 * pA

    # Pump parameters — all cells use AVAL anchor × per-cell scale
    group.pump_NaK_I_max_mAcm2 = pump_scale_arr * PUMP_ANCHOR_AVAL["I_NaK_max"]
    apply_na_k_atpase_params(group, I_max_mAcm2=PUMP_ANCHOR_AVAL["I_NaK_max"])
    # override I_max with scaled values
    group.pump_NaK_I_max_mAcm2 = pump_scale_arr * PUMP_ANCHOR_AVAL["I_NaK_max"]
    apply_ca_clearance_params(group, I_max_mAcm2=PUMP_ANCHOR_AVAL["I_Ca_clear_max"])
    apply_kcc2_params(group, I_max_mAcm2=PUMP_ANCHOR_AVAL["I_kcc2_max"])
    apply_abts1_params(group, I_max_mAcm2=PUMP_ANCHOR_AVAL["I_abts1_max"])

    # Channel parameters — kinetics from defaults, per-cell gbar
    for ch_name, mod, _, _, _ in ALL_CHANNELS:
        # Build per-cell gbar array (0 for cells without this channel)
        gbar_arr = np.array([p["channels"].get(ch_name, 0.0) for p in per_cell_params])
        # Get kinetics from default params, set on group
        if ch_name in _CHANNEL_APPLIES:
            cfg = _CHANNEL_APPLIES[ch_name]
            params = getattr(mod, cfg["params_attr"])
            for src, dst in cfg["pairs"]:
                if src in cfg["skip_keys"]:
                    continue
                if src == cfg["gbar_key"]:
                    setattr(group, dst, gbar_arr)
                else:
                    setattr(group, dst, params[src])
        elif ch_name == "nca":
            nca_mod.nca_apply_params(group, gbar_Scm2=gbar_arr[0])
            group.nca_gbar = gbar_arr

    # Initialize channel state variables at v_init per cell
    # For homogeneous group, init each state var as steady-state at each cell's v_init
    init_channel_states_vectorized(group, v_init_arr)


def init_channel_states_vectorized(group, v_init_arr):
    """Initialize all channel state variables to SAFE closed/available values.

    Strategy: set activation gates (m_*) to 0 (channels closed → I=0
    regardless of inactivation state), inactivation gates (h_*) to 1
    (available — won't matter while m=0 but matches biology). Brian2's
    integration will drive them to steady-state within a few ms.

    This avoids replicating each channel's m_inf/h_inf formula and
    eliminates the risk of init-induced numerical blow-up from
    inappropriate initial conductances.
    """
    # Activation gates — closed (m=0 → channel current = 0)
    group.m_egl19  = 0.0
    group.h_egl19  = 1.0
    group.m_irk    = 0.0
    group.m_unc103 = 0.0
    group.h_unc103 = 1.0
    group.m_shl1   = 0.0
    group.hf_shl1  = 1.0
    group.hs_shl1  = 1.0
    group.m_cca1   = 0.0
    group.h_cca1   = 1.0
    group.m_unc2   = 0.0
    group.h_unc2   = 1.0
    group.m_egl2   = 0.0
    group.m_kqt1   = 0.0
    group.s_kqt1   = 0.0
    group.m_exp2   = 0.0
    group.h_exp2   = 1.0
    group.m_shk1   = 0.0
    group.h_shk1   = 1.0
    group.m_egl36  = 0.0
    group.m_kvs1   = 0.0
    # TWK: no gating
    # SLO-2: Ca-activated, computed instantaneously from Ca_in
