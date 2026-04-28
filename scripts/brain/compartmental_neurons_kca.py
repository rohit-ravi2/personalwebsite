#!/usr/bin/env python3
"""Wave 1 sandbox: compartmental scaffold + h-inactivation + I_KCa variants.

NOT a production patch. Sandbox for empirically testing whether γ
(compartmental scaffold extended with Ca-activated K+ current) closes the
plateau termination gap surfaced by Phase 0 analytic verification.

Companion to graded_brain_h_kca.py (Session 2's β' sandbox). Uses identical
parameter values for the K_Ca + h patch so cross-architecture comparison
is artifact-free:
  - alpha_Ca   = 0.0005 (Session 3 corrected; Session 2's first version had
                  0.05 which produced K_Ca dominance via [Ca] saturation)
  - tau_Ca_decay = 200 ms
  - n_Hill     = 4 (BK-class; Salkoff 2006)
  - K_d_KCa    = 1.0 µM (BK; Yuan 2000)
  - g_KCa      = 2.0 nS (default; Phase 1.5 sensitivity-checks this)
  - E_K        = -90 mV
  - I_Ca with h as direct multiplicative gate: I_ca = g_ca * m_inf * h * (E_Ca - v_d)
    (mathematically equivalent to scaffold's I_ca - I_ca_inact form, but
    matches Session 2's sandbox structure for cross-session diff)

Differences from Session 2's sandbox (the load-bearing architectural test):
  - Two-compartment soma + dendrite (vs Session 2's single compartment)
  - L-type Ca + h + I_KCa + [Ca] all live on the dendrite
  - Soma feels K_Ca only via axial coupling (g_ax 0.8-1.5 nS per cell)
  - v_rest set to -25 mV (Mellem AVA up-state; scaffold's default -65 mV
    is mammalian template and overridden here)

Per-cell parameters sourced from compartmental_neurons.COMPARTMENTAL_ROSTER
(single source of truth for the scaffold). Original scaffold not modified.

Three variants supported (parallel structure with Session 2's sandbox):
  - 'base'   : compartmental scaffold equations as-is (no h, no K_Ca)
               — but with v_rest correction to -25 mV applied. Reference
               for "no plateau termination machinery".
  - 'h_only' : adds h-inactivation following compartmental_neurons.py
               pattern, with v_rest = -25. Phase 0 predicts this fails
               because h_ss = 0.3/(0.3+m_inf) leaves 23-30% I_Ca uninactivated.
  - 'h_kca'  : adds h-inactivation + intracellular [Ca] pool dynamics
               + Ca-activated K+ current (I_KCa) on dendrite. Primary
               patch hypothesis. Tests the compartmental architectural-
               comparison claim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from brian2 import (
    NeuronGroup, StateMonitor, Network,
    ms, mV, nS, pF, pA,
    prefs, seed as brian2_seed,
    start_scope,
)


prefs.codegen.target = "cython"


# Single source of truth for per-cell parameters: import from production scaffold.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compartmental_neurons import (  # noqa: E402
    CompartmentalParams,
    COMPARTMENTAL_ROSTER,
)


# ---------------------------------------------------------------------
# Sandbox parameter overrides (Wave 1 calibration scope only)
# ---------------------------------------------------------------------

# Mellem 2008 AVA up-state per project T0 closure work.
# Production scaffold uses -65 mV (mammalian template); sandbox overrides
# to test cellular validation against Mellem-grounded targets.
V_REST_MELLEM_MV = -25.0

# K_Ca + Ca-pool parameters (literature-grounded, fixed for cross-session
# comparability with Session 2's graded_brain_h_kca.py).
ALPHA_CA = 0.0005          # dimensionless-µM per (pA × ms). Session 3 corrected.
TAU_CA_DECAY_MS = 200.0    # PMCA + buffering aggregate (literature mid-range)
N_HILL = 4.0               # BK Hill coefficient (Salkoff 2006)
K_D_KCA = 1.0              # µM-equivalent (Yuan 2000 BK Ca affinity)
G_KCA_NS_DEFAULT = 2.0     # nS; Phase 1.5 sensitivity-checks this on AVA
E_K_MV = -90.0             # standard K+ reversal


# ---------------------------------------------------------------------
# Equations per variant
# ---------------------------------------------------------------------

# Variant 0 (base): scaffold's equations with no h, no K_Ca.
# Just m_inf-gated I_Ca; no inactivation, no termination mechanism.
EQS_BASE = """
# Soma
dv_s/dt = (v_rest - v_s)/tau_s + (I_axial + I_syn + I_ext)/C_mem : volt
# Dendrite
dv_d/dt = (v_rest - v_d)/tau_d + (-I_axial + I_ca_eff)/C_mem : volt
# Axial coupling
I_axial = g_ax * (v_d - v_s) : amp
# L-type Ca on dendrite (no h-gating, no K_Ca)
m_inf = 1 / (1 + exp(-(v_d - v_ca_half)/k_ca)) : 1
I_ca_eff = g_ca * m_inf * (e_ca - v_d) : amp
# Per-neuron parameters
tau_s : second
tau_d : second
g_ax : siemens
v_rest : volt
g_ca : siemens
e_ca : volt
v_ca_half : volt
k_ca : volt
# External drive
I_ext : amp
I_syn : amp
"""


# Variant A (h_only): adds h-inactivation as direct multiplicative gate.
# Phase 0 predicts this fails — h_ss = 0.3/(0.3+m_inf) ≈ 0.30 leaves 70%
# uninactivated. Documented-failure reference.
EQS_H_ONLY = """
dv_s/dt = (v_rest - v_s)/tau_s + (I_axial + I_syn + I_ext)/C_mem : volt
dv_d/dt = (v_rest - v_d)/tau_d + (-I_axial + I_ca_eff)/C_mem : volt
I_axial = g_ax * (v_d - v_s) : amp
m_inf = 1 / (1 + exp(-(v_d - v_ca_half)/k_ca)) : 1
# h as direct multiplicative gate (matches Session 2's pattern):
I_ca_eff = g_ca * m_inf * h * (e_ca - v_d) : amp
# Scaffold's existing h equation form (hardcoded 0.3 ratio):
dh/dt = (1 - h)/tau_h - (m_inf * h)/(tau_h * 0.3) : 1
tau_s : second
tau_d : second
g_ax : siemens
v_rest : volt
g_ca : siemens
e_ca : volt
v_ca_half : volt
k_ca : volt
tau_h : second
I_ext : amp
I_syn : amp
"""


# Variant B (h_kca): adds h + Ca pool + I_KCa on the dendrite.
# Tests whether compartmentalization (dendrite-soma isolation) allows
# K_Ca to provide plateau termination without the K_Ca dominance issue
# that Session 2 surfaced in single-compartment graded mode.
EQS_H_KCA = """
dv_s/dt = (v_rest - v_s)/tau_s + (I_axial + I_syn + I_ext)/C_mem : volt
dv_d/dt = (v_rest - v_d)/tau_d + (-I_axial + I_ca_eff + I_kca)/C_mem : volt
I_axial = g_ax * (v_d - v_s) : amp
m_inf = 1 / (1 + exp(-(v_d - v_ca_half)/k_ca)) : 1
I_ca_eff = g_ca * m_inf * h * (e_ca - v_d) : amp
dh/dt = (1 - h)/tau_h - (m_inf * h)/(tau_h * 0.3) : 1
# Ca pool on the dendrite (Session 3 sign fix: +alpha_Ca, inward Ca → [Ca] up)
dCa_int/dt = alpha_Ca * I_ca_eff - Ca_int/tau_Ca_decay : 1
# Hill activation (n=4 BK-class)
f_Ca = Ca_int**n_Hill / (K_d_Hill**n_Hill + Ca_int**n_Hill) : 1
# K_Ca current on the dendrite
I_kca = g_kca * f_Ca * (E_K - v_d) : amp
tau_s : second
tau_d : second
g_ax : siemens
v_rest : volt
g_ca : siemens
e_ca : volt
v_ca_half : volt
k_ca : volt
tau_h : second
g_kca : siemens
I_ext : amp
I_syn : amp
"""


# ---------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------


def build_compartmental_kca_group(
    variant: str,
    cells: list[str] | None = None,
    g_kca_override: dict[str, float] | None = None,
    v_rest_mv: float = V_REST_MELLEM_MV,
    C_mem_pf: float = 50.0,
):
    """Construct a Brian2 NeuronGroup for the chosen variant.

    Parameters
    ----------
    variant : 'base' | 'h_only' | 'h_kca'
    cells : list of neuron names from COMPARTMENTAL_ROSTER. Default = all 15.
    g_kca_override : dict of name → g_KCa in nS, only used for 'h_kca' variant.
        Default: G_KCA_NS_DEFAULT for plateau-equipped cells, 0 otherwise.
    v_rest_mv : per-neuron resting potential override. Default -25 (Mellem).
    C_mem_pf : membrane capacitance in pF. Default 50.

    Returns
    -------
    (group, names) — Brian2 NeuronGroup + ordered list of cell names.
    """
    if variant not in ("base", "h_only", "h_kca"):
        raise ValueError(f"Unknown variant: {variant}")

    if cells is None:
        names = list(COMPARTMENTAL_ROSTER.keys())
    else:
        names = list(cells)
        for n in names:
            if n not in COMPARTMENTAL_ROSTER:
                raise KeyError(f"{n} not in COMPARTMENTAL_ROSTER")
    N = len(names)

    eqs = {"base": EQS_BASE, "h_only": EQS_H_ONLY, "h_kca": EQS_H_KCA}[variant]

    # Namespace for K_Ca params (used in EQS_H_KCA only, but harmless in others)
    ns = {
        "alpha_Ca": ALPHA_CA / (pA * ms),
        "tau_Ca_decay": TAU_CA_DECAY_MS * ms,
        "n_Hill": N_HILL,
        "K_d_Hill": K_D_KCA,
        "E_K": E_K_MV * mV,
    }

    grp = NeuronGroup(
        N,
        model=eqs,
        method="exponential_euler",
        namespace={"C_mem": C_mem_pf * pF, **ns},
        name=f"compartmental_kca_{variant}",
    )

    # Initial conditions
    grp.v_s = v_rest_mv * mV
    grp.v_d = v_rest_mv * mV
    if variant in ("h_only", "h_kca"):
        grp.h = 1.0
    if variant == "h_kca":
        grp.Ca_int = 0.0

    # Per-neuron parameters from the roster, with v_rest override
    for i, nm in enumerate(names):
        p = COMPARTMENTAL_ROSTER[nm]
        grp.tau_s[i] = p.soma_tau_ms * ms
        grp.tau_d[i] = p.dend_tau_ms * ms
        grp.g_ax[i] = p.g_axial_ns * nS
        grp.v_rest[i] = v_rest_mv * mV          # OVERRIDE: Mellem -25 mV
        grp.g_ca[i] = (p.g_ca_ns if p.has_plateau else 0.0) * nS
        grp.e_ca[i] = p.e_ca_mv * mV
        grp.v_ca_half[i] = p.v_ca_half_mv * mV
        grp.k_ca[i] = 6 * mV                     # scaffold's k_ca default
        if variant in ("h_only", "h_kca"):
            grp.tau_h[i] = p.plateau_tau_ms * ms
        if variant == "h_kca":
            # Default g_kca: G_KCA_NS_DEFAULT for plateau cells, 0 for non-plateau
            if g_kca_override is not None and nm in g_kca_override:
                g_kca_val = g_kca_override[nm]
            else:
                g_kca_val = G_KCA_NS_DEFAULT if p.has_plateau else 0.0
            grp.g_kca[i] = g_kca_val * nS

    return grp, names


# ---------------------------------------------------------------------
# Mellem-protocol single-cell test runner
# ---------------------------------------------------------------------


def run_mellem_protocol(
    cell_name: str,
    variant: str,
    inject_pa: float = 50.0,
    inject_ms: float = 100.0,
    settle_ms: float = 200.0,
    post_ms: float = 1500.0,
    g_kca_override: dict[str, float] | None = None,
    v_rest_mv: float = V_REST_MELLEM_MV,
    record_dt_ms: float = 1.0,
):
    """Run Mellem 2008 current-injection protocol on one cell × variant.

    Returns
    -------
    dict with keys:
      - cell, variant, inject_pa, v_rest_mv, g_kca_ns
      - v_s_settle_mv, v_d_settle_mv
      - v_s_peak_mv, v_d_peak_mv (during injection window)
      - v_s_post_mv, v_d_post_mv (at end of post-window)
      - plateau_amplitude_mv (v_d_peak - v_d_settle)
      - plateau_duration_ms (time from injection-release to v_d within 5mV of settle)
      - h_min, Ca_max (if applicable)
      - settled (bool: did v_d return to baseline ±5mV by end of post-window)
      - traces (dict of state variable arrays for plotting if needed)
    """
    start_scope()
    grp, names = build_compartmental_kca_group(
        variant, cells=[cell_name],
        g_kca_override=g_kca_override,
        v_rest_mv=v_rest_mv,
    )

    record_vars = ["v_s", "v_d", "I_ca_eff"]
    if variant in ("h_only", "h_kca"):
        record_vars.append("h")
    if variant == "h_kca":
        record_vars.extend(["Ca_int", "f_Ca", "I_kca"])

    mon = StateMonitor(grp, record_vars, record=[0], dt=record_dt_ms * ms)
    net = Network(grp, mon)

    # Phase 1: settle
    net.run(settle_ms * ms)
    v_s_settle = float(grp.v_s[0] / mV)
    v_d_settle = float(grp.v_d[0] / mV)

    # Phase 2: inject
    grp.I_ext[0] = inject_pa * pA
    net.run(inject_ms * ms)
    v_s_at_release = float(grp.v_s[0] / mV)
    v_d_at_release = float(grp.v_d[0] / mV)

    # Phase 3: post-injection observation
    grp.I_ext[0] = 0 * pA
    net.run(post_ms * ms)
    v_s_post = float(grp.v_s[0] / mV)
    v_d_post = float(grp.v_d[0] / mV)

    # Extract trace metrics from monitor
    t_arr = mon.t / ms
    v_s_trace = mon.v_s[0] / mV
    v_d_trace = mon.v_d[0] / mV

    # Peak v_d during injection window
    inject_mask = (t_arr >= settle_ms) & (t_arr < settle_ms + inject_ms)
    if inject_mask.any():
        v_s_peak = float(np.max(v_s_trace[inject_mask]))
        v_d_peak = float(np.max(v_d_trace[inject_mask]))
    else:
        v_s_peak = v_s_at_release
        v_d_peak = v_d_at_release

    # Plateau amplitude
    plateau_amp = v_d_peak - v_d_settle

    # Plateau duration: time from injection-release to v_d within 5 mV of settle
    post_mask = t_arr >= (settle_ms + inject_ms)
    post_v_d = v_d_trace[post_mask]
    post_t = t_arr[post_mask]
    plateau_dur = None
    for k, (t_k, v_k) in enumerate(zip(post_t, post_v_d)):
        if abs(v_k - v_d_settle) <= 5.0:
            plateau_dur = float(t_k - (settle_ms + inject_ms))
            break
    if plateau_dur is None:
        plateau_dur = float(post_ms)  # never settled within window
        settled = False
    else:
        settled = True

    g_kca_ns = 0.0
    if variant == "h_kca":
        g_kca_ns = float(grp.g_kca[0] / nS)

    result = {
        "cell": cell_name,
        "variant": variant,
        "inject_pa": inject_pa,
        "v_rest_mv": v_rest_mv,
        "g_kca_ns": g_kca_ns,
        "v_s_settle_mv": v_s_settle,
        "v_d_settle_mv": v_d_settle,
        "v_s_peak_mv": v_s_peak,
        "v_d_peak_mv": v_d_peak,
        "v_s_post_mv": v_s_post,
        "v_d_post_mv": v_d_post,
        "plateau_amplitude_mv": plateau_amp,
        "plateau_duration_ms": plateau_dur,
        "settled_within_window": settled,
    }

    if variant in ("h_only", "h_kca"):
        h_trace = mon.h[0]
        result["h_min"] = float(np.min(h_trace))
        result["h_final"] = float(h_trace[-1])
    if variant == "h_kca":
        Ca_trace = mon.Ca_int[0]
        f_Ca_trace = mon.f_Ca[0]
        result["Ca_max"] = float(np.max(Ca_trace))
        result["Ca_final"] = float(Ca_trace[-1])
        result["f_Ca_max"] = float(np.max(f_Ca_trace))

    return result


# ---------------------------------------------------------------------
# Smoke test — verify equations parse and run
# ---------------------------------------------------------------------


def _smoke_test():
    print("=" * 76)
    print("compartmental_neurons_kca smoke test")
    print("=" * 76)
    print(f"v_rest = {V_REST_MELLEM_MV} mV (Mellem AVA up-state)")
    print(f"alpha_Ca = {ALPHA_CA} dimensionless-µM/(pA·ms)")
    print(f"K_d = {K_D_KCA} µM, n_Hill = {N_HILL}, g_KCa = {G_KCA_NS_DEFAULT} nS")
    print()

    for variant in ("base", "h_only", "h_kca"):
        print(f"--- AVAL × {variant} (50 pA × 100 ms protocol) ---")
        r = run_mellem_protocol("AVAL", variant)
        print(f"  v_s settle/peak/post = {r['v_s_settle_mv']:+.1f} / {r['v_s_peak_mv']:+.1f} / {r['v_s_post_mv']:+.1f} mV")
        print(f"  v_d settle/peak/post = {r['v_d_settle_mv']:+.1f} / {r['v_d_peak_mv']:+.1f} / {r['v_d_post_mv']:+.1f} mV")
        print(f"  plateau_amp = {r['plateau_amplitude_mv']:+.1f} mV   plateau_dur = {r['plateau_duration_ms']:.0f} ms   settled? {r['settled_within_window']}")
        if "h_final" in r:
            print(f"  h_min/final = {r['h_min']:.3f} / {r['h_final']:.3f}")
        if "Ca_max" in r:
            print(f"  [Ca]_max/final = {r['Ca_max']:.3f} / {r['Ca_final']:.3f} µM   f_Ca_max = {r['f_Ca_max']:.3f}")
        print()


if __name__ == "__main__":
    _smoke_test()
