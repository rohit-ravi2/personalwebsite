"""
AVA cell with dynamic [Ca]_i (caintra1 pool) and Ca-coupled SLO-1 isolated.

Phase F follow-on after density-sensitivity sweep produced
VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS (see
`artifacts/density_sensitivity_analysis.md`). This module rebuilds the AVA
cell from `validate_phase_f_gate2.build_brian2_ava_2b` with one architectural
change: the dynamic Ca-pool `caintra1` is wired in, EGL-19's I_Ca drives it,
and SLO-1 isolated reads its dynamic [Ca]_i (instead of static cai = 5e-5 mM).

Architecture decisions (documented in companion artifact)
---------------------------------------------------------
1. **caintra1 (not cadiff)** because Nicoletti pairs caintra1 with slo1iso in
   her AIY/RIM models when she does pair them. cadiff is a Yale Purkinje
   adaptation used by VA5; it would be a non-Nicoletti choice for AVA.

2. **AVA-scaled vol/surf** — caintra1's calibration (in `calcium_pool.py`)
   uses AIY geometry (vol=7.42e-12 cm³, surf=65.89e-8 cm²) but applies a
   linear scaling for other geometries. AVA: vol=129.6e-12 cm³,
   surf=1123.84e-8 cm². The scaled effective coefficient is computed in
   `caintra1_eqs()`.

3. **SLO-1+EGL-19 coupled keeps closed-form `calcium(V)` (Option A)** —
   Nicoletti's published encoding of the coupled variant uses a
   nanodomain Ca formula (Lluís-Buchholz / Alvarez), not bulk Ca. Replacing
   that with bulk dynamic [Ca]_i would (a) be a different biophysical claim
   (nanodomain ≠ bulk), and (b) confound the test — we want to isolate the
   effect of *adding the SLO-1-isolated Ca-feedback loop* per the F12
   mechanism diagnosis, not change two things at once. Option A preserves
   slo1egl19 as-is and only modifies slo1iso.

4. **Conductances unchanged from Phase F 2b baseline** — same g vector
   that produced 46.8 mV / 21.4 ms before. Any phenotype change is then
   attributable to the Ca-coupling change (clean controlled comparison).

5. **EGL-19 → caintra1 wiring** — the parent eqs sums all channel ica into
   `ica_mAcm2` and the pool reads that. In our cell, only EGL-19 contributes
   to ica (NCA's current is treated as `ik_nca_mAcm2` in the existing code,
   not Ca-carrying). So `ica_mAcm2 = ica_egl19_mAcm2` exactly as in the
   static-cai cell.

Builder
-------
`build_brian2_ava_ca_coupled(...)` mirrors the signature of
`validate_phase_f_gate2.build_brian2_ava_2b` and returns a factory closure
producing a Brian2 NeuronGroup + StateMonitor + Network bundle.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure wave2/ is on sys.path so submodules import cleanly whether this is
# run as a script or imported.
_WAVE2_DIR = Path(__file__).resolve().parent
if str(_WAVE2_DIR) not in sys.path:
    sys.path.insert(0, str(_WAVE2_DIR))

import numpy as np

from calcium_pool import caintra1_eqs

from channels import egl19 as egl19_mod
from channels import nca as nca_mod
from channels import shk1 as shk1_mod
from channels import shl1 as shl1_mod
from channels import kqt3 as kqt3_mod
from channels import slo1_iso_dynamic_ca as slo1iso_dyn_mod
from channels import slo1_egl19_coupled as slo1egl19_mod


# AVA geometry (Nicoletti AVAL_simulations.py, AVAL_simulation_iclamp.py)
AVA_SURF_CM2 = 1123.84e-8
AVA_VOL_CM3 = 129.6e-12
AVA_CM_UFCM2 = 0.859551
AVA_E_LEAK_MV = -39.0


def build_brian2_ava_ca_coupled(
    surf_cm2: float = AVA_SURF_CM2,
    vol_cm3: float = AVA_VOL_CM3,
    cm_uFcm2: float = AVA_CM_UFCM2,
    g_leak_Scm2: float = 0.150164e-9 / AVA_SURF_CM2,
    e_leak_mV: float = AVA_E_LEAK_MV,
    g_egl19_Scm2: float = 0.104385e-9 / AVA_SURF_CM2,
    g_nca_Scm2: float = 0.0,
    g_slo1iso_Scm2: float = 1.0e-9 / 65.89e-8,         # AIY-derived intensive density
    g_slo1egl19_Scm2: float = 0.92e-9 / 65.89e-8,
    g_shl1_Scm2: float = 0.5e-9 / 65.89e-8,
    g_shk1_Scm2: float = 1e-4,
    g_kqt3_Scm2: float = 1e-4,
    eca_mV: float = 60.0,
    ek_mV: float = -80.0,
    v_init_mV: float = -60.0,
    # Ca-pool (caintra1) parameters — Nicoletti defaults from caintra1.mod,
    # geometry scaled to AVA.
    caintra1_fca: float = 0.001,
    caintra1_tca_ms: float = 50.0,
    # caintra1's NMODL declares ca_eq = 0.05e-6 (M). NEURON stores raw 5e-8.
    # We pass that raw 5e-8 to the pool eqs (which is calibrated in those units).
    caintra1_ca_eq_raw: float = 5e-8,
    # Unit-conversion factor: caintra (raw NMODL state, declared "M") → mM that
    # slo1iso's formula expects. Since 1 M = 1000 mM, factor = 1000.
    # (Documented finding from this work block; see ca_coupling_test_results.md.)
    caintra_to_mM_scale: float = 1000.0,
    method: str = "rk4",
):
    """Build factory for AVA cell with dynamic caintra1 + Ca-coupled SLO-1 iso.

    Returns a zero-arg callable that, when called, performs `start_scope()`,
    constructs the NeuronGroup + StateMonitor + Network, applies parameters,
    and returns a bundle dict with keys: `group`, `monitor`, `network`, and
    `pool_meta`.
    """
    # Pool produces a state variable that the calibration named `cai_mM` but is
    # numerically in raw NMODL units (caintra1's "M" declaration; numeric 5e-8
    # at rest). To feed slo1iso (which expects cai in mM, ie. multiplies by 1e3
    # internally to get μM), we rename the pool's state to `caintra_raw` and
    # define `cai_mM = caintra_raw * caintra_to_mM_scale`. With scale=1000,
    # caintra=5e-8 (raw) → cai_mM = 5e-5 (mM, ≡ 50 nM ≡ NEURON's cai default).
    pool = caintra1_eqs(
        vol_cm3=vol_cm3,
        surf_cm2=surf_cm2,
        fca=caintra1_fca,
        tca_ms=caintra1_tca_ms,
        ca_eq_mM=caintra1_ca_eq_raw,
    )
    # Note: `caintra1_eqs()` uses an empirical calibration coefficient that
    # absorbs fca = 0.001 (the AIY calibration value). It does NOT rescale the
    # coefficient when callers pass a different `fca`. Per the NMODL formula,
    # rate_inward is linear in fca, so we apply the scaling explicitly:
    #     coef_in_eff_scaled = coef_in_eff_baseline * (fca / 0.001)
    # This keeps `calcium_pool.py` untouched while letting the cell builder
    # honor the `fca` axis. (Documented finding from this work block.)
    FCA_CALIB = 0.001
    pool_params = dict(pool["params"])
    pool_params["caintra1_coef_in_eff"] = (
        pool_params["caintra1_coef_in_eff"] * (caintra1_fca / FCA_CALIB)
    )
    # Rename the pool's `cai_mM` differential variable to `caintra_raw` so we
    # can reserve `cai_mM` as the *exposed* (mM-scale) value for channels.
    pool_eqs_renamed = pool["eqs"].replace("cai_mM", "caintra_raw")

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network,
            ms, mV, prefs, start_scope, pA, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        # Cell-level eqs. Mirrors validate_phase_f_gate2.build_brian2_ava_2b
        # except:
        #  - splices the pool eqs (defines `cai_mM` as a state)
        #  - uses SLO1_ISO_DYNAMIC_CA_EQS (does NOT declare cai_mM as parameter)
        #  - keeps slo1egl19's closed-form calcium(V) (Option A)
        cell_eqs = f"""
        v_mV = v / mV : 1
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        # K-channels: shk1, shl1, kqt3, slo1iso (Ca-dynamic), slo1egl19 (V-only).
        ik_total_mAcm2 = ik_shk1_mAcm2 + ik_shl1_mAcm2 + ik_kqt3_mAcm2 + ik_slo1iso_mAcm2 + ik_slo1egl19_mAcm2 : 1
        # Non-specific leak and nca:
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_nca_mAcm2 = ik_nca_mAcm2 : 1
        i_total_mAcm2 = i_leak_mAcm2 + i_nca_mAcm2 + ica_mAcm2 + ik_total_mAcm2 : 1
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA - I_inj : amp
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        I_inj : amp
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        # cai_mM exposed to channels = pool's raw caintra * scale.
        # caintra (NMODL state, declared "M") at rest is 5e-8 raw → cai_mM = 5e-5
        # (= 50 nM, matching NEURON's default). See ca_coupling_test_results.md.
        cai_mM = caintra_raw * {caintra_to_mM_scale} : 1
        """ \
            + pool_eqs_renamed \
            + egl19_mod.EGL19_EQS \
            + nca_mod.NCA_EQS \
            + shk1_mod.SHK1_EQS \
            + shl1_mod.SHL1_EQS \
            + kqt3_mod.KQT3_EQS \
            + slo1iso_dyn_mod.SLO1_ISO_DYNAMIC_CA_EQS \
            + slo1egl19_mod.SLO1_EGL19_EQS

        G = NeuronGroup(1, cell_eqs, method=method)

        # Cell-level parameters
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV
        G.I_inj = 0 * pA

        # Ca-pool (caintra1) parameters and init.
        # Note pool dict has init key "cai_mM" by historical name — we mapped
        # the eqs to `caintra_raw`, so init must go to `caintra_raw` instead.
        # We use the locally-rescaled `pool_params` (fca-aware) instead of
        # `pool["params"]` directly.
        for k, v in pool_params.items():
            setattr(G, k, v)
        for k, (unit_str, v) in pool.get("params_with_units", {}).items():
            from brian2 import ms as _ms, second as _sec
            unit_map = {"ms": _ms, "second": _sec}
            setattr(G, k, v * unit_map[unit_str])
        for k, v in pool["init"].items():
            target = "caintra_raw" if k == "cai_mM" else k
            setattr(G, target, v)

        # Channel parameters and init
        egl19_mod.egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=eca_mV)
        egl19_mod.egl19_init_states(G, v_mV=v_init_mV)
        nca_mod.nca_apply_params(G, gbar_Scm2=g_nca_Scm2)
        # nca has no SS init function in Phase F's pattern; no-op safe.
        shk1_mod.shk1_apply_params(G, gbar_Scm2=g_shk1_Scm2, ek_mV=ek_mV)
        shk1_mod.shk1_init_states(G, v_mV=v_init_mV)
        shl1_mod.shl1_apply_params(G, gbar_Scm2=g_shl1_Scm2, ek_mV=ek_mV)
        shl1_mod.shl1_init_states(G, v_mV=v_init_mV)
        kqt3_mod.kqt3_apply_params(G, gbar_Scm2=g_kqt3_Scm2, ek_mV=ek_mV)
        kqt3_mod.kqt3_init_states(G, v_mV=v_init_mV)

        # SLO-1 isolated: Ca-dynamic variant. Init [Ca]_i (mM) = caintra_raw at
        # rest * scale. Default ca_eq raw 5e-8 * 1000 = 5e-5 mM (= 50 nM,
        # NEURON's default cai_ca_ion).
        slo1iso_dyn_mod.slo1iso_dynca_apply_params(
            G, gbar_Scm2=g_slo1iso_Scm2, ek_mV=ek_mV,
        )
        cai_mM_at_rest = caintra1_ca_eq_raw * caintra_to_mM_scale
        slo1iso_dyn_mod.slo1iso_dynca_init_states(
            G, v_mV=v_init_mV, cai_mM_init=cai_mM_at_rest,
        )

        # SLO-1+EGL-19 coupled: kept as closed-form calcium(V) (Option A).
        slo1egl19_mod.slo1egl19_apply_params(G, gbar_Scm2=g_slo1egl19_Scm2, ek_mV=ek_mV)
        slo1egl19_mod.slo1egl19_init_states(G, v_mV=v_init_mV)

        mon = StateMonitor(
            G,
            ["v", "I_total", "I_inj", "cai_mM", "caintra_raw", "ica_egl19_mAcm2",
             "m_slo1iso", "ik_slo1iso_mAcm2", "ik_slo1egl19_mAcm2"],
            record=True,
        )
        net = Network(G, mon)

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "pool_meta": pool["metadata"],
            "pool_params_effective": pool_params,
            "channel_densities": {
                "g_leak_Scm2": g_leak_Scm2,
                "g_egl19_Scm2": g_egl19_Scm2,
                "g_nca_Scm2": g_nca_Scm2,
                "g_slo1iso_Scm2": g_slo1iso_Scm2,
                "g_slo1egl19_Scm2": g_slo1egl19_Scm2,
                "g_shl1_Scm2": g_shl1_Scm2,
                "g_shk1_Scm2": g_shk1_Scm2,
                "g_kqt3_Scm2": g_kqt3_Scm2,
            },
            "geometry": {
                "surf_cm2": surf_cm2,
                "vol_cm3": vol_cm3,
                "cm_uFcm2": cm_uFcm2,
            },
            "ca_pool_settings": {
                "fca": caintra1_fca,
                "tca_ms": caintra1_tca_ms,
                "ca_eq_raw": caintra1_ca_eq_raw,
                "caintra_to_mM_scale": caintra_to_mM_scale,
                "cai_mM_at_rest": caintra1_ca_eq_raw * caintra_to_mM_scale,
            },
        }

    return _factory


def leak_tau_ms(cm_uFcm2: float = AVA_CM_UFCM2,
                g_leak_Scm2: float = 0.150164e-9 / AVA_SURF_CM2) -> float:
    """Pure-leak τ_m for AVA. τ_ms = (cm/g_leak) * 1e-3."""
    return (cm_uFcm2 / g_leak_Scm2) * 1e-3
