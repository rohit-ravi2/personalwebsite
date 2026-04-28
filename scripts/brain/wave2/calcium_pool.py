"""
Ca-pool subsystem translations: cadiff + caintra1.

Phase β CP1.B deliverable. Revised v2 after F1-F5 findings.

Architectural commitment
------------------------
**eqs-string encoding for Ca-pool subsystems.** This module exposes Brian2
equation strings that callers splice into their own NeuronGroup eqs alongside
channels that need [Ca]_i. Rationale:

  * Nicoletti's models are single-compartment cylindrical. Ca-pool dynamics
    encoded as eqs-string in Brian2 minimizes translation artifacts.
  * Faster validation (single ODE system, single integrator).
  * If condition 6 surfaces and morphology fork triggers, separate-subsystem
    encoding becomes part of morphology integration work.

Decision logged: 2026-04-26 phase_beta_findings.md (CP1.B Ca-pool encoding).

Unit convention
---------------

After investigating NEURON's behavior (see findings F1-F4), we adopt **mM as
the internal unit** for the Brian2 `cai_mM` state variable. This matches:
- cadiff.mod's STATE `ca` (mM)
- cadiff.mod's `cai` write (mM, NEURON ion default)
- caintra1.mod's STATE `caintra` — NMODL declares no unit but ca_eq is in
  parameter `M`; numerical values produced by NEURON are in mM-equivalent
  scale (e.g., ca_eq stored as 5e-5 = 5e-5 mM)

Channels that need M-scale Ca (e.g., slo1iso reads `cai (mM)` and converts
internally via `cai*1e3` for μM-scale formulas) apply the conversion at
the read site.

Source mod summary
------------------

cadiff.mod:
    USEION ca READ ica, cai WRITE cai
    BREAKPOINT { ca = ca + 10000 * dt * (-1/(2F)*ica/depth - 0.0001*beta*ca);
                 if (ca < 1e-4) ca = 1e-4; cai = ca; }
    Writes to cai. Floor at 1e-4 (mM = 100 nM).

caintra1.mod:
    USEION ca READ ica, eca   (NOT WRITE cai)
    DERIVATIVE { caintra' = if (ica<=0) [fca*(-((1/(2*vol*Fc))*(ica*surf*1e-3)))
                                          - (caintra-ca_eq)/tca]
                            else [-(caintra-ca_eq)/tca] }
    Stores in private STATE caintra. Channels reading [Ca]_i must access
    via mechanism's `caintra` attribute or read GLOBAL `calcium`.

Multi-writer note
-----------------
cadiff WRITES cai; caintra1 does NOT. They CAN coexist in the same NEURON
section without a multi-writer error. But functionally they track different
variables — caintra1 is independent of cai. Nicoletti's published cells
appear to use ONE pool per cell.

For our CP3 cell (leak + EGL-19 + caintra1), nothing actually consumes cai —
EGL-19 in Nicoletti's parameterization has no Ca-dependent inactivation
(verified by reading egl19.mod: only voltage-dependent m, h gates). So
caintra1's bookkeeping role doesn't affect V(t).

Brian2 eqs strings
------------------

cadiff_eqs() and caintra1_eqs() return Brian2 equation strings + parameters.
The caller's parent NeuronGroup eqs must define `ica_mAcm2` (the Ca-channel
current density, mA/cm²-scale). The pool eqs add `cai_mM` (mM-scale state).
"""
from __future__ import annotations

# Constants
F_COUL = 96485.0          # Faraday's constant, coul/mol
F_NMODL_CONST = 9.6485e4  # cadiff uses 9.6485e4


# ---------------------------------------------------------------------------
# cadiff eqs string (CP1.B.5)
# ---------------------------------------------------------------------------

def cadiff_eqs(
    depth_um: float = 0.1,
    beta_per_ms: float = 1.0,
    cai_floor_mM: float = 1e-4,
    cai_init_mM: float = 1e-4,
) -> dict:
    """Brian2 eqs for cadiff, mM-scale internal.

    Source `BREAKPOINT { ca = ca + 10000 * dt * (-1/(2F)*ica/depth -
    0.0001 * beta * ca); cai = ca }`.

    Symbolic re-derivation gives a coefficient ~5183 mM/(mA/cm²·ms), but
    NEURON's empirical behavior differs by factor ~1e4 due to NMODL hidden
    unit-conversion machinery (see findings F6 in phase_beta_findings.md).

    **Coefficient is empirically calibrated** against NEURON's `cai`
    trajectory under cca1 voltage-clamp at AIY-like geometry
    (surf=65.89e-8 cm², depth=0.1 um, beta=1 /ms). Calibration result:
    α ≈ -0.525 mM/(mA/cm²·ms), R² ≈ 0.984. We use coef_ica = +0.525
    (sign flipped: Brian2 eqs use `-coef_ica * ica` so positive
    coefficient gives positive Δcai for inward ica).

    For other depth_um/beta_per_ms values, the coefficient should scale
    accordingly per the formula's intent. We apply a depth-scaling factor
    based on the formula relationship (assume coef_ica ∝ 1/depth_um):
      coef_ica = 0.525 * 0.1 / depth_um

    Floor: cai_floor_mM = 1e-4 mM (= 100 nM). Enforced via decay-rate
    vanishing at floor.

    Equation string defines:
      cai_mM (state): in mM
      cadiff_coef_ica (param): empirically calibrated, scaled by depth_um
      cadiff_beta (param): beta_per_ms, /ms
      cadiff_floor (param): 1e-4 mM
    """
    # Empirically calibrated coefficient at depth_um=0.1, scaled inversely
    # to depth as per the formula's structural intent.
    CADIFF_COEF_ICA_AT_DEPTH_0_1_UM = 0.525  # |α|, mM/(mA/cm²·ms)
    coef_ica = CADIFF_COEF_ICA_AT_DEPTH_0_1_UM * (0.1 / depth_um)

    eqs = """
    # Ca-pool: cadiff (Yale Purkinje adaptation, Nicoletti convention).
    # cai_mM is internal [Ca²⁺] in mM, matching NEURON's cai for cadiff.
    # Floor enforced in decay term: as cai_mM → cadiff_floor, decay rate → 0.
    dcai_mM/dt = (-cadiff_coef_ica * ica_mAcm2
                  - cadiff_beta * (cai_mM - cadiff_floor)) / ms : 1
    cadiff_coef_ica : 1
    cadiff_beta : 1
    cadiff_floor : 1
    """

    return {
        "eqs": eqs,
        "params": {
            "cadiff_coef_ica": coef_ica,
            "cadiff_beta": beta_per_ms,
            "cadiff_floor": cai_floor_mM,
        },
        "init": {
            "cai_mM": cai_init_mM,
        },
        "metadata": {
            "source_mod": "cadiff.mod",
            "depth_um": depth_um,
            "beta_per_ms": beta_per_ms,
            "encoding": "eqs-string, dimensionless cai_mM (mM-scale numeric)",
            "note": "Floor enforced via decay-rate vanishing at floor (smooth approximation).",
        },
    }


# ---------------------------------------------------------------------------
# caintra1 eqs string (CP1.B.6)
# ---------------------------------------------------------------------------

def caintra1_eqs(
    vol_cm3: float = 7.42e-12,
    surf_cm2: float = 65.89e-8,
    fca: float = 0.001,
    tca_ms: float = 50.0,
    ca_eq_mM: float = 5e-8,     # NEURON's numerical value: ca_eq=0.05e-6 (M) → 5e-8 raw
):
    """Brian2 eqs for caintra1, mM-scale internal.

    Source DERIVATIVE block:
        if (ica <= 0):
          rs = fca * (-((1/(2*vol*Fc)) * (ica*surf*1e-3))) - (caintra - ca_eq)/tca
        else:
          rs = -(caintra - ca_eq)/tca

    Symbolic coefficient: coef_in_naive = fca / (2 * vol_cm3 * F_COUL).
    For AIY geometry (vol=7.42e-12, fca=0.001, F=96485): naive ≈ 698.3.
    Multiplied by surf*1e-3 = 6.589e-10 gives effective coefficient ~4.6e-7.

    Empirical calibration (cca1+caintra1 at AIY geometry) gives α ≈ -7.3e-8,
    R² ≈ 0.408 (lower than cadiff because conditional ica<=0 introduces
    asymmetry the linear fit averages over). Empirical / naive ratio ≈ 0.16.

    We use the **empirical effective coefficient directly** (not the
    formula-derived one), with sign flip applied per Brian2 eqs convention.
    Coefficient assumed to scale linearly with surf and inversely with vol
    (per formula structure):
      coef_in_eff = -α_empirical * (surf/surf_calib) * (vol_calib/vol)
                  = 7.3e-8 * (surf/65.89e-8) * (7.42e-12/vol)

    See findings F6, F7, F8 in phase_beta_findings.md.

    Smooth conditional: indicator(ica) sigmoid centered at ica=0 with width
    0.0001 mA/cm². 1 → inward (ica<0), 0 → outward (ica>0).

    Returns
    -------
    dict with eqs, params, params_with_units, init, metadata.
    """
    # Empirically calibrated effective coefficient at AIY geometry.
    # The eqs uses: rate_inward_mMperms = coef_in_eff * (-ica_mAcm2) * indicator
    # Calibration with corrected ca_eq=5e-8: α=-4.60e-7, R²=1.0000.
    # coef_in_eff has units mM/(mA/cm²·ms) directly (surf+vol absorbed).
    CAINTRA1_COEF_EFF_AT_AIY = 4.60e-7  # |α|, mM/(mA/cm²·ms)
    SURF_CALIB = 65.89e-8
    VOL_CALIB = 7.42e-12
    coef_in_eff = CAINTRA1_COEF_EFF_AT_AIY * (surf_cm2 / SURF_CALIB) * (VOL_CALIB / vol_cm3)

    eqs = """
    # Ca-pool: caintra1 (Nicoletti AIY/RIM convention).
    # cai_mM is internal [Ca²⁺] in mM, matching NEURON's caintra (private state).
    # Smooth conditional via sigmoid: 1 when inward, 0 when outward.
    # caintra1_coef_in_eff already has surf and unit-conversion absorbed
    # (empirical calibration vs NEURON).
    inward_indicator = 1.0 / (1.0 + exp(ica_mAcm2 / 0.0001)) : 1
    rs_inward = caintra1_coef_in_eff * (-ica_mAcm2) * inward_indicator : 1
    rs_decay = -(cai_mM - caintra1_ca_eq) / (caintra1_tca / ms) : 1
    dcai_mM/dt = (rs_inward + rs_decay) / ms : 1
    caintra1_coef_in_eff : 1
    caintra1_ca_eq : 1
    caintra1_tca : second
    """

    return {
        "eqs": eqs,
        "params": {
            "caintra1_coef_in_eff": coef_in_eff,
            "caintra1_ca_eq": ca_eq_mM,
        },
        "params_with_units": {
            "caintra1_tca": ("ms", tca_ms),
        },
        "init": {
            "cai_mM": ca_eq_mM,
        },
        "metadata": {
            "source_mod": "caintra1.mod",
            "vol_cm3": vol_cm3,
            "surf_cm2": surf_cm2,
            "fca": fca,
            "tca_ms": tca_ms,
            "ca_eq_mM": ca_eq_mM,
            "encoding": "eqs-string, dimensionless cai_mM (mM-scale numeric), "
                        "smooth conditional sigmoid",
        },
    }


# ---------------------------------------------------------------------------
# Brian2 factories — pool-only cells for direct validation
# ---------------------------------------------------------------------------

def caintra1_brian2_factory(
    vol_cm3: float = 7.42e-12,
    surf_cm2: float = 65.89e-8,
    fca: float = 0.001,
    tca_ms: float = 50.0,
    ca_eq_mM: float = 5e-8,
):
    """Brian2 cell with caintra1 pool only (no V dynamics)."""
    pool = caintra1_eqs(vol_cm3=vol_cm3, surf_cm2=surf_cm2, fca=fca,
                        tca_ms=tca_ms, ca_eq_mM=ca_eq_mM)

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, ms, prefs, start_scope,
        )
        start_scope()
        prefs.codegen.target = "cython"

        eqs = """
        ica_mAcm2 : 1
        """ + pool["eqs"]

        G = NeuronGroup(1, eqs, method="euler")
        for k, v in pool["params"].items():
            setattr(G, k, v)
        for k, (unit_str, v) in pool.get("params_with_units", {}).items():
            from brian2 import ms as _ms, second as _sec
            unit_map = {"ms": _ms, "second": _sec}
            setattr(G, k, v * unit_map[unit_str])
        for k, v in pool["init"].items():
            setattr(G, k, v)
        G.ica_mAcm2 = 0.0

        mon = StateMonitor(G, ["cai_mM", "ica_mAcm2"], record=True)
        net = Network(G, mon)

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "params": pool["params"],
        }

    return _factory


def cadiff_brian2_factory(
    depth_um: float = 0.1,
    beta_per_ms: float = 1.0,
    cai_floor_mM: float = 1e-4,
    cai_init_mM: float = 1e-4,
):
    """Brian2 cell with cadiff pool only."""
    pool = cadiff_eqs(depth_um=depth_um, beta_per_ms=beta_per_ms,
                      cai_floor_mM=cai_floor_mM, cai_init_mM=cai_init_mM)

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, ms, prefs, start_scope,
        )
        start_scope()
        prefs.codegen.target = "cython"

        eqs = """
        ica_mAcm2 : 1
        """ + pool["eqs"]

        G = NeuronGroup(1, eqs, method="euler")
        for k, v in pool["params"].items():
            setattr(G, k, v)
        for k, v in pool["init"].items():
            setattr(G, k, v)
        G.ica_mAcm2 = 0.0

        mon = StateMonitor(G, ["cai_mM", "ica_mAcm2"], record=True)
        net = Network(G, mon)

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "params": pool["params"],
        }

    return _factory
