"""
Empirically calibrate cadiff and caintra1 Brian2 coefficients against NEURON.

Per finding F6/F7/F8 in phase_beta_findings.md: NMODL hidden unit-conversion
machinery makes symbolic re-derivation of the BREAKPOINT/DERIVATIVE formulas
unreliable. We instead fit the Brian2 coefficient empirically against
NEURON's behavior at known ica regimes.

Calibration strategy
--------------------

For each pool (cadiff, caintra1):

  1. Construct a NEURON section with [cca1 + pool] (cadiff writes cai;
     caintra1 stores in private state).
  2. Voltage-clamp at multiple holding potentials → multiple ica values.
  3. Record (t, ica, cai_state) where cai_state is `_ref_cai` for cadiff
     or `mech._ref_caintra` for caintra1.
  4. Compute, per timestep, the NUMERICAL Δca/Δt (with cai_state in mM
     and Δt in ms).
  5. Linear regression: Δca/Δt = α · ica + β · (cai - eq) + γ
     (γ ≈ 0 if formula is well-formed).
  6. Brian2 eqs use α, β as fitted parameters.

Output: writes calibrated coefficients to JSON for downstream validation.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from neuron_reference import NEURONReference, _nicoletti_env


def collect_trajectory(
    ca_pool: str,
    pool_kwargs: dict,
    holding_potentials: list[float],
    duration_ms: float = 200.0,
    dt_ms: float = 0.025,
    surf_cm2: float = 65.89e-8,
    cm_uFcm2: float = 1.0,
    ca_channel_gbar: float = 0.7,
) -> list[dict]:
    """Return list of per-hold dicts {hold_mV, t, ica, cai}."""
    custom_spec = {
        "channels": ["cca1", ca_pool],
        "params": {("cca1", "gbar"): ca_channel_gbar},
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": 60.0,
        "ek_mV": -80.0,
        "v_init_mV": -60.0,
    }
    for k, v in pool_kwargs.items():
        custom_spec["params"][(ca_pool, k)] = v

    nref = NEURONReference("custom", custom_spec=custom_spec)

    with _nicoletti_env():
        h = nref._h
        soma = nref._soma

        stim = h.VClamp(soma(0.5))
        prestep_ms = 50.0
        stim.dur[0] = prestep_ms
        stim.dur[1] = duration_ms
        stim.dur[2] = 0.0
        stim.amp[0] = -60.0
        stim.amp[2] = -60.0

        t_vec = h.Vector()
        ica_vec = h.Vector()
        cai_vec = h.Vector()
        t_vec.record(h._ref_t)
        ica_vec.record(soma(0.5)._ref_ica)
        if ca_pool == "cadiff":
            cai_vec.record(soma(0.5)._ref_cai)
        elif ca_pool == "caintra1":
            cai_vec.record(getattr(soma(0.5), ca_pool)._ref_caintra)
        else:
            cai_vec.record(soma(0.5)._ref_cai)

        results = []
        for v_hold in holding_potentials:
            stim.amp[1] = float(v_hold)
            h.tstop = prestep_ms + duration_ms
            h.dt = dt_ms
            h.finitialize(-60.0)
            h.run()

            t_full = np.array(t_vec.to_python())
            ica_full = np.array(ica_vec.to_python())
            cai_full = np.array(cai_vec.to_python())

            step_mask = (t_full >= prestep_ms) & (t_full <= prestep_ms + duration_ms)
            t = t_full[step_mask] - prestep_ms
            ica = ica_full[step_mask]
            cai = cai_full[step_mask]

            results.append({
                "hold_mV": float(v_hold),
                "t_ms": t,
                "ica_mAcm2": ica,
                "cai_mM": cai,
            })
    nref.cleanup()
    return results


def fit_pool_coefficients(traj: list[dict], cai_eq: float) -> dict:
    """Fit Δcai/Δt = α · ica + β · (cai - cai_eq) across all trajectory points.

    Uses central differences for Δcai/Δt and trims unreliable endpoints.
    Returns {alpha, beta, r2, n_samples}.
    """
    all_dcai_dt = []
    all_ica = []
    all_cai_dev = []

    for r in traj:
        t = r["t_ms"]
        ica = r["ica_mAcm2"]
        cai = r["cai_mM"]

        if len(t) < 5:
            continue

        # Central differences
        dcai_dt = np.gradient(cai, t)
        # Use only the bulk of the trajectory (skip first/last 5%)
        n = len(t)
        lo = int(0.05 * n)
        hi = int(0.95 * n)
        all_dcai_dt.extend(dcai_dt[lo:hi])
        all_ica.extend(ica[lo:hi])
        all_cai_dev.extend(cai[lo:hi] - cai_eq)

    X = np.column_stack([
        np.array(all_ica),
        np.array(all_cai_dev),
    ])
    y = np.array(all_dcai_dt)

    # Filter NaN/Inf
    mask = np.isfinite(y) & np.isfinite(X[:, 0]) & np.isfinite(X[:, 1])
    X = X[mask]
    y = y[mask]

    # Least squares
    coefs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coefs[0]), float(coefs[1])

    y_pred = X @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "alpha_mMperms_per_mAcm2": alpha,
        "beta_perms_decay": beta,
        "r2": r2,
        "n_samples": int(len(y)),
        "rmse": float(np.sqrt(ss_res / len(y))),
        "ica_range": [float(np.min(X[:, 0])), float(np.max(X[:, 0]))],
        "cai_dev_range": [float(np.min(X[:, 1])), float(np.max(X[:, 1]))],
        "dcai_dt_range": [float(np.min(y)), float(np.max(y))],
    }


def main():
    print("=== CALIBRATION: cadiff (cell: cca1 + cadiff at AIY-like geometry) ===")
    cadiff_traj = collect_trajectory(
        "cadiff",
        pool_kwargs={"depth": 0.1, "beta": 1.0},
        holding_potentials=[-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0],
        duration_ms=200.0,
        dt_ms=0.025,
        surf_cm2=65.89e-8,
        cm_uFcm2=1.0,
    )
    cadiff_fit = fit_pool_coefficients(cadiff_traj, cai_eq=1e-4)
    print(f"  α (mM/(mA/cm²·ms)) = {cadiff_fit['alpha_mMperms_per_mAcm2']:.6e}")
    print(f"  β (1/ms decay)     = {cadiff_fit['beta_perms_decay']:.6e}")
    print(f"  R² = {cadiff_fit['r2']:.4f}  N = {cadiff_fit['n_samples']}  RMSE = {cadiff_fit['rmse']:.3e}")
    print(f"  ica range: {cadiff_fit['ica_range']}")
    print(f"  dcai/dt range: {cadiff_fit['dcai_dt_range']}")

    print("\n=== CALIBRATION: caintra1 (cell: cca1 + caintra1 at AIY-like geometry) ===")
    surf = 65.89e-8
    vol = 7.42e-12
    caintra_traj = collect_trajectory(
        "caintra1",
        pool_kwargs={"vol": vol, "surf": surf},
        holding_potentials=[-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0],
        duration_ms=200.0,
        dt_ms=0.025,
        surf_cm2=surf,
        cm_uFcm2=1.0,
    )
    # caintra1 uses ca_eq = 5e-8 (raw NEURON numerical value of 0.05e-6 (M))
    caintra_fit = fit_pool_coefficients(caintra_traj, cai_eq=5e-8)
    print(f"  α (mM/(mA/cm²·ms)) = {caintra_fit['alpha_mMperms_per_mAcm2']:.6e}")
    print(f"  β (1/ms decay)     = {caintra_fit['beta_perms_decay']:.6e}")
    print(f"  R² = {caintra_fit['r2']:.4f}  N = {caintra_fit['n_samples']}  RMSE = {caintra_fit['rmse']:.3e}")
    print(f"  ica range: {caintra_fit['ica_range']}")
    print(f"  dcai/dt range: {caintra_fit['dcai_dt_range']}")

    out_path = Path(__file__).parent / "artifacts" / "calcium_pool_calibration.json"
    out = {
        "cadiff": {
            "fit": cadiff_fit,
            "geometry": {"surf_cm2": 65.89e-8, "cm_uFcm2": 1.0,
                          "depth_um": 0.1, "beta_per_ms": 1.0},
            "method": "Linear LSQ on Δcai/Δt = α·ica + β·(cai - ca_eq) "
                      "from cca1+cadiff voltage-clamp trajectory.",
            "ca_eq_mM": 1e-4,
        },
        "caintra1": {
            "fit": caintra_fit,
            "geometry": {"surf_cm2": surf, "vol_cm3": vol, "cm_uFcm2": 1.0,
                          "fca": 0.001, "tca_ms": 50.0, "ca_eq_mM": 5e-5},
            "method": "Linear LSQ on Δcai/Δt = α·ica + β·(cai - ca_eq) "
                      "from cca1+caintra1 voltage-clamp trajectory.",
            "ca_eq_mM": 5e-5,
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
