#!/usr/bin/env python3
"""
Phase I — JAX differentiable simulator + inverse design (stretch).

Status: SCAFFOLDED, DEFERRED. Activates only if Phase H >= 6/8 anchors.

Purpose
-------
Build a JAX-differentiable version of the simulator (LIF + 7 channels + Markov synapse
+ metabolic layer) and solve the inverse problem:

    min ||simulator(occupancy_vec) - empirical_data||^2  over occupancy_vec in [0,1]^N

Compare inverse occupancy against Phase C's structural-prediction occupancy.

Inputs
------
- artifacts/runs/<config>.npz (Phase G traces for cross-validation)
- artifacts/occupancy/occupancy_matrix.npz (Phase C; comparison target)
- External: Atanas 2023 / Hallinen 2021 / Yemini NeuroPAL anesthesia recordings

Outputs
-------
- artifacts/runs/inverse_occupancy.npz
- artifacts/runs/inverse_validation.md
- artifacts/runs/phase_i_completion.md

Reference: preregistration/phase_i_inverse_design.md
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "artifacts" / "runs"
LOG_DIR = ROOT / "artifacts" / "logs"


def setup_logger(verbose: bool = False) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"phase_i_{date.today().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("phase_i")


def jax_simulator_skeleton(log: logging.Logger):
    """SCAFFOLD skeleton showing intended JAX simulator structure."""
    log.warning("JAX simulator is a SCAFFOLD; manual reimplementation pending (~2 weeks)")
    code = '''
    @jax.jit
    def simulate(occupancy_vec, params, T):
        state = init_state(params)
        def step(state, t):
            kinetic_shifts = apply_occupancy(occupancy_vec, params)
            state = step_dynamics(state, kinetic_shifts, params)
            return state, (state.V, state.Ca)
        state, (V, Ca) = jax.lax.scan(step, state, jnp.arange(T))
        return V, Ca

    @jax.jit
    def loss(occupancy_vec, empirical, params):
        V, Ca = simulate(occupancy_vec, params, len(empirical.V))
        return jnp.mean((V - empirical.V) ** 2) + jnp.mean((Ca - empirical.Ca) ** 2)

    grad_fn = jax.grad(loss)
    optimizer = optax.adam(1e-3)
    '''
    return code


def gate_i1_evaluation(log: logging.Logger) -> dict:
    log.warning("Gate I.1 evaluation is SCAFFOLD")
    return {
        "I.1.1_jax_brian2_agreement_within_5pct": "PENDING",
        "I.1.2_loss_decreases_50x": "PENDING",
        "I.1.3_inverse_phase_c_spearman_geq_0.5": "PENDING",
        "I.1.4_synthetic_recovery_within_0.1": "PENDING",
        "overall": "PENDING",
    }


def run(args: argparse.Namespace, log: logging.Logger) -> int:
    if args.dry_run:
        log.info("[dry-run] Phase I requires manual JAX reimplementation; ~2 weeks engineering")
        return 0

    if args.show_skeleton:
        print(jax_simulator_skeleton(log))
        return 0

    if args.cross_validate:
        log.error("JAX-vs-Brian2 cross validation not implemented in scaffold")
        return 2

    if args.identifiability:
        log.error("Identifiability test (synthetic data) not implemented in scaffold")
        return 2

    if args.inverse_fit:
        log.error("Inverse fit on real data not implemented in scaffold")
        return 2

    if args.gate_evaluation:
        verdict = gate_i1_evaluation(log)
        with open(RUN_DIR / "gate_i1_evaluation.json", "w") as f:
            json.dump(verdict, f, indent=2)
        log.info("Gate I.1 evaluation written (scaffold)")
        return 0

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase I JAX inverse design (stretch)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--show-skeleton", action="store_true",
                        help="Print intended JAX simulator skeleton")
    parser.add_argument("--cross-validate", action="store_true",
                        help="Cross-validate JAX vs Brian2 (scaffold)")
    parser.add_argument("--identifiability", action="store_true",
                        help="Run identifiability test on synthetic data (scaffold)")
    parser.add_argument("--inverse-fit", action="store_true",
                        help="Inverse fit on real anesthesia data (scaffold)")
    parser.add_argument("--empirical", type=str, default=None,
                        help="Path to empirical NPZ data file")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gate-evaluation", action="store_true")
    args = parser.parse_args()
    log = setup_logger(args.verbose)

    print("PHASE I SCAFFOLD (DEFERRED) — implementation pending — see preregistration/phase_i_inverse_design.md")

    if not any([args.dry_run, args.show_skeleton, args.cross_validate,
                args.identifiability, args.inverse_fit, args.gate_evaluation]):
        parser.print_help()
        return 0

    return run(args, log)


if __name__ == "__main__":
    sys.exit(main())
