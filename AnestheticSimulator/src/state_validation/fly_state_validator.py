"""fly_state_validator — fly cross-species wrapper around phase_g_state_validator.

Provides fly-specific defaults (FlyLarvaBrain factory, fly quiescent threshold,
fly command-neuron set) and reuses the worm validator's run_single, apply_genotype,
apply_anesthetic, compute_metrics, and Hill-fit machinery without modification.

Usage:
    from state_validation.fly_state_validator import run_fly_single, FLY_DATA_DIR
    m = run_fly_single('halothane', dose_uM=340.0, seed=42, sim_duration_s=60.0,
                       alpha_calib=0.13)
"""
from __future__ import annotations

from pathlib import Path
import sys

ANESTH = Path('/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator')
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, '/home/rohit/Desktop/website/personalwebsite/scripts')

from state_validation.phase_g_state_validator import (
    load_perturbation_table, load_mutant_table, run_single as _run_single,
    hill_fit_ec50, MutantBaseline, PerturbationRow,
)
from state_validation.fly_larva_brain import FlyLarvaBrain


# ===== Fly-specific configuration =====

FLY_DATA_DIR = ANESTH / 'data' / 'state_validation_fly'

FLY_PERTURBATION_TABLE = FLY_DATA_DIR / 'fly_anesthetic_perturbation_table.csv'
FLY_MUTANT_TABLE       = FLY_DATA_DIR / 'fly_mutant_baseline_perturbations.csv'
FLY_WORM_ANCHORS       = FLY_DATA_DIR / 'fly_immobilization_anchors.csv'

# Fly baseline command-neuron rate ~2 Hz; quiescent threshold ~50% suppression = 1 Hz
FLY_QUIESCENT_THRESHOLD_HZ = 1.0


def make_fly_brain_factory():
    """Returns a callable(seed) → FlyLarvaBrain (subclassed with seed)."""
    def _factory(seed):
        class SeededFly(FlyLarvaBrain):
            _brian2_seed = seed
        return SeededFly()
    return _factory


def run_fly_single(anesthetic: str, dose_uM: float, seed: int, sim_duration_s: float,
                    mutant_gene: str | None = None,
                    alpha_calib: float = 0.13,
                    perturbation_table_path: Path | None = None,
                    mutant_table_path: Path | None = None) -> dict:
    """Run one fly cross-species sim (anesthetic × dose × seed × mutant).

    Returns the same metrics dict as worm run_single — including mutant_gene,
    network firing rate, command rate, quiescent fraction.
    """
    perturbation_table_path = perturbation_table_path or FLY_PERTURBATION_TABLE
    mutant_table_path = mutant_table_path or FLY_MUTANT_TABLE

    profiles = load_perturbation_table(perturbation_table_path)
    if anesthetic not in profiles:
        raise KeyError(f"Anesthetic {anesthetic!r} not in perturbation table")
    profile = profiles[anesthetic]

    mut_obj = None
    if mutant_gene and mutant_gene != 'WT':
        muts = load_mutant_table(mutant_table_path)
        mut_obj = muts.get(mutant_gene)
        if mut_obj is None:
            raise KeyError(f"Mutant {mutant_gene!r} not in fly mutant table")

    factory = make_fly_brain_factory()
    # run_single now auto-discovers brain.command_neurons_idx if the brain has it,
    # so we don't need a separate probe brain.
    metrics = _run_single(
        anesthetic=anesthetic,
        dose_uM=dose_uM,
        seed=seed,
        sim_duration_s=sim_duration_s,
        profile=profile,
        mutant=mut_obj,
        alpha_calib=alpha_calib,
        brain_factory=factory,
        quiescent_threshold_hz=FLY_QUIESCENT_THRESHOLD_HZ,
    )
    metrics['organism'] = 'fly_larva'
    return metrics


def smoke_test_validator(seed: int = 42, sim_duration_s: float = 15.0):
    """End-to-end smoke: WT baseline + halothane @ 340 µM at α=0.13.

    Expected:
      - baseline (dose ≈ 0): fly cmd rate ~ 2 Hz, qf low (with threshold=1.0)
      - halothane @ 340: cmd rate suppressed, qf rises
    """
    import time
    print(f'=== fly_state_validator smoke test ===')
    print(f'  seed={seed}  sim_duration={sim_duration_s}s  alpha_calib=0.13')
    print()

    t0 = time.time()
    print('Test 1: WT baseline (dose ≈ 0)')
    m1 = run_fly_single('halothane', dose_uM=0.001, seed=seed,
                         sim_duration_s=sim_duration_s, mutant_gene='WT', alpha_calib=0.13)
    print(f'  net_rate={m1["network_mean_firing_rate_hz"]:.2f} Hz  '
          f'cmd_rate={m1["command_mean_firing_rate_hz"]:.2f} Hz  '
          f'qf={m1["quiescent_fraction"]:.3f}  '
          f'autocorr={m1["state_autocorrelation_lag1"]:.3f}  [{time.time()-t0:.0f}s]')

    t1 = time.time()
    print('\nTest 2: halothane @ 1×fly_MAC = 340 µM, α=0.13')
    m2 = run_fly_single('halothane', dose_uM=340.0, seed=seed,
                         sim_duration_s=sim_duration_s, mutant_gene='WT', alpha_calib=0.13)
    print(f'  net_rate={m2["network_mean_firing_rate_hz"]:.2f} Hz  '
          f'cmd_rate={m2["command_mean_firing_rate_hz"]:.2f} Hz  '
          f'qf={m2["quiescent_fraction"]:.3f}  [{time.time()-t1:.0f}s]')

    t2 = time.time()
    print('\nTest 3: halothane @ 1000 µM (super-MAC), α=0.13')
    m3 = run_fly_single('halothane', dose_uM=1000.0, seed=seed,
                         sim_duration_s=sim_duration_s, mutant_gene='WT', alpha_calib=0.13)
    print(f'  net_rate={m3["network_mean_firing_rate_hz"]:.2f} Hz  '
          f'cmd_rate={m3["command_mean_firing_rate_hz"]:.2f} Hz  '
          f'qf={m3["quiescent_fraction"]:.3f}  [{time.time()-t2:.0f}s]')

    print(f'\nTotal wall: {time.time()-t0:.0f}s')

    # Smoke verdict
    ok = (
        m1['command_mean_firing_rate_hz'] >= 1.0 and  # baseline biological
        m2['command_mean_firing_rate_hz'] < m1['command_mean_firing_rate_hz'] and  # 340µM reduces firing
        m3['quiescent_fraction'] >= max(m2['quiescent_fraction'], 0.5)  # super-MAC is more quiescent
    )
    print(f'\nSmoke verdict: {"PASS — architecture wired" if ok else "INVESTIGATE"}')


if __name__ == '__main__':
    smoke_test_validator()
