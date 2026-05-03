"""mouse_state_validator — V6 mammalian wrapper around phase_g_state_validator."""
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
from state_validation.mouse_brain import MouseBrain

MOUSE_DATA_DIR = ANESTH / 'data' / 'state_validation_mouse'
MOUSE_PERTURBATION_TABLE = MOUSE_DATA_DIR / 'mouse_anesthetic_perturbation_table.csv'
MOUSE_MUTANT_TABLE       = MOUSE_DATA_DIR / 'mouse_mutant_baseline_perturbations.csv'

# Mouse baseline ~1.5 Hz; threshold = 50% suppression = 0.75 Hz
MOUSE_QUIESCENT_THRESHOLD_HZ = 0.75


def make_mouse_brain_factory():
    def _factory(seed):
        class SeededMouse(MouseBrain):
            _brian2_seed = seed
        return SeededMouse()
    return _factory


def run_mouse_single(anesthetic: str, dose_uM: float, seed: int, sim_duration_s: float,
                     mutant_gene: str | None = None, alpha_calib: float = 0.13,
                     perturbation_table_path: Path | None = None,
                     mutant_table_path: Path | None = None) -> dict:
    perturbation_table_path = perturbation_table_path or MOUSE_PERTURBATION_TABLE
    mutant_table_path = mutant_table_path or MOUSE_MUTANT_TABLE
    profiles = load_perturbation_table(perturbation_table_path)
    if anesthetic not in profiles:
        raise KeyError(f"Anesthetic {anesthetic!r} not in mouse perturbation table")
    profile = profiles[anesthetic]
    mut_obj = None
    if mutant_gene and mutant_gene != 'WT':
        muts = load_mutant_table(mutant_table_path)
        mut_obj = muts.get(mutant_gene)
    factory = make_mouse_brain_factory()
    metrics = _run_single(
        anesthetic=anesthetic, dose_uM=dose_uM, seed=seed,
        sim_duration_s=sim_duration_s, profile=profile,
        mutant=mut_obj, alpha_calib=alpha_calib,
        brain_factory=factory,
        quiescent_threshold_hz=MOUSE_QUIESCENT_THRESHOLD_HZ,
    )
    metrics['organism'] = 'mouse'
    return metrics


def smoke_test_validator(seed: int = 42, sim_duration_s: float = 15.0):
    import time
    print('=== mouse_state_validator smoke test ===')
    t0 = time.time()
    print('Test 1: WT baseline (dose ≈ 0)')
    m1 = run_mouse_single('halothane', dose_uM=0.001, seed=seed,
                          sim_duration_s=sim_duration_s, mutant_gene='WT', alpha_calib=0.10)
    print(f'  net_rate={m1["network_mean_firing_rate_hz"]:.2f} Hz  '
          f'cmd_rate={m1["command_mean_firing_rate_hz"]:.2f} Hz  '
          f'qf={m1["quiescent_fraction"]:.3f}')

    t1 = time.time()
    print('\nTest 2: halothane @ mouse MAC (350 µM), α=0.10')
    m2 = run_mouse_single('halothane', dose_uM=350.0, seed=seed,
                          sim_duration_s=sim_duration_s, mutant_gene='WT', alpha_calib=0.10)
    print(f'  net_rate={m2["network_mean_firing_rate_hz"]:.2f} Hz  '
          f'cmd_rate={m2["command_mean_firing_rate_hz"]:.2f} Hz  '
          f'qf={m2["quiescent_fraction"]:.3f}')

    t2 = time.time()
    print('\nTest 3: halothane @ 1000 µM (super-MAC), α=0.10')
    m3 = run_mouse_single('halothane', dose_uM=1000.0, seed=seed,
                          sim_duration_s=sim_duration_s, mutant_gene='WT', alpha_calib=0.10)
    print(f'  net_rate={m3["network_mean_firing_rate_hz"]:.2f} Hz  '
          f'cmd_rate={m3["command_mean_firing_rate_hz"]:.2f} Hz  '
          f'qf={m3["quiescent_fraction"]:.3f}')
    print(f'\nTotal wall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    smoke_test_validator()
