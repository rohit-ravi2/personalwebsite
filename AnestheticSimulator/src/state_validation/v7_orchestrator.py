"""v7_orchestrator — Run V7 pipeline phases sequentially with notifications.

Usage:
    python v7_orchestrator.py [phase ...]

    Phases: stage2, stage3, verdict, subq1, m3, m4, all-after-stage1

    'all-after-stage1' runs: stage2 → stage3 → subq1 → m3 → m4
    (Stage 1 must already have completed; this script does NOT re-run Stage 1.)

Each phase notifies start + end + result. Phases run sequentially, NOT in
parallel — they all use mp.Pool(12) and would contend for cores.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/rohit/Desktop/website/personalwebsite')
ANESTH = ROOT / 'AnestheticSimulator'
sys.path.insert(0, str(ANESTH / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

NOTIFY = '/home/rohit/bin/notify'


def notify(msg: str, title: str = 'V7', priority: str = 'default'):
    try:
        subprocess.run([NOTIFY, msg, title, priority], check=False, timeout=10)
    except Exception as e:
        print(f'  notify failed: {e}', flush=True)


def phase_stage2():
    notify('V7 Stage 2 starting (iso held-out on Stage 1 passers)')
    from state_validation.v7_subset_stages23 import run_stage2
    t0 = time.time()
    run_stage2()
    notify(f'V7 Stage 2 done in {(time.time()-t0)/60:.0f}m')


def phase_stage3():
    notify('V7 Stage 3 starting (Eger non-immobilizers)')
    from state_validation.v7_subset_stages23 import run_stage3
    t0 = time.time()
    run_stage3()
    notify(f'V7 Stage 3 done in {(time.time()-t0)/60:.0f}m')


def phase_verdict():
    from state_validation.v7_subset_stages23 import write_final_subset_verdict
    write_final_subset_verdict()


def phase_subq1():
    notify('V7 Sub-Q1 starting (random ensembles, Match 1 + 2)')
    from state_validation.v7_random_ensemble import (
        main_match_level, write_random_ensemble_verdict,
    )
    t0 = time.time()
    m1 = main_match_level(1)
    m2 = main_match_level(2)
    write_random_ensemble_verdict(m1, m2)
    notify(f'V7 Sub-Q1 done in {(time.time()-t0)/60:.0f}m')


def phase_m3():
    notify('V7 M3 starting (sensitivity OAT + LHS)')
    from state_validation.v7_m3_sensitivity import (
        run_oat, run_lhs, write_verdict, LHS_ORGANISM, LHS_N_SAMPLES,
    )
    t0 = time.time()
    oat_rows = run_oat()
    lhs_summary = run_lhs(LHS_ORGANISM, LHS_N_SAMPLES)
    write_verdict(oat_rows, lhs_summary)
    notify(f'V7 M3 done in {(time.time()-t0)/60:.0f}m')


def phase_m4():
    notify('V7 M4 starting (anchor-swap cross-cal)')
    from state_validation.v7_m4_cross_cal import main as m4_main
    t0 = time.time()
    m4_main()
    notify(f'V7 M4 done in {(time.time()-t0)/60:.0f}m')


PHASES = {
    'stage2': phase_stage2,
    'stage3': phase_stage3,
    'verdict': phase_verdict,
    'subq1': phase_subq1,
    'm3': phase_m3,
    'm4': phase_m4,
}

DEFAULT_SEQUENCE = ['stage2', 'stage3', 'verdict', 'subq1', 'm3', 'm4']


def main():
    if len(sys.argv) < 2:
        print('Usage: v7_orchestrator.py [phase ...]')
        print(f'  Phases: {", ".join(PHASES.keys())}')
        print(f"  Or 'all-after-stage1' = {' → '.join(DEFAULT_SEQUENCE)}")
        sys.exit(1)
    args = sys.argv[1:]
    if args == ['all-after-stage1']:
        sequence = DEFAULT_SEQUENCE
    else:
        sequence = args
    notify(f'V7 orchestrator starting: {" → ".join(sequence)}')
    t_total = time.time()
    for phase in sequence:
        if phase not in PHASES:
            print(f'Unknown phase: {phase}')
            notify(f'V7 orchestrator FAIL: unknown phase {phase}', priority='urgent')
            sys.exit(1)
        print(f'\n=== Phase: {phase} ===\n', flush=True)
        try:
            PHASES[phase]()
        except Exception as e:
            print(f'Phase {phase} FAILED: {e!r}', flush=True)
            import traceback
            traceback.print_exc()
            notify(f'V7 phase {phase} FAILED: {e!r}', priority='urgent')
            sys.exit(2)
    total_min = (time.time() - t_total) / 60.0
    notify(f'V7 orchestrator complete: {total_min:.0f}m')
    print(f'\n=== V7 orchestrator complete in {total_min:.0f}m ===')


if __name__ == '__main__':
    main()
