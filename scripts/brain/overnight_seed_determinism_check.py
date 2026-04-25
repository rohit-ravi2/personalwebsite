#!/usr/bin/env python3
"""Overnight Task 0 — verify seed determinism via ClosedLoopEnv reuse.

Runs the same short simulation (spontaneous 5s, seed=42) twice via
ClosedLoopEnv. Compares FSM-state traces and full-spike-buffer
checksums. If identical, seed determinism is safe for parallel Brian2
work. If not, D1 must serialize.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts" / "overnight_20260421"
OUT = ART / "seed_determinism.json"


def one_run(seed_val: int) -> dict:
    from closed_loop_env import ClosedLoopEnv
    env = ClosedLoopEnv(seed=seed_val, enable_modulation=True,
                        brain_class="lif")
    env.run(5.0)
    fsm_str = ",".join(str(s) for s in env.fsm_states)
    # Hash of spike buffer
    if env.full_spike_buffer:
        fsb = np.stack(env.full_spike_buffer)
        fsb_hash = hashlib.md5(fsb.tobytes()).hexdigest()
    else:
        fsb_hash = "empty"
    return {
        "fsm_hash": hashlib.md5(fsm_str.encode()).hexdigest(),
        "fsb_hash": fsb_hash,
        "n_fsm_states": len(env.fsm_states),
    }


def main():
    print("Seed determinism check: two ClosedLoopEnv runs with seed=42")
    t0 = time.time()
    r1 = one_run(42)
    r2 = one_run(42)
    identical = (r1["fsm_hash"] == r2["fsm_hash"]
                 and r1["fsb_hash"] == r2["fsb_hash"])
    result = {
        "identical": identical,
        "run1": r1, "run2": r2,
        "wall_s": round(time.time() - t0, 2),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"  Identical: {identical}")
    print(f"  Run1 FSM hash: {r1['fsm_hash'][:16]}")
    print(f"  Run2 FSM hash: {r2['fsm_hash'][:16]}")
    print(f"  Wrote {OUT}")
    if not identical:
        print("WARNING: non-deterministic. Serialize Brian2 tasks.")
    else:
        print("OK: deterministic. Parallel Brian2 safe.")


if __name__ == "__main__":
    main()
