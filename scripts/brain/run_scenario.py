#!/usr/bin/env python3
"""Phase 3c-4 — Run validation scenarios and export as site-ready JSON.

Four scenarios, each ~30 s of closed-loop brain-body simulation:

  spontaneous    — no stimulus, baseline behavioral distribution
  touch          — touch_anterior stim at t=5s (ALM/AVM mechanoreceptors)
  osmotic_shock  — osmotic_shock stim at t=5s (ASH polymodal avoidance)
  food           — food_signal tonic stim from t=2s to t=25s (ASI/ASJ/ADF)

Outputs to public/data/wormbody-brain-{scenario}.json for the new
BrainBody React component.

Usage: `python run_scenario.py [scenario_name...]`
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_env import ClosedLoopEnv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "public" / "data"


SCENARIOS = {
    "spontaneous": {
        "duration_s": 30.0,
        "stim": [],
        "description": "Spontaneous behaviour, no stimulus. Expect mixed "
                       "forward-run + reversal distribution.",
        "environment": None,
    },
    "touch": {
        "duration_s": 30.0,
        "stim": [(5.0, "touch_anterior", 1.0)],
        "description": "Gentle head touch at t=5s. ALM/AVM mechanoreceptors "
                       "fire → expected reversal within ~1s (Chalfie 1981).",
        "environment": None,
    },
    "osmotic_shock": {
        "duration_s": 30.0,
        "stim": [(5.0, "osmotic_shock", 1.0)],
        "description": "Osmotic shock at t=5s. ASH polymodal avoidance "
                       "sensor fires → expected reversal + possible "
                       "pirouette/omega (Hart 1995).",
        "environment": None,
    },
    "food": {
        "duration_s": 30.0,
        "stim": [(2.0, "food_signal", 1.0)],
        "description": "Food-availability signal from t=2s onwards. "
                       "ASI/ASJ/ADF feeding-state neurons tonically "
                       "active → expected extended forward runs and "
                       "reduced reversal rate.",
        "environment": None,
    },
    "chemotaxis": {
        "duration_s": 60.0,
        "stim": [],
        "description": "Chemotaxis scenario: 2D agar with attractant food "
                       "patch at (4 mm, 0). ASE/AWC/AWA sensory drive "
                       "from the real concentration field at the worm's "
                       "head position. Pierce-Shimomura 1999 navigation.",
        "environment": {
            "food_xy_m": (4e-3, 0.0),
            "peak_conc": 1.0,
            "sigma_m": 3e-3,
        },
    },
}


def run_scenario(name: str) -> None:
    if name not in SCENARIOS:
        print(f"Unknown scenario: {name}")
        return

    sc = SCENARIOS[name]
    print(f"=== {name} ===  {sc['description']}")
    t0 = time.time()

    # T1e — optionally attach a chemical-gradient environment
    env_obj = None
    if sc.get("environment"):
        from environment import Environment, ChemoGradient  # noqa
        e_cfg = sc["environment"]
        grad = ChemoGradient(
            food_xy=e_cfg["food_xy_m"],
            peak_conc=e_cfg["peak_conc"],
            sigma_m=e_cfg["sigma_m"],
        )
        env_obj = Environment(grad, initial_head_xy=(0.0, 0.0))

    env = ClosedLoopEnv(environment=env_obj)
    env.run(sc["duration_s"], stim_schedule=sc["stim"])
    out = OUT_DIR / f"wormbody-brain-{name}.json"
    env.export(out, name)

    # State distribution summary
    state_names = ["(none)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"]
    tally = {}
    for s in env.fsm_states:
        tally[state_names[s]] = tally.get(state_names[s], 0) + 1
    print(f"  Wall time: {time.time()-t0:.1f}s")
    print(f"  State distribution: "
          f"{', '.join(f'{k}={v}' for k, v in tally.items())}")


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else list(SCENARIOS)
    for n in names:
        run_scenario(n)


if __name__ == "__main__":
    main()
