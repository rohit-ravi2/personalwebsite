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
    # P0 #3 — aerotaxis (O2 avoidance)
    "aerotaxis": {
        "duration_s": 60.0,
        "stim": [],
        "description": "Aerotaxis: linear O2 gradient (7% at x=−10 mm to "
                       "21% at x=+10 mm). URX/AQR/PQR fire at high O2, "
                       "BAG at low O2 and on CO2 rise. Wild-type worms "
                       "prefer ~12% O2 and navigate away from high-O2 "
                       "zones (Gray 2004, Zimmer 2009).",
        "environment": {
            # Reuse the chemotaxis fields as dummies (no food patch needed)
            "food_xy_m": (0.0, 0.0),
            "peak_conc": 0.0,
            "sigma_m": 1e-3,
            "aerotaxis": {
                "kind": "linear_o2",
                "o2_min": 0.07,
                "o2_max": 0.21,
                "x_min_m": -10e-3,
                "x_max_m":  10e-3,
                "preferred_o2": 0.12,
            },
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
        from environment import (
            Environment, ChemoGradient,
            LinearGasField, RadialGasField, AerotaxisSensory,
        )  # noqa
        e_cfg = sc["environment"]
        grad = ChemoGradient(
            food_xy=e_cfg["food_xy_m"],
            peak_conc=e_cfg["peak_conc"],
            sigma_m=e_cfg["sigma_m"],
        )
        # P0 #3 — optional aerotaxis overlay
        aero = None
        if e_cfg.get("aerotaxis"):
            a = e_cfg["aerotaxis"]
            if a.get("kind") == "linear_o2":
                o2_field = LinearGasField(
                    min_frac=a.get("o2_min", 0.07),
                    max_frac=a.get("o2_max", 0.21),
                    x_min_m=a.get("x_min_m", -10e-3),
                    x_max_m=a.get("x_max_m", 10e-3),
                )
            elif a.get("kind") == "radial_o2":
                o2_field = RadialGasField(
                    center_xy=a.get("center_xy_m", (0.0, 0.0)),
                    baseline_frac=a.get("o2_baseline", 0.21),
                    peak_frac=a.get("o2_peak", 0.07),
                    sigma_m=a.get("sigma_m", 5e-3),
                )
            else:
                o2_field = None
            # CO2 field — optional radial (worm-colony emission)
            co2_field = None
            if a.get("co2"):
                c = a["co2"]
                co2_field = RadialGasField(
                    center_xy=c.get("center_xy_m", (0.0, 0.0)),
                    baseline_frac=c.get("co2_baseline", 0.0004),
                    peak_frac=c.get("co2_peak", 0.02),
                    sigma_m=c.get("sigma_m", 3e-3),
                )
            aero = AerotaxisSensory(
                o2_field=o2_field, co2_field=co2_field,
                preferred_o2_frac=a.get("preferred_o2", 0.12),
            )
        env_obj = Environment(
            grad, initial_head_xy=(0.0, 0.0), aerotaxis=aero,
        )

    # P1 #4 / #8 — FSM_MODE and SENSORY_MODE env-vars switch between
    # legacy (classifier, injection) and upgraded (activity,
    # transduction) modes. Defaults preserve v3 shipped behaviour.
    import os
    fsm_mode = os.environ.get("FSM_MODE", "classifier")
    sensory_mode = os.environ.get("SENSORY_MODE", "injection")
    env = ClosedLoopEnv(
        environment=env_obj,
        fsm_mode=fsm_mode, sensory_mode=sensory_mode,
    )
    env.run(sc["duration_s"], stim_schedule=sc["stim"])
    suffix_parts = []
    if fsm_mode != "classifier": suffix_parts.append(fsm_mode)
    if sensory_mode != "injection": suffix_parts.append(sensory_mode)
    suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else ""
    out = OUT_DIR / f"wormbody-brain-{name}{suffix}.json"
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
