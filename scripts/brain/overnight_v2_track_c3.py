#!/usr/bin/env python3
"""Track C3 — FLP-11 scenario-scenario Mode stability.

Run FLP-11 CONTROL + RIS_ABLATE at n=3 seeds × 60s on 3 scenarios
(osmotic_shock, food, spontaneous). Does Mode 1 classification hold
across scenarios, or is it scenario-specific?

Output: task_c_parallel_analysis/c3_scenario_stability/
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_env import ClosedLoopEnv
from phase0_audit import state_props

ART = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ART / "overnight_20260422_v2" / "task_c_parallel_analysis" / "c3_scenario_stability"

SCENARIOS = {
    "osmotic_shock": [(5.0, "osmotic_shock", 1.0)],
    "food": [(2.0, "food_signal", 1.0)],
    "spontaneous": [],
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    for scen, stim in SCENARIOS.items():
        scen_results = {"CONTROL": [], "RIS_ABLATE": []}
        for seed in [42, 43, 44]:
            for cond, ablate in [("CONTROL", None),
                                  ("RIS_ABLATE", ["RIS"])]:
                env = ClosedLoopEnv(seed=seed, enable_modulation=True,
                                    ablate=ablate, brain_class="lif")
                env.run(60.0, stim_schedule=stim)
                props = state_props(env.fsm_states)
                STATE_NAMES = ["FORWARD", "REVERSE", "OMEGA", "PIROUETTE",
                               "QUIESCENT"]
                sp = [props.get(s, 0) for s in STATE_NAMES]
                scen_results[cond].append(sp)
                print(f"  {scen} {cond} seed={seed}: "
                      f"QUI={props.get('QUIESCENT', 0):.2f} "
                      f"REV={props.get('REVERSE', 0):.2f}")
        ctrl = np.array(scen_results["CONTROL"]).mean(axis=0)
        abl = np.array(scen_results["RIS_ABLATE"]).mean(axis=0)
        delta = abl - ctrl
        max_delta = float(np.max(np.abs(delta)))
        mode = "Mode 1" if max_delta < 0.15 else ("Mode 3" if max_delta > 0.15 else "ambiguous")
        results[scen] = {
            "ctrl_mean": {s: float(ctrl[i])
                           for i, s in enumerate(STATE_NAMES)},
            "abl_mean": {s: float(abl[i])
                          for i, s in enumerate(STATE_NAMES)},
            "delta": {s: float(delta[i])
                       for i, s in enumerate(STATE_NAMES)},
            "max_abs_delta": round(max_delta, 3),
            "mode": mode,
        }
        print(f"→ {scen}: {mode} (max |Δ|={max_delta:.3f})")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Track C3 — FLP-11 scenario-scenario Mode stability",
        "",
        f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Wall: {(time.time()-t0)/60:.1f} min",
        "",
        "| scenario | ctrl QUI | abl QUI | ΔQUI | max |Δ| | Mode |",
        "|---|---|---|---|---|---|",
    ]
    for scen, r in results.items():
        lines.append(
            f"| {scen} | {r['ctrl_mean']['QUIESCENT']:.2f} | "
            f"{r['abl_mean']['QUIESCENT']:.2f} | "
            f"{r['delta']['QUIESCENT']:+.3f} | "
            f"{r['max_abs_delta']} | **{r['mode']}** |"
        )
    lines.append("")
    modes = set(r["mode"] for r in results.values())
    if len(modes) == 1:
        lines.append(f"**Mode stable across scenarios: {list(modes)[0]}**")
    else:
        lines.append(f"**Mode varies across scenarios: {modes}** — "
                     "Mode classification is scenario-dependent")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))

    status_md = ART / "overnight_20260422_v2" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Track C3: FLP-11 scenario stability\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Modes observed: {list(modes)}\n")


if __name__ == "__main__":
    main()
