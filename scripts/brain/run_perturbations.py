#!/usr/bin/env python3
"""Phase 3d-3 — Perturbation validation.

For each of 6 canonical ablation targets, run paired control + ablated
scenarios and compare state distributions to published phenotypes.
If the model captures the right circuits, ablation should produce the
behavioural changes predicted by experimental literature.

Ablation targets & expected phenotypes (from literature):

  RIS     → loss of stress-induced quiescence (Turek 2016)
  NSM     → loss of feeding-state dwelling bias (Flavell 2013)
  RIM     → altered reversal dynamics (Alkema 2005, Donnelly 2013)
  AVA     → loss of reversal behaviour (Chalfie 1985, Gordus 2015)
  AVB     → loss of forward locomotion (Chalfie 1985)
  PDE     → altered roaming/pirouette dynamics (Chase 2004)

For each target we pick the scenario most sensitive to that ablation,
run control + ablated with the same random seed, and report state
proportion changes.

Outputs: artifacts/perturbation_report.md with the comparison table
and qualitative pass/fail verdicts.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_env import ClosedLoopEnv  # noqa: E402
from behavioral_fsm import State  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
OUT_MD = ART / "perturbation_report.md"

STATE_NAMES = ["(unused)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"]


# Ablation experiments: (label, neurons, scenario, stim_schedule, expected)
EXPERIMENTS = [
    ("RIS ablation / osmotic shock",
     ["RIS"],
     "osmotic_shock", [(5.0, "osmotic_shock", 1.0)],
     "QUIESCENCE ↓ — RIS drives sleep-like quiescence via FLP-11 "
     "(Turek 2016). Ablating RIS should abolish the quiescence "
     "surge under aversive stimulation."),

    ("NSM ablation / food",
     ["NSML", "NSMR"],
     "food", [(2.0, "food_signal", 1.0)],
     "QUIESCENCE/dwelling ↓ — NSM serotonin drives dwelling state "
     "under food (Flavell 2013). Ablating NSM should reduce feeding-"
     "state quiescence (if our 5HT pathway is connected)."),

    ("RIM ablation / touch",
     ["RIML", "RIMR"],
     "touch", [(5.0, "touch_anterior", 1.0)],
     "REVERSE altered — RIM tyramine biases reversal bout duration "
     "(Alkema 2005, Donnelly 2013). Ablating RIM should change the "
     "reversal response profile to mechanosensory stimulus."),

    ("AVA ablation / touch",
     ["AVAL", "AVAR"],
     "touch", [(5.0, "touch_anterior", 1.0)],
     "REVERSE ↓ — AVA is the primary reversal command interneuron "
     "(Chalfie 1985). Ablating AVA should drastically reduce reversal."),

    ("AVB ablation / spontaneous",
     ["AVBL", "AVBR"],
     "spontaneous", [],
     "FORWARD ↓ — AVB drives forward locomotion (Chalfie 1985). "
     "Ablating AVB should reduce forward-run time."),

    ("PDE ablation / spontaneous",
     ["PDEL", "PDER"],
     "spontaneous", [],
     "PIROUETTE / roaming altered — PDE dopamine modulates pirouette "
     "duration (Chase 2004, Ben Arous 2009)."),
]


DURATION_S = 20.0  # shorter than normal to keep 12 runs tractable


def state_proportions(fsm_states: list[int]) -> dict[str, float]:
    """Return frequency of each state."""
    if not fsm_states:
        return {}
    total = len(fsm_states)
    out = {}
    for i, name in enumerate(STATE_NAMES[1:], start=1):
        c = sum(1 for s in fsm_states if s == i)
        out[name] = c / total
    return out


def run_one(scenario_name: str, stim_schedule: list, ablate: list[str] | None,
            seed: int = 42, duration_s: float = DURATION_S) -> dict:
    np.random.seed(seed)
    env = ClosedLoopEnv(seed=seed, enable_modulation=True, ablate=ablate)
    t0 = time.time()
    env.run(duration_s, stim_schedule=stim_schedule)
    elapsed = time.time() - t0
    return {
        "scenario": scenario_name,
        "ablated": ablate or [],
        "ablated_actual": env.ablated,
        "duration_s": duration_s,
        "wall_time_s": elapsed,
        "state_props": state_proportions(env.fsm_states),
        "fsm_states": env.fsm_states,
        "modulator_concentrations": (
            {m: float(c) for m, c in zip(
                env.modulation.modulators, env.modulation.concentrations)}
            if env.modulation else {}
        ),
    }


def main() -> None:
    print("=" * 70)
    print(f"Phase 3d-3 perturbation suite — {len(EXPERIMENTS)} ablations")
    print(f"Duration: {DURATION_S} s per run, seed=42 (shared)")
    print("=" * 70)

    results = []
    for label, neurons, scen, schedule, expected in EXPERIMENTS:
        print(f"\n--- {label} ---")
        # Control
        print(f"  control [{scen}]…")
        ctrl = run_one(scen, schedule, ablate=None)
        print(f"    state props: "
              f"{ ', '.join(f'{k}={v:.2f}' for k, v in ctrl['state_props'].items() if v > 0) }")
        # Ablated
        print(f"  ablated {neurons} [{scen}]…")
        ablt = run_one(scen, schedule, ablate=neurons)
        print(f"    state props: "
              f"{ ', '.join(f'{k}={v:.2f}' for k, v in ablt['state_props'].items() if v > 0) }")
        # Delta
        delta = {k: ablt["state_props"].get(k, 0) - ctrl["state_props"].get(k, 0)
                 for k in ["FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"]}
        print(f"    Δ (ablated - control): "
              f"{ ', '.join(f'{k}={v:+.2f}' for k, v in delta.items()) }")

        results.append({
            "label": label,
            "neurons": neurons,
            "scenario": scen,
            "expected": expected,
            "control": ctrl,
            "ablated": ablt,
            "delta": delta,
        })

    # Write markdown report
    lines = ["# Phase 3d-3 — Perturbation validation report",
             "",
             f"In-silico neuron ablations run on the v3 model with full ",
             f"neuromodulation layer enabled. Each experiment pairs a ",
             f"control and an ablated run ({DURATION_S:.0f} s each, shared ",
             f"random seed). State proportions are fractions of simulation ",
             f"time spent in each FSM state.",
             ""]

    for r in results:
        lines.append(f"## {r['label']}")
        lines.append("")
        lines.append(f"**Target neurons:** {', '.join(r['neurons'])}  ")
        lines.append(f"**Scenario:** `{r['scenario']}`")
        lines.append("")
        lines.append(f"**Expected phenotype:** {r['expected']}")
        lines.append("")
        lines.append("| state | control | ablated | Δ |")
        lines.append("|---|---:|---:|---:|")
        for k in ["FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"]:
            c = r["control"]["state_props"].get(k, 0.0)
            a = r["ablated"]["state_props"].get(k, 0.0)
            d = a - c
            marker = "**" if abs(d) >= 0.05 else ""
            lines.append(f"| {k} | {c:.2f} | {a:.2f} | {marker}{d:+.2f}{marker} |")
        lines.append("")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| ablation | expected change | observed largest |")
    lines.append("|---|---|---|")
    for r in results:
        largest_key = max(r["delta"], key=lambda k: abs(r["delta"][k]))
        largest_delta = r["delta"][largest_key]
        lines.append(f"| {r['label']} | see above | "
                     f"{largest_key} {largest_delta:+.2f} |")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nwrote {OUT_MD} ({OUT_MD.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
