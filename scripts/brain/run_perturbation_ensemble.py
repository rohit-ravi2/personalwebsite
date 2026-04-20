#!/usr/bin/env python3
"""Phase 3d-6 — Reproducibility audit for perturbation suite.

For each of three v3 configurations, run the 6-ablation suite across
multiple random seeds and report mean ± standard deviation per
ablation Δ. This distinguishes "real but small" effects from
"Brian2-noise" (stochastic variance within a single configuration).

Configurations tested:
  v3.0: per-neuron NT signs, no 5HT pharyngeal exclusion
        (modulator_tables_v30.npz, use_per_edge_glu_signs=False)
  v3.1: per-neuron NT signs, 5HT pharyngeal exclusion
        (modulator_tables.npz,  use_per_edge_glu_signs=False)
  v3.2: per-edge Glu signs + 5HT pharyngeal exclusion
        (modulator_tables.npz,  use_per_edge_glu_signs=True)

Output: artifacts/ensemble_report.md with mean ± std per
(config, ablation) pair. Deltas smaller than ±2σ should be treated
as null.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_env import ClosedLoopEnv  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
OUT_MD = ART / "ensemble_report.md"

STATE_NAMES = ["(unused)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"]

CONFIGS = [
    ("v3.0",
     {"modulator_tables_path": ART / "modulator_tables_v30.npz",
      "use_per_edge_glu_signs": False},
     "per-neuron NT signs, all 5HT targets incl pharyngeal"),
    ("v3.1",
     {"modulator_tables_path": ART / "modulator_tables.npz",
      "use_per_edge_glu_signs": False},
     "per-neuron NT signs, 5HT pharyngeal excluded"),
    ("v3.2",
     {"modulator_tables_path": ART / "modulator_tables.npz",
      "use_per_edge_glu_signs": True},
     "per-edge Glu signs from CeNGEN + 5HT excluded"),
]

ABLATIONS = [
    ("RIS / osmotic_shock",  ["RIS"],
     "osmotic_shock", [(5.0, "osmotic_shock", 1.0)]),
    ("NSM / food",           ["NSML", "NSMR"],
     "food",          [(2.0, "food_signal", 1.0)]),
    ("RIM / touch",          ["RIML", "RIMR"],
     "touch",         [(5.0, "touch_anterior", 1.0)]),
    ("AVA / touch",          ["AVAL", "AVAR"],
     "touch",         [(5.0, "touch_anterior", 1.0)]),
    ("AVB / spontaneous",    ["AVBL", "AVBR"],
     "spontaneous",   []),
    ("PDE / spontaneous",    ["PDEL", "PDER"],
     "spontaneous",   []),
]

SEEDS = [42, 43, 44]
DURATION_S = 20.0


def state_props(fsm_states):
    if not fsm_states:
        return {n: 0.0 for n in STATE_NAMES[1:]}
    total = len(fsm_states)
    return {name: sum(1 for s in fsm_states if s == i) / total
            for i, name in enumerate(STATE_NAMES[1:], start=1)}


def run_one(scenario, stim, ablate, seed, cfg_kwargs):
    env = ClosedLoopEnv(seed=seed, enable_modulation=True, ablate=ablate,
                        **cfg_kwargs)
    env.run(DURATION_S, stim_schedule=stim)
    return state_props(env.fsm_states)


def main():
    # Full matrix: ~ len(CONFIGS) × len(ABLATIONS) × 2 × len(SEEDS)
    total = len(CONFIGS) * len(ABLATIONS) * 2 * len(SEEDS)
    print(f"Ensemble audit: {total} runs "
          f"({len(CONFIGS)} configs × {len(ABLATIONS)} ablations × "
          f"2 control/ablated × {len(SEEDS)} seeds)")
    t0 = time.time()

    # results[(cfg_name, ablation_label, seed)] = {"control": {...}, "ablated": {...}, "delta": {...}}
    results = {}

    run_idx = 0
    for cfg_name, cfg_kwargs, _ in CONFIGS:
        for abl_label, neurons, scen, stim in ABLATIONS:
            for seed in SEEDS:
                run_idx += 1
                t_r = time.time()

                ctrl = run_one(scen, stim, None, seed, cfg_kwargs)
                ablt = run_one(scen, stim, neurons, seed, cfg_kwargs)
                delta = {s: ablt[s] - ctrl[s] for s in STATE_NAMES[1:]}

                results[(cfg_name, abl_label, seed)] = {
                    "control": ctrl, "ablated": ablt, "delta": delta,
                }
                dt = time.time() - t_r
                eta = (total * 2 - run_idx * 2) * dt / 2 / 60
                print(f"  [{run_idx}/{total//2}] {cfg_name} | "
                      f"{abl_label} | seed={seed} | "
                      f"ΔREV={delta['REVERSE']:+.2f} "
                      f"ΔQUI={delta['QUIESCENT']:+.2f} | "
                      f"{dt:.0f}s, ETA {eta:.0f} min")

    print(f"\nTotal ensemble time: {(time.time()-t0)/60:.1f} min")

    # Aggregate: for each (cfg, ablation, state), compute mean+std of delta across seeds
    agg = {}
    for cfg_name, _, _ in CONFIGS:
        for abl_label, *_ in ABLATIONS:
            deltas = [results[(cfg_name, abl_label, s)]["delta"]
                      for s in SEEDS]
            per_state = {}
            for state in STATE_NAMES[1:]:
                vals = [d[state] for d in deltas]
                per_state[state] = {
                    "mean": mean(vals),
                    "std": stdev(vals) if len(vals) > 1 else 0.0,
                    "vals": vals,
                }
            agg[(cfg_name, abl_label)] = per_state

    # Write markdown report
    lines = ["# Phase 3d-6 — Perturbation ensemble audit",
             "",
             f"Reproducibility audit of the v3 perturbation suite. Each ",
             f"(config × ablation) cell is run across {len(SEEDS)} seeds ",
             f"({', '.join(str(s) for s in SEEDS)}), with Brian2 internal ",
             f"RNG locked alongside `np.random.seed()`. For each state, ",
             f"reports mean ± std of the (ablated - control) delta across ",
             f"seeds. Deltas with **|mean| < 2σ** should be treated as null.",
             ""]

    for abl_label, *_ in ABLATIONS:
        lines.append(f"## {abl_label}")
        lines.append("")
        lines.append("| state | "
                     + " | ".join(c for c, *_ in CONFIGS)
                     + " |")
        lines.append("|---|" + "---|" * len(CONFIGS))
        for state in STATE_NAMES[1:]:
            row = [f"**{state}**"]
            for cfg_name, *_ in CONFIGS:
                cell = agg[(cfg_name, abl_label)][state]
                m, s = cell["mean"], cell["std"]
                # Mark significant (|mean| >= 2σ AND |mean| > 0.05) vs null
                if abs(m) >= max(2 * s, 0.05) and s < 0.5:
                    marker = "**"
                else:
                    marker = ""
                row.append(f"{marker}{m:+.2f} ± {s:.2f}{marker}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Headline summary: for each ablation, is the expected effect
    # significantly detected in any config?
    lines.append("## Summary: which effects survive variance?")
    lines.append("")
    lines.append("| ablation | expected state | v3.0 Δ (μ±σ) | v3.1 Δ | v3.2 Δ |")
    lines.append("|---|---|---|---|---|")

    EXPECTED = {
        "RIS / osmotic_shock":  ("QUIESCENT", -0.30, "Turek 2016"),
        "NSM / food":           ("QUIESCENT", -0.20, "Flavell 2013"),
        "RIM / touch":          ("REVERSE",   None,  "Alkema 2005 / Gordus 2015"),
        "AVA / touch":          ("REVERSE",   -0.15, "Chalfie 1985"),
        "AVB / spontaneous":    ("FORWARD",   -0.15, "Chalfie 1985"),
        "PDE / spontaneous":    ("PIROUETTE", None,  "Chase 2004"),
    }

    for abl_label, *_ in ABLATIONS:
        expected_state, _, _ = EXPECTED[abl_label]
        row = [abl_label, expected_state]
        for cfg_name, *_ in CONFIGS:
            cell = agg[(cfg_name, abl_label)][expected_state]
            row.append(f"{cell['mean']:+.2f} ± {cell['std']:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    # Interpretability note
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"With {len(SEEDS)} seeds, a Δ is credible if |μ| ≥ 2σ and ")
    lines.append(f"|μ| ≥ 0.05. Bold entries in the tables above pass that ")
    lines.append(f"threshold. Everything else is noise at this run length ")
    lines.append(f"({DURATION_S:.0f} s per run).")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nwrote {OUT_MD} ({OUT_MD.stat().st_size / 1024:.1f} KB)")

    # Print concise summary to stdout
    print("\n" + "=" * 70)
    print("ENSEMBLE AUDIT RESULTS")
    print("=" * 70)
    print(f"{'ablation':<25} {'state':<12} {'v3.0':>15} {'v3.1':>15} {'v3.2':>15}")
    for abl_label, *_ in ABLATIONS:
        expected_state, _, _ = EXPECTED[abl_label]
        vals = []
        for cfg_name, *_ in CONFIGS:
            cell = agg[(cfg_name, abl_label)][expected_state]
            vals.append(f"{cell['mean']:+.2f} ± {cell['std']:.2f}")
        print(f"{abl_label:<25} {expected_state:<12} "
              f"{vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")


if __name__ == "__main__":
    main()
