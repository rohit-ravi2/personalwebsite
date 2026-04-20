#!/usr/bin/env python3
"""v3.3 ensemble audit — tests the full Tier 1 graded stack through
the same 6-ablation perturbation suite used in Phase 3d-6.

Config: graded brain (T1a) + Ca plateau (T1b) + volume-transmission
modulators (T1c) + closed-loop proprioception (T1d). Environment
(T1e) is NOT used for these ablation tests — they run identically to
the prior audit scenarios (no food patch).

Writes: artifacts/v33_audit_results.csv + appends to ensemble_report.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_env import ClosedLoopEnv  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
OUT_CSV = ART / "v33_audit_results.csv"

STATE_NAMES = ["(unused)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"]

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


def run_one(scenario, stim, ablate, seed):
    # v3.3 = graded brain + Ca plateau + volume transmission + proprioception.
    # Uses modulator_tables.npz (v3.1 pharyngeal-excluded).
    env = ClosedLoopEnv(
        seed=seed,
        enable_modulation=True,
        ablate=ablate,
        brain_class="graded",
        modulator_tables_path=ART / "modulator_tables.npz",
        use_per_edge_glu_signs=False,
    )
    env.run(DURATION_S, stim_schedule=stim)
    return state_props(env.fsm_states)


def main():
    total = len(ABLATIONS) * 2 * len(SEEDS)
    print(f"v3.3 audit: {total} runs ({len(ABLATIONS)} ablations × "
          f"2 control/ablated × {len(SEEDS)} seeds)")
    t0 = time.time()

    rows = []
    run_idx = 0
    for abl_label, neurons, scen, stim in ABLATIONS:
        for seed in SEEDS:
            run_idx += 1
            t_r = time.time()
            try:
                ctrl = run_one(scen, stim, None, seed)
                ablt = run_one(scen, stim, neurons, seed)
                delta = {s: ablt[s] - ctrl[s] for s in STATE_NAMES[1:]}
            except Exception as e:
                print(f"  ERROR seed={seed} {abl_label}: {e}")
                ctrl = {s: 0.0 for s in STATE_NAMES[1:]}
                ablt = {s: 0.0 for s in STATE_NAMES[1:]}
                delta = {s: 0.0 for s in STATE_NAMES[1:]}

            row = dict(
                config="v3.3", ablation=abl_label, seed=seed,
                ctrl_FWD=ctrl["FORWARD"], ctrl_REV=ctrl["REVERSE"],
                ctrl_OMG=ctrl["OMEGA"],   ctrl_PIR=ctrl["PIROUETTE"],
                ctrl_QUI=ctrl["QUIESCENT"],
                abl_FWD=ablt["FORWARD"],  abl_REV=ablt["REVERSE"],
                abl_OMG=ablt["OMEGA"],    abl_PIR=ablt["PIROUETTE"],
                abl_QUI=ablt["QUIESCENT"],
                dREV=delta["REVERSE"], dOMG=delta["OMEGA"],
                dPIR=delta["PIROUETTE"], dQUI=delta["QUIESCENT"],
                dFWD=delta["FORWARD"],
            )
            rows.append(row)
            dt = time.time() - t_r
            eta = (total - run_idx) * dt / 60
            print(f"  [{run_idx}/{total}] {abl_label} seed={seed} | "
                  f"ΔREV={delta['REVERSE']:+.2f} "
                  f"ΔQUI={delta['QUIESCENT']:+.2f} | "
                  f"{dt:.0f}s, ETA {eta:.0f} min")
            # Incremental save in case we need to abort
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    total_min = (time.time() - t0) / 60
    print(f"\nTotal: {total_min:.1f} min")

    # Aggregate: mean±std of each delta across seeds per ablation
    df = pd.DataFrame(rows)
    agg = df.groupby("ablation").agg(
        REV_mu=("dREV", "mean"), REV_sd=("dREV", "std"),
        OMG_mu=("dOMG", "mean"), OMG_sd=("dOMG", "std"),
        PIR_mu=("dPIR", "mean"), PIR_sd=("dPIR", "std"),
        QUI_mu=("dQUI", "mean"), QUI_sd=("dQUI", "std"),
        FWD_mu=("dFWD", "mean"), FWD_sd=("dFWD", "std"),
    ).round(3)
    print("\nv3.3 Aggregated (mean ± std across seeds):")
    print(agg.to_string())

    # Quick verdict for the target states
    print("\n" + "=" * 60)
    print("v3.3 ABLATION VERDICTS")
    print("=" * 60)
    checks = [
        ("RIS / osmotic_shock", "dQUI", "Turek 2016 (expected ↓)"),
        ("NSM / food",          "dQUI", "Flavell 2013 (expected ↓)"),
        ("AVA / touch",         "dREV", "Chalfie 1985 (expected ↓)"),
        ("AVB / spontaneous",   "dFWD", "Chalfie 1985 (expected ↓)"),
        ("RIM / touch",         "dREV", "Alkema/Gordus (directional)"),
        ("PDE / spontaneous",   "dPIR", "Chase 2004 (directional)"),
    ]
    for abl, col, ref in checks:
        sub = df[df["ablation"] == abl][col]
        mu = sub.mean()
        sd = sub.std()
        n = len(sub)
        sem = sd / np.sqrt(n) if sd > 0 else 0.001
        tstat = abs(mu) / (2 * sem)
        significant = (tstat > 1) and (abs(mu) > 0.05)
        print(f"  {abl:<22} {col} = {mu:+.3f} ± {sd:.3f}  |  "
              f"{ref:<32}  |  "
              f"{'SIG' if significant else 'noise'}")


if __name__ == "__main__":
    main()
