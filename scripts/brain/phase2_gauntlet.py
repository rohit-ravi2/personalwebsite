#!/usr/bin/env python3
"""Phase 2 gauntlet — re-test 3 sign modes under recalibrated FSM stack.

Following sub-tasks 2.1-2.4 (readout decision, classifier retrain,
M2-pure calibration, FSM threshold recalibration), this runs the same
9-test gauntlet as Phase 1 but with ALL 3 modes feeding through the same
downstream stack:

  bank: classifier_bank_v2_a2balanced.npz (21-cell, leave-one-worm-out CV)
  calibration: calibration_m2pure.npz (per-neuron affine, M2-pure)
  fsm thresholds: phase2_fsm_thresholds_*_m2pure.json (95th-percentile-tuned)

Modes tested (per Rohit's "re-test M1 and M2-current under new classifier"):
  M1         per-presynaptic-neuron + DOCUMENTED_SIGN_EXCEPTIONS
  M2-pure    per-edge + sign_exceptions={} (LOCKED CHOICE per Phase 1)
  M2-current per-edge + DOCUMENTED_SIGN_EXCEPTIONS

Tier defaults to 'screen' (n=5×30s, ~15 hr wall total) for first-pass
comparison. After review, can promote winning mode to 'default' tier
(n=10×60s, ~17 hr per mode).

Wraps `phase0_audit.py` functions; differs from `phase1_gauntlet.py`:
  - Plumbs new bank/cal/fsm_thresholds paths
  - Drops M3a (never tested in Phase 1; not gating Phase 2)
  - Output: phase2_gauntlet_*.{csv,json,md} (Phase 1 artifacts preserved)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from phase0_audit import (  # noqa: E402
    ABLATIONS,
    COMMAND_CASCADE_NEURONS,
    STATE_NAMES,
    seed_list,
    state_props,
    run_phenotype_one,
    run_scenario_one,
)
from phase1_gauntlet import (  # noqa: E402
    SCENARIOS_FOR_STABILITY,
    summarize_phenotype,
    summarize_cascade,
    summarize_scenarios,
    write_decision_matrix,
)

ART = THIS_DIR / "artifacts"

# Phase 2 artifact paths (all of these must exist before running)
BANK_PATH = ART / "classifier_bank_v2_a2balanced.npz"
CAL_PATH = ART / "calibration_m2pure.npz"
FSM_THRESH_BEHAVIORAL = ART / "phase2_fsm_thresholds_behavioral_m2pure.json"
FSM_THRESH_ACTIVITY = ART / "phase2_fsm_thresholds_activity_m2pure.json"

# 3 candidate modes (drops M3a — Phase 1 stopped before testing it)
MODES = {
    "M1": {
        "label": "default + 7 exceptions [PROD default]",
        "use_per_edge_glu_signs": False,
        "sign_exceptions": None,  # → DOCUMENTED_SIGN_EXCEPTIONS
    },
    "M2-pure": {
        "label": "per-edge pure [Phase 1 LOCKED]",
        "use_per_edge_glu_signs": True,
        "sign_exceptions": {},  # explicit empty
    },
    "M2-current": {
        "label": "per-edge + 7 exceptions [PROD per-edge]",
        "use_per_edge_glu_signs": True,
        "sign_exceptions": None,
    },
}

TIER_CONFIGS = {
    "screen": {"seeds": 5, "phenotype_dur_s": 30.0, "scenario_dur_s": 30.0},
    "default": {"seeds": 10, "phenotype_dur_s": 60.0, "scenario_dur_s": 30.0},
}


def cfg_kwargs_for_mode(mode_id: str, fsm_mode: str = "classifier") -> dict:
    """Build cfg kwargs with new Phase 2 stack — bank, cal, fsm_thresholds."""
    spec = MODES[mode_id]
    cfg = {
        "modulator_tables_path": ART / "modulator_tables.npz",
        "use_per_edge_glu_signs": spec["use_per_edge_glu_signs"],
        "brain_class": "lif",
        "fsm_mode": fsm_mode,
        "bank_path": BANK_PATH,
        "cal_path": CAL_PATH,
        "fsm_thresholds_path": (
            FSM_THRESH_BEHAVIORAL if fsm_mode == "classifier"
            else FSM_THRESH_ACTIVITY
        ),
    }
    if spec["sign_exceptions"] is not None:
        cfg["sign_exceptions"] = spec["sign_exceptions"]
    return cfg


def run_phenotype_for_mode(mode_id: str, tier_cfg: dict) -> pd.DataFrame:
    """6 ablations × 2 conditions × N seeds for one mode under new stack."""
    seeds = seed_list(tier_cfg["seeds"])
    duration_s = tier_cfg["phenotype_dur_s"]
    cfg_kwargs = cfg_kwargs_for_mode(mode_id, fsm_mode="classifier")

    rows = []
    total = len(ABLATIONS) * 2 * len(seeds)
    print(f"\n  [{mode_id}] phenotype audit: {len(ABLATIONS)} ablations × "
          f"2 conds × {len(seeds)} seeds × {duration_s}s = {total} runs")
    t0 = time.time()
    run_idx = 0
    for abl_label, neurons, scen, stim in ABLATIONS:
        for seed in seeds:
            run_idx += 1
            t_r = time.time()
            try:
                ctrl_props, ctrl_rates = run_phenotype_one(
                    scen, stim, None, seed, cfg_kwargs, duration_s)
                abl_props, abl_rates = run_phenotype_one(
                    scen, stim, neurons, seed, cfg_kwargs, duration_s)
                delta = {s: abl_props[s] - ctrl_props[s]
                         for s in STATE_NAMES[1:]}
                err = ""
            except Exception as e:
                ctrl_props = {s: 0.0 for s in STATE_NAMES[1:]}
                abl_props = {s: 0.0 for s in STATE_NAMES[1:]}
                delta = {s: 0.0 for s in STATE_NAMES[1:]}
                ctrl_rates, abl_rates = {}, {}
                err = repr(e)

            row = dict(
                mode=mode_id,
                ablation=abl_label, scenario=scen, seed=seed,
                duration_s=duration_s,
                ctrl_FWD=ctrl_props["FORWARD"], ctrl_REV=ctrl_props["REVERSE"],
                ctrl_OMG=ctrl_props["OMEGA"], ctrl_PIR=ctrl_props["PIROUETTE"],
                ctrl_QUI=ctrl_props["QUIESCENT"],
                abl_FWD=abl_props["FORWARD"], abl_REV=abl_props["REVERSE"],
                abl_OMG=abl_props["OMEGA"], abl_PIR=abl_props["PIROUETTE"],
                abl_QUI=abl_props["QUIESCENT"],
                dREV=delta["REVERSE"], dOMG=delta["OMEGA"],
                dPIR=delta["PIROUETTE"], dQUI=delta["QUIESCENT"],
                dFWD=delta["FORWARD"],
                ctrl_rates_json=json.dumps(ctrl_rates),
                abl_rates_json=json.dumps(abl_rates),
                error=err,
            )
            rows.append(row)
            dt = time.time() - t_r
            if run_idx % 5 == 0 or run_idx == total:
                elapsed = time.time() - t0
                est_total = elapsed / run_idx * total
                print(f"    [{mode_id}]  {run_idx}/{total}  "
                      f"({100*run_idx/total:.0f}%)  "
                      f"last={dt:.1f}s  elapsed={elapsed/60:.1f}min  "
                      f"eta={est_total/60:.1f}min")

    return pd.DataFrame(rows)


def run_scenarios_for_mode(mode_id: str, tier_cfg: dict) -> pd.DataFrame:
    """Non-touch stability + RIS baseline scenarios under new stack."""
    seeds = seed_list(tier_cfg["seeds"])
    duration_s = tier_cfg["scenario_dur_s"]
    cfg_kwargs = cfg_kwargs_for_mode(mode_id, fsm_mode="classifier")

    rows = []
    total = len(SCENARIOS_FOR_STABILITY) * len(seeds)
    print(f"\n  [{mode_id}] scenario audit: {len(SCENARIOS_FOR_STABILITY)} scens × "
          f"{len(seeds)} seeds × {duration_s}s = {total} runs")
    t0 = time.time()
    run_idx = 0
    for scen_name, stim in SCENARIOS_FOR_STABILITY:
        for seed in seeds:
            run_idx += 1
            t_r = time.time()
            try:
                env = run_scenario_one(scen_name, stim, seed, cfg_kwargs, duration_s)
                props = state_props(env.fsm_states)
                from closed_loop_env import BRAIN_SYNC_MS
                if len(env.full_spike_buffer) > 0:
                    fsb = np.stack(env.full_spike_buffer)
                    dt_s = BRAIN_SYNC_MS / 1000.0
                    n_cells = fsb.shape[1]
                    per_cell_rate_hz = fsb.sum(axis=0) / (fsb.shape[0] * dt_s)
                    mean_rate = float(per_cell_rate_hz.mean())
                    max_rate = float(per_cell_rate_hz.max())
                    n_zero = int((per_cell_rate_hz == 0).sum())
                    n_above_100hz = int((per_cell_rate_hz > 100).sum())
                else:
                    mean_rate = max_rate = 0.0
                    n_zero = n_above_100hz = 0
                    n_cells = 0

                ris_rate = float("nan")
                if "RIS" in env.brain.idx and len(env.full_spike_buffer) > 0:
                    fsb = np.stack(env.full_spike_buffer)
                    ris_idx = env.brain.idx["RIS"]
                    ris_rate = float(fsb[:, ris_idx].sum() / (fsb.shape[0] * BRAIN_SYNC_MS / 1000.0))

                err = ""
            except Exception as e:
                props = {s: 0.0 for s in STATE_NAMES[1:]}
                mean_rate = max_rate = ris_rate = float("nan")
                n_cells = n_zero = n_above_100hz = 0
                err = repr(e)

            rows.append(dict(
                mode=mode_id, scenario=scen_name, seed=seed,
                duration_s=duration_s,
                FWD=props["FORWARD"], REV=props["REVERSE"],
                OMG=props["OMEGA"], PIR=props["PIROUETTE"],
                QUI=props["QUIESCENT"],
                mean_rate_hz=mean_rate, max_rate_hz=max_rate,
                n_silent=n_zero, n_above_100hz=n_above_100hz,
                ris_rate_hz=ris_rate,
                error=err,
            ))
            dt = time.time() - t_r
            if run_idx % 5 == 0 or run_idx == total:
                elapsed = time.time() - t0
                print(f"    [{mode_id}/scen]  {run_idx}/{total}  "
                      f"({100*run_idx/total:.0f}%)  "
                      f"last={dt:.1f}s  elapsed={elapsed/60:.1f}min")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=list(TIER_CONFIGS.keys()), default="screen")
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    parser.add_argument("--skip-scenarios", action="store_true")
    args = parser.parse_args()

    # Sanity-check the new artifacts exist
    for p in [BANK_PATH, CAL_PATH, FSM_THRESH_BEHAVIORAL]:
        assert p.exists(), f"required Phase 2 artifact missing: {p}"

    tier_cfg = TIER_CONFIGS[args.tier]
    print("=" * 78)
    print("  Phase 2 gauntlet — recalibrated FSM stack across 3 sign modes")
    print(f"  bank: {BANK_PATH.name}")
    print(f"  cal:  {CAL_PATH.name}")
    print(f"  fsm thresholds (behavioral): {FSM_THRESH_BEHAVIORAL.name}")
    print(f"  Tier: {args.tier}  (seeds={tier_cfg['seeds']}, "
          f"phen={tier_cfg['phenotype_dur_s']}s, scen={tier_cfg['scenario_dur_s']}s)")
    print(f"  Modes: {args.modes}")
    print("=" * 78)

    t_start = time.time()
    all_phen_dfs = []
    all_scen_dfs = []
    for mode_id in args.modes:
        if mode_id not in MODES:
            print(f"Unknown mode: {mode_id}. Skipping.")
            continue
        print(f"\n>>> MODE {mode_id}: {MODES[mode_id]['label']}")
        phen_df = run_phenotype_for_mode(mode_id, tier_cfg)
        phen_path = ART / f"phase2_gauntlet_{mode_id}_{args.tier}_phenotype.csv"
        phen_df.to_csv(phen_path, index=False)
        print(f"  [{mode_id}] phenotype CSV → {phen_path}")
        all_phen_dfs.append(phen_df)

        if not args.skip_scenarios:
            scen_df = run_scenarios_for_mode(mode_id, tier_cfg)
            scen_path = ART / f"phase2_gauntlet_{mode_id}_{args.tier}_scenario.csv"
            scen_df.to_csv(scen_path, index=False)
            print(f"  [{mode_id}] scenario CSV → {scen_path}")
            all_scen_dfs.append(scen_df)

    if all_phen_dfs:
        phen_all = pd.concat(all_phen_dfs, ignore_index=True)
        phen_summary = summarize_phenotype(phen_all)
        cascade_summary = summarize_cascade(phen_all)
    else:
        phen_summary = {}
        cascade_summary = {}

    if all_scen_dfs:
        scen_all = pd.concat(all_scen_dfs, ignore_index=True)
        scen_summary = summarize_scenarios(scen_all)
    else:
        scen_summary = {}

    summary_path = ART / f"phase2_gauntlet_{args.tier}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "tier": args.tier,
            "phase2_stack": {
                "bank": str(BANK_PATH.name),
                "cal": str(CAL_PATH.name),
                "fsm_thresholds_behavioral": str(FSM_THRESH_BEHAVIORAL.name),
                "fsm_thresholds_activity": str(FSM_THRESH_ACTIVITY.name),
            },
            "modes": {
                m: {**MODES[m], "sign_exceptions": (
                    "DOCUMENTED_SIGN_EXCEPTIONS" if MODES[m]["sign_exceptions"] is None
                    else {f"{a}->{b}": v for (a, b), v in MODES[m]["sign_exceptions"].items()}
                )} for m in args.modes
            },
            "phenotype_summary": phen_summary,
            "cascade_summary": cascade_summary,
            "scenario_summary": scen_summary,
            "wall_time_s": time.time() - t_start,
        }, f, indent=2, default=str)
    print(f"\n[summary JSON written] {summary_path}")

    decision_path = ART / f"phase2_gauntlet_{args.tier}_decision_matrix.md"
    write_decision_matrix(phen_summary, cascade_summary, scen_summary, decision_path)
    print(f"[decision matrix written] {decision_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*78}")
    print(f"  Phase 2 gauntlet complete. Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
