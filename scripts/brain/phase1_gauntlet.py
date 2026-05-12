#!/usr/bin/env python3
"""Phase 1 sign-mode decision gauntlet.

Runs the 9-test gauntlet (or a subset, see --tier) under 4 candidate sign modes:
  M1         per-presynaptic-neuron signs + DEFAULT_SIGN_OVERRIDES + DOCUMENTED_SIGN_EXCEPTIONS
             (current production default)
  M2-pure    pure per-edge CeNGEN, NO sign_exceptions
             (what T0 §5 measured: cascade fires +60 Hz)
  M2-current per-edge + DOCUMENTED_SIGN_EXCEPTIONS
             (current production per-edge; cascade collapses per commit aea4c79 verification)
  M3a        per-edge + AIY-only exceptions (drop 5 PVC entries, keep 2 AIY entries)
             (proposed: tests whether PVC entries are the cascade-collapsing factor)

Wraps `phase0_audit.py` functions to avoid duplicating run logic. Each mode gets:
  - phenotype audit: 6 ablations × 2 conditions × N seeds (covers C-13, C-21, C-22,
    C-25, C-27 in single sweep)
  - scenario audit: 4 non-touch scenarios × N seeds (covers C-24 RIS baseline, test #6
    network stability)

Tiers:
  screen     n=5 × 30s    — first-pass viability screen across all 4 modes (~4 hr wall total)
  default    n=10 × 60s   — decision-grade tier for finalist modes (~6 hr per mode)

Output:
  artifacts/phase1_gauntlet_<mode>_<tier>_phenotype.csv
  artifacts/phase1_gauntlet_<mode>_<tier>_scenario.csv
  artifacts/phase1_gauntlet_<tier>_summary.json   (cross-mode aggregate)
  artifacts/phase1_gauntlet_<tier>_decision_matrix.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

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
    _build_environment,
)
from lif_brain import DOCUMENTED_SIGN_EXCEPTIONS  # noqa: E402

ART = THIS_DIR / "artifacts"

# AIY exceptions — the 2 entries that are independent of PVC question
AIY_EXCEPTIONS_ONLY = {
    ("AIYL", "AIZL"): -1,
    ("AIYR", "AIZR"): -1,
}

# Empty exceptions = pure mode
NO_EXCEPTIONS: dict[tuple[str, str], int] = {}

MODES = {
    "M1": {
        "label": "default + 7 exceptions [PROD default]",
        "use_per_edge_glu_signs": False,
        "sign_exceptions": None,  # None → LIFBrain uses DOCUMENTED_SIGN_EXCEPTIONS
    },
    "M2-pure": {
        "label": "per-edge pure [§5 +60 Hz cascade]",
        "use_per_edge_glu_signs": True,
        "sign_exceptions": NO_EXCEPTIONS,  # explicitly empty
    },
    "M2-current": {
        "label": "per-edge + 7 exceptions [PROD per-edge]",
        "use_per_edge_glu_signs": True,
        "sign_exceptions": None,
    },
    "M3a": {
        "label": "per-edge + AIY-only (drop 5 PVC)",
        "use_per_edge_glu_signs": True,
        "sign_exceptions": AIY_EXCEPTIONS_ONLY,
    },
}

TIER_CONFIGS = {
    "screen": {"seeds": 5, "phenotype_dur_s": 30.0, "scenario_dur_s": 30.0},
    "default": {"seeds": 10, "phenotype_dur_s": 60.0, "scenario_dur_s": 30.0},
}

# Subset of scenarios for non-touch network stability + RIS baseline
SCENARIOS_FOR_STABILITY = [
    ("spontaneous", []),
    ("osmotic_shock", [(5.0, "osmotic_shock", 1.0)]),
    ("food", [(2.0, "food_signal", 1.0)]),
    ("chemotaxis", []),  # _build_environment returns Environment for chemotaxis
]


def cfg_kwargs_for_mode(mode_id: str) -> dict:
    spec = MODES[mode_id]
    cfg = {
        "modulator_tables_path": ART / "modulator_tables.npz",
        "use_per_edge_glu_signs": spec["use_per_edge_glu_signs"],
        "brain_class": "lif",
    }
    # Only set sign_exceptions if mode wants to override default
    if spec["sign_exceptions"] is not None:
        cfg["sign_exceptions"] = spec["sign_exceptions"]
    return cfg


def run_phenotype_for_mode(mode_id: str, tier_cfg: dict) -> pd.DataFrame:
    """Run all 6 ablations × 2 conditions × N seeds under one mode."""
    seeds = seed_list(tier_cfg["seeds"])
    duration_s = tier_cfg["phenotype_dur_s"]
    cfg_kwargs = cfg_kwargs_for_mode(mode_id)
    cfg_label = f"{mode_id}"

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
                mode=cfg_label,
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
                      f"last_run={dt:.1f}s  elapsed={elapsed/60:.1f}min  "
                      f"eta={est_total/60:.1f}min")

    df = pd.DataFrame(rows)
    return df


def run_scenarios_for_mode(mode_id: str, tier_cfg: dict) -> pd.DataFrame:
    """Run non-touch scenarios for network stability + RIS baseline."""
    seeds = seed_list(tier_cfg["seeds"])
    duration_s = tier_cfg["scenario_dur_s"]
    cfg_kwargs = cfg_kwargs_for_mode(mode_id)

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
                # FSM state distribution
                props = state_props(env.fsm_states)
                # Network stability: per-cell rate over the run
                from closed_loop_env import BRAIN_SYNC_MS
                if len(env.full_spike_buffer) > 0:
                    fsb = np.stack(env.full_spike_buffer)
                    dt_s = BRAIN_SYNC_MS / 1000.0
                    n_cells = fsb.shape[1]
                    per_cell_rate_hz = fsb.sum(axis=0) / (fsb.shape[0] * dt_s)
                    # Health metrics
                    mean_rate = float(per_cell_rate_hz.mean())
                    max_rate = float(per_cell_rate_hz.max())
                    n_zero = int((per_cell_rate_hz == 0).sum())
                    n_above_100hz = int((per_cell_rate_hz > 100).sum())
                else:
                    mean_rate = max_rate = 0.0
                    n_zero = n_above_100hz = 0
                    n_cells = 0

                # RIS baseline rate (test #4)
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

            row = dict(
                mode=mode_id, scenario=scen_name, seed=seed,
                duration_s=duration_s,
                FWD=props["FORWARD"], REV=props["REVERSE"],
                OMG=props["OMEGA"], PIR=props["PIROUETTE"],
                QUI=props["QUIESCENT"],
                mean_rate_hz=mean_rate, max_rate_hz=max_rate,
                n_silent=n_zero, n_above_100hz=n_above_100hz,
                ris_rate_hz=ris_rate,
                error=err,
            )
            rows.append(row)
            dt = time.time() - t_r
            if run_idx % 5 == 0 or run_idx == total:
                elapsed = time.time() - t0
                print(f"    [{mode_id}/scen]  {run_idx}/{total}  "
                      f"({100*run_idx/total:.0f}%)  "
                      f"last={dt:.1f}s  elapsed={elapsed/60:.1f}min")

    df = pd.DataFrame(rows)
    return df


def summarize_phenotype(df: pd.DataFrame) -> dict:
    """Per-mode × per-ablation summary: dREV/dPIR/dQUI mean ± SEM, neg-seed count."""
    summary = {}
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        per_abl = {}
        for abl in sub["ablation"].unique():
            r = sub[sub["ablation"] == abl]
            n = len(r)
            per_abl[abl] = {
                "n_seeds": int(n),
                "dREV_mean": float(r["dREV"].mean()),
                "dREV_sem": float(r["dREV"].std() / np.sqrt(n)) if n > 1 else 0.0,
                "dREV_neg_seeds": int((r["dREV"] < 0).sum()),
                "dPIR_mean": float(r["dPIR"].mean()),
                "dPIR_sem": float(r["dPIR"].std() / np.sqrt(n)) if n > 1 else 0.0,
                "dPIR_neg_seeds": int((r["dPIR"] < 0).sum()),
                "dQUI_mean": float(r["dQUI"].mean()),
                "dQUI_sem": float(r["dQUI"].std() / np.sqrt(n)) if n > 1 else 0.0,
                "dQUI_neg_seeds": int((r["dQUI"] < 0).sum()),
                "dOMG_mean": float(r["dOMG"].mean()),
                "dFWD_mean": float(r["dFWD"].mean()),
            }
        summary[mode] = per_abl
    return summary


def summarize_cascade(df: pd.DataFrame) -> dict:
    """Pull AVDL/AVAL/PVCL per-cell touch-cascade rates from AVA-touch ablation rows."""
    cascade = {}
    for mode in df["mode"].unique():
        sub = df[(df["mode"] == mode) & (df["ablation"] == "AVA / touch")]
        # Only use control rows (ablation=False); ctrl_rates_json has the cascade firing data
        per_seed = []
        for _, row in sub.iterrows():
            try:
                rates = json.loads(row["ctrl_rates_json"])
                per_seed.append(rates)
            except (json.JSONDecodeError, TypeError):
                continue
        if not per_seed:
            continue
        # Aggregate per-cell mean delta_hz across seeds
        all_cells = set()
        for r in per_seed:
            all_cells.update(r.keys())
        cell_summary = {}
        for cell in sorted(all_cells):
            deltas = [r[cell]["delta_hz"] for r in per_seed if cell in r]
            pre = [r[cell]["pre_hz"] for r in per_seed if cell in r]
            peri = [r[cell]["peri_hz"] for r in per_seed if cell in r]
            if deltas:
                cell_summary[cell] = {
                    "delta_hz_mean": float(np.mean(deltas)),
                    "delta_hz_sem": float(np.std(deltas) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
                    "pre_hz_mean": float(np.mean(pre)),
                    "peri_hz_mean": float(np.mean(peri)),
                    "n_seeds": int(len(deltas)),
                }
        cascade[mode] = cell_summary
    return cascade


def summarize_scenarios(df: pd.DataFrame) -> dict:
    """Per-mode × per-scenario stability + RIS baseline summary."""
    summary = {}
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        per_scen = {}
        for scen in sub["scenario"].unique():
            r = sub[sub["scenario"] == scen]
            per_scen[scen] = {
                "n_seeds": int(len(r)),
                "mean_rate_hz_mean": float(r["mean_rate_hz"].mean()),
                "max_rate_hz_mean": float(r["max_rate_hz"].mean()),
                "n_silent_mean": float(r["n_silent"].mean()),
                "n_above_100hz_mean": float(r["n_above_100hz"].mean()),
                "ris_rate_hz_mean": float(r["ris_rate_hz"].mean()),
                "ris_rate_hz_sem": float(r["ris_rate_hz"].std() / np.sqrt(len(r))) if len(r) > 1 else 0.0,
                "errors": int(r["error"].fillna("").astype(str).ne("").sum()),
            }
        summary[mode] = per_scen
    return summary


def write_decision_matrix(phen_summary, cascade_summary, scen_summary, out_path: Path):
    lines = ["# Phase 1 gauntlet — decision matrix",
             "",
             f"Generated 2026-05-02. Tier: see metadata in JSON sidecar.",
             ""]

    # Cascade firing table — test #1 (C-13)
    lines.append("## Test 1 — Touch cascade firing (C-13 broadening)")
    lines.append("")
    lines.append("Per-cell Δ peri-touch (Hz, mean across seeds). 'AVA / touch' control runs only.")
    lines.append("")
    cascade_cells = ["ALML", "AVM", "PVCL", "AVDL", "AVAL", "AVAR", "AVBL", "AIBL", "RIML"]
    header = "| cell | " + " | ".join(MODES.keys()) + " |"
    sep = "|---|" + "---|" * len(MODES)
    lines.append(header)
    lines.append(sep)
    for cell in cascade_cells:
        cells = []
        for mode in MODES.keys():
            v = cascade_summary.get(mode, {}).get(cell, {})
            if v:
                cells.append(f"{v['delta_hz_mean']:+.2f}")
            else:
                cells.append("-")
        lines.append(f"| {cell} | " + " | ".join(cells) + " |")
    lines.append("")

    # Phenotype tables — tests #2, #3, #5, #9
    for ablation, channel, claim in [
        ("AVA / touch", "dREV", "Test 2 — AVA→dREV (C-22 default-mode reproduction)"),
        ("AVA / touch", "dPIR", "Test 3 — AVA→dPIR (C-21 per-edge channel shift)"),
        ("RIS / osmotic_shock", "dQUI", "Test 5 — RIS→dQUI (C-25 Turek)"),
        ("NSM / food", "dQUI", "Test 9 — NSM→dQUI counter-finding (C-27)"),
    ]:
        lines.append(f"## {claim}")
        lines.append("")
        lines.append(f"Mean ± SEM, neg-seed count for {channel}.")
        lines.append("")
        lines.append("| mode | mean | SEM | neg/N |")
        lines.append("|---|---|---|---|")
        for mode in MODES.keys():
            v = phen_summary.get(mode, {}).get(ablation, {})
            if v:
                m = v[f"{channel}_mean"]
                s = v[f"{channel}_sem"]
                neg = v[f"{channel}_neg_seeds"]
                n = v["n_seeds"]
                lines.append(f"| {mode} | {m:+.4f} | {s:.4f} | {neg}/{n} |")
        lines.append("")

    # Scenario stability
    lines.append("## Test 4 + 6 — RIS baseline + non-touch network stability")
    lines.append("")
    lines.append("RIS baseline rate (Hz, spontaneous) and stability metrics across non-touch scenarios.")
    lines.append("")
    for scen in ["spontaneous", "osmotic_shock", "food", "chemotaxis"]:
        lines.append(f"### {scen}")
        lines.append("")
        lines.append("| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |")
        lines.append("|---|---|---|---|---|")
        for mode in MODES.keys():
            v = scen_summary.get(mode, {}).get(scen, {})
            if v:
                lines.append(
                    f"| {mode} | {v['mean_rate_hz_mean']:.2f} | "
                    f"{v['ris_rate_hz_mean']:.2f} | "
                    f"{v['n_above_100hz_mean']:.1f} | {v['errors']} |"
                )
        lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=list(TIER_CONFIGS.keys()), default="screen",
                        help="screen=n5×30s (~4 hr), default=n10×60s (~26 hr)")
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()),
                        help="Subset of modes to run. Default: all 4.")
    parser.add_argument("--skip-scenarios", action="store_true",
                        help="Skip the scenario audit (only do phenotype).")
    args = parser.parse_args()

    tier_cfg = TIER_CONFIGS[args.tier]
    print("=" * 78)
    print(f"  Phase 1 sign-mode decision gauntlet")
    print(f"  Tier: {args.tier} (seeds={tier_cfg['seeds']}, "
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
        phen_path = ART / f"phase1_gauntlet_{mode_id}_{args.tier}_phenotype.csv"
        phen_df.to_csv(phen_path, index=False)
        print(f"  [{mode_id}] phenotype CSV → {phen_path}")
        all_phen_dfs.append(phen_df)

        if not args.skip_scenarios:
            scen_df = run_scenarios_for_mode(mode_id, tier_cfg)
            scen_path = ART / f"phase1_gauntlet_{mode_id}_{args.tier}_scenario.csv"
            scen_df.to_csv(scen_path, index=False)
            print(f"  [{mode_id}] scenario CSV → {scen_path}")
            all_scen_dfs.append(scen_df)

    # Aggregate + decision matrix
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

    summary_path = ART / f"phase1_gauntlet_{args.tier}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "tier": args.tier,
            "modes": {m: {**MODES[m], "sign_exceptions": (
                "DOCUMENTED_SIGN_EXCEPTIONS" if MODES[m]["sign_exceptions"] is None
                else {f"{a}->{b}": v for (a, b), v in MODES[m]["sign_exceptions"].items()}
            )} for m in args.modes},
            "phenotype_summary": phen_summary,
            "cascade_summary": cascade_summary,
            "scenario_summary": scen_summary,
            "wall_time_s": time.time() - t_start,
        }, f, indent=2, default=str)
    print(f"\n[summary JSON written] {summary_path}")

    decision_path = ART / f"phase1_gauntlet_{args.tier}_decision_matrix.md"
    write_decision_matrix(phen_summary, cascade_summary, scen_summary, decision_path)
    print(f"[decision matrix written] {decision_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*78}")
    print(f"  Gauntlet complete. Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
