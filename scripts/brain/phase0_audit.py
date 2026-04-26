#!/usr/bin/env python3
"""Phase 0 — Tiered ensemble audit runner.

Supersedes run_perturbation_ensemble.py (v3.0/3.1/3.2 sweep) and
run_v33_audit.py (single graded config) for Tier 2+ baseline work.
Preserves their ablation matrix and control/ablated pairing.

Tiers (--tier):
  quick       n=5 seeds × 30s   — dev iteration, signal-visible
  default     n=10 seeds × 60s  — phase-gate standard (Phase 0 baseline)
  audit-long  n=10 seeds × 120s — final phenotype claims (RIS-style longer runs)
  v33-compat  n=3 seeds × 20s × 3 configs — historical reproducibility

Modes (--mode):
  phenotype   6 ablations × 2 conditions (control/ablated) × N seeds
              Used for Chalfie/Turek/Flavell/Gordus phenotype baselines.
  scenario    6 scenarios × N seeds (no ablation)
              Used for state-distribution + cross-correlation baselines.
  calibration 1 run × duration × 1 seed — measures wall/simulated ratio.

Brain class: default "lif" (v3, shipped). Override with --brain-class graded
for Tier 1 stack comparison.

Output: artifacts/<prefix>_<mode>_<tier>.csv + markdown summary.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_env import ClosedLoopEnv  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"

STATE_NAMES = ["(unused)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE",
               "QUIESCENT"]

# --- Ablation matrix (imported structure from run_perturbation_ensemble) --
ABLATIONS = [
    ("RIS / osmotic_shock", ["RIS"],
     "osmotic_shock", [(5.0, "osmotic_shock", 1.0)]),
    ("NSM / food", ["NSML", "NSMR"],
     "food", [(2.0, "food_signal", 1.0)]),
    ("RIM / touch", ["RIML", "RIMR"],
     "touch", [(5.0, "touch_anterior", 1.0)]),
    ("AVA / touch", ["AVAL", "AVAR"],
     "touch", [(5.0, "touch_anterior", 1.0)]),
    ("AVB / spontaneous", ["AVBL", "AVBR"],
     "spontaneous", []),
    ("PDE / spontaneous", ["PDEL", "PDER"],
     "spontaneous", []),
]

# --- Scenarios for --mode scenario ------------------------------------
SCENARIOS = [
    ("spontaneous", []),
    ("touch", [(5.0, "touch_anterior", 1.0)]),
    ("osmotic_shock", [(5.0, "osmotic_shock", 1.0)]),
    ("food", [(2.0, "food_signal", 1.0)]),
    # Chemotaxis + aerotaxis require Environment setup — handled at run
    # time if scenario matches.
    ("chemotaxis", []),
    ("aerotaxis", []),
]

ENV_CONFIGS = {
    "chemotaxis": {"food_xy_m": (4e-3, 0.0), "peak_conc": 1.0,
                   "sigma_m": 3e-3},
    "aerotaxis": {"food_xy_m": (0.0, 0.0), "peak_conc": 0.0,
                  "sigma_m": 1e-3,
                  "aerotaxis": {"kind": "linear_o2", "o2_min": 0.07,
                                "o2_max": 0.21, "x_min_m": -10e-3,
                                "x_max_m": 10e-3, "preferred_o2": 0.12}},
}


# --- v33-compat config definitions ------------------------------------
V33_CONFIGS = [
    ("v3.0", {"modulator_tables_path": ART / "modulator_tables_v30.npz",
              "use_per_edge_glu_signs": False, "brain_class": "lif"}),
    ("v3.1", {"modulator_tables_path": ART / "modulator_tables.npz",
              "use_per_edge_glu_signs": False, "brain_class": "lif"}),
    ("v3.2", {"modulator_tables_path": ART / "modulator_tables.npz",
              "use_per_edge_glu_signs": True, "brain_class": "lif"}),
]

TIERS = {
    "quick": {"seeds": 5, "duration_s": 30.0, "configs": "single"},
    "default": {"seeds": 10, "duration_s": 60.0, "configs": "single"},
    "audit-long": {"seeds": 10, "duration_s": 120.0, "configs": "single"},
    "v33-compat": {"seeds": 3, "duration_s": 20.0, "configs": "v33"},
}


def seed_list(n: int) -> list[int]:
    """Deterministic seed sequence starting at 42 (matches legacy scripts)."""
    return list(range(42, 42 + n))


def state_props(fsm_states):
    if not fsm_states:
        return {n: 0.0 for n in STATE_NAMES[1:]}
    total = len(fsm_states)
    return {name: sum(1 for s in fsm_states if s == i) / total
            for i, name in enumerate(STATE_NAMES[1:], start=1)}


def _build_environment(scenario: str):
    """Construct an Environment object for chemotaxis/aerotaxis scenarios."""
    if scenario not in ENV_CONFIGS:
        return None
    from environment import (Environment, ChemoGradient, LinearGasField,
                             AerotaxisSensory)
    cfg = ENV_CONFIGS[scenario]
    grad = ChemoGradient(food_xy=cfg["food_xy_m"],
                         peak_conc=cfg["peak_conc"], sigma_m=cfg["sigma_m"])
    aero = None
    if cfg.get("aerotaxis"):
        a = cfg["aerotaxis"]
        o2_field = LinearGasField(
            min_frac=a["o2_min"], max_frac=a["o2_max"],
            x_min_m=a["x_min_m"], x_max_m=a["x_max_m"],
        )
        aero = AerotaxisSensory(o2_field=o2_field, co2_field=None,
                                preferred_o2_frac=a["preferred_o2"])
    return Environment(grad, initial_head_xy=(0.0, 0.0), aerotaxis=aero)


def _command_neuron_stats(env, names_of_interest):
    """For the scenario run, compute pre/peri-touch firing-rate stats
    on the listed neurons. Returns dict per neuron.

    Pre window: 1-5s (assumes stim at t=5s for touch scenarios).
    Peri window: 5-7s.
    Returns {} if fewer than 10 sync buckets — baseline only meaningful
    on full scenarios.
    """
    if len(env.full_spike_buffer) < 20:
        return {}
    fsb = np.stack(env.full_spike_buffer)  # (T, N) uint8
    from closed_loop_env import BRAIN_SYNC_MS
    dt_s = BRAIN_SYNC_MS / 1000.0
    times = np.arange(fsb.shape[0]) * dt_s
    pre_mask = (times >= 1.0) & (times < 5.0)
    peri_mask = (times >= 5.0) & (times < 7.0)
    out = {}
    for n in names_of_interest:
        if n not in env.brain.idx:
            continue
        i = env.brain.idx[n]
        pre_rate = float(fsb[pre_mask, i].sum()) / max(1, pre_mask.sum()) / dt_s
        peri_rate = float(fsb[peri_mask, i].sum()) / max(1, peri_mask.sum()) / dt_s
        out[n] = {"pre_hz": round(pre_rate, 3),
                  "peri_hz": round(peri_rate, 3),
                  "delta_hz": round(peri_rate - pre_rate, 3)}
    return out


COMMAND_CASCADE_NEURONS = [
    "ALML", "ALMR", "AVM",  # sensory
    "AIBL", "AIBR",         # first-order interneuron
    "AVAL", "AVAR",         # primary reversal command
    "AVEL", "AVER",         # secondary reversal command
    "AVDL", "AVDR",         # tertiary reversal command
    "AVBL", "AVBR",         # forward command
    "RIML", "RIMR",         # tyraminergic gate
    "RIS",                  # quiescence/FLP-11
    "PVCL", "PVCR",         # forward command
]


def run_phenotype_one(scen, stim, ablate, seed, cfg_kwargs, duration_s):
    """One phenotype run. Returns (state_props dict, command_rates dict)."""
    env = ClosedLoopEnv(seed=seed, enable_modulation=True, ablate=ablate,
                        **cfg_kwargs)
    env.run(duration_s, stim_schedule=stim)
    props = state_props(env.fsm_states)
    rates = _command_neuron_stats(env, COMMAND_CASCADE_NEURONS)
    return props, rates


def run_scenario_one(scenario, stim, seed, cfg_kwargs, duration_s):
    """One scenario run. Returns environment object (caller extracts stats)."""
    env_obj = _build_environment(scenario)
    env = ClosedLoopEnv(seed=seed, enable_modulation=True,
                        environment=env_obj, **cfg_kwargs)
    env.run(duration_s, stim_schedule=stim)
    return env


# --- Main dispatch ---------------------------------------------------


def mode_phenotype(args, tier_cfg):
    seeds = seed_list(tier_cfg["seeds"])
    duration_s = tier_cfg["duration_s"]
    if tier_cfg["configs"] == "v33":
        configs = V33_CONFIGS
    else:
        # Single config — use default modulator tables + shipped brain
        cfg_kwargs = {
            "modulator_tables_path": ART / "modulator_tables.npz",
            "use_per_edge_glu_signs": bool(args.use_per_edge_glu),
            "brain_class": args.brain_class,
        }
        if args.g_gap_ns is not None:
            cfg_kwargs["g_gap_ns"] = args.g_gap_ns
        cfg_label = f"{args.brain_class}_default"
        if args.use_per_edge_glu:
            cfg_label += "_peredge"
        configs = [(cfg_label, cfg_kwargs)]

    # Filter ablations by --ablations flag (substring match on labels)
    ablations = ABLATIONS
    if args.ablations:
        filt = [a.lower() for a in args.ablations]
        ablations = [a for a in ABLATIONS
                     if any(f in a[0].lower() for f in filt)]
        if not ablations:
            print(f"No ablations matched filter {args.ablations}")
            print(f"Available: {[a[0] for a in ABLATIONS]}")
            return
        print(f"Filtered to ablations: {[a[0] for a in ablations]}")

    total = len(configs) * len(ablations) * 2 * len(seeds)
    print(f"PHENOTYPE audit: tier={args.tier}  configs={len(configs)}  "
          f"ablations={len(ABLATIONS)}  seeds={len(seeds)}  "
          f"duration={duration_s}s")
    print(f"Total runs: {total}")
    t0 = time.time()

    rows = []
    run_idx = 0
    for cfg_name, cfg_kwargs in configs:
        for abl_label, neurons, scen, stim in ablations:
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
                    err = str(e)

                row = dict(
                    config=cfg_name, ablation=abl_label, scenario=scen,
                    seed=seed, duration_s=duration_s,
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
                eta_s = (total - run_idx) * dt
                print(f"  [{run_idx}/{total}] {cfg_name} | {abl_label} | "
                      f"seed={seed} | ΔREV={delta['REVERSE']:+.2f} "
                      f"ΔQUI={delta['QUIESCENT']:+.2f} | "
                      f"{dt:.0f}s wall (1 control + 1 ablated) | "
                      f"ETA {eta_s/60:.0f} min")
                out_csv = ART / f"{args.output_prefix}_phenotype_{args.tier}.csv"
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    total_wall = time.time() - t0
    total_sim = total * duration_s * 2  # 2 runs per row (ctrl + ablated)
    ratio = total_wall / total_sim if total_sim > 0 else 0.0
    print(f"\nPhenotype audit done: {total_wall/60:.1f} min wall, "
          f"{total_sim:.0f}s simulated, ratio={ratio:.2f}× wall/sim")
    return rows, {"wall_s": total_wall, "sim_s": total_sim, "ratio": ratio}


def mode_scenario(args, tier_cfg):
    seeds = seed_list(tier_cfg["seeds"])
    duration_s = tier_cfg["duration_s"]
    cfg_kwargs = {
        "modulator_tables_path": ART / "modulator_tables.npz",
        "use_per_edge_glu_signs": bool(args.use_per_edge_glu),
        "brain_class": args.brain_class,
    }
    if args.g_gap_ns is not None:
        cfg_kwargs["g_gap_ns"] = args.g_gap_ns
    if getattr(args, "disable_documented_sign_exceptions", False):
        cfg_kwargs["sign_exceptions"] = {}

    # Filter scenarios by --scenarios flag (substring match; case-insensitive)
    scenarios = SCENARIOS
    if getattr(args, "scenarios", None):
        filt = [s.lower() for s in args.scenarios]
        scenarios = [s for s in SCENARIOS if any(f == s[0].lower() for f in filt)]
        if not scenarios:
            print(f"No scenarios matched filter {args.scenarios}")
            print(f"Available: {[s[0] for s in SCENARIOS]}")
            return
        print(f"Filtered to scenarios: {[s[0] for s in scenarios]}")

    total = len(scenarios) * len(seeds)
    print(f"SCENARIO audit: tier={args.tier}  scenarios={len(scenarios)}  "
          f"seeds={len(seeds)}  duration={duration_s}s  "
          f"per_edge={cfg_kwargs['use_per_edge_glu_signs']}  "
          f"sign_exceptions_disabled={getattr(args, 'disable_documented_sign_exceptions', False)}")
    print(f"Total runs: {total}")
    t0 = time.time()

    rows = []
    run_idx = 0
    # For each (scenario, seed), we also save the per-neuron time-series
    # needed for T4-6 trajectory correlation baselines — stored as an
    # npz side-car keyed by (scenario, seed).
    trace_dir = ART / f"{args.output_prefix}_scenario_traces"
    trace_dir.mkdir(exist_ok=True)

    for scenario, stim in scenarios:
        for seed in seeds:
            run_idx += 1
            t_r = time.time()
            try:
                env = run_scenario_one(scenario, stim, seed, cfg_kwargs,
                                       duration_s)
                props = state_props(env.fsm_states)
                rates = _command_neuron_stats(env, COMMAND_CASCADE_NEURONS)

                # Save the full spike buffer + body frames for later
                # trajectory analysis (T4-1, T4-6).
                fsb = np.stack(env.full_spike_buffer).astype(np.uint8)
                neuron_names = np.array(env.brain.names)
                # Body positions (num_frames, num_segments, 2)
                body_xy = np.array([
                    [[float(p[0]), float(p[1])] for p in fr["positions"]]
                    for fr in env.body_frames
                ], dtype=np.float32)
                fsm_states = np.array(env.fsm_states, dtype=np.int8)
                mod_conc = (np.stack(env.modulator_buffer).astype(np.float32)
                            if env.modulator_buffer else np.zeros((0,),
                                                                  dtype=np.float32))
                np.savez_compressed(
                    trace_dir / f"{scenario}_seed{seed}.npz",
                    full_raster=fsb,
                    neuron_names=neuron_names,
                    body_xy=body_xy,
                    fsm_states=fsm_states,
                    modulator_conc=mod_conc,
                    duration_s=duration_s,
                )
                err = ""
            except Exception as e:
                props = {s: 0.0 for s in STATE_NAMES[1:]}
                rates = {}
                err = str(e)

            row = dict(
                scenario=scenario, seed=seed, duration_s=duration_s,
                FWD=props["FORWARD"], REV=props["REVERSE"],
                OMG=props["OMEGA"], PIR=props["PIROUETTE"],
                QUI=props["QUIESCENT"],
                rates_json=json.dumps(rates),
                error=err,
            )
            rows.append(row)
            dt = time.time() - t_r
            eta_s = (total - run_idx) * dt
            print(f"  [{run_idx}/{total}] {scenario} seed={seed} | "
                  f"FWD={props['FORWARD']:.2f} REV={props['REVERSE']:.2f} "
                  f"QUI={props['QUIESCENT']:.2f} | {dt:.0f}s wall | "
                  f"ETA {eta_s/60:.0f} min")
            out_csv = ART / f"{args.output_prefix}_scenario_{args.tier}.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    total_wall = time.time() - t0
    total_sim = total * duration_s
    ratio = total_wall / total_sim if total_sim > 0 else 0.0
    print(f"\nScenario audit done: {total_wall/60:.1f} min wall, "
          f"{total_sim:.0f}s simulated, ratio={ratio:.2f}× wall/sim")
    return rows, {"wall_s": total_wall, "sim_s": total_sim, "ratio": ratio}


def mode_calibration(args, tier_cfg):
    """Single-run wall-time calibration. One phenotype ablation,
    single seed, at the tier's duration. Prints measured ratio."""
    duration_s = tier_cfg["duration_s"]
    cfg_kwargs = {
        "modulator_tables_path": ART / "modulator_tables.npz",
        "use_per_edge_glu_signs": False,
        "brain_class": args.brain_class,
    }
    abl_label, neurons, scen, stim = ABLATIONS[3]  # AVA / touch (representative)
    seed = 42
    print(f"CALIBRATION run: {abl_label} seed={seed} duration={duration_s}s")
    t0 = time.time()
    ctrl_props, _ = run_phenotype_one(scen, stim, None, seed, cfg_kwargs,
                                      duration_s)
    t_ctrl = time.time() - t0
    abl_props, _ = run_phenotype_one(scen, stim, neurons, seed, cfg_kwargs,
                                     duration_s)
    t_abl = time.time() - t0 - t_ctrl
    total_wall = time.time() - t0
    total_sim = 2 * duration_s
    ratio = total_wall / total_sim
    print(f"\nControl run: {t_ctrl:.1f}s wall for {duration_s}s simulated "
          f"(ratio {t_ctrl/duration_s:.2f}×)")
    print(f"Ablated run: {t_abl:.1f}s wall for {duration_s}s simulated "
          f"(ratio {t_abl/duration_s:.2f}×)")
    print(f"Combined:    {total_wall:.1f}s wall for {total_sim}s simulated "
          f"(ratio {ratio:.2f}× wall/sim)")
    delta_rev = abl_props["REVERSE"] - ctrl_props["REVERSE"]
    print(f"ΔREV (seed=42) = {delta_rev:+.3f}")

    # Save summary
    out = ART / f"{args.output_prefix}_calibration.json"
    out.write_text(json.dumps({
        "ablation": abl_label,
        "seed": seed,
        "duration_s": duration_s,
        "wall_s_control": t_ctrl,
        "wall_s_ablated": t_abl,
        "wall_s_total": total_wall,
        "sim_s_total": total_sim,
        "ratio_wall_per_sim": ratio,
        "control_state_props": ctrl_props,
        "ablated_state_props": abl_props,
        "delta_REVERSE": delta_rev,
        "brain_class": args.brain_class,
    }, indent=2))
    print(f"Wrote {out}")
    return ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier",
                    choices=list(TIERS.keys()), default="default",
                    help="Ensemble tier (quick/default/audit-long/v33-compat)")
    ap.add_argument("--mode",
                    choices=["phenotype", "scenario", "calibration"],
                    default="phenotype",
                    help="Audit mode")
    ap.add_argument("--brain-class",
                    choices=["lif", "graded"], default="lif",
                    help="Brain implementation (default: shipped v3 LIF)")
    ap.add_argument("--output-prefix", default="phase0",
                    help="Output filename prefix (default: phase0)")
    ap.add_argument("--ablations", nargs="+", default=None,
                    help="Filter to ablations containing these substrings "
                         "(case-insensitive). E.g. --ablations AVA")
    ap.add_argument("--g-gap-ns", type=float, default=None,
                    help="Override gap-junction conductance in nS "
                         "(default: lif_brain G_GAP_DEFAULT = 0.1 nS).")
    ap.add_argument("--use-per-edge-glu", action="store_true",
                    help="Use CeNGEN-derived per-edge Glu receptor signs "
                         "(W_chem_per_edge) instead of per-neuron NT-sign "
                         "default. Flips ~518 chemical edges where Glu "
                         "sources target iGluR-dominant neurons.")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="Filter to scenarios by exact-match name. "
                         "E.g. --scenarios touch food. Default: all 6.")
    ap.add_argument("--disable-documented-sign-exceptions",
                    action="store_true",
                    help="Pass sign_exceptions={} to disable the "
                         "DOCUMENTED_SIGN_EXCEPTIONS overlay. Use for "
                         "pure per-edge / pure default-mode diagnostic "
                         "runs without curated overrides.")
    args = ap.parse_args()
    tier_cfg = TIERS[args.tier]

    if args.mode == "phenotype":
        mode_phenotype(args, tier_cfg)
    elif args.mode == "scenario":
        mode_scenario(args, tier_cfg)
    elif args.mode == "calibration":
        mode_calibration(args, tier_cfg)


if __name__ == "__main__":
    main()
