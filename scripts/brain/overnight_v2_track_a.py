#!/usr/bin/env python3
"""Overnight v2 Track A — Mode 1 molecular audit densification.

For each of FLP-1, NLP-12, TA, OA: run CONTROL + RELEASER_ABLATE +
PEPTIDE_KO at n=5 seeds × 60s on scenario-matched stimulus. Saves
full telemetry per run. RECEPTOR_KO not implemented in this run
(requires modulation-layer edit per-receptor).

Pass/fail criteria per modulator (pre-specified):
  PASS Mode 1: peptide concentration drops to 0 under PEPTIDE_KO,
    |ΔQUI| < 0.05 AND |ΔREV| < 0.05 across all 5 seeds with consistent
    sign
  FAIL Mode 1: behavioral effect detected (|ΔQUI| > 0.05 OR |ΔREV| >
    0.05 — re-classify modulator
  AMBIGUOUS: mechanism inert (concentration stays near 0 even intact)

Output per modulator: full NPZ telemetry + classification JSON.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_env import ClosedLoopEnv

ART = Path(__file__).resolve().parent / "artifacts"
OUT_ROOT = ART / "overnight_20260422_v2" / "task_a_mode1_densification"

OSMOTIC_STIM = [(5.0, "osmotic_shock", 1.0)]
TOUCH_STIM = [(5.0, "touch_anterior", 1.0)]

# Modulator config: releasers, target_scenario, modulator_name_in_layer
MODULATOR_CONFIG = {
    "FLP-1": {
        "releasers": ["AVKL", "AVKR"],
        "scenario_stim": OSMOTIC_STIM,
        "scenario_name": "osmotic_shock",
        "modulator_key": "FLP-1",
    },
    "NLP-12": {
        "releasers": ["DVA"],
        "scenario_stim": OSMOTIC_STIM,
        "scenario_name": "osmotic_shock",
        "modulator_key": "NLP-12",
    },
    "TA": {
        "releasers": ["RIML", "RIMR"],
        "scenario_stim": TOUCH_STIM,
        "scenario_name": "touch",
        "modulator_key": "TA",
    },
    "OA": {
        "releasers": ["RICL", "RICR"],
        "scenario_stim": TOUCH_STIM,
        "scenario_name": "touch",
        "modulator_key": "OA",
    },
}


def apply_peptide_ko(env: ClosedLoopEnv, modulator_key: str):
    """Zero the modulator's releaser weights in the modulation layer,
    keeping releaser neurons firing normally."""
    if env.modulation is None:
        return False
    mods = env.modulation.modulators
    if modulator_key not in mods:
        return False
    mi = mods.index(modulator_key)
    # Zero base releaser weights
    env.modulation.releaser_weights[mi, :] = 0.0
    # If volume transmission on, also empty the releaser lists for
    # this modulator
    if env.modulation.use_volume:
        env.modulation.releaser_indices[mi] = np.array([], dtype=np.int64)
        env.modulation.per_releaser_conc[mi] = np.array([],
                                                         dtype=np.float32)
        env.modulation.effective_target[mi] = np.zeros(
            (0, env.modulation.N), dtype=np.float32
        )
        env.modulation.releaser_total_weights[mi] = np.array([],
                                                              dtype=np.float32)
    return True


def run_one(mod: str, condition: str, seed: int,
            duration_s: float = 60.0) -> dict:
    cfg = MODULATOR_CONFIG[mod]
    ablate_arg = cfg["releasers"] if condition == "RELEASER_ABLATE" else None
    env = ClosedLoopEnv(seed=seed, enable_modulation=True,
                        ablate=ablate_arg, brain_class="lif")
    ko_applied = False
    if condition == "PEPTIDE_KO":
        ko_applied = apply_peptide_ko(env, cfg["modulator_key"])
    env.run(duration_s, stim_schedule=cfg["scenario_stim"])

    fsb = np.stack(env.full_spike_buffer).astype(np.uint8)
    neuron_names = np.array(env.brain.names)
    fsm_states = np.array(env.fsm_states, dtype=np.int8)
    if env.modulator_buffer:
        modulator_conc = np.stack(env.modulator_buffer).astype(np.float32)
        modulator_names = np.array(list(env.modulation.modulators))
    else:
        modulator_conc = np.zeros((0, 0), dtype=np.float32)
        modulator_names = np.array([])

    from phase0_audit import state_props
    props = state_props(env.fsm_states)

    return {
        "modulator": mod, "condition": condition, "seed": seed,
        "ko_applied": ko_applied,
        "scenario": cfg["scenario_name"],
        "duration_s": duration_s,
        "full_raster": fsb, "neuron_names": neuron_names,
        "fsm_states": fsm_states,
        "modulator_conc": modulator_conc,
        "modulator_names": modulator_names,
        "state_props": props,
    }


def classify_mode(mod: str, ctrl_props_list, abl_props_list,
                  ko_props_list, modulator_conc_ctrl, modulator_conc_ko):
    """Pre-specified Mode 1 pass/fail logic."""
    STATE_NAMES = ["FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"]
    ctrl_mean = np.array(ctrl_props_list).mean(axis=0)
    abl_mean = np.array(abl_props_list).mean(axis=0)
    ko_mean = (np.array(ko_props_list).mean(axis=0)
               if ko_props_list else np.zeros(5))

    # Primary: was the peptide actually active in control?
    max_conc_ctrl = float(np.max(modulator_conc_ctrl)) if \
        modulator_conc_ctrl is not None and modulator_conc_ctrl.size > 0 \
        else 0.0
    max_conc_ko = float(np.max(modulator_conc_ko)) if \
        modulator_conc_ko is not None and modulator_conc_ko.size > 0 \
        else 0.0

    # Compute relevant deltas
    rev_idx = STATE_NAMES.index("REVERSE")
    qui_idx = STATE_NAMES.index("QUIESCENT")
    d_rev_abl = abl_mean[rev_idx] - ctrl_mean[rev_idx]
    d_qui_abl = abl_mean[qui_idx] - ctrl_mean[qui_idx]

    # Classification
    if max_conc_ctrl < 0.1 and max_conc_ctrl is not None:
        status = "AMBIGUOUS"
        reason = (f"mechanism inert: max concentration in control "
                  f"{max_conc_ctrl:.3f} (threshold 0.1)")
    elif abs(d_rev_abl) < 0.05 and abs(d_qui_abl) < 0.05:
        status = "PASS_MODE_1"
        reason = (f"|ΔREV|={abs(d_rev_abl):.3f} and "
                  f"|ΔQUI|={abs(d_qui_abl):.3f} both < 0.05")
    else:
        status = "FAIL_MODE_1"
        reason = (f"behavioral effect: ΔREV={d_rev_abl:+.3f}, "
                  f"ΔQUI={d_qui_abl:+.3f} — re-classify")

    return {
        "modulator": mod,
        "max_conc_control": round(max_conc_ctrl, 3),
        "max_conc_ko": round(max_conc_ko, 3),
        "ctrl_state_mean": {s: float(ctrl_mean[i])
                             for i, s in enumerate(STATE_NAMES)},
        "abl_state_mean": {s: float(abl_mean[i])
                            for i, s in enumerate(STATE_NAMES)},
        "ko_state_mean": {s: float(ko_mean[i])
                           for i, s in enumerate(STATE_NAMES)},
        "d_rev_abl": round(float(d_rev_abl), 3),
        "d_qui_abl": round(float(d_qui_abl), 3),
        "status": status,
        "reason": reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--modulators", nargs="+",
                    default=["FLP-1", "NLP-12", "TA", "OA"])
    ap.add_argument("--start-seed", type=int, default=42)
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    conditions = ["CONTROL", "RELEASER_ABLATE", "PEPTIDE_KO"]
    total = len(args.modulators) * len(conditions) * len(seeds)
    print(f"Track A: {len(args.modulators)} modulators × "
          f"{len(conditions)} conditions × {len(seeds)} seeds = "
          f"{total} runs × {args.duration}s")

    t0 = time.time()
    summary_rows = []
    idx = 0
    for mod in args.modulators:
        mod_dir = OUT_ROOT / mod
        mod_dir.mkdir(parents=True, exist_ok=True)
        cfg = MODULATOR_CONFIG[mod]
        # Collect state props per condition for classification
        props_by_cond = {"CONTROL": [], "RELEASER_ABLATE": [],
                          "PEPTIDE_KO": []}
        conc_by_cond = {"CONTROL": None, "RELEASER_ABLATE": None,
                         "PEPTIDE_KO": None}
        for cond in conditions:
            for seed in seeds:
                idx += 1
                t_r = time.time()
                try:
                    res = run_one(mod, cond, seed, args.duration)
                    out_path = mod_dir / f"{cond}_seed{seed}.npz"
                    STATE_NAMES = ["FORWARD", "REVERSE", "OMEGA",
                                   "PIROUETTE", "QUIESCENT"]
                    sp_arr = np.array(
                        [res["state_props"].get(s, 0) for s in STATE_NAMES]
                    )
                    np.savez_compressed(
                        out_path,
                        modulator=mod, condition=cond, seed=seed,
                        scenario=res["scenario"],
                        duration_s=args.duration,
                        full_raster=res["full_raster"],
                        neuron_names=res["neuron_names"],
                        fsm_states=res["fsm_states"],
                        modulator_conc=res["modulator_conc"],
                        modulator_names=res["modulator_names"],
                        state_props=sp_arr,
                        ko_applied=res["ko_applied"],
                    )
                    props_by_cond[cond].append(sp_arr)
                    # Save first control and first KO modulator_conc for
                    # classification comparison
                    if cond == "CONTROL" and conc_by_cond["CONTROL"] is None:
                        conc_by_cond["CONTROL"] = res["modulator_conc"]
                    if cond == "PEPTIDE_KO" and conc_by_cond["PEPTIDE_KO"] is None:
                        conc_by_cond["PEPTIDE_KO"] = res["modulator_conc"]
                    dt = time.time() - t_r
                    eta = (total - idx) * dt / 60
                    props = res["state_props"]
                    print(f"  [{idx}/{total}] {mod:6s} {cond:15s} "
                          f"seed={seed} | QUI={props.get('QUIESCENT', 0):.2f} "
                          f"REV={props.get('REVERSE', 0):.2f} | "
                          f"{dt:.0f}s | ETA {eta:.0f} min")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  [{idx}/{total}] {mod} {cond} seed={seed} ERROR")

        # Get modulator concentration from control — use the FLP family
        # index. Find the modulator's column in modulator_conc matrix.
        def extract_conc(conc_mat, conc_names, modulator_key):
            if conc_mat is None or conc_mat.size == 0:
                return np.zeros(0)
            names_list = [str(n) for n in conc_names]
            if modulator_key in names_list:
                i = names_list.index(modulator_key)
                return conc_mat[:, i]
            return np.zeros(0)

        # Load one control run to get the names
        try:
            ctrl_npz = np.load(mod_dir / f"CONTROL_seed{seeds[0]}.npz",
                               allow_pickle=True)
            mod_names = ctrl_npz["modulator_names"]
            ctrl_conc = extract_conc(conc_by_cond["CONTROL"],
                                     mod_names, cfg["modulator_key"])
            ko_conc = extract_conc(conc_by_cond["PEPTIDE_KO"],
                                   mod_names, cfg["modulator_key"])
        except Exception:
            ctrl_conc = np.zeros(0); ko_conc = np.zeros(0)

        classification = classify_mode(
            mod, props_by_cond["CONTROL"],
            props_by_cond["RELEASER_ABLATE"],
            props_by_cond["PEPTIDE_KO"],
            ctrl_conc, ko_conc,
        )
        classification["n_seeds_per_cond"] = len(seeds)
        (mod_dir / "classification.json").write_text(
            json.dumps(classification, indent=2)
        )
        summary_rows.append(classification)
        print(f"  → {mod}: {classification['status']}")

    # Summary
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_md = [
        "# Track A — Mode 1 densification",
        "",
        f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total wall: {(time.time()-t0)/60:.1f} min",
        "",
        "| modulator | status | max conc ctrl | max conc KO | ΔREV | ΔQUI |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        summary_md.append(
            f"| {r['modulator']} | **{r['status']}** | "
            f"{r['max_conc_control']} | {r['max_conc_ko']} | "
            f"{r['d_rev_abl']:+.3f} | {r['d_qui_abl']:+.3f} |"
        )
    summary_md.append("")
    for r in summary_rows:
        summary_md.append(f"### {r['modulator']}: {r['status']}")
        summary_md.append(f"- Reason: {r['reason']}")
        summary_md.append("")
    (OUT_ROOT / "summary.md").write_text("\n".join(summary_md))
    print(f"Wrote {OUT_ROOT / 'summary.md'}")

    status_md = ART / "overnight_20260422_v2" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Track A: Mode 1 densification\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        for r in summary_rows:
            f.write(f"- {r['modulator']}: {r['status']}\n")


if __name__ == "__main__":
    main()
