#!/usr/bin/env python3
"""Phase 0 — Analysis + threshold ratification.

Ingests the outputs of:
  - phase0_calibration.json           (wall/sim ratio)
  - phase0_phenotype_default.csv      (T4-3, T4-5 baselines)
  - phase0_scenario_default.csv       (T4-1, T4-4, T4-6 state distributions)
  - phase0_scenario_traces/           (per-seed spike raster, body xy, mod conc)
  - phase0_plateau_baseline.csv       (T4-2 baseline)
  - phase0_cascade_baseline.npz       (T2-#4 baseline shapes)
  - phase0_swap_jitter.json           (Cython-migration decision)

Produces the ratified threshold table for each phase and writes:
  - artifacts/phase0_baseline_report.md  (primary measurement document)
  - docs/current-state-summary.md         (narrative summary)

Run after all baseline measurements are collected.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts"
REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

REPORT_MD = ART / "phase0_baseline_report.md"
SUMMARY_MD = DOCS / "current-state-summary.md"


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def safe_load_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def analyze_ratio(calib):
    """Extract wall/sim ratio + feasibility for W0.2/W0.3."""
    if calib is None:
        return None
    ratio = calib["ratio_wall_per_sim"]
    # Predictions at this ratio
    w02_s = 120 * 60 * ratio  # 120 runs × 60s sim
    w03_s = 60 * 60 * ratio   # 60 runs × 60s sim
    return {
        "measured_ratio": ratio,
        "calibration_wall_s": calib["wall_s_total"],
        "calibration_sim_s": calib["sim_s_total"],
        "w02_predicted_hours": round(w02_s / 3600, 2),
        "w03_predicted_hours": round(w03_s / 3600, 2),
        "vs_v33_extrapolation": ratio / 2.69,
    }


def analyze_phenotype(df):
    """Extract T4-3 + T4-5 baselines."""
    if df is None or len(df) == 0:
        return None
    out = {"n_runs": len(df), "n_seeds": df["seed"].nunique(),
           "duration_s": df["duration_s"].iloc[0],
           "per_ablation": {}}
    for abl in df["ablation"].unique():
        sub = df[df["ablation"] == abl]
        state_stats = {}
        for col, label in [("dREV", "REVERSE"), ("dFWD", "FORWARD"),
                           ("dOMG", "OMEGA"), ("dPIR", "PIROUETTE"),
                           ("dQUI", "QUIESCENT")]:
            vals = sub[col].values
            n_neg = int(np.sum(vals < 0))
            n_pos = int(np.sum(vals > 0))
            state_stats[label] = {
                "mean": round(float(vals.mean()), 3),
                "std": round(float(vals.std()), 3),
                "sem": round(float(vals.std() / np.sqrt(len(vals))), 3),
                "n_neg": n_neg, "n_pos": n_pos, "n_total": len(vals),
            }
        # Aggregate rates if any have them
        out["per_ablation"][abl] = state_stats
    return out


def analyze_scenario(df, traces_dir):
    """Extract T4-1 body-curvature baseline candidates, T4-4 firing
    variance, T4-6 scenario-level state distributions."""
    if df is None or len(df) == 0:
        return None
    out = {"n_runs": len(df), "n_seeds": df["seed"].nunique(),
           "duration_s": df["duration_s"].iloc[0],
           "per_scenario": {}}
    for scen in df["scenario"].unique():
        sub = df[df["scenario"] == scen]
        state_stats = {}
        for col, label in [("FWD", "FORWARD"), ("REV", "REVERSE"),
                           ("OMG", "OMEGA"), ("PIR", "PIROUETTE"),
                           ("QUI", "QUIESCENT")]:
            vals = sub[col].values
            state_stats[label] = {
                "mean": round(float(vals.mean()), 3),
                "std": round(float(vals.std()), 3),
            }
        out["per_scenario"][scen] = state_stats
    # Load touch traces to extract AVA/AVE/AVD pre/peri statistics
    # (T4-3 scenario baseline, higher-resolution than phenotype-audit
    # neuron-rate output).
    touch_stats = {}
    if traces_dir.exists():
        command_rates_by_seed = {}
        for trace_file in traces_dir.glob("touch_seed*.npz"):
            seed = int(trace_file.stem.split("seed")[-1])
            data = np.load(trace_file)
            fsb = data["full_raster"]
            names = list(data["neuron_names"])
            duration_s = float(data["duration_s"])
            dt_s = duration_s / fsb.shape[0]
            times = np.arange(fsb.shape[0]) * dt_s
            pre_mask = (times >= 1.0) & (times < 5.0)
            peri_mask = (times >= 5.0) & (times < 7.0)
            for n in ["ALML", "ALMR", "AVM", "AIBL", "AIBR",
                      "AVAL", "AVAR", "AVEL", "AVER",
                      "AVDL", "AVDR", "RIML", "RIMR"]:
                if n not in names:
                    continue
                i = names.index(n)
                pre_rate = float(fsb[pre_mask, i].sum()) / max(1, pre_mask.sum()) / dt_s
                peri_rate = float(fsb[peri_mask, i].sum()) / max(1, peri_mask.sum()) / dt_s
                command_rates_by_seed.setdefault(n, []).append({
                    "seed": seed, "pre_hz": pre_rate, "peri_hz": peri_rate,
                    "delta_hz": peri_rate - pre_rate,
                })
        for n, recs in command_rates_by_seed.items():
            pre_vals = np.array([r["pre_hz"] for r in recs])
            peri_vals = np.array([r["peri_hz"] for r in recs])
            delta_vals = np.array([r["delta_hz"] for r in recs])
            touch_stats[n] = {
                "n_seeds": len(recs),
                "pre_mean_hz": round(float(pre_vals.mean()), 2),
                "pre_std_hz": round(float(pre_vals.std()), 2),
                "peri_mean_hz": round(float(peri_vals.mean()), 2),
                "peri_std_hz": round(float(peri_vals.std()), 2),
                "delta_mean_hz": round(float(delta_vals.mean()), 2),
                "delta_std_hz": round(float(delta_vals.std()), 2),
            }
    out["touch_command_rates"] = touch_stats
    return out


def build_threshold_table(calib_stats, phen_stats, scen_stats):
    """Given the baselines, construct ratified pass thresholds per phase."""
    rows = []

    # T4-3: AVA peri-touch synaptic calibration
    if scen_stats and "AVAL" in scen_stats.get("touch_command_rates", {}):
        ava = scen_stats["touch_command_rates"]["AVAL"]
        aver = scen_stats["touch_command_rates"].get("AVER", {})
        baseline_delta = ava["delta_mean_hz"]
        baseline_peri = ava["peri_mean_hz"]
        rows.append({
            "phase": "T4-3 synaptic calibration",
            "baseline": f"AVAL peri={baseline_peri:.1f}±{ava['peri_std_hz']:.1f}Hz, "
                        f"Δ={baseline_delta:+.1f}±{ava['delta_std_hz']:.1f}Hz",
            "threshold": "AVAL peri ≥ 20 Hz AND Δ ≥ +15 Hz on ≥ 8/10 seeds",
            "reference": "Chalfie 1985 command cascade biology",
        })
    else:
        rows.append({
            "phase": "T4-3 synaptic calibration",
            "baseline": "touch scenario traces not loaded",
            "threshold": "AVAL peri ≥ 20 Hz AND Δ ≥ +15 Hz (provisional)",
            "reference": "Chalfie 1985 command cascade biology",
        })

    # T4-3 phenotype side: ΔREV from AVA ablation
    if phen_stats and "AVA / touch" in phen_stats.get("per_ablation", {}):
        ava_rev = phen_stats["per_ablation"]["AVA / touch"]["REVERSE"]
        rows.append({
            "phase": "T4-3 ActivityFSM phenotype",
            "baseline": f"ΔREV={ava_rev['mean']:+.2f}±{ava_rev['std']:.2f} "
                        f"(classifier-mode; see T0 finding for why)",
            "threshold": "ActivityFSM-mode ΔREV ≤ −0.40 at n=5/10 seeds, all negative",
            "reference": "Chalfie 1985",
        })

    # T4-5 RIS ablation
    if phen_stats and "RIS / osmotic_shock" in phen_stats.get("per_ablation", {}):
        ris_qui = phen_stats["per_ablation"]["RIS / osmotic_shock"]["QUIESCENT"]
        rows.append({
            "phase": "T4-5 RIS/Turek phenotype",
            "baseline": f"ΔQUI={ris_qui['mean']:+.2f}±{ris_qui['std']:.2f} "
                        f"(n={ris_qui['n_total']} seeds)",
            "threshold": "ΔQUI ≤ −0.30 with 95% CI excluding zero",
            "reference": "Turek 2016",
        })

    # Plateau baseline (loaded separately)
    rows.append({
        "phase": "T4-2 AVA plateau (Gao & Hobert 2020)",
        "baseline": "see phase0_plateau_baseline.csv — AVA g_ca_ns, tau_h_ms",
        "threshold": "plateau duration ∈ [480, 720] ms (20% of 600 ms target); "
                     "amplitude ∈ [18, 22] mV above rest",
        "reference": "Gao & Hobert 2020 Fig 3",
    })

    # Sensory cascades
    rows.append({
        "phase": "T2-#4 sensory cascade calibration",
        "baseline": "see phase0_cascade_baseline.npz — uncalibrated shape per cascade",
        "threshold": "Each cascade's rate trace ≤ 10% Frechet distance to "
                     "digitised published ΔF/F (Thiele 2009, Chalasani 2007, "
                     "Hilliard 2005, Clark 2006, O'Hagan 2005)",
        "reference": "per-cascade primary refs",
    })

    # T4-1 curvature (from scenario body traces)
    rows.append({
        "phase": "T4-1 motor coupling (curvature ρ)",
        "baseline": "CPG-driven forward bout body trace in scenario_traces/",
        "threshold": "median ρ vs Tierpsy pool ≥ max(0.6, CPG_baseline + 0.15)",
        "reference": "Tierpsy centerlines in data/external/wormpose",
    })

    # T4-6 trajectory correlation
    rows.append({
        "phase": "T4-6 trajectory correlation",
        "baseline": "per-neuron × per-event ρ distribution vs Atanas (current v3 LIF)",
        "threshold": "median ρ increases ≥ +0.10 relative to baseline; "
                     "tail named in paper",
        "reference": "Atanas 2023 (10 worms)",
    })

    return rows


def write_report(calib_stats, phen_stats, scen_stats, plateau_df,
                 cascade_npz_path, jitter_stats, thresholds):
    """Write the primary baseline report."""
    lines = ["# Phase 0 — Baseline measurement report",
             "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             "",
             "Current-state measurements across the Tier 2 / Tier 4 "
             "validation surface. Every downstream phase's pass threshold is "
             "ratified against these numbers — see the *Ratified thresholds* "
             "section.",
             ""]

    # --- Compute ----------------------------------------------------
    lines.append("## 1. Wall/simulated compute ratio")
    lines.append("")
    if calib_stats:
        lines.append(f"- **Measured ratio:** {calib_stats['measured_ratio']:.2f}× "
                     f"wall seconds per simulated second "
                     f"({calib_stats['calibration_wall_s']:.0f} s wall for "
                     f"{calib_stats['calibration_sim_s']:.0f} s sim).")
        lines.append(f"- **vs v3.3-audit extrapolation (2.69×):** "
                     f"{calib_stats['vs_v33_extrapolation']:.2f}× "
                     f"({'within' if abs(calib_stats['vs_v33_extrapolation'] - 1) < 0.25 else 'significantly above'} "
                     f"tolerance).")
        lines.append(f"- **W0.2 full phenotype audit predicted:** "
                     f"{calib_stats['w02_predicted_hours']:.1f} hours wall "
                     f"(120 runs × 60s).")
        lines.append(f"- **W0.3 full scenario audit predicted:** "
                     f"{calib_stats['w03_predicted_hours']:.1f} hours wall "
                     f"(60 runs × 60s).")
    else:
        lines.append("- Calibration data not yet available.")
    lines.append("")

    # --- Swap jitter ------------------------------------------------
    lines.append("## 2. Swap-jitter (Cython-migration decision)")
    lines.append("")
    if jitter_stats:
        lines.append(f"- Mean wall: **{jitter_stats['mean_ms']} ms** "
                     f"(σ = {jitter_stats['std_ms']} ms, CV = "
                     f"{jitter_stats['cv_pct']}%)")
        lines.append(f"- Tolerance for T4-2 plateau discrimination: "
                     f"{jitter_stats['jitter_tolerance_ms']} ms.")
        if jitter_stats['exceeds_tolerance']:
            lines.append("- ⚠️ **Tolerance exceeded.** Phase 1.5 Cython-target "
                         "migration recommended before T4-2 calibration.")
        else:
            lines.append("- ✅ Within tolerance. T4-2 calibration can proceed "
                         "on current hardware.")
    lines.append("")

    # --- Phenotype ablation baseline (T4-3, T4-5) ------------------
    lines.append("## 3. Phenotype ablation baseline (n=10 × 60s)")
    lines.append("")
    if phen_stats:
        lines.append("Current-state ablation deltas, v3 LIF brain, "
                     "classifier-mode FSM.")
        lines.append("")
        lines.append("| ablation | ΔREV | ΔQUI | ΔFWD | ΔOMG | ΔPIR |")
        lines.append("|---|---|---|---|---|---|")
        for abl, stats in phen_stats["per_ablation"].items():
            row = [abl]
            for state in ["REVERSE", "QUIESCENT", "FORWARD", "OMEGA",
                          "PIROUETTE"]:
                s = stats[state]
                row.append(f"{s['mean']:+.2f} ± {s['std']:.2f} "
                           f"({s['n_neg']}/{s['n_total']}↓)")
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("- Phenotype audit (W0.2) not yet complete.")
    lines.append("")

    # --- Scenario + touch command rates (T4-3 sensory cascade) -----
    lines.append("## 4. Touch-scenario command-neuron rates (T4-3 baseline)")
    lines.append("")
    if scen_stats and scen_stats.get("touch_command_rates"):
        lines.append("n=10 seeds, touch_anterior at t=5s. Pre window 1-5s, "
                     "peri window 5-7s.")
        lines.append("")
        lines.append("| neuron | pre (Hz) | peri (Hz) | Δ (Hz) | cascade role |")
        lines.append("|---|---|---|---|---|")
        roles = {
            "ALML": "sensory", "ALMR": "sensory", "AVM": "sensory",
            "AIBL": "1st-order interneuron", "AIBR": "1st-order interneuron",
            "AVAL": "reversal command", "AVAR": "reversal command",
            "AVEL": "secondary reversal", "AVER": "secondary reversal",
            "AVDL": "tertiary reversal", "AVDR": "tertiary reversal",
            "RIML": "tyraminergic gate", "RIMR": "tyraminergic gate",
        }
        for n, s in scen_stats["touch_command_rates"].items():
            role = roles.get(n, "")
            lines.append(
                f"| {n} | {s['pre_mean_hz']:.1f} ± {s['pre_std_hz']:.1f} | "
                f"{s['peri_mean_hz']:.1f} ± {s['peri_std_hz']:.1f} | "
                f"{s['delta_mean_hz']:+.1f} ± {s['delta_std_hz']:.1f} | "
                f"{role} |"
            )
    else:
        lines.append("- Scenario audit (W0.3) not yet complete.")
    lines.append("")

    # --- Scenario state distribution -------------------------------
    lines.append("## 5. Scenario state distributions (T4-6 baseline)")
    lines.append("")
    if scen_stats:
        lines.append("| scenario | FWD | REV | OMG | PIR | QUI |")
        lines.append("|---|---|---|---|---|---|")
        for scen, states in scen_stats["per_scenario"].items():
            row = [scen]
            for state in ["FORWARD", "REVERSE", "OMEGA", "PIROUETTE",
                          "QUIESCENT"]:
                s = states[state]
                row.append(f"{s['mean']:.2f} ± {s['std']:.2f}")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- Plateau baseline ------------------------------------------
    lines.append("## 6. T4-2 plateau baseline (15 neurons × 50 pA / 100 ms)")
    lines.append("")
    if plateau_df is not None:
        n_pass = int((plateau_df["status"] == "PASS").sum())
        n_total = len(plateau_df)
        lines.append(f"- **{n_pass}/{n_total} neurons** within ±20% of "
                     f"Gao & Hobert / Wang / Kawano targets.")
        failing = plateau_df[plateau_df["status"] == "FAIL"][
            "neuron"].tolist()
        if failing:
            lines.append(f"- Failing neurons: {', '.join(failing)}")
        lines.append("")
        lines.append("See `phase0_plateau_baseline.md` for per-neuron "
                     "voltages and gaps.")
    else:
        lines.append("- Plateau baseline not yet run.")
    lines.append("")

    # --- Cascade baseline ------------------------------------------
    lines.append("## 7. T2-#4 sensory cascade baseline (5 cascades)")
    lines.append("")
    if cascade_npz_path.exists():
        data = np.load(cascade_npz_path)
        peaks = {
            "ASE": float(np.max(data["ASE_rates"])),
            "AWC": float(np.max(data["AWC_rates"])),
            "ASH": float(np.max(data["ASH_rates"])),
            "AFD": float(np.max(data["AFD_rates"])),
            "ALM": float(np.max(data["ALM_rates"])),
        }
        lines.append("Uncalibrated cascade peak rates under canonical stimuli:")
        lines.append("")
        for k, v in peaks.items():
            lines.append(f"- **{k}**: peak {v:.1f} Hz")
        lines.append("")
        lines.append("References in `references/` — digitisation pending "
                     "for Frechet-distance evaluation.")
    else:
        lines.append("- Cascade baseline not yet run.")
    lines.append("")

    # --- Ratified thresholds ---------------------------------------
    lines.append("## 8. Ratified pass thresholds per phase")
    lines.append("")
    lines.append("| phase | current baseline | pass threshold | reference |")
    lines.append("|---|---|---|---|")
    for r in thresholds:
        lines.append(f"| **{r['phase']}** | {r['baseline']} | "
                     f"{r['threshold']} | {r['reference']} |")
    lines.append("")

    # --- Audit infrastructure status -------------------------------
    lines.append("## 9. Audit infrastructure status")
    lines.append("")
    lines.append("`phase0_audit.py` implements the 3-tier config:")
    lines.append("")
    lines.append("| tier | seeds | duration | use |")
    lines.append("|---|---|---|---|")
    lines.append("| `--quick` | 5 | 30 s | dev iteration (signal-visible) |")
    lines.append("| `--default` | 10 | 60 s | phase gate / Phase 0 baseline |")
    lines.append("| `--audit-long` | 10 | 120 s | final phenotype claims |")
    lines.append("| `--v33-compat` | 3 | 20 s × 3 configs | historical reproducibility |")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines))
    print(f"Wrote {REPORT_MD}")


def write_summary(calib_stats, thresholds):
    """~300-word narrative layer on top of the baseline report."""
    lines = ["# Current state summary",
             "",
             f"*Last updated: {time.strftime('%Y-%m-%d')}*",
             "",
             "Narrative layer on top of `scripts/brain/artifacts/"
             "phase0_baseline_report.md`. Updated at phase boundaries.",
             "",
             "## Simulator execution profile",
             ""]
    if calib_stats:
        lines.append(
            f"Brian2 2.9.0 on CPU (numpy codegen target). Measured "
            f"wall/simulated ratio on the shipped v3 LIF brain: "
            f"**{calib_stats['measured_ratio']:.2f}×**. Full phenotype audit "
            f"at n=10 × 60s: ~{calib_stats['w02_predicted_hours']:.1f} "
            f"hours wall."
        )
    else:
        lines.append("Execution ratio pending calibration run.")
    lines.append("")

    lines.append("## Phase roadmap status")
    lines.append("")
    status = [
        ("Phase 0", "in progress", "baseline measurement + audit infra"),
        ("T2-#4 sensory cascade calibration", "pending", "baseline captured; "
         "digitisation pending for Frechet eval"),
        ("T4-2 compartmental plateau calibration", "pending", "baseline "
         "captured; Gao & Hobert 2020 digitisation pending"),
        ("T4-1 motor coupling", "pending", "CPG baseline captured for "
         "curvature-ρ comparison"),
        ("T4-3 synaptic calibration (T0 fix)", "pending", "baseline captured; "
         "currently AVA doesn't fire on touch"),
        ("T4-4 CeNGEN-conductance coupling", "pending", "end of sequence; "
         "architectural overlay"),
        ("T4-5 INS-family peptide expansion", "pending", "6-peptide "
         "selection confirmed"),
        ("T4-6 trajectory correlation", "pending", "baseline ρ distribution "
         "captured; capstone"),
    ]
    for phase, st, note in status:
        lines.append(f"- **{phase}** — {st}. {note}")
    lines.append("")

    lines.append("## Ratified thresholds")
    lines.append("")
    for r in thresholds:
        lines.append(f"- **{r['phase']}** — {r['threshold']}")
    lines.append("")

    lines.append("## References")
    lines.append("")
    lines.append("- Primary measurement: `scripts/brain/artifacts/"
                 "phase0_baseline_report.md`")
    lines.append("- Per-subsystem: `phase0_plateau_baseline.md`, "
                 "`phase0_cascade_baseline.md`, `phase0_swap_jitter.md`")
    lines.append("- Digitised reference traces: `scripts/brain/references/`")
    lines.append("")

    DOCS.mkdir(exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines))
    print(f"Wrote {SUMMARY_MD}")


def main():
    calib = load_json(ART / "phase0_calibration.json")
    phen_df = safe_load_csv(ART / "phase0_phenotype_default.csv")
    scen_df = safe_load_csv(ART / "phase0_scenario_default.csv")
    plateau_df = safe_load_csv(ART / "phase0_plateau_baseline.csv")
    jitter = load_json(ART / "phase0_swap_jitter.json")
    cascade_path = ART / "phase0_cascade_baseline.npz"
    traces_dir = ART / "phase0_scenario_traces"

    calib_stats = analyze_ratio(calib)
    phen_stats = analyze_phenotype(phen_df)
    scen_stats = analyze_scenario(scen_df, traces_dir)

    thresholds = build_threshold_table(calib_stats, phen_stats, scen_stats)
    write_report(calib_stats, phen_stats, scen_stats, plateau_df,
                 cascade_path, jitter, thresholds)
    write_summary(calib_stats, thresholds)


if __name__ == "__main__":
    main()
