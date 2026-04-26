#!/usr/bin/env python3
"""Compare AVA/touch phenotype results across the post-voltage-fix
g_gap sweep.

Reads:
  artifacts/phase0_postvolt_phenotype_default.csv          (g_gap=0.1)
  artifacts/phase0_postvolt_gap03_phenotype_default.csv    (g_gap=0.3)
  artifacts/phase0_postvolt_gap10_phenotype_default.csv    (g_gap=1.0)

Compares against the historical Phase 0 baseline (pre-voltage-fix,
embedded below from phase0_baseline_report.md section 4).

Output: side-by-side table of per-neuron pre/peri/delta firing rates
across the cascade neurons, plus phenotype delta summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"

# Historical (pre-voltage-fix, g_gap=0.1) — from phase0_baseline_report.md
# section 4. Touch_anterior at t=5s, n=10, pre 1-5s peri 5-7s.
HISTORICAL = {
    "ALML":  {"pre": 2.4,  "peri": 88.2, "d": +85.8},
    "ALMR":  {"pre": 0.8,  "peri": 85.8, "d": +85.1},
    "AVM":   {"pre": 0.7,  "peri": 88.5, "d": +87.8},
    "AIBL":  {"pre": 8.3,  "peri": 10.2, "d":  +1.8},
    "AIBR":  {"pre": 14.8, "peri": 14.3, "d":  -0.4},
    "AVAL":  {"pre": 45.2, "peri": 41.1, "d":  -4.0},
    "AVAR":  {"pre": 46.0, "peri": 42.9, "d":  -3.1},
    "AVEL":  {"pre": 29.3, "peri": 24.4, "d":  -5.0},
    "AVER":  {"pre": 33.9, "peri": 29.4, "d":  -4.4},
    "AVDL":  {"pre": 42.2, "peri": 35.3, "d":  -6.9},
    "AVDR":  {"pre": 41.6, "peri": 34.4, "d":  -7.2},
    "RIML":  {"pre": 32.1, "peri": 31.6, "d":  -0.6},
    "RIMR":  {"pre": 32.2, "peri": 34.4, "d":  +2.1},
    "AVBL":  {"pre": None, "peri": None, "d": None},
    "AVBR":  {"pre": None, "peri": None, "d": None},
    "PVCL":  {"pre": None, "peri": None, "d": None},
    "PVCR":  {"pre": None, "peri": None, "d": None},
    "RIS":   {"pre": None, "peri": None, "d": None},
}

CASCADE_ORDER = [
    # Sensory
    "ALML", "ALMR", "AVM",
    # First-order interneuron (in textbook circuit; not actually wired
    # from ALM in this connectome)
    "AIBL", "AIBR",
    # The actual touch-cascade bridge
    "AVDL", "AVDR",
    # Reversal command
    "AVAL", "AVAR", "AVEL", "AVER",
    # Forward command (inhibited by ALM/AVM via Glu)
    "AVBL", "AVBR", "PVCL", "PVCR",
    # Tyraminergic gate
    "RIML", "RIMR",
    # Sleep
    "RIS",
]

CONDITIONS = [
    ("g_gap=0.1 (post-volt baseline)", "phase0_postvolt_phenotype_default.csv"),
    ("g_gap=0.3",                       "phase0_postvolt_gap03_phenotype_default.csv"),
    ("g_gap=1.0",                       "phase0_postvolt_gap10_phenotype_default.csv"),
    ("per-edge Glu signs (g_gap=0.1)",  "phase0_postvolt_peredge_phenotype_default.csv"),
]


def load_rates(csv_path: Path) -> dict:
    """Returns {neuron: {pre_mean, pre_sem, peri_mean, peri_sem, d_mean, d_sem, n_seeds}}."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df = df[df["ablation"].str.contains("AVA", na=False)]
    if len(df) == 0:
        return {}
    out: dict = {n: {"pre": [], "peri": [], "d": []} for n in CASCADE_ORDER}
    for _, row in df.iterrows():
        try:
            rates = json.loads(row["ctrl_rates_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        for n in CASCADE_ORDER:
            if n in rates:
                out[n]["pre"].append(rates[n]["pre_hz"])
                out[n]["peri"].append(rates[n]["peri_hz"])
                out[n]["d"].append(rates[n]["delta_hz"])
    summary = {}
    for n, vals in out.items():
        if not vals["pre"]:
            summary[n] = None
            continue
        pre_a = np.array(vals["pre"])
        peri_a = np.array(vals["peri"])
        d_a = np.array(vals["d"])
        n_seeds = len(pre_a)
        summary[n] = {
            "pre_mean": float(pre_a.mean()),
            "pre_sem":  float(pre_a.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
            "peri_mean": float(peri_a.mean()),
            "peri_sem":  float(peri_a.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
            "d_mean":    float(d_a.mean()),
            "d_sem":     float(d_a.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
            "n":         int(n_seeds),
        }
    return summary


def load_phenotype_summary(csv_path: Path) -> dict | None:
    """Returns {dREV_mean, dREV_sem, dQUI_mean, dQUI_sem, n_seeds, signs_neg}.

    For AVA / touch, biology says ablation should reduce reversal
    (ΔREV ≤ −0.40, all 10 seeds negative) per Chalfie 1985.
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df["ablation"].str.contains("AVA", na=False)]
    if len(df) == 0:
        return None
    drev = df["dREV"].to_numpy()
    dqui = df["dQUI"].to_numpy()
    return {
        "dREV_mean": float(drev.mean()),
        "dREV_sem":  float(drev.std(ddof=1) / np.sqrt(len(drev))) if len(drev) > 1 else 0.0,
        "dQUI_mean": float(dqui.mean()),
        "dQUI_sem":  float(dqui.std(ddof=1) / np.sqrt(len(dqui))) if len(dqui) > 1 else 0.0,
        "n_seeds":   int(len(drev)),
        "neg_seeds": int((drev < 0).sum()),
    }


def fmt_neuron_row(name: str, hist, conds: list[dict | None]) -> str:
    parts = [f"{name:<6}"]
    if hist["pre"] is not None:
        parts.append(f"{hist['pre']:6.1f} → {hist['peri']:6.1f}  Δ={hist['d']:+6.2f}")
    else:
        parts.append("                              ")
    for c in conds:
        if c is None:
            parts.append("       — no data —          ")
        else:
            parts.append(
                f"{c['pre_mean']:5.1f}±{c['pre_sem']:4.1f} → "
                f"{c['peri_mean']:5.1f}±{c['peri_sem']:4.1f}  "
                f"Δ={c['d_mean']:+5.2f}±{c['d_sem']:4.2f}"
            )
    return "  ".join(parts)


def main():
    print("=" * 130)
    print("AVA/touch baseline — voltage-fix + g_gap sweep")
    print("Comparison: control runs only (n=10 × 60s, touch_anterior at t=5s)")
    print("Pre window 1-5s, peri window 5-7s")
    print("=" * 130)

    cond_data = []
    for label, fname in CONDITIONS:
        rates = load_rates(ART / fname)
        cond_data.append((label, rates))

    # Header
    header = f"{'neuron':<6}  {'historical (pre-volt)':<30}"
    for label, _ in cond_data:
        header += f"  {'POST: ' + label:<32}"
    print(header)
    print("-" * len(header))

    for n in CASCADE_ORDER:
        hist = HISTORICAL[n]
        rows_for_neuron = []
        for label, rates in cond_data:
            rows_for_neuron.append(rates.get(n))
        print(fmt_neuron_row(n, hist, rows_for_neuron))

    # Phenotype summary (AVA ablation reproduces Chalfie reversal abolition?)
    print()
    print("=" * 130)
    print("Phenotype: AVA-ablation effect on touch (Chalfie 1985 expects ΔREV ≤ −0.40 across all seeds)")
    print("=" * 130)
    print(f"{'condition':<28}  {'dREV (mean ± SEM)':<22}  {'neg seeds':<10}  "
          f"{'dQUI':<22}  {'verdict':<25}")
    print("-" * 130)
    for label, fname in CONDITIONS:
        ph = load_phenotype_summary(ART / fname)
        if ph is None:
            print(f"{label:<28}  {'(no data yet)':<22}")
            continue
        verdict = "PASS" if ph["dREV_mean"] <= -0.40 and ph["neg_seeds"] >= 8 else (
            "DIRECTIONAL" if ph["dREV_mean"] < -0.10 and ph["neg_seeds"] >= 6 else "NULL"
        )
        print(f"{label:<28}  {ph['dREV_mean']:+5.2f} ± {ph['dREV_sem']:.2f}        "
              f"{ph['neg_seeds']}/{ph['n_seeds']}        "
              f"{ph['dQUI_mean']:+5.2f} ± {ph['dQUI_sem']:.2f}        "
              f"{verdict}")

    # Highlight key cascade movements vs historical
    print()
    print("=" * 130)
    print("Cascade-firing assessment (key question: does AVA go UP on touch in any condition?)")
    print("=" * 130)
    for label, rates in cond_data:
        if not rates:
            print(f"{label}: (no data yet)")
            continue
        avd_d = []
        ava_d = []
        aib_d = []
        for n in ("AVDL", "AVDR"):
            if rates.get(n) is not None:
                avd_d.append(rates[n]["d_mean"])
        for n in ("AVAL", "AVAR"):
            if rates.get(n) is not None:
                ava_d.append(rates[n]["d_mean"])
        for n in ("AIBL", "AIBR"):
            if rates.get(n) is not None:
                aib_d.append(rates[n]["d_mean"])
        avd_mean = np.mean(avd_d) if avd_d else float("nan")
        ava_mean = np.mean(ava_d) if ava_d else float("nan")
        aib_mean = np.mean(aib_d) if aib_d else float("nan")
        ava_dir = "UP   ✓" if ava_mean > +1.0 else ("flat" if abs(ava_mean) <= 1.0 else "DOWN ✗")
        avd_dir = "UP   ✓" if avd_mean > +1.0 else ("flat" if abs(avd_mean) <= 1.0 else "DOWN ✗")
        print(f"{label:<28}  AIB Δ={aib_mean:+5.2f}  AVD Δ={avd_mean:+5.2f} ({avd_dir})  "
              f"AVA Δ={ava_mean:+5.2f} ({ava_dir})")

    print()
    print("=" * 130)


if __name__ == "__main__":
    main()
