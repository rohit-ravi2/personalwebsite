#!/usr/bin/env python3
"""Track C2 — Molecular-layer baseline for all 9 modulators.

Loads D1 traces from last night's run (overnight_20260421/task1_d1/)
and for each modulator's CONTROL runs extracts:
  - peak modulator concentration
  - time to peak (kinetics)
  - per-target firing rate distribution

Classifies each modulator as:
  MECHANISM_OPERATING — conc rises during scenario (peak > 0.1 of cap)
  MECHANISM_INERT — conc stays near zero even in intact control

Output: task_c_parallel_analysis/c2_molecular_baseline/
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts"
D1_TRACES = ART / "overnight_20260421" / "task1_d1"
OUT_DIR = ART / "overnight_20260422_v2" / "task_c_parallel_analysis" / "c2_molecular_baseline"
OUT_MD = OUT_DIR / "summary.md"
OUT_CSV = OUT_DIR / "molecular_baseline.csv"

OPERATING_THRESHOLD = 0.1  # fraction of concentration cap (10.0)


def main():
    import time
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading D1 CONTROL traces from {D1_TRACES}")

    if not D1_TRACES.exists():
        print("D1 traces not found — task cannot run")
        sys.exit(1)

    by_mod = {}
    for p in sorted(D1_TRACES.glob("*_CONTROL_seed*.npz")):
        d = np.load(p, allow_pickle=True)
        mod = str(d["modulator"])
        by_mod.setdefault(mod, []).append(p)

    print(f"Found CONTROL traces for {len(by_mod)} modulators")

    rows = []
    for mod, paths in sorted(by_mod.items()):
        # Pull concentration trajectory for this modulator from first trace
        d = np.load(paths[0], allow_pickle=True)
        conc_mat = d["modulator_conc"]
        mod_names = [str(n) for n in d["modulator_names"]]
        if mod not in mod_names:
            rows.append({
                "modulator": mod, "status": "NO_LAYER_ENTRY",
                "peak_conc": 0, "time_to_peak_s": None,
                "reason": f"{mod} not in modulator_names list of trace",
            })
            continue
        mi = mod_names.index(mod)
        # Aggregate peak + time-to-peak across seeds
        peaks = []
        times_to_peak = []
        for p in paths:
            d2 = np.load(p, allow_pickle=True)
            conc = d2["modulator_conc"][:, mi]
            if conc.size == 0:
                continue
            peak = float(conc.max())
            t_peak_step = int(conc.argmax())
            t_peak_s = t_peak_step * 0.05  # 50 ms per step
            peaks.append(peak)
            times_to_peak.append(t_peak_s)

        mean_peak = float(np.mean(peaks)) if peaks else 0.0
        mean_ttp = float(np.mean(times_to_peak)) if times_to_peak else 0.0
        std_peak = float(np.std(peaks)) if peaks else 0.0

        if mean_peak > OPERATING_THRESHOLD * 10.0:  # >10% of cap=10
            status = "MECHANISM_OPERATING"
        else:
            status = "MECHANISM_INERT"

        rows.append({
            "modulator": mod,
            "status": status,
            "n_seeds_analyzed": len(peaks),
            "peak_conc_mean": round(mean_peak, 3),
            "peak_conc_std": round(std_peak, 3),
            "time_to_peak_s_mean": round(mean_ttp, 2),
            "reason": (f"peak {mean_peak:.2f} > threshold 1.0"
                       if status == "MECHANISM_OPERATING"
                       else f"peak {mean_peak:.2f} < threshold 1.0"),
        })
        print(f"  {mod}: peak={mean_peak:.2f}, ttp={mean_ttp:.1f}s → {status}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # Markdown summary
    lines = [
        "# Track C2 — Molecular baseline for 9 modulators",
        "",
        "For each modulator in the v3 layer, extract peak modulator ",
        "concentration from D1 CONTROL runs. Classify as ",
        "MECHANISM_OPERATING if mean peak > 10% of concentration cap, ",
        "MECHANISM_INERT otherwise.",
        "",
        "| modulator | status | mean peak | std | time-to-peak (s) | n seeds |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['modulator']}** | {r['status']} | "
            f"{r.get('peak_conc_mean', '-')} | "
            f"{r.get('peak_conc_std', '-')} | "
            f"{r.get('time_to_peak_s_mean', '-')} | "
            f"{r.get('n_seeds_analyzed', '-')} |"
        )
    lines.append("")
    operating = [r for r in rows if r.get("status") == "MECHANISM_OPERATING"]
    inert = [r for r in rows if r.get("status") == "MECHANISM_INERT"]
    lines.append(f"Operating: {len(operating)}/{len(rows)}")
    lines.append(f"Inert: {len(inert)}/{len(rows)}")
    lines.append("")
    lines.append("## Classification reason per modulator")
    lines.append("")
    for r in rows:
        lines.append(f"- **{r['modulator']}**: {r['reason']}")
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")

    status_md = ART / "overnight_20260422_v2" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Track C2: molecular baseline\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Headline: {len(operating)}/{len(rows)} operating, "
                f"{len(inert)}/{len(rows)} inert\n")


if __name__ == "__main__":
    main()
