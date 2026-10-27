"""RNAFold-v2 results figure + fact recompute, from a clone of github.com/rohit-ravi2/rnafold-v2.

Usage: python rnafold-v2_results.py <repo_clone_dir> <out_png>

Sources (all inside the repo):
- evaluation/results/*.csv: the TM-align harness run. Recomputed here: how many sample evaluations
  produced a TM-score (none do).
- Notebook OUTPUT cells, each tied to the code in the same cell (the predicted CIFs and reference
  structures are not in the repo, so the scores cannot be recomputed from coordinates):
    pipeline5.ipynb cell 75: RNAPro sample 0 vs reference, residues paired by index (all 24 per-target values)
    pipeline5.ipynb cell 77: RNAPro sample 0 vs reference, residues paired by local sequence alignment
                             (24-target summary + the first 10 per-target rows the notebook displayed)
    pipeline3.ipynb cell 17: RNAPro best-of-5 (seed_101), backbone atoms paired by index (24 per-target values)
    pipeline5.ipynb cell 40-43: template retrieval top-1 identity (shows the target's own entry is in the bank)
    pipeline5.ipynb cell 98: Pipeline5 stage benchmark (leaky, reported only to be excluded)
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo, out_png = Path(sys.argv[1]), sys.argv[2]


def outputs(nb, i):
    cell = json.load(open(repo / nb))["cells"][i]
    txt = []
    for o in cell.get("outputs", []):
        txt.append("".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", [])))
    return "\n".join(txt)


# ---- 1. TM-align harness CSVs -------------------------------------------------------------
res = repo / "evaluation/results"
allv = pd.read_csv(res / "all_sample_evaluations.csv")
fail = pd.read_csv(res / "evaluation_failures.csv")
best = pd.read_csv(res / "best_per_target.csv")
n_rows, n_scored = len(allv), int(allv["TM_score"].notna().sum())
distinct = allv.drop_duplicates(["target_id", "sample_name"])
parse_fail = int(allv["raw_output"].str.contains("Cannot parse file").sum())
print(f"harness: {n_rows} rows, {allv.target_id.nunique()} targets, {len(distinct)} distinct samples, "
      f"{n_scored} with a TM-score, {parse_fail} 'Cannot parse file'; failures file {len(fail)} rows "
      f"({fail.target_id.unique().tolist()}, {fail.reason.unique().tolist()}); best_per_target rows {len(best)}")
print("rows per target:", allv.groupby("target_id").size().value_counts().to_dict())

# ---- 2. Notebook output cells ------------------------------------------------------------------
c77 = outputs("pipeline5.ipynb", 77)
rnapro_block = c77.split("Pipeline5 sequence-matched")[0]
seq = {m.group(1): (float(m.group(3)), int(m.group(2)))
       for m in re.finditer(r"^\s*\d+\s+(\w{4})\s+RNAPro\s+(\d+)\s+([\d.]+)", rnapro_block, re.M)}
summ = {k: float(v) for k, v in re.findall(r"(Mean TM-score|Median TM-score|Best TM-score|Mean matched residues): ([\d.]+)", rnapro_block)}
p3 = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\d+\s+(\w{4})\s+seed_\d+\s+\d\s+([\d.]+)", outputs("pipeline3.ipynb", 17), re.M)}
idx = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\w{4})\s+TM=([\d.]+)", outputs("pipeline5.ipynb", 75), re.M)}

print(f"\nRNAPro sample 0, index-paired residues (p5 cell 75): n={len(idx)} mean={np.mean(list(idx.values())):.4f} "
      f"median={np.median(list(idx.values())):.4f}")
print(f"RNAPro best-of-5 seed_101, index-paired atoms (p3 cell 17): n={len(p3)} mean={np.mean(list(p3.values())):.4f}")
print(f"RNAPro sample 0, sequence-aligned (p5 cell 77) printed summary over 24: {summ}")
v = np.array([s[0] for s in seq.values()]); L = np.array([s[1] for s in seq.values()])
print(f"  displayed rows: n={len(seq)} mean={v.mean():.4f}; rows: {seq}")
print(f"  Spearman(matched residues, TM) on displayed rows: {pd.Series(L).corr(pd.Series(v), method='spearman'):.3f}")
for k in seq:
    print(f"  {k}: index {idx[k]:.4f} -> seq-aligned {seq[k][0]:.4f} (n_matched {seq[k][1]})")

ret = outputs("pipeline5.ipynb", 43)
print("\nretrieval:", re.findall(r"Mean top-1 identity: [\d.]+", ret))
print("stage benchmark (leaky):", re.findall(r"(Stage\d+_\w+)\s+[\d.]+\s+([\d.]+)", outputs("pipeline5.ipynb", 98)))
print("pipeline2 ranking:", re.findall(r"Average ranking score: [\d.]+", outputs("pipeline2.ipynb", 11)))

# ---- 3. Figure: per-target effect of pairing residues by sequence --------------------------
INK, MUTED, GRID, BAR, OLD = "#1f2328", "#59636e", "#e6e8eb", "#4a6fa5", "#b8c0c8"
order = sorted(seq, key=lambda k: seq[k][0])
fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
for y, k in enumerate(order):
    a, b = idx[k], seq[k][0]
    ax.plot([a, b], [y, y], color=GRID, lw=2.2, zorder=1)
    ax.scatter([a], [y], s=46, color=OLD, zorder=2)
    ax.scatter([b], [y], s=58, color=BAR, zorder=3)
    ax.text(max(a, b) + 0.008, y, f"{seq[k][1]} res", va="center", fontsize=9, color=MUTED)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=10.5, color=INK)
ax.axvline(summ["Mean TM-score"], color=BAR, ls="--", lw=1.1)
ax.text(summ["Mean TM-score"] + 0.003, 1.55, f"mean over all 24 targets\n(sequence-aligned): {summ['Mean TM-score']:.3f}",
        fontsize=9.5, color=BAR)
ax.scatter([], [], s=46, color=OLD, label="residues paired by position (earlier evaluation)")
ax.scatter([], [], s=58, color=BAR, label="residues paired by sequence alignment")
ax.legend(loc="center right", frameon=False, fontsize=9.5)
ax.set_xlim(0, 0.30); ax.set_xlabel("TM-score (Kabsch superposition, one sample per target)", color=MUTED, fontsize=11)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=10)
ax.set_title("RNAPro, one sample per target: pairing residues by sequence changes the score\n"
             "(10 of 24 targets; the notebook printed per-target rows only for these)", fontsize=12, color=INK, pad=10)
fig.tight_layout(); fig.savefig(out_png, facecolor="white")
print("wrote", out_png)
