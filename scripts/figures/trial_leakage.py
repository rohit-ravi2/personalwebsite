"""Trial-failure leakage figure from saved results only.

Sources: /mnt/ssd4tb/research/trial-failure/results/day2/lgbm_v2_summary.json (Day-2 p3/time test AUROC) and
day3/leakage_audit.json (audit retrain AUROC, null-label AUROC). The 5-fold CV figure in the old version had no
saved result file and is omitted.
"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/mnt/ssd4tb/research/trial-failure/results/"
d2 = json.load(open(R + "day2/lgbm_v2_summary.json"))["summaries"]["p3/time"]
au = json.load(open(R + "day3/leakage_audit.json"))
vals = [d2["AUROC"], au["verdict"]["audited_auroc"], au["null_label_experiment_p3_time"]["null_shuffled_auroc"]]
labels = ["Day-2 model\n(held-out, time split)", "Audit retrain\n(held-out, time split)", "Shuffled training\nlabels (null)"]
INK, MUTED, GRID, BAR, NULL = "#1f2328", "#59636e", "#e6e8eb", "#4a6fa5", "#9aa3ad"

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
bars = ax.bar(labels, vals, width=0.6, color=[BAR, BAR, NULL])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=11, color=INK)
ax.axhline(0.5, color=MUTED, lw=1, ls="--"); ax.text(-0.42, 0.51, "chance", color=MUTED, fontsize=9, ha="left")
ax.set_ylim(0, 1); ax.set_ylabel("Phase-3 success AUROC", color=MUTED, fontsize=11)
ax.set_title("Leakage audit: the shuffled-label control lands at chance", fontsize=13, color=INK, pad=10)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=10)
fig.tight_layout(); fig.savefig(sys.argv[1], facecolor="white")
print([round(v, 4) for v in vals])
