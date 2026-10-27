"""WormLab Graph Dynamics cover: the L2 learned-GNN result and its gate-stability control, from saved JSON only.

Sources (repo github.com/rohit-ravi2/worm-graph-dynamics, local copy ~/projects/wormlab):
  results/gd/benchmarks/L2/L2_results.json        summary.r2 (held-out R^2 per graph) and paired contrasts
  results/gd/benchmarks/L2/L2_gate_stability.json contrast.{chem,gap}.gates (mean pairwise fold correlation)
The complete-graph variant (R^2 -0.17) is off-scale and is named in the caption instead of plotted.
"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/home/rohit/projects/wormlab/results/gd/benchmarks/L2/"
r2 = json.load(open(B + "L2_results.json"))["summary"]["r2"]
gs = json.load(open(B + "L2_gate_stability.json"))["contrast"]

rows = [("gnn_geometry", "Nearest physical\nneighbours"), ("gnn_shuffled_1", "Shuffle 1"), ("gnn_connectome", "Connectome"),
        ("gnn_shuffled_2", "Shuffle 2"), ("gnn_shuffled_0", "Shuffle 0"), ("gnn_none", "No edges")]
INK, MUTED, GRID, BAR, NULL, ACC = "#1f2328", "#59636e", "#e6e8eb", "#4a6fa5", "#9aa3ad", "#c0703a"
colour = {"gnn_connectome": BAR, "gnn_geometry": ACC, "gnn_none": MUTED}

fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
for i, (k, lab) in enumerate(rows):
    m, (lo, hi) = r2[k]["mean"], r2[k]["ci"]
    c = colour.get(k, NULL)
    ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="round")
    ax.plot(m, i, "o", color=c, ms=8)
    ax.text(hi + 0.0008, i, f"{m:.4f}", va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows]); ax.invert_yaxis()
ax.set_xlim(0.553, 0.603)
ax.set_xlabel("held-out R² at 2.4 s (38 animals, 95% CI)", color=MUTED, fontsize=10.5)
ax.set_title("Same GNN, different edges", fontsize=12, color=INK, pad=10)

ax = axes[1]
groups = [("Chemical", gs["chem"]["gates"]), ("Gap junction", gs["gap"]["gates"])]
w = 0.16
for g, (name, d) in enumerate(groups):
    vals = [d["connectome"]] + d["shuffled_each"] + [d["geometry"]]
    cols = [BAR, NULL, NULL, NULL, ACC]
    for j, (v, c) in enumerate(zip(vals, cols)):
        x = g + (j - 2) * w
        ax.bar(x, v, width=w * 0.9, color=c)
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels([n for n, _ in groups])
ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("edge-gate agreement across folds (r)", color=MUTED, fontsize=10.5)
ax.set_title("Learned edge gates: as stable on shuffles", fontsize=12, color=INK, pad=10)
ax.grid(axis="y", color=GRID, lw=0.8)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BAR, NULL, ACC)]
ax.legend(handles, ["connectome", "degree-preserving shuffles", "physical neighbours"], frameon=False, fontsize=8.3,
          loc="upper center", ncol=3, handlelength=1.0, columnspacing=0.8)

for ax in axes:
    ax.set_axisbelow(True)
    if ax is axes[0]: ax.grid(axis="x", color=GRID, lw=0.8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
fig.suptitle("The wiring diagram does no better than a shuffled copy of itself", fontsize=14, color=INK, y=0.995)
fig.text(0.01, 0.01, "Atanas et al. 2023, 5-fold animal split. Complete graph (not shown): R² "
         f"{r2['gnn_full']['mean']:.2f}. CIs show animal spread; paired contrasts in text.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else
            "/mnt/ssd4tb/Desktop/website/wt-stage1/public/images/projects/worm-graph-dynamics-cover.png",
            facecolor="white")
print({k: round(r2[k]["mean"], 4) for k, _ in rows}, {n: [round(x, 2) for x in [d["connectome"]] + d["shuffled_each"]
      + [d["geometry"]]] for n, d in groups})
