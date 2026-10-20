"""AMR project cover: like-for-like canonical-AMR top-3 from the Day-3 result JSON.

Source: /mnt/ssd4tb/research/amr/results/day3/phase_d_results.json. The reranker only scores transitions whose true
mutation is in LightGBM's top-10 (n=72 of 571), so its unconditional value is top3 * 72/571 (misses count as wrong).
"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/mnt/ssd4tb/research/amr/results/day3/phase_d_results.json"))
N, n = d["split_info"]["n_test_transitions"], d["reranker_top10"]["canonical"]["n"]
full = [d["lgbm_only_full_vocab"]["canonical"]["top3"], d["transformer_only_full_vocab"]["canonical"]["top3"],
        d["reranker_top10"]["canonical"]["top3"] * n / N]
sub = [d["lgbm_only_top10_subset"]["canonical"]["top3"], d["transformer_only_top10_subset"]["canonical"]["top3"],
       d["reranker_top10"]["canonical"]["top3"]]
labels = ["LightGBM", "Transformer", "Reranker"]
INK, MUTED, GRID, BAR = "#1f2328", "#59636e", "#e6e8eb", "#4a6fa5"

fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.2), dpi=200, sharey=True)
for ax, vals, title in ((axes[0], full, f"All {N} test transitions"),
                        (axes[1], sub, f"The {n} where LightGBM's top-10\ncontains the answer")):
    bars = ax.bar(labels, vals, width=0.6, color=BAR)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_title(title, fontsize=12, color=INK, pad=10)
    ax.set_ylim(0, 0.5); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10.5)
axes[0].set_ylabel("canonical-AMR top-3", color=MUTED, fontsize=11)
fig.suptitle("The reranker beats neither of its components", fontsize=14, color=INK, y=0.99)
fig.tight_layout()
fig.savefig(sys.argv[1], facecolor="white")
print("full", [round(v, 4) for v in full], "subset", [round(v, 4) for v in sub])
