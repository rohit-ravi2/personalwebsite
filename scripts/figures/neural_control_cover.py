"""neural-control cover: state-free identity model vs Creamer's connectome model, and where the connectome helps.

Sources (all in ~/projects/neural-control/results/, also in the public repo):
  blind_final.json                       state-free identity ensemble (A+B) and shuffled-identity control, fixed split
  headroom.json                          noiseless ceiling of the 30-animal test map (key a.noiseless_ceiling)
  creamer_reconcile.json                 Creamer models scored on the same split and estimator (T_auth_ours)
  cv_combine_worm_causal_final.json      5-fold CV over 110 animals: identity model, ceiling, Creamer models
  cv_combine_worm_causal_cvshuf.json     5-fold CV shuffled-identity control
  near_far_cc.json                       Creamer models on their own shipped test matrices, near (<10 um) vs far (>=15 um)
Usage: python neural_control_cover.py <out.png>
"""
import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/projects/neural-control/results/")
J = lambda f: json.load(open(R + f))
bf, hr, rc = J("blind_final.json"), J("headroom.json"), J("creamer_reconcile.json")
cv, cvs, nf = J("cv_combine_worm_causal_final.json"), J("cv_combine_worm_causal_cvshuf.json"), J("near_far_cc.json")

fixed = {"ceil": hr["a"]["noiseless_ceiling"],
         "vals": [bf["A+B ensemble | state-free"]["all"],
                  rc["models"]["connectome_constrained"]["full"]["T_auth_ours"],
                  bf["shuffled A | state-free"]["all"][0]]}
cvd = {"ceil": cv["ceiling_110"],
       "vals": [cv["A+B_ensemble"], cv["Creamer connectome_constrained (contaminated)"], cvs["A+B_ensemble"]]}
near = [nf["T_ship | connectome_constrained"]["near<10"], nf["T_ship | shuffled_constrained"]["near<10"]]
far = [nf["T_ship | connectome_constrained"]["far>=15"], nf["T_ship | shuffled_constrained"]["far>=15"]]

INK, MUTED, GRID, BAR, ALT, NULL = "#1f2328", "#59636e", "#e6e8eb", "#4a6fa5", "#c9822b", "#9aa3ad"
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=200, gridspec_kw={"width_ratios": [1.45, 1]})

ax = axes[0]
names = ["Identity model\n(no connectome)", "Connectome-\nconstrained (Creamer)", "Identity model,\nnames shuffled"]
cols, w = [BAR, ALT, NULL], 0.26
for gi, g in enumerate((fixed, cvd)):
    for k, v in enumerate(g["vals"]):
        x = gi + (k - 1) * w
        ax.bar(x, v, width=w * 0.92, color=cols[k], label=names[k] if gi == 0 else None)
        ax.text(x, max(v, 0) + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.plot([gi - 1.6 * w, gi + 1.6 * w], [g["ceil"]] * 2, color=INK, lw=1.2, ls="--")
    ax.text(gi + 1.6 * w, g["ceil"] + 0.01, f"noise ceiling {g['ceil']:.2f}", ha="right", va="bottom", fontsize=9, color=MUTED)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Fixed split (30 held-out animals)", "5-fold CV (110 animals)"])
ax.set_ylim(0, 0.68); ax.set_ylabel("correlation with measured causal map (state-free)", color=MUTED, fontsize=10.5)
ax.set_title("Who the neurons are predicts the map", fontsize=12, color=INK, pad=10)
ax.legend(frameon=False, fontsize=8.8, loc="upper left", labelcolor=MUTED)

ax = axes[1]
xs, w2 = [0, 1], 0.36
for k, (lab, c) in enumerate((("Connectome-constrained", ALT), ("Shuffled connectome", NULL))):
    vs = [near[k], far[k]]
    for i, v in enumerate(vs):
        x = i + (k - 0.5) * w2
        ax.bar(x, v, width=w2 * 0.92, color=c, label=lab if i == 0 else None)
        ax.text(x, max(v, 0) + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(["Responder < 10 µm\nfrom stimulated cell", "Responder ≥ 15 µm\nfrom stimulated cell"])
ax.set_ylim(-0.03, 0.26); ax.set_ylabel("correlation (Creamer's own test maps)", color=MUTED, fontsize=10.5)
ax.set_title("The wiring diagram helps only up close", fontsize=12, color=INK, pad=10)
ax.legend(frameon=False, fontsize=8.8, loc="upper right", labelcolor=MUTED)

for ax in axes:
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.5)
fig.tight_layout()
fig.savefig(sys.argv[1], facecolor="white")
print("fixed", [round(v, 4) for v in fixed["vals"]], round(fixed["ceil"], 4))
print("cv", [round(v, 4) for v in cvd["vals"]], round(cvd["ceil"], 4))
print("near", [round(v, 4) for v in near], "far", [round(v, 4) for v in far])
