"""NeurIPS 2025 polymer page figure: local out-of-fold scores against a predict-the-fold-mean baseline.

Sources (public repo github.com/rohit-ravi2/neurips-polymerprediction; pass a clone as argv[1]):
  outputs/predictions/oof_lgb.csv, oof_cat.csv   phase4 OOF predictions (5-fold GroupKFold on canonical SMILES, 3 seeds)
  train.csv labels from the repo's first commit (efa27c3), read with `git show`
Baseline: training-fold mean per target on the reconstructed phase4 GroupKFold folds (sizes match fold_sizes.json).
Metric: competition_wmae from phase1.ipynb (weights 1/(sqrt(n_t) * range_t), normalised to sum 1).
Usage: python neurips-polymer-prediction_results.py <repo clone> public/images/projects/neurips-polymer-prediction-results.png
"""
import io, os, subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupKFold

REPO, OUT = sys.argv[1], sys.argv[2]
T = ["Tg", "FFV", "Tc", "Density", "Rg"]
train = pd.read_csv(io.BytesIO(subprocess.run(["git", "-C", REPO, "show", "efa27c3:train.csv"],
                                              capture_output=True, check=True).stdout))
Y = train[T]
lgb = pd.read_csv(os.path.join(REPO, "outputs/predictions/oof_lgb.csv"))
cat = pd.read_csv(os.path.join(REPO, "outputs/predictions/oof_cat.csv"))
RDLogger.DisableLog("rdApp.*")
groups = np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in train["SMILES"]])
base = pd.DataFrame(np.nan, index=train.index, columns=T)
for tr, va in GroupKFold(5).split(train, groups=groups):
    for t in T:
        base.loc[va, t] = Y.loc[tr, t].mean()

mae = lambda p: {t: float(np.mean(np.abs(Y[t] - p[t])[Y[t].notna()])) for t in T}
w = {t: 1 / (np.sqrt(Y[t].notna().sum()) * (Y[t].max() - Y[t].min())) for t in T}
w = {t: v / sum(w.values()) for t, v in w.items()}
models = {"Fold-mean\nbaseline": base, "CatBoost": cat, "LightGBM": lgb, "50/50\nblend": (lgb + cat) / 2}
M = {k: mae(p) for k, p in models.items()}
W = {k: sum(w[t] * m[t] for t in T) for k, m in M.items()}

INK, MUTED, GRID, BAR, ALT, NULL = "#1f2328", "#59636e", "#e6e8eb", "#4a6fa5", "#c9822b", "#9aa3ad"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 5.2), dpi=200, gridspec_kw={"width_ratios": [1, 1.5]})

names = list(M)
cols = [NULL, ALT, BAR, "#2f4f7a"]
bars = a1.bar(names, [W[k] for k in names], width=0.62, color=cols)
for b, k in zip(bars, names):
    a1.text(b.get_x() + b.get_width() / 2, W[k] + 0.003, f"{W[k]:.4f}", ha="center", va="bottom", fontsize=10.5, color=INK)
a1.set_ylabel("weighted MAE (lower is better)", color=MUTED, fontsize=11)
a1.set_title("Competition-weighted MAE, 5-fold OOF", fontsize=12, color=INK, pad=10)
a1.set_ylim(0, max(W.values()) * 1.15)

# per-target error as a fraction of the baseline error, targets ordered by metric weight
order = sorted(T, key=lambda t: -w[t])
x = np.arange(len(order)); bw = 0.26
for i, (k, c) in enumerate([("CatBoost", ALT), ("LightGBM", BAR), ("50/50\nblend", "#2f4f7a")]):
    vals = [M[k][t] / M["Fold-mean\nbaseline"][t] for t in order]
    a2.bar(x + (i - 1) * bw, vals, width=bw, color=c, label=k.replace("\n", " "))
a2.axhline(1.0, color=NULL, lw=1.2, ls="--"); a2.text(len(order) - 0.55, 1.01, "baseline", color=MUTED, fontsize=9.5, ha="right", va="bottom")
a2.set_xticks(x, [f"{t}\nn={int(Y[t].notna().sum())}\nw={w[t]:.3f}" for t in order], fontsize=9.5)
a2.set_ylabel("MAE / fold-mean-baseline MAE", color=MUTED, fontsize=11)
a2.set_title("Per target, ordered by metric weight", fontsize=12, color=INK, pad=10)
a2.set_ylim(0, 1.12); a2.legend(frameon=False, fontsize=9.5, loc="upper left")
for ax in (a1, a2):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
fig.suptitle("Gradient boosting on descriptors cuts the baseline error by about 60%", fontsize=13.5, color=INK, y=0.995)
fig.tight_layout()
fig.savefig(OUT, facecolor="white")
print({k.replace("\n", " "): round(v, 5) for k, v in W.items()})
