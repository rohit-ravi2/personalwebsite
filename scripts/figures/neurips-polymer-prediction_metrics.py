"""Recompute the local (out-of-fold) scores for the NeurIPS 2025 polymer page from the public repo's files.

Repo: github.com/rohit-ravi2/neurips-polymerprediction (clone it; pass the clone path as argv[1]).
Inputs (all from the repo):
  outputs/predictions/oof_lgb.csv, oof_cat.csv   phase4 OOF predictions (5-fold GroupKFold on canonical SMILES, 3 seeds averaged)
  outputs/predictions/fold_sizes.json            phase4 fold sizes (used to check the reconstructed folds)
  outputs/predictions/feature_names_pref.json    2,641 features after variance + correlation prefilters
  outputs/predictions/pruned_feature_names.json  938 features kept by gain pruning
  artifacts/models_lgb/lgb_<T>_manifest.json     phase5 scaffold-fold LightGBM CV MAE
  train.csv labels: the competition file, which the repo's first commit (efa27c3) shipped; read via `git show`.
  artifacts/models/<T>_fold<k>.pkl + artifacts/mordred_cols.pkl from the same first commit: the phase-1-style
    LightGBM fold models (Mordred 1,412 + Morgan r2 counts 2,048 + RDKit basics 8 = 3,468 features), used for gain.
Metric: competition_wmae as written in phase1.ipynb / track2.ipynb: weight_t = 1 / (sqrt(n_t) * range_t), weights
normalised to sum 1, wMAE = sum_t weight_t * MAE_t, with n_t and range_t from the labelled training rows.
Writes metrics JSON to argv[2].
"""
import json, os, subprocess, sys, io, pickle
import numpy as np, pandas as pd

REPO, OUT = sys.argv[1], sys.argv[2]
T = ["Tg", "FFV", "Tc", "Density", "Rg"]
git = lambda path: subprocess.run(["git", "-C", REPO, "show", f"efa27c3:{path}"], capture_output=True, check=True).stdout
P = lambda f: os.path.join(REPO, f)

train = pd.read_csv(io.BytesIO(git("train.csv")))
Y = train[T]
lgb = pd.read_csv(P("outputs/predictions/oof_lgb.csv"))
cat = pd.read_csv(P("outputs/predictions/oof_cat.csv"))
assert len(lgb) == len(cat) == len(train) == 7973


def maes(pred):
    out = {}
    for t in T:
        m = Y[t].notna() & pred[t].notna()
        out[t] = float(np.mean(np.abs(Y.loc[m, t] - pred.loc[m, t])))
    return out


n = {t: int(Y[t].notna().sum()) for t in T}
rng = {t: float(Y[t].max() - Y[t].min()) for t in T}
raw_w = {t: 1.0 / (np.sqrt(n[t]) * rng[t]) for t in T}
W = {t: raw_w[t] / sum(raw_w.values()) for t in T}
wmae = lambda m: float(sum(W[t] * m[t] for t in T))

# coverage: every labelled row should have an OOF prediction
cover = {t: int((Y[t].notna() & lgb[t].notna()).sum()) for t in T}

res = {"n_labelled": n, "range": rng, "weights": W, "oof_coverage": cover, "models": {}}
blend = (lgb + cat) / 2
for name, pred in [("LightGBM", lgb), ("CatBoost", cat), ("50/50 blend", blend)]:
    m = maes(pred); res["models"][name] = {"mae": m, "wmae": wmae(m)}

# Baseline: predict the training-fold mean (and median) on reconstructed phase4 folds.
# phase4 grouped by train_df["SMILES_CANON"]; the cell that builds it is not in the saved notebook, so I use RDKit
# canonical SMILES of the raw string. GroupKFold sizes are checked against fold_sizes.json.
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupKFold
RDLogger.DisableLog("rdApp.*")
canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in train["SMILES"]]
folds = list(GroupKFold(n_splits=5).split(train, groups=np.array(canon)))
recorded = json.load(open(P("outputs/predictions/fold_sizes.json")))
res["fold_sizes_reconstructed"] = [[len(a), len(b)] for a, b in folds]
res["fold_sizes_recorded"] = recorded
res["n_unique_canonical"] = len(set(canon))
for stat in ("mean", "median"):
    pred = pd.DataFrame(np.nan, index=train.index, columns=T)
    for tr, va in folds:
        for t in T:
            pred.loc[va, t] = getattr(Y.loc[tr, t], stat)()
    m = maes(pred); res["models"][f"fold-{stat} baseline"] = {"mae": m, "wmae": wmae(m)}
# Same baseline with plain shuffled KFold, to show the number does not hinge on the fold reconstruction.
from sklearn.model_selection import KFold
pred = pd.DataFrame(np.nan, index=train.index, columns=T)
for tr, va in KFold(5, shuffle=True, random_state=0).split(train):
    for t in T:
        pred.loc[va, t] = Y.loc[tr, t].mean()
m = maes(pred); res["models"]["fold-mean baseline (plain KFold check)"] = {"mae": m, "wmae": wmae(m)}

# Skill relative to the fold-mean baseline (1 - MAE_model / MAE_baseline)
b = res["models"]["fold-mean baseline"]["mae"]
res["skill_vs_mean"] = {k: {t: 1 - v["mae"][t] / b[t] for t in T} for k, v in res["models"].items()
                        if k in ("LightGBM", "CatBoost", "50/50 blend")}

# Which model wins per row-target (share of rows where LGB error < CatBoost error)
res["lgb_better_share"] = {t: float(np.mean(np.abs(Y[t] - lgb[t])[Y[t].notna()] < np.abs(Y[t] - cat[t])[Y[t].notna()]))
                           for t in T}
res["lgb_cat_corr"] = {t: float(np.corrcoef(lgb.loc[Y[t].notna(), t], cat.loc[Y[t].notna(), t])[0, 1]) for t in T}

# Phase 5 scaffold-fold LightGBM manifests
res["phase5_manifest"] = {t: json.load(open(P(f"artifacts/models_lgb/lgb_{t}_manifest.json"))) for t in T}
res["phase5_manifest"] = {t: {"cv_mae": v["cv_mae"], "folds_used": v["folds_used"], "label_count": v["label_count"],
                              "min_data_in_leaf": v["params"]["min_data_in_leaf"]} for t, v in res["phase5_manifest"].items()}
scaff = json.load(open(P("artifacts/ingest/folds.json")))["folds"]
res["scaffold_fold_sizes"] = [len(f) for f in scaff]


# Feature families
def family(name):
    # rdkit8 / phys / MACCS blocks were reloaded from parquet caches with integer column names (0..N), so they
    # appear in the json as bare ints and cannot be told apart; they are reported together.
    if not isinstance(name, str): return "MACCS / RDKit / physics (int-named)"
    for p, f in [("morgan_r2c_", "Morgan r2 counts"), ("morgan_r3p_", "Morgan r3 bits"), ("maccs_", "MACCS keys"),
                 ("rdkit8_", "RDKit basics"), ("phys_", "Physics block")]:
        if name.startswith(p): return f
    return "Mordred 2D"
pref = json.load(open(P("outputs/predictions/feature_names_pref.json")))
pruned = json.load(open(P("outputs/predictions/pruned_feature_names.json")))
cnt = lambda names: pd.Series([family(x) for x in names]).value_counts().to_dict()
res["families_prefiltered"], res["families_pruned"] = cnt(pref), cnt(pruned)
res["family_keep_rate"] = {f: res["families_pruned"].get(f, 0) / c for f, c in res["families_prefiltered"].items()}

# Gain importance from the first-commit LightGBM fold models (Tg, Tc, Density, Rg; FFV models are 58 MB each, also used)
import joblib, tempfile
mord = pickle.loads(git("artifacts/mordred_cols.pkl"))
names = list(mord) + [f"morgan_r2c_{i}" for i in range(2048)] + ["MolWt", "TPSA", "LogP", "NumRings", "NumAromaticRings",
                                                                  "NumAliphaticRings", "NumHBD", "NumHBA"]
fam1 = ["Mordred 2D"] * len(mord) + ["Morgan r2 counts"] * 2048 + ["RDKit basics"] * 8
gain = {}
with tempfile.TemporaryDirectory() as d:
    for t in T:
        g = np.zeros(len(names))
        for k in range(5):
            fp = os.path.join(d, "m.pkl"); open(fp, "wb").write(git(f"artifacts/models/{t}_fold{k}.pkl"))
            mdl = joblib.load(fp); assert mdl.n_features_in_ == len(names)
            g += mdl.booster_.feature_importance("gain")
        g = g / g.sum()
        fam_share = pd.Series(g).groupby(pd.Series(fam1)).sum().to_dict()
        top = np.argsort(-g)[:8]
        gain[t] = {"family_share": fam_share, "top": [[names[i], float(g[i])] for i in top],
                   "top20_share": float(np.sort(g)[::-1][:20].sum())}
res["phase1_models_gain"] = gain

json.dump(res, open(OUT, "w"), indent=1)
for k, v in res["models"].items():
    print(f"{k:42s} wMAE {v['wmae']:.5f}  " + "  ".join(f"{t} {v['mae'][t]:.5g}" for t in T))
print("weights", {t: round(W[t], 5) for t in T}); print("cover", cover, "n", n)
print("folds", res["fold_sizes_reconstructed"], "recorded", recorded, "unique", res["n_unique_canonical"])
print("skill", json.dumps(res["skill_vs_mean"], indent=0))
print("lgb better", res["lgb_better_share"], "corr", res["lgb_cat_corr"])
print("fam pref", res["families_prefiltered"]); print("fam pruned", res["families_pruned"])
for t in T: print(t, {k: round(v, 3) for k, v in gain[t]["family_share"].items()}, gain[t]["top"][:5], round(gain[t]["top20_share"], 3))
