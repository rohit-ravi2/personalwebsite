import type { InvestigationEntry } from "./types";

export const ENTRIES: InvestigationEntry[] = [
  {
    n: 1,
    date: "2025-08-18",
    q: "How far does one gradient-boosted model per target get on stock descriptors?",
    verdict: "kept",
    tag: "phase1.ipynb",
    tried:
      "Parse every SMILES with the polymer wildcard replaced, compute 1,412 Mordred 2D descriptors, 1,024 Morgan radius-2 count bits and 8 RDKit basics, and fit one LightGBM per target on only the rows where that target is labelled, with 5-fold CV.",
    found:
      "The notebook's own output for this run is a weighted MAE of 0.06827 (Tg 49.3, FFV 0.0059, Tc 0.0269, density 0.0270, Rg 1.65). The folds were plain shuffled KFold, so a polymer that appears twice could sit on both sides of a split.",
    ruled: "The idea that the descriptor route was too weak to bother with.",
    why: "A few hundred labels per target is too few for a model to learn chemistry from raw structure; descriptors hand it the chemistry for free.",
    next: "Check how sparse the labels really are, and what a do-nothing model scores, before tuning anything.",
  },
  {
    n: 2,
    q: "How sparse are the labels, and what does predicting the mean score?",
    verdict: "partial",
    tag: "phase2.ipynb",
    tried:
      "Audit label coverage per target, rebuild the feature blocks with a small physics-motivated set (sp3 fraction, heteroatom ratio, halogen/N/O/S counts, aromatic rings), score a predict-the-mean baseline, and collect external Tg tables.",
    found:
      "Only FFV is well covered (7,030 of 7,973 polymers). Tg has 511 labels, Tc 737, density 613, Rg 614. The external tables were sorted but never entered a scored model in this repo.",
    ruled: "Treating the problem as one multi-output regression with a shared loss.",
    why: "With 88% of the Tg column empty, every target needs its own masked training set and its own validation count.",
    next: "Build one clean pipeline with grouped folds, feature selection and a second model family.",
  },
  {
    n: 3,
    q: "Does a grouped, pruned, two-model pipeline hold up?",
    verdict: "kept",
    tag: "phase4.ipynb",
    tried:
      "Add MACCS keys and Morgan radius-3 bits, drop near-constant and |r| > 0.995 features (2,641 left), keep the union of the top 600 LightGBM-gain features per target (938 left), and train LightGBM and CatBoost per target on 5 GroupKFold folds keyed on canonical SMILES, averaged over 3 seeds.",
    found:
      "Out-of-fold weighted MAE 0.0649 for LightGBM and 0.0674 for CatBoost, against 0.168 for predicting the training-fold mean. LightGBM wins on Tg, density and Rg; the two tie on Tc; CatBoost is slightly ahead on FFV. A 50/50 average reaches 0.0647.",
    ruled: "CatBoost as a replacement for LightGBM. Its errors track LightGBM's closely (prediction correlation 0.97 to 0.99), so it is useful only as a blend partner.",
    why: "Two tree ensembles on the same features make nearly the same mistakes; the blend gain is small for that reason.",
    next: "Try signals the descriptors cannot carry: a pretrained SMILES language model and a graph network, under stricter scaffold folds.",
  },
  {
    n: 4,
    q: "Do learned embeddings or a GNN add anything?",
    verdict: "partial",
    tag: "phase5.ipynb, hybridpipeline.ipynb",
    tried:
      "Append 768-dimensional ChemBERTa mean-pooled embeddings to the tabular block (4,237 features), switch to Murcko-scaffold folds, and set up a GINE graph network and a stacking layer.",
    found:
      "Unresolved. The scored LightGBM run used min_data_in_leaf = 200 on folds with 316 to 489 training rows, and in 2 of 4 Tg folds and 1 of 2 Tc folds LightGBM found zero usable features, so those folds predicted a constant. Its Tg MAE of 92.1 is worse than the 88.4 of the fold-mean baseline. The GNN and stacking cells did not produce scores, and hybridpipeline.ipynb was never executed.",
    ruled: "Nothing about embeddings themselves; the run that would test them was broken by a leaf-size setting.",
    why: "A guardrail tuned for FFV's 7,000 rows silently disables the model on the small targets.",
    next: "Rerun phase5 with a leaf size scaled to each target's row count, on the same folds as phase4, so the embedding columns are the only change.",
  },
  {
    n: 5,
    q: "Is the 0.02 weighted MAE from track3 real?",
    verdict: "ruled-out",
    tag: "track3.ipynb",
    tried: "A modular run over precomputed feature blocks, including a block called 'physics'.",
    found:
      "That block is train[['Tg','FFV','Tc','Density','Rg']]: the five targets themselves, merged into the features. Every track3 score is leakage.",
    ruled: "All track3 numbers. None appear on this page.",
    why: "A model that can read the answer column scores near zero error without learning any chemistry.",
    next: "Keep phase4 as the reference result.",
  },
];
