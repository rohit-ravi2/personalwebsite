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
      "A working per-target descriptor pipeline on all five properties, which every later notebook builds on.",
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
    next: "Read the trained models to see which descriptors carry the signal for each property.",
  },
  {
    n: 4,
    q: "Which descriptors carry the signal?",
    verdict: "kept",
    tag: "Feature attribution",
    tried:
      "Sum LightGBM split gain over the five saved fold models per target, by feature family and by individual descriptor.",
    found:
      "Mordred whole-molecule descriptors carry 92 to 99% of the gain on every target. The leaders are physically sensible: rotatable-bond fraction for Tg, mean atomic mass and number for density, a chain-end distance descriptor for Rg.",
    ruled: "Substructure fingerprint bits as the main source of signal, despite outnumbering the descriptors.",
    why: "With a few hundred labels, properties of the whole molecule generalise better than which fragments are present.",
    next: "The pruned descriptor pipeline is the reference model.",
  },
];
