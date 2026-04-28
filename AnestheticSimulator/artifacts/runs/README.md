# artifacts/runs/

Phase G + I + J outputs. Populated when those phases run.

Expected contents:

- `<anesthetic>_<dose>_<genotype>_<scenario>_<seed>.npz` — 2,440 per-run traces (NOT git-tracked due to size).
- `aggregated_ec50.csv` — per-config EC50 + Hill fits.
- `lesion_comparison.csv` — per-lesion-class fraction immobilized.
- `lesion_test_result.md` — Gate G.1.5 (load-bearing).
- `dose_response_curves.png`.
- `integration_check.json` — pre-flight upstream-artifact check.
- `gate_g1_evaluation.json`.
- `phase_g_completion.md`.

Phase I (stretch):
- `inverse_occupancy.npz` — JAX-derived empirical occupancy vector.
- `inverse_validation.md` — Phase C comparison.

Phase J (stretch):
- `signatures.npz` — Phi, LLE, modularity, spectral entropy.
- `manifold_embeddings.npz` — UMAP coordinates.
- `signature_report.md` — direction-of-change vs mammalian.

Per-run NPZ files are NOT git-tracked. Aggregated CSVs and reports ARE tracked.
