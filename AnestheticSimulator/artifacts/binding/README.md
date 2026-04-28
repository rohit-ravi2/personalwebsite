# artifacts/binding/

Phase B outputs. Populated when Phase B runs.

Expected contents:

- `<TARGET>_<ANESTHETIC>_vina.pdbqt` — Vina top-9 poses.
- `<TARGET>_<ANESTHETIC>_diffdock/` — DiffDock 40-pose ensemble.
- `<TARGET>_<ANESTHETIC>_gnina.sdf` — GNINA-rescored poses.
- `<TARGET>_<ANESTHETIC>_consensus.json` — consensus pose + cross-method metrics.
- `binding_matrix.csv` — 25 × 6 grid of Kd estimates with uncertainty bracket (consumed by Phase C).
- `photolabel_match.md` — photolabel cross-validation report.
- `top10_gnina.csv` — top-10 GNINA hits with cross-method-agreement metrics (Gate B.1.4). FEP results for the top 10 are DEFERRED — see `preregistration/phase_b_binding_pose.md` §13.
- `coverage_report.md` — Gate B.1 evaluation.
- `gate_b1_evaluation.json`.
- `phase_b_completion.md`.

Big files (PDBQT, SDF) are NOT git-tracked. The CSV/JSON/MD reports ARE tracked.
