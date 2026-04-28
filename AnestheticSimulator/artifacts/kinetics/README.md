# artifacts/kinetics/

Phase D outputs. Populated when Phase D runs.

Expected contents:

- `literature_shifts.csv` — per-(target, anesthetic) literature-direct shift values.
- `<TARGET>_<ANESTHETIC>_prod.dcd` — OpenMM production trajectories (NOT git-tracked).
- `<TARGET>_<ANESTHETIC>_shift.json` — per-pair MD-derived shift extracted from trajectory analysis.
- `anesthetic_kinetic_shifts.npz` — **master output**: per-target shift form + parameters.
- `calibration_report.md` — mammalian-control MD vs literature.
- `gate_d1_evaluation.json`.
- `phase_d_completion.md`.

MD trajectories (DCD, XTC, TRR) are NOT git-tracked due to size. The JSON / NPZ / MD reports ARE tracked.
