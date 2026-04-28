# artifacts/metabolic/

Phase F outputs. Populated when Phase F runs.

Expected contents:

- `metabolic_layer_module.py` — importable Brian2 module (consumed by Phase G).
- `wt_baseline.npz` — WT [ATP] + K-ATP traces.
- `gas1_baseline.npz` — gas-1(fc21) baseline.
- `mev1_baseline.npz` — mev-1 baseline (Complex II control).
- `atp2_baseline.npz` — atp-2 severe ATP synthase baseline.
- `wt_vs_gas1_halothane.npz` — EC50 comparison.
- `calibration_report.md` — Gate F.1 evaluation.
- `gate_f1_evaluation.json`.
- `smoke_test_results.json` — quick sanity check from `phase_f_metabolic.py --smoke-test`.
- `phase_f_completion.md`.
