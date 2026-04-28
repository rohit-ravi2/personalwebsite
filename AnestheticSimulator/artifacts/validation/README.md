# artifacts/validation/

Phase H outputs. Populated when Phase H runs.

Expected contents:

- `anchor_table.csv` — 8-row anchor evaluation.
- `anchor_evaluation.md` — per-anchor pass/fail with diagnosis.
- `lesion_test_program_level.md` — Gate G.1.5 program-level re-evaluation.
- `program_verdict.md` — **Wave P program-level verdict** (STRONG_PASS / PASS / PARTIAL_FAIL / FAIL).
- `program_verdict.json`.
- `phase_h_completion.md`.

This directory contains the **headline output of Wave P**. The `program_verdict.md` is what feeds the paper.
