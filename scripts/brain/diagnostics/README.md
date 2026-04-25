# Diagnostics

One-time measurement scripts organised by the phase that produced them.
Kept in the repo for reproducibility of each phase's findings; not
intended to be re-run except to validate regressions.

## Layout

- `phase0/` — Phase 0 baseline measurement + equation sanity checks.
  Each script was run once against the v3 LIF brain to establish
  ratified thresholds for downstream phases.

## When to re-run

- **Regression check:** if the compartmental scaffold or LIF brain
  parameters change, re-running the Phase 0 plateau diagnostic should
  produce different results (ideally, closer to biological targets).
- **Reviewer questions:** if a reviewer asks whether a result is
  reproducible, rerunning the relevant script should produce the same
  numbers (deterministic Brian2 seeds are locked).

## What belongs here vs at `scripts/brain/` top level

- **`scripts/brain/`:** active modules imported by the simulator or
  tooling that is re-run each phase (e.g., `phase0_audit.py`,
  `phase0_analyze.py`, `muscle_driver.py`).
- **`scripts/brain/diagnostics/<phase>/`:** scripts that produced a
  specific finding during that phase. Running them again should
  reproduce the numbers in the corresponding phase artifacts.

## Artifacts stay in `scripts/brain/artifacts/`

Diagnostics write their outputs to `scripts/brain/artifacts/` (same
location as ensemble-audit outputs). Moving diagnostic code here
doesn't move the outputs; the paths in the scripts still point at
`../artifacts/`.
