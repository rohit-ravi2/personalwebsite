# artifacts/logs/

Per-phase execution logs. NOT git-tracked.

Expected naming:

- `phase_<letter>_<YYYYMMDD>.log` — per-day execution log per phase.
- `prepare_ligands_<YYYYMMDD>.log` — anesthetic prep logs.

Logs include INFO-level entries for tool invocations, gate-criteria evaluations, and surfaced findings. Verbose `-v` runs add DEBUG entries.

Logs are not version-controlled but should be archived alongside phase completion reports for posterity. The completion reports (`phase_*_completion.md`) carry the human-readable summary; the log is the operational trace.
