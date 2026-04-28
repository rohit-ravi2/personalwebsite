# artifacts/occupancy/

Phase C outputs. **GATE C.1 LIVES HERE.** This is the load-bearing falsifiability checkpoint of Wave P.

Expected contents:

- `occupancy_matrix.npz` — 25 × 6 × 4 occupancy matrix with central + bracket. **Master output.**
- `occupancy_table.csv` — same data, human-readable.
- `multitarget_count.csv` — per-(anesthetic, multiplier) count of targets > 10% occupancy.
- `gate_c1_evaluation.md` — **load-bearing**: C.1.1-C.1.4 evaluation with pass/fail per criterion.
- `gate_c1_evaluation.json`.
- `occupancy_heatmap_*.png` — visualization.
- `occupancy_dose_response.png`.
- `compartment_uncertainty.md` — targets with ambiguous compartment.
- `phase_c_completion.md` — end-of-block report.

If Gate C.1 fails, Wave P pivots. The verdict here is the program's first major decision point.
