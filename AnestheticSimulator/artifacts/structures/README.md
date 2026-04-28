# artifacts/structures/

Phase A outputs. Populated when Phase A runs.

Expected contents:

- `<TARGET>_monomer_AFDB.pdb` — pre-computed AlphaFold DB monomer pulls (25 files).
- `<TARGET>_multimer/*rank_001*.pdb` — ColabFold/AlphaFold-Multimer outputs for oligomers (~12 directories).
- `<TARGET>_pocket_plddt.json` — per-target pocket pLDDT distribution.
- `<TARGET>_homolog_alignment.json` — TM-score + RMSD vs experimental homolog.
- `<TARGET>_foldseek.tsv` — FoldSeek hits.
- `coverage_report.md` — Gate A.1 summary.
- `gate_a1_evaluation.json` — Gate A.1 pass/fail verdict.
- `phase_a_completion.md` — end-of-block report.

Big files (PDBs, multimer tarballs) are NOT git-tracked per `.gitignore`. The reports and JSON files ARE tracked.
