# Session 1 WB1 — Commit proposal (CP1 pause gate)

**Pause for review.** Don't apply `.gitignore` changes or commits until approved.

---

## Pre-flight scope inventory

**97 uncommitted items at repo root.** Mix of:

- **4 modified files** (production simulator):
  - `scripts/brain/graded_brain.py` (cython codegen flip)
  - `scripts/brain/lif_brain.py` (cython codegen flip)
  - `scripts/brain/overnight_v2_track_f.py` (cython codegen flip)
  - `scripts/brain/artifacts/modulator_tables.json` (unknown — needs investigation)

- **93 untracked items** spanning multiple work-block scopes

**Recent committed work** (head 5): T0 closure, T0 resolution propagation, T0 consolidation, T0 voltage-regime fix, dashboard NaN guards. Wave 2 work is entirely uncommitted.

**Branch:** main.

---

## Scope boundary clarifications needed

### Confirmed Session 1 scope (commit in this WB1)

- **Wave 2 work** (`scripts/brain/wave2/` entire tree, 105MB total)
- **Production simulator cython migration** (4 modified files above)
- **Wave 2-related artifact docs in scripts/brain/artifacts/** (the `phase_v_w*` markdown files — Wave 2 prompts, plans, reports)
- **session handoff docs** (`scripts/brain/artifacts/handoffs/session_{1,3}_handoff_2026-04-26.md`)
- **Two Wave 1 sandbox files** (`compartmental_neurons_kca.py`, `graded_brain_h_kca.py`) modified during the cython migration cleanup

### Confirmed exclude (Session 2 territory or unrelated)

- **`AnestheticSimulator/`** (81MB) — Session 2 territory, separate commit work block
- **`scripts/build_wormbody_v3.py`** — body/MJCF work, unclear if Session 1

### Ambiguous scope — needs your call

The following are uncommitted but **may be from prior or parallel work blocks I lack context on:**

**Phase 0 / Path B engineering** (likely separate work block, NOT today's Wave 2 work):
- `scripts/brain/phase0_*.py` (~16 Python files: adversarial worker, daf7 sweep, layer1 validation, modulator d1, param_lhs, path_b diagnostic, peptide expression audit, postvolt peredge analyzers, ris pathway, ris scenario scan)
- `scripts/brain/artifacts/phase0_*.{json,csv,txt,npz,png}` (numerous artifacts)
- `scripts/brain/artifacts/phase0_*/` (config/log/trace directories)
- `scripts/brain/artifacts/phase0_layer1_validation/`, `phase0_param_lhs_configs/`, `phase0_param_lhs_logs/`, `phase0_param_lhs_traces/` (236MB!)
- `scripts/brain/artifacts/phase0_postvolt_peredge_*_traces/` (~46MB total across 7 scenarios)

**Phase 1 / Phase 3 / Phase 6 work** (also possibly separate):
- `scripts/brain/phase1_calibrate_cascades.py`, `phase1_curvature_compare.py`, `phase1_plateau_calibrate.py`, `phase1_tierpsy_reference.py`
- `scripts/brain/phase3_cascade_diagnostic.py`
- `scripts/brain/phase6_trajectory_correlate.py`

**Wave 1 audit artifacts:**
- `scripts/brain/artifacts/phase_v_w1_biophysical_audit_matrix.md` (36 KB)
- `scripts/brain/artifacts/phase_v_w1_research_tool_roadmap.md` (24 KB)
- `scripts/brain/artifacts/phase_v_w1_graded_kca.json`

**docs/ tree** (10 markdown files + `specs/` subdir):
- `docs/audit-strategy.md`, `docs/citation-audit-checklist.md`, `docs/claude-chat-context.md`, `docs/current-state-summary.md`, `docs/new-session-primer.md`, `docs/path_b_engineering_spec.md`, `docs/project-history.md`, `docs/t0_resolution_report.md`, `docs/tier2-4-execution-plan.md`, `docs/specs/`
- These look load-bearing for project documentation. `claude-chat-context.md` is the §5 file WB2 will edit — committing it now lets WB2 work against tracked baseline.

**`scripts/brain/references/`** — directory contents not enumerated above.

**Q for you:** Which of these ambiguous categories are Session 1 scope vs separate work blocks? Three options I see:

1. **Conservative:** commit only the explicitly-Wave-2 stuff (wave2/, the 4 cython-flipped files, Wave 2 artifact docs, session handoffs). Leave Phase 0/1/3/6/Wave 1 + docs/ untracked for separate work blocks.
2. **Wide:** commit everything in `scripts/brain/` that's not AnestheticSimulator, treating Phase 0/1/etc. as Session 1 work that just happened earlier today/yesterday.
3. **Hybrid:** commit Wave 2 + cython migration + docs/ (since `claude-chat-context.md` is needed for WB2), defer Phase 0/1/3/6 and `references/` to separate commit work blocks.

**Recommendation:** option 3. WB2 needs `claude-chat-context.md` tracked; deferring Phase 0/1/3/6 to separate commit work blocks respects the methodology of "honest scope labels" (committing them under Wave 2 work block messages would mislabel the scope).

---

## .gitignore proposal

### Proposed additions (with rationale)

```gitignore
# Wave 2 — large validation result JSONs containing per-sweep trajectories
# (regeneratable from wave2/run_*.py drivers; canonical findings live in .md files)
scripts/brain/wave2/artifacts/option_b_*_results.json
scripts/brain/wave2/artifacts/avar_validation_results.json
scripts/brain/wave2/artifacts/option_alpha_phase_f_results.json
scripts/brain/wave2/artifacts/ca_coupling_test_results.json

# Wave 2 — Nicoletti 2024 PDF + page rasters (external IP; PDFs available from PLOS ONE)
# Per-panel PNGs in figures/ root (not source_pdfs/) are kept as digitization refs
scripts/brain/wave2/artifacts/figures/source_pdfs/

# Phase 0 — large LHS traces (regeneratable from phase0_param_lhs_runner.py)
scripts/brain/artifacts/phase0_param_lhs_traces/

# Phase 0 — per-edge postvolt scenario traces (regeneratable from postvolt_peredge audit scripts)
scripts/brain/artifacts/phase0_postvolt_peredge_*_traces/

# Brian2 / NEURON build artifacts
**/x86_64/
**/__pycache__/
*.pyc
*.so
```

### Rationale per entry

| Entry | Size impact | Justification |
|---|---|---|
| `option_b_*_results.json` | ~24 MB (RIM 15 + AIY 8 + others 1) | Per-sweep voltage trajectories; canonical results in `cellular_validation_findings.md` and per-cell `*_construction.md` summaries; regeneratable from `wave2/run_*.py` drivers |
| `figures/source_pdfs/` | 30 MB | Nicoletti 2024 PDF + 16 page rasters; PDF licensing convention is to link not redistribute even with CC BY; rasters regenerate via `pdftoppm` |
| `phase0_param_lhs_traces/` | 236 MB | Largest single artifact in repo; LHS sweep traces, regeneratable; analysis output already in `phase0_param_lhs_synthesis.json` (688 KB, kept) |
| `phase0_postvolt_peredge_*_traces/` | ~46 MB | Per-scenario trace bundles; analysis output in `phase0_postvolt_peredge_baseline_synthesis.{json,txt}` (kept) |
| `**/x86_64/`, `__pycache__/`, etc. | varies | NEURON compiled mods + Python bytecode; the existing `.gitignore` covers `__pycache__/` but not `**/x86_64/` |

### Verification — won't break currently-tracked files

Checked: none of the proposed glob patterns match files currently tracked in git. Existing `.gitignore` already handles `data/external/`, `atanas_worm_*.npz`, `phase0_*.log`, `phase0_scenario_traces/`. No conflict.

### Files that would now be ignored

- 30 MB in `wave2/artifacts/figures/source_pdfs/`
- 236 MB in `phase0_param_lhs_traces/`
- ~46 MB across `phase0_postvolt_peredge_*_traces/`
- ~24 MB in Wave 2 large JSONs
- **Total ignored: ~336 MB**

Without these additions, committing would either bloat repo substantially OR require selective `git add` per group (slower, error-prone).

---

## Open question on `modulator_tables.json` modification

`scripts/brain/artifacts/modulator_tables.json` is modified but not explicitly part of any documented Wave 2 work. Could be:
- Side effect of a Wave 2 run (unlikely — Wave 2 doesn't write to this path)
- Side effect of a Phase 0 / Path B run
- Manual edit from an earlier session

**Recommendation:** investigate this diff before committing. If it's a side effect of a previous session, group with that work; if substantive, surface for review. I'll inspect during execution unless you'd prefer to do it yourself first.

---

## Decisions needed at this CP1 pause gate

1. **Confirm scope option** (1=conservative, 2=wide, 3=hybrid). I lean **3 (hybrid)**.
2. **Approve `.gitignore` additions** or specify changes.
3. **Adjudicate `modulator_tables.json`** — investigate or skip the modification.
4. **Confirm `scripts/build_wormbody_v3.py`** is Session 1 scope or defer.

After your call, I'll apply the `.gitignore` updates, verify staging is correct, and proceed to CP2 (commit groupings proposal).

---

## What CP2 will look like (preview)

Based on option 3 hybrid + approved `.gitignore`:

- **Group A:** Wave 2 channel library (14 channels + translation_patterns.md + F-catalog work)
- **Group B:** Wave 2 cell builders (AVAL, AIY, RIM, AVAR — 4 production-grade cells)
- **Group C:** Wave 2 validation harnesses + drivers (voltage-clamp, plateau, NEURON reference wrapper, run_*.py drivers)
- **Group D:** Mellem 2008 investigation + citation audit + literature scoping (the methodology trail)
- **Group E:** Cython migration (4 production simulator files + 2 Wave 1 sandbox files + cython_migration/ wrapper)
- **Group F:** Phase δ scoping + WB1 + WB2 findings + Wave2HybridBrain scaffold (F20 capacitance mismatch documented)
- **Group G:** Stage I-IV overnight outputs (literature scoping, AVAR Stage II, Stage IV §5 reframing, overnight summary)
- **Group H:** Wave 2 prompt artifacts (the `phase_v_w2_*_prompt.md` files) — these document what work blocks were deployed; useful for reproducibility
- **Group I:** Session handoff docs (`session_{1,3}_handoff_2026-04-26.md`)
- **Group J:** docs/ tree (claude-chat-context.md, etc.) — needed for WB2

Total: 10 commits proposed for Session 1 WB1, all with primary-source-disciplined honest messages.

Standing by for your approval at this CP1 pause gate.
