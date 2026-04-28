# Wave 2 Brian2 cython codegen migration

**Mode:** infrastructure work block. Single session, file-based pause-and-wait.

**Strategic positioning:** Phase δ network integration is the next major Wave 2 deliverable. Compute-bound enough that migration before scoping produces meaningfully better scoping output (accurate compute-cost estimates, accurate architectural constraints). RIM CP6 took ~50 min for 11 × 1000ms current-clamp sweeps under current codegen — Phase δ will compound this substantially.

---

## Out of scope

- Phase δ scoping or implementation (separate work blocks)
- RMD validation (Nicoletti 2019 acquisition needed)
- Methodology paper, architectural plan revisions, AVAR upstream
- **Performance optimization beyond cython codegen target switch** (no algorithmic optimization, no parallelization, no code restructuring beyond what migration requires)

If cython migration produces speedup substantially below expected (5-10×), document the finding but don't expand scope into broader optimization.

---

## Working environment

- Venv: `~/venvs/wave2-neuron/`
- Code: `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- 3 production-grade cells:
  - `wave2/option_alpha_ava_cell.py` (4-channel AVAL)
  - `wave2/option_alpha_aiy_cell.py` (7-channel AIY)
  - `wave2/option_alpha_rim_cell.py` (7-channel RIM)
- Validation harnesses: voltage_clamp_harness, plateau_harness with Layer A compare
- F1-F18 catalog (with F18 refinement from RIM): `wave2/translation_patterns.md`
- 14 channel translations at `wave2/channels/`

---

## Pre-flight pushback expected

Specific verification items:

1. **C compiler availability** in venv. Check `gcc --version`. If missing, scope question.
2. **Current Brian2 codegen target.** Don't assume numpy. Check actual `prefs.codegen.target` value. If already cython, this work block has different shape.
3. **Cython compatibility scan** of 3 cell construction files. Standard Brian2 string equations cleanly compatible. Custom Python functions in equations or unusual conditional logic may not translate.

If pre-flight surfaces concerns: write to `wave2/artifacts/cython_migration_pushback.md` + create `PAUSED_FOR_REVIEW.txt`. Otherwise proceed.

---

## Methodology continuity

- Mid-flight findings to `wave2/artifacts/cython_migration_findings.md`
- F1-F18 lessons applied: F18-class concerns under cython (multi-USEION-ca eca handling, GLOBAL semantics, S/cm² conventions) need verification if any cell shows divergence post-migration
- Don't fudge parameters or tolerances to make validation pass post-migration
- Document, don't fabricate

**Validation residuals must match pre-migration baselines.** Cython codegen should produce numerically identical or near-identical results to numpy codegen for the same model. Post-migration validation should produce same residuals as pre-migration (within ~10% of baseline values, not absolute equivalence — cython may produce tiny floating-point differences):

- **AVAL:** 11/11 voltage-clamp holds at max div ≤0.0035, 7/7 current-clamp sweeps with V agreement ~5 decimal places
- **AIY:** 11/11 voltage-clamp holds at max div ≤0.0113, 10/11 current-clamp sweeps (the -15 pA KQT-1 numerical drift) with same residuals
- **RIM:** 11/11 voltage-clamp holds at max div ≤0.0043, 11/11 current-clamp sweeps with 0.000 mV residuals across 55,000 timepoints

If post-migration residuals are meaningfully different from pre-migration baselines, that's a finding requiring investigation — not a sign to relax tolerances.

---

## CP1 — Environment verification + pre-migration baseline

1. Verify C compiler available (`gcc --version` or platform equivalent)
2. Verify Brian2 cython codegen dependencies (Cython package installed, runtime imports work)
3. Document current `prefs.codegen.target` value before any changes
4. Run minimal smoke test: tiny Brian2 model under current codegen, time to execute. Establishes pre-migration baseline.
5. Quick scan of 3 cell construction files for cython-incompatible patterns

**CP1 acceptance:** C compiler verified; Cython verified; current codegen target documented; pre-migration baseline timing recorded; no incompatible patterns OR identified patterns documented; status output.

**CP1 failure modes:**
- C compiler missing → pause, may require venv rebuild
- Cython package missing → install if simple, pause if complex
- Cython-incompatible patterns → document, surface before migration

---

## CP2 — Codegen target switch + minimal smoke test

1. Switch Brian2 prefs to cython codegen target (document exact change)
2. Re-run minimal smoke test under cython
3. Document timing comparison vs CP1 baseline

**CP2 acceptance:** cython codegen active; smoke test runs without errors; speedup factor documented; status output.

**CP2 failure modes:**
- First-run cython compile errors → document specifics, may indicate environment or pattern issue
- Cython runtime errors → investigate per Brian2 docs
- No measurable speedup on minimal model → document; small models may not show speedup

---

## CP3 — AVAL re-validation under cython

1. Run AVAL voltage-clamp Layer A under cython using existing harness
2. Run AVAL current-clamp Layer A under cython using existing harness
3. Compare residuals against pre-migration baseline (max div within ~10% of baseline)
4. Time the runs; compute speedup vs pre-migration

**CP3 acceptance:** voltage-clamp validation passes with residuals matching baseline (max div ≤~0.004); current-clamp validation passes with residuals matching baseline; speedup factor documented; status output.

**CP3 failure modes:**
- Validation fails post-migration → investigate (cython edge case, FP semantics differences, incompatible pattern). Pause if cause unclear.
- Residuals meaningfully worse → investigation required, no tolerance relaxation
- No speedup → document as finding (may indicate cells aren't compute-bound as expected, or cython has overhead for short runs)

---

## CP4 — AIY re-validation under cython

Same workflow as CP3 applied to AIY.

**Specific concerns:**
- AIY has multi-USEION-ca asymmetric pattern (slo1egl19 reads but doesn't write ica). F18 required explicit `eca_mV = 127.59`. Verify this still works under cython.
- AIY has KQT-1 slow integrator drift at -15 pA in current-clamp (10/11 sweep result). Cython may or may not change integrator behavior. Same residuals expected; document if different.

**CP4 acceptance:** voltage-clamp passes (11/11, max div ≤~0.012); current-clamp passes (10/11; same -15 pA failure if drift consistent across codegens); speedup documented; status output.

**CP4 failure modes:**
- Multi-USEION-ca eca handling breaks under cython → investigate F18 pattern. Pause if unclear.
- KQT-1 drift character changes (different injection fails, all 11/11 pass under cython, systematic divergence) → document as cython behavioral finding

---

## CP5 — RIM re-validation under cython

Same workflow as CP3 applied to RIM.

**Specific concerns:**
- RIM has multi-USEION-ca symmetric pattern (3 USEION ca, eca preserved at 60 mV per F18 refinement). Verify ion_style behavior matches under cython.
- RIM has UNC-2 with GLOBAL declarations that auto-resolved under Brian2's per-cell-by-default. Verify GLOBAL handling still auto-resolves under cython (cython namespace handling sometimes differs).
- RIM is the cleanest validation result (CP6 was 11/11 with 0.000 mV residuals across 55,000 timepoints). Meaningfully worse residuals = finding.

**CP5 acceptance:** voltage-clamp passes (11/11, max div ≤~0.005); current-clamp passes (11/11, ~0.000 mV residuals); speedup documented; status output.

**CP5 failure modes:**
- F18 ion_style behavior differs under cython → investigate, document. May require per-codegen verification standing requirement.
- UNC-2 GLOBAL semantics break under cython → investigate F2-class. May require explicit per-cell handling under cython.
- Residual divergence vs baseline → investigation required, no tolerance relaxation

---

## CP6 — Benchmark + outcome documentation

1. **Representative Phase-δ-like workload:** simulate all 3 cells (AVAL + AIY + RIM) for 10 seconds simulated time, with current injection schedules. Approximates Phase δ workload character.
2. Compare wall-clock time against equivalent numpy codegen run (re-run with numpy for direct comparison).
3. Document speedup factor for representative workload.
4. Document any cython-specific findings extending F1-F18 catalog (F19+ if patterns surfaced).
5. Write outcome summary `wave2/artifacts/cython_migration_summary.md`.

### Outcome summary structure

**Section 1: Verdict** (one of):
- VERDICT_CYTHON_PRODUCTION_READY: all 3 cells validate post-migration with residuals matching baselines; speedup at expected magnitude (5-10× or actual measured). Cython is production target.
- VERDICT_CYTHON_PARTIAL: some cells validate, others have issues. Document specifics.
- VERDICT_CYTHON_INSUFFICIENT_SPEEDUP: all cells validate but speedup meaningfully below expected. Surface decision.
- VERDICT_CYTHON_BLOCKED: cython produces validation issues unresolved this work block. Defer migration, document blockers.

**Section 2: Speedup measurements** — minimal smoke test, AVAL, AIY, RIM, representative workload, net assessment

**Section 3: Findings extending F1-F18** — F19+ if applicable. Particularly: F18 ion_style under cython, GLOBAL under cython, any cython-specific patterns.

**Section 4: Implications for Phase δ scoping** — performance baseline, architectural constraints, compute envelope

**Section 5: Standing followups** — update from previous list

**Section 6: Recommendations:**
- PRODUCTION_READY → proceed to Phase δ scoping with cython baseline
- PARTIAL/INSUFFICIENT → surface decision (proceed with current state, or invest more in cython)
- BLOCKED → Phase δ scoping proceeds against numpy baseline; cython deferred

**CP6 acceptance:** representative benchmark run with measured speedup; outcome summary complete; all findings documented; final state file `wave2/artifacts/checkpoints/cython_run_state.json`.

---

## Failure modes and recovery

**Environment:**
- C compiler missing or broken → abort, document, may require venv rebuild
- Cython package issues → install if simple, document if complex

**Migration (CP2):**
- Codegen switch fails → document errors, investigate Brian2 cython docs
- Smoke test fails → pause, investigate

**Validation (CP3-CP5):**
- Per-cell validation divergence → pause, investigate. Don't proceed to next cell until current cell understood.
- Systematic divergence across cells → pause, investigate cython broadly. May indicate environment or fundamental incompatibility.
- Speedup not observed → document, continue (speedup desired but not strictly required for PRODUCTION_READY if validation passes)

**Benchmark (CP6):**
- Benchmark fails → investigate, may indicate Phase-δ-scale issues
- Speedup substantially below expected → document, surface for review

**General principle:** pause-with-documentation > fabricate-completion. Cython migration is infrastructure — better to have CP1-CP3 cleanly completed and pause at CP4 with documented divergence than ship migration with hidden uncertainty.

---

## Infrastructure robustness

Same as previous overnight runs: timeouts, NaN/Inf detection, memory monitoring, **disk space verification** (cython codegen produces compiled .so files in cache directory; verify sufficient disk space), catch and document errors.

---

## On time scoping

CP1-CP2 should complete quickly (env check + codegen switch + smoke test). CP3-CP5 each re-run validations that already worked under numpy — same compute. CP6 adds representative benchmark.

Total expected: 1-2 hours including measurement and documentation.

---

## Output file structure

```
wave2/
└── artifacts/
    ├── checkpoints/
    │   ├── cython_CP1_status.json - CP6
    │   └── cython_run_state.json
    ├── cython_migration_summary.md                      # CP6 entry point
    ├── cython_migration_findings.md                     # mid-flight findings
    ├── cython_migration_pushback.md                     # only if pre-flight concerns
    └── PAUSED_FOR_REVIEW.txt                            # only if paused
```

May produce minor edits to `wave2/option_alpha_*_cell.py` files if cython compatibility requires (document any changes).

---

## Final framing

This work block is infrastructure preparation for Phase δ scoping. After this completes with PRODUCTION_READY verdict, Phase δ scoping runs against accurate compute characteristics. If verdict is PARTIAL or BLOCKED, Phase δ scoping proceeds with explicit awareness of compute baseline.

Apply the discipline. Pre-flight verification of environment. Don't relax tolerances. Document, don't fabricate. F1-F18 lessons under cython codegen — particularly F18 ion_style and GLOBAL semantics.

You have full implementation context. Single-session focused infrastructure work.

Standing by for pre-flight pushback or completion notification.
