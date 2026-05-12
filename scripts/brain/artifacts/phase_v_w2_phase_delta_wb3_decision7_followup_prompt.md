# Phase δ WB3 follow-up — Decision 7 readout artifact resolution (Option α)

**Mode:** focused follow-up work block. Bounded ~60-90 min. Architectural decision determined (σ-magnitude readout per `graded_brain.py output_rates()` precedent). No pre-flight pushback required.

**Predecessor:** WB3 CP2-CP6 completed locally with 4 commits queued (`29b88e3`, `539b546`, `83a2a72`, `617cdad`); push blocked by sandbox protection. Cross-coupling biology propagates correctly through chemical pathway. Caveat 1 (V_half sensitivity) resolved positively. Caveat 2 (W_graded_I retune) trajectory followed honestly. One load-bearing readout artifact surfaced and needs resolution before WB4 multi-cell drop-in or Phase G LIFBrain integration deploy.

---

## The artifact this work block resolves

Decision 7(a) — σ > 0.5 rising-threshold pseudo-spikes — breaks at saturation. At the network's operating point, σ stays continuously above 0.5 for active Wave 2 cells. There's no rising-threshold crossing → pseudo-spike rate = 0. The biology IS reaching AVA (V driven from -40 → -16 mV across W_graded_I ladder 0.3 → 10 pA), but the `firing_rates()` readout API reports zero.

Downstream impact (silent corruption if uncorrected):
- Phase G FSM classifier sees Wave 2 cells as silent even when maximally active
- WB4 multi-cell drop-in inherits the readout issue
- Ablation harness consumes Phase G output, inherits corruption
- Dashboard at /projects/c-elegans-multimodal would render Wave 2 cells as silent in saturation

**Resolution:** Option α — revise Decision 7 to use σ-magnitude continuous output as canonical Wave 2 cell readout, matching `graded_brain.py output_rates()` line 378 pattern.

---

## Out of scope

- WB4 multi-cell drop-in (separate work block; unblocked by this follow-up)
- Phase G LIFBrain integration (Session 2 territory; unblocked by this follow-up)
- Modifying Wave 2 cell-builder code (read-only; readout change happens at Wave2HybridBrain layer)
- Bifurcated API approaches (Option β rejected; commits to unified σ-magnitude readout for Wave 2 cells)

---

## Working environment

- Brain code: `~/Desktop/website/personalwebsite/scripts/brain/`
- Wave 2 work: `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- Wave2HybridBrain: `wave2/integration/wave2_hybrid_brain.py`
- Reference: `scripts/brain/graded_brain.py output_rates()` at line 378
- WB3 findings: `wave2/artifacts/phase_delta_wb3_findings.md`
- F20 catalog: `wave2/translation_patterns.md`
- Wave 2 venv: `~/venvs/wave2-neuron/`
- Cython is production default

---

## CP1 — σ-magnitude continuous readout implementation

1. **Read `graded_brain.py output_rates()` at line 378** to understand the production σ-magnitude pattern. Document the API contract:
   - Input: cell state at time t
   - Output: continuous activity level reflecting σ value
   - Units: dimensionless [0,1] OR converted to pseudo-rate Hz; verify graded_brain.py convention
   - Per-cell vs population: matches existing population firing_rates() shape

2. **Modify `wave2/integration/wave2_hybrid_brain.py`** to expose σ-magnitude continuous readout as canonical Wave 2 cell activity API:
   - Add `wave2_activities()` method (or equivalent name matching project conventions) returning σ-magnitude per Wave 2 cell
   - Preserve existing `firing_rates()` API for LIF cells unchanged
   - For mixed brain readouts (LIF + Wave 2), provide unified activity API that returns appropriate readout per cell type:
     - LIF cells: firing rate via existing infrastructure
     - Wave 2 cells: σ-magnitude continuous activity
   - Document per-cell-type readout pattern in code comments + class docstring

3. **Update Decision 7 implementation site** (σ > 0.5 rising-threshold pseudo-spike emission):
   - Remove pseudo-spike emission entirely if no consumer needs it, OR
   - Mark as legacy with deprecation note pointing to σ-magnitude readout
   - Pick whichever the existing code structure makes cleaner. Don't preserve the artifact under "legacy" framing if no consumer actually uses it.

**CP1 acceptance criteria:**
- σ-magnitude continuous readout exposed as canonical Wave 2 activity API
- LIF firing_rates() API preserved unchanged
- Unified mixed-brain activity readout pattern documented
- Decision 7 pseudo-spike emission cleaned up appropriately
- Code matches graded_brain.py output_rates() precedent

---

## CP2 — Re-validate CP4 touch cascade with σ-magnitude readout

The original WB3 CP4 showed AVAL/AVAR pseudo-spike Δ < ±1 Hz across W_graded_I ladder — the artifact symptom. Re-run validation with σ-magnitude readout to verify biology actually reaches Wave 2 command cells.

1. Run touch_anterior 30s under WB3 cross-coupling at the W_graded_I value WB3 settled on (likely 1.0 pA or 3.0 pA from retune ladder; check WB3 findings for actual final value)
2. Profile per-neuron readouts pre-touch (1-5s) vs peri-touch (5-7s):
   - LIF cells (ALM/AVM/AIB/AVB/AVD) via firing_rates() — should show same cascade as previous CP4
   - Wave 2 cells (AVAL/AVAR/AIY/RIM) via σ-magnitude readout — this is the new validation
3. Verify σ-magnitude readout shows meaningful AVA Δ peri-touch:
   - Expected: AVAL/AVAR σ increases substantively from baseline to peri-touch (biology was confirmed mechanistically via V trajectory; σ should reflect this)
   - Compare to per-edge LIF baseline AVA Δ+7.5 Hz from Stage IV (units: σ-magnitude isn't directly Hz, but relative change should be consistent)
4. Document the mapping between σ-magnitude and "equivalent firing rate" if needed for downstream FSM consumers. graded_brain.py output_rates() may already do this conversion; reuse if so.

**CP2 acceptance criteria:**
- Touch cascade re-run with σ-magnitude readout for Wave 2 cells
- AVA Δ peri-touch now visible via σ-magnitude (artifact resolved)
- Comparison to per-edge LIF baseline documented
- σ-magnitude to firing-rate conversion documented if relevant

**CP2 failure modes:**
- AVA σ Δ peri-touch is small even with corrected readout: real biological finding, not artifact. Document and surface — would suggest cross-coupled biology isn't propagating as mechanistically confirmed. Pause for review.
- σ-magnitude API doesn't expose granularity needed for cascade analysis: pause, refine API

---

## CP3 — F20 catalog update + WB3 findings amendment

1. **Update `wave2/translation_patterns.md` F20 entry** with readout pattern lesson:
   - Pattern: saturating activation functions (σ Boltzmann) break rising-threshold detectors
   - Why: in saturated regime, no rising crossings means rate-based readouts report zero despite active dynamics
   - Resolution: continuous output APIs (σ-magnitude) for graded cells; rising-threshold readouts only for genuinely spiking cells
   - Generalization: any time discrete-event readouts are applied to continuous-output systems, check whether saturation is in the operating range

2. **Amend `wave2/artifacts/phase_delta_wb3_findings.md`** with readout artifact resolution:
   - Original Decision 7(a) finding (pseudo-spike rate = 0 in saturation) preserved as historical record
   - New section documenting σ-magnitude resolution
   - CP2 re-validation result (AVA Δ peri-touch via σ-magnitude readout)
   - Confidence rating updates: AVAL/AVAR cascade propagation now confirmed via biology AND readout (was previously confirmed via biology only with readout artifact obscuring it)

3. **Update `wave2/integration/wave2_hybrid_brain.py` docstring/header** to document per-cell-type readout pattern as canonical API contract going forward.

**CP3 acceptance criteria:**
- F20 catalog entry updated with readout pattern lesson
- WB3 findings amended with resolution
- Code documentation reflects new API contract

---

## CP4 — Commit + push

1. Stage changes in 3 logical groups:
   - Group A: Wave2HybridBrain σ-magnitude readout implementation (CP1)
   - Group B: CP2 re-validation outputs
   - Group C: F20 catalog update + WB3 findings amendment + docstring updates

2. Honest commit messages:
   - **Group A:** `Wave 2 σ-magnitude readout — resolves Decision 7 pseudo-spike artifact in saturated regime. Adds canonical wave2_activities() API matching graded_brain.py output_rates() precedent. LIF firing_rates() API preserved. Per-cell-type readout pattern documented as canonical contract for downstream consumers (Phase G FSM classifier, WB4 multi-cell drop-in, ablation harness).`
   - **Group B:** `WB3 CP4 re-validation with σ-magnitude readout — touch cascade AVA Δ peri-touch now visible (was obscured by rising-threshold detector blindness in saturated regime). Cross-coupled biology propagation confirmed via both V trajectory AND readout API.`
   - **Group C:** `F20 catalog + WB3 findings amendment — saturating activation functions break rising-threshold detectors as translation pattern. Readout API contract for graded cells documented.`

3. **Push the original 4 WB3 commits + these 3 follow-up commits together to remote:**
   ```
   git -C ~/Desktop/website/personalwebsite push origin main
   ```
   Should land 7 commits total at origin/main.

**CP4 acceptance criteria:**
- All work committed with honest messages
- Remote push succeeds
- WB3 + follow-up history clean on origin/main

---

## Failure modes and recovery

- Implementation surfaces that `graded_brain.py output_rates()` pattern doesn't compose cleanly with mixed brain readouts: pause, document integration challenge, propose adapted approach
- CP2 re-validation shows AVA cascade isn't propagating biologically (not just artifact): substantive finding, pause; would invalidate WB3 Caveat 2 retune trajectory
- Push fails for non-trivial reason (auth, branch protection, conflicts): pause, document, escalate
- σ-magnitude to firing-rate conversion ambiguity: document multiple approaches, recommend the one matching graded_brain.py precedent

**General principle:** σ-magnitude readout matches established graded_brain.py precedent. This isn't novel architecture, it's bringing cross-coupling implementation into alignment with project's existing graded-cell idiom. Pause-with-documentation > push-through if anything unexpected surfaces.

---

## On time scoping

- CP1 readout implementation: ~30-45 min
- CP2 re-validation: ~15-30 min
- CP3 F20 + findings update: ~15 min
- CP4 commit + push: ~10 min

Total: ~60-90 min. Bounded follow-up work block.

Begin with CP1: read `graded_brain.py output_rates()` line 378 first, document the precedent, then implement matching pattern in Wave2HybridBrain.
