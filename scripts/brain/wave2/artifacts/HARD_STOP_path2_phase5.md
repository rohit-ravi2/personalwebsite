# HARD STOP — Path 2 Phase 5 (2026-05-12)

**Triggered:** §5.2 Tier 3 — ≥50% of (channel, cell) combinations beyond
5× Nicoletti agreement. **Observed: 75% (12 of 16 evaluated combinations).**

**Per methodology §5.2:** Methodology fundamentally inadequate at the formula
level — refinements within the v1 framework won't recover acceptable
validation. Surface for architectural direction; do not proceed to Phase 6.

---

## Diagnosis

### Failure structure

Two distinct, structured failure patterns (not random):

**Pattern A — Small-cell systematic under-channeling (10/12 cases).**

All 4 AIY channels + all 6 RIM channels show derived gbar UNDER Nicoletti
by factors of 6× to 1000×. Cause: AVAL-anchored C_global × linear TPM
scaling produces small absolute channel counts in small cells (AIY surface
65.9 μm² is ~17× smaller than AVAL 1124 μm²). Nicoletti's parameterization
compensates with high per-cm² gbars in small cells to fit published
whole-cell currents.

**Pattern B — IRK over-channeling in large cells (2/12 cases).**

AVAL IRK 8× over, AVAR IRK 19× over Nicoletti. Cause: γ_IRK = 25 pS (Kir2.1
chord conductance) is the largest γ in the inventory; combined with high
IRK paralog-sum TPM (165.6 in AVA), Path 2 predicts high density that
exceeds Nicoletti's per-cm² gbar.

### Methodologically important finding (Phase 5 pre-flight Nicoletti uncertainty check)

Direct query of Nicoletti 2024 (PMC10980225) on the FRAC-flagged parameters:

- **NO error bars / confidence intervals reported** for any fitted gbar
- **NO sensitivity analyses** reported
- Authors **explicitly acknowledge "non-uniqueness of the set of parameters"**

**Implication:** Nicoletti's specific per-cell gbars are point estimates
from local optima of under-constrained fits, not biophysical ground truth.
Multiple parameter combinations likely reproduce the same I-V curves
equivalently well. Path 2's 75% disagreement is partly a critique of
Nicoletti's parameterization uniqueness rather than a methodology failure
of Path 2.

This supports Rohit's interpretation (a) from the Phase 4 framework:
"Nicoletti overfit AIY/RIM parameters (multi-channel cells, fewer
constraints per parameter)." Empirically validated by direct paper-reading.

---

## What Path 2 v1 DOES ship

Despite the hard stop on cell-builder deployment, Path 2 v1 ships:

1. **Methodology document** (`docs/channel_parameter_derivation_methodology.md`)
   — reference material for the entire substrate redesign trajectory,
   including the new failure-mode category "biophysically derived under
   assumptions inconsistent with substrate" + recurring "parameter audit
   before integration" methodology step.
2. **γ inventory** (`docs/channel_gamma_inventory.md`) — primary-source
   verified single-channel conductance values for 9 Wave 2 channels.
3. **TPM inventory** (`docs/channel_tpm_inventory.md`) — per-(channel, cell)
   CeNGEN T2 data + heteromer-vs-paralog decisions + Phase 3.5 false-
   negative disambiguation.
4. **Calibration protocol** (`docs/channel_calibration_protocol.md`) —
   C_global computed from EGL-19/AVAL reference with biophysical
   plausibility verification.
5. **Derivation module** (`scripts/brain/wave2/channels/derived_channel_parameters.py`)
   — production code with calibrated constants. Available for use under
   any architectural direction Rohit selects.
6. **Phase 5 validation document** (`docs/path2_channel_validation.md`)
   — full validation table + tier classification + architectural-direction
   options.

The methodology contribution (audit + derive-not-inherit + minimal-
calibration pattern) transfers to subsequent layers regardless of Path 2
v1 deployment decision. See methodology doc §7 forward-looking applications.

---

## What Path 2 v1 does NOT ship without architectural direction

- **Phase 6 cell-builder integration BLOCKED.** Path 2 derived gbar values
  not deployed into AVAL/AVAR/AIY/RIM cell builders.
- **§7.4 Phase F restructure still BLOCKED** (per §7.3.5 dependency).
- **Layer 1 v1 cell-level validation pending architectural direction.**

---

## Architectural-direction options (full discussion in path2_channel_validation.md §6)

### Option α — Path 2 v1 ships under reframed validation criteria

Validate against cell-level rest stability + I-V envelope match (not
per-channel gbar match against Nicoletti's specific values). Acceptance
criterion shift acknowledges that Nicoletti's gbars are degenerate fits
without uncertainty quantification.

**Phase 6 conditional on:** Rohit authorizes reframed validation criteria.

### Option β — Path 2 v2 methodology refinement

Deploy per-cell-family C_global (AVA / AIY / RIM each get their own
C_global value) + refit γ_IRK. Increases free parameter count from 1 to
~4 across cell families.

**Phase 5b conditional on:** Rohit authorizes v2 scope; requires new
calibration anchors (more than EGL-19 in AVAL alone).

### Option γ — Path 1 fallback (refit Nicoletti under physiological Nernst)

The originally-considered Path 1 from pre-Path-2 scoping. Refits
Nicoletti's gbars to match published I-V curves under physiological E_Ca.
Preserves Nicoletti's per-cell parameterization while addressing §7.3
finding.

**§7.3.5 Path 1 work block:** estimated 3-5 work blocks; original scope
preserved in `docs/substrate_redesign_roadmap.md` git history pre-Path 2
update.

### Option δ — Hybrid derivation framework

γ-per-channel as free parameter calibrated against Nicoletti's I-V curves;
TPM × E_translation × surface as cross-cell scaling primitive. Hybrid of
Path 1 (data-fit) and Path 2 (biology-derived). Substantial new scope.

**Phase 0 conditional on:** New work block scoping; not bounded as a
modification to existing Path 2.

---

## Recommendation

**Option α (Path 2 v1 ships under reframed validation criteria) followed
by Option β (v2 refinement) if Phase 6 reveals load-bearing issues.**

The Nicoletti uncertainty finding genuinely shifts validation interpretation.
Per-cell integrated rest stability + I-V envelope match is more biologically
meaningful than per-channel gbar point-match. If Path 2 v1 cells achieve
stable rest with physiological [Ca]_in and reproduce Nicoletti's I-V
envelopes (within reported SEM), the v1 methodology contribution is
preserved — and the substrate redesign delivers its first major
demonstration of "physiological Nernst + biology-derived parameters
produce equivalent biology to Nicoletti's degenerate per-cell fits."

If Phase 6 fails, the failure pattern informs which v2 refinement (Option
β) to deploy. Skipping straight to v2 risks fitting to a different aspect
of Nicoletti's degenerate parameter space.

---

## Standing-by state

- Path 2 Phases 1-5 SHIPPED with documented hard stop
- All Phase 1-5 deliverables in git (commits 9d19672 → 4b68467 → Phase 5)
- Phase 6 BLOCKED pending architectural-direction decision
- §7.4 Phase F restructure remains BLOCKED (§7.3.5 dependency)

**Awaiting Rohit's direction on Options α / β / γ / δ.**
