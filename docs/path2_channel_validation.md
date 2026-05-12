# Path 2 channel validation — Phase 5 deliverable

**Status:** Phase 5 of §7.3.5 Path 2. Per-channel validation against
Nicoletti's published gbar values. **Tier 3 HARD STOP triggered: 75% of
(channel, cell) combinations beyond 5× Nicoletti agreement.**

Per Path 2 methodology §5.2 + Rohit's Phase 5 authorization decision rule:
hard stop at ≥50% beyond 5×. Phase 6 does NOT proceed until architectural-
level decision per `scripts/brain/wave2/artifacts/HARD_STOP_path2_phase5.md`.

**Date:** 2026-05-12

---

## 1 · Methodology

For each (channel, cell) combination in Wave 2 cell builders:
- `gbar_derived = γ × TPM × E_translation × C_global` (Path 2 formula)
- `ratio = gbar_derived / gbar_Nicoletti`
- Band classification per methodology §4.1:
  - **Within 2× (good):** ratio ∈ [0.5, 2.0]
  - **Within 5× (acceptable):** ratio ∈ [0.2, 5.0]
  - **Beyond 5× (substantive finding):** ratio < 0.2 or > 5.0

Under Path 2 v1: γ values from Phase 2 inventory (mammalian homolog
fallback), TPMs from Phase 3 inventory (CeNGEN T2), C_global from Phase 4
calibration (1.7297e4 channels/(cm²·TPM)). Channel kinetic parameters
(V_half, k, time constants) preserved from Nicoletti — only gbar derives.

Under linear-conductance approximation, the gbar ratio at every voltage
equals the I-V ratio (because kinetics are preserved). Per-channel
voltage-clamp simulation is therefore unnecessary — gbar ratios are
the I-V validation metric directly. (Per-cell integrated VC/CC validation
is Phase 6 scope, which is blocked by this hard stop.)

---

## 2 · Per-(channel, cell) validation table

```
cell    channel    gbar_derived  gbar_Nicoletti   ratio    band
AVAL    EGL-19     9.288e-6     9.288e-6        1.0000   2× (reference, by construction)
AVAL    IRK        7.161e-5     8.898e-6        8.0479   BEYOND-5× (OVER)
AVAL    NCA        1.325e-5     0.000e+0           —     excluded (Nicoletti g=0)
AVAR    EGL-19     9.288e-6     5.735e-6        1.6195   2×
AVAR    IRK        7.161e-5     3.751e-6       19.0909   BEYOND-5× (OVER)
AVAR    NCA        1.325e-5     4.398e-6        3.0127   5×
AVAR    UNC-103    1.595e-6     4.294e-6        0.3714   5×
AIY     EGL-19     3.145e-6     1.518e-4        0.0207   BEYOND-5× (UNDER)
AIY     KQT-1      3.290e-6     3.035e-4        0.0108   BEYOND-5× (UNDER)
AIY     SHL-1      0.000e+0     7.588e-4        0.0000   BEYOND-5× (UNDER, T2 false neg per §3.5)
AIY     NCA        2.525e-6     9.106e-5        0.0277   BEYOND-5× (UNDER)
RIM     EGL-19     1.379e-5     3.200e-4        0.0431   BEYOND-5× (UNDER)
RIM     SHL-1      1.589e-5     9.048e-4        0.0176   BEYOND-5× (UNDER)
RIM     IRK        5.202e-5     3.273e-4        0.1589   BEYOND-5× (UNDER)
RIM     CCA-1      1.884e-6     8.452e-4        0.0022   BEYOND-5× (UNDER)
RIM     UNC-2      4.947e-6     9.677e-5        0.0511   BEYOND-5× (UNDER)
RIM     EGL-2      9.105e-6     1.412e-4        0.0645   BEYOND-5× (UNDER)
```

---

## 3 · Band distribution + tier classification

| band | count | percentage |
|---|---:|---:|
| Within 2× | 2 | 12.5% |
| Within 5× | 2 | 12.5% |
| **Beyond 5×** | **12** | **75.0%** |
| Total evaluated | 16 | — |
| Excluded (Nicoletti g=0) | 1 | — |

**§5.2 Tier classification: Tier 3 (≥50% beyond 5×) — HARD STOP.**

Per methodology §5.2 + Rohit's Phase 5 decision rule, this triggers:
- Write `HARD_STOP.txt` in artifacts with diagnostic data
- Surface for architectural direction; do not proceed to Phase 6
- Path 2 v1 does not ship until architectural-level decision lands

---

## 4 · Failure pattern analysis (per Rohit's framework)

User's framework distinguishes:
- **<30% beyond 5× + random:** ship v1 with limitations
- **<30% beyond 5× + systematic:** investigate systematic before shipping
- **≥30%:** v2 refinement required
- **≥50%:** hard stop, architectural direction needed

Here we're at 75%. But the failure pattern is highly structured:

### 4.1 Two distinct failure patterns

**Pattern A: Small-cell systematic under-channeling (10/12 beyond-5× cases)**

All 10 of these are UNDER (derived < Nicoletti), and all occur in AIY (small cell, 65.9 μm²) or RIM (small cell, 103 μm²):

| cell | n channels beyond 5× | ratio range | direction |
|---|---:|---|---|
| AIY | 4/4 (100%) | 0.000 - 0.028 | UNDER (50-1000× lower than Nicoletti) |
| RIM | 6/6 (100%) | 0.002 - 0.159 | UNDER (6-450× lower than Nicoletti) |

**Cause:** AVAL-anchored C_global with linear-TPM scaling produces fractional
total channels per cell in small cells (Phase 4 audit: 5/18 combinations <1
channel/cell, all in AIY or RIM). Nicoletti compensates with HIGH per-cm²
gbar values in small cells to reproduce published whole-cell currents.

**Pattern B: IRK over-channeling in large cells (2/12 beyond-5× cases)**

| cell | channel | ratio | direction |
|---|---|---:|---|
| AVAL | IRK | 8.05 | OVER (8× higher than Nicoletti) |
| AVAR | IRK | 19.1 | OVER (19× higher) |

**Cause:** γ_IRK = 25 pS (Kir2.1 chord conductance at V=-100 mV) is the
largest γ in the inventory. Combined with high IRK paralog-sum TPM (165.6
in AVA), Path 2 derives high IRK density. Nicoletti's per-cm² IRK gbar is
small (8.9e-6 S/cm²), implying Nicoletti's IRK channel has lower per-cm²
density than γ × TPM × C_global predicts.

### 4.2 Cells with successful agreement

| cell | channel | ratio | band |
|---|---|---:|---|
| AVAL | EGL-19 | 1.00 | 2× (reference) |
| AVAR | EGL-19 | 1.62 | 2× |
| AVAR | NCA | 3.01 | 5× |
| AVAR | UNC-103 | 0.37 | 5× |

All large-cell channels EXCEPT IRK. Pattern B applies only to IRK in
large cells.

---

## 5 · Nicoletti uncertainty finding (Phase 5 pre-flight per Rohit's authorization)

Direct query of Nicoletti 2024 paper (PMC10980225) for error bars / sensitivity
analyses on the FRAC-flagged parameters:

| channel-cell | gbar | error bars | fitting procedure |
|---|---:|---|---|
| AIY EGL-19 | 0.1 nS | **none reported** | least-squares optimization |
| AIY KQT-1 | 0.2 nS | **none reported** | least-squares optimization |
| AIY SHL-1 | 0.5 nS | **none reported** | least-squares optimization |
| AIY NCA | 0.06 nS | **none reported** | least-squares optimization |
| RIM CCA-1 | 0.87 nS | **none reported** | hybrid evolutionary + least-squares |
| RIM EGL-19 | 0.1 nS | **none reported** | same as RIM CCA-1 |

**Key methodological finding from Nicoletti 2024:**

> Authors explicitly acknowledge "non-uniqueness of the set of parameters,"
> suggesting parameter space degeneracy without formal characterization.

**Implication for Path 2 interpretation:**

Nicoletti's specific per-cell gbar values are **point estimates from local
optima of under-constrained fits**, not biophysical ground truth with
confidence intervals. Multiple combinations of gbars + kinetic parameters
likely reproduce the same I-V curves. Path 2 derivation produces a
**consistent cross-cell scaling** under a single global constant; Nicoletti's
per-cell fits produce inconsistent cross-cell scaling because each cell
was fit independently.

**This shifts the interpretation of the 75% beyond-5× result:** the
discrepancy isn't necessarily "Path 2 wrong, Nicoletti right." It's
**"Path 2 produces consistent biology-derived parameters; Nicoletti
produces inconsistent cell-fit parameters; reconciling requires choosing
which constraint is more biologically meaningful."**

This is interpretation **(a)** from Rohit's four possible explanations,
empirically supported by direct paper-reading evidence.

---

## 6 · Architectural-direction options for hard-stop resolution

Per methodology §5.2 Tier 3, the architectural decision options are:

### 6.1 Option α — Path 2 v1 ships under reframed validation criteria

**Rationale:** Nicoletti's gbars are degenerate fits without error bars.
"Match Nicoletti's specific gbar values" is the wrong validation criterion.
The right criterion is: do Layer 1 cells using Path 2 derived gbars
**produce stable rest with physiological [Ca]_in and reproduce Nicoletti's
published I-V envelopes** (envelopes, not point-fit values)?

**Action:** Proceed to Phase 6 cell-level validation under the reframed
acceptance criteria. If Phase 6 succeeds, Path 2 v1 ships with documented
methodology contribution: "Substrate redesign produced consistent cross-cell
biology-derived gbars that reproduce Nicoletti's I-V envelopes via different
parameter values than Nicoletti's specific fit; this is a methodologically
cleaner parameterization."

**Risk:** Phase 6 may fail because absolute current magnitudes are 50-450×
off in small cells. Even if I-V envelope shapes are preserved by kinetics,
absolute amplitudes affect rest stability (pump balance, leak balance).

### 6.2 Option β — v2 methodology refinement (per-cell or per-channel-family C_global)

**Rationale:** Pattern A (small-cell systematic) is exactly what the methodology
designed v2 refinement candidates to address. Deploy **per-cell-family C_global**:
- AVA-class C_global (= 1.73e4, current)
- AIY-class C_global (estimated ~5e5; ~30× larger to compensate for small surface)
- RIM-class C_global (estimated ~2e5)

Pattern B (IRK over) addresses separately by refining γ_IRK (current 25 pS
chord; alternative ~10 pS slope or family-average).

**Action:** Write Path 2 v2 spec; deploy refined per-cell-family
C_global. Re-run Phase 5 validation with v2 parameters. If <30% beyond
5× under v2: ship Path 2 v2.

**Risk:** v2 has multiple free parameters (one C_global per cell-family
+ refined γ_IRK = 4 free parameters across 4 cells). With ~16 validated
combinations, the methodology may become parameter-rich enough to fit
Nicoletti's published values trivially, losing the v1 minimal-calibration
methodology contribution. Increases methodology surface area without
clear biological grounding for cell-family-specific C_global differences.

### 6.3 Option γ — Reconsider Path 1 (refit Nicoletti under physiological Nernst)

**Rationale:** If both Path 2 v1 and Path 2 v2 have issues, fallback is
the originally-considered Path 1: refit Nicoletti's gbar values under
physiological E_X (mainly E_Ca = 134 mV vs Nicoletti's 60 mV). Preserves
Nicoletti's per-cell parameterization while fixing the §7.3 finding.

**Action:** Pause Path 2 work; deploy Path 1 (refit) work block.
Estimated 3-5 work blocks separately scoped.

**Risk:** Loses the "biology-derived methodology" contribution. Path 1 ships
a parameter-refit substrate (the original §7.3.5 scope before user authorized
Path 2). The cross-cell consistency benefit of Path 2 is forfeited;
Nicoletti's degenerate fits remain the substrate's parameterization basis.

### 6.4 Option δ — Restructure derivation framework (gbar as per-channel free parameter)

**Rationale:** Per methodology §5.2 Tier 3 listed option. Retain TPM ×
E_translation × surface as cross-cell scaling primitive, but make per-channel
γ a free parameter calibrated against Nicoletti's published I-V curves
(one γ per channel, not per cell). This is a hybrid of Path 1 (refit) and
Path 2 (biology-derived) — gbar's cell-scaling derives from biology, but
the channel's "intrinsic conductance per cell" calibrates against data.

**Action:** Write Path 2 v3 (or Path 3 entirely) spec with this hybrid
approach. Substantial scope; needs deliberate scoping work block.

**Risk:** Increases methodology complexity; less clearly motivated than
either pure Path 2 or pure Path 1.

---

## 7 · Recommendation

**Option α (Path 2 v1 ships under reframed validation) followed by
Option β (v2 refinement) if Phase 6 reveals load-bearing issues.**

Reasoning:
1. The Nicoletti uncertainty finding (no error bars; "non-uniqueness"
   acknowledged) genuinely shifts the validation interpretation. Path 2's
   75% disagreement with Nicoletti's specific point estimates is partially
   a critique of Nicoletti's parameterization uniqueness rather than Path
   2 methodology failure.
2. Phase 6 (per-cell integrated rest stability + voltage-clamp envelope
   match) is the more biologically meaningful validation. If Path 2 v1
   passes Phase 6, the v1 methodology contribution is preserved.
3. If Phase 6 fails, the failure pattern itself informs which v2 refinement
   to deploy (Option β). Skipping straight to v2 without Phase 6 evidence
   risks fitting to a different aspect of Nicoletti's degenerate fits.
4. Option γ (Path 1 fallback) is preserved as backup if both Path 2 v1
   and v2 fail.

**However, this is an architectural-level decision requiring Rohit's
judgment.** Path 2 v1 does not autonomously proceed to Phase 6 per methodology
§5.2. Surface for direction.

---

## 8 · Phase 5 acceptance criteria status

Per methodology / roadmap:

- [x] Every channel in every cell has derived gbar with documented
      comparison to Nicoletti (18 combinations, 1 excluded)
- [x] Substantive findings explicitly captured (75% beyond 5×, two
      structured failure patterns documented)
- [x] §5.2 Tier classification applied (Tier 3 HARD STOP triggered)
- [x] HARD_STOP file written (`scripts/brain/wave2/artifacts/HARD_STOP_path2_phase5.md`)
- [x] derived_channel_parameters.py module shipped as Path 2 v1
      infrastructure (verified by construction at AVAL EGL-19 reference)
- [ ] **Phase 6 BLOCKED** pending architectural-direction decision

**Phase 5 SHIPPED with hard stop.** Path 2 v1 infrastructure complete;
deployment in cell builders awaits architectural direction.

---

## 9 · Files of record

- This document: `docs/path2_channel_validation.md`
- HARD_STOP record: `scripts/brain/wave2/artifacts/HARD_STOP_path2_phase5.md`
- Path 2 v1 derivation module: `scripts/brain/wave2/channels/derived_channel_parameters.py`
- Methodology reference: `docs/channel_parameter_derivation_methodology.md` §4.1 (bands) + §5.2 (tier triggers)
- Progress tracking: `scripts/brain/wave2/artifacts/path2_progress_summary.md`
- Phase 5 checkpoint: `scripts/brain/wave2/artifacts/path2_phase5_checkpoint.json`
- Nicoletti uncertainty source: PMC10980225 (Nicoletti et al. 2024)
