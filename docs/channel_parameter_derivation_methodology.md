# Channel parameter derivation methodology (Path 2)

**Status:** Reference material for the substrate redesign trajectory.
Codifies the gene-expression-derived channel parameterization adopted in
§7.3.5 Path 2.

**Date:** 2026-05-12 (Phase 1 of Path 2)

**Scope of authority:** This document is referenced from:
- `docs/layer1_design_decisions.md` §8 (inherited parameter audit motivating Path 2)
- `docs/substrate_redesign_roadmap.md` §7.3.5 (Path 2 work block scope)
- `scripts/brain/wave2/artifacts/path2_progress_summary.md` (work-block progress tracking)

Path 2 methodology becomes step zero of any "inherit-and-compose" work
block in subsequent layers (§7) per the standing "parameter audit before
integration" cross-cutting track.

---

## 1 · Architectural principle

Channel parameters in the substrate are NOT directly fit to cell-specific
measurements. They derive from four ingredients:

1. **Intrinsic channel properties** — single-channel conductance γ
   (picosiemens), a property of the channel type, ideally measured by
   single-channel patch clamp in C. elegans or homologous heterologous
   expression. γ is a biophysical constant, not a fit parameter.

2. **Cell-specific expression** — mRNA abundance from the CeNGEN
   integrated bulk + single-cell RNA-seq atlas, queried per (gene, cell).
   TPM (transcripts per million) is empirically grounded.

3. **Cell geometry** — surface area `A_cell`, derived from Nicoletti's
   published capacitance via the standard specific-capacitance
   assumption (1 μF/cm² for biological membrane). NOTE: under Path B
   formulation (§2), A_cell does **not** enter the gbar derivation; it
   enters only via the cell builder's membrane equation downstream.

4. **A single global scaling constant** `C_global` — calibrated once
   against a well-characterized reference channel-cell combination
   (EGL-19 in AVAL; AVA voltage anchor from Liu/Chen/Wang 2020 Nat
   Commun 11:5076). C_global is the **only free parameter** in v1; per-
   channel-family or per-cell calibration is deferred to v2 refinement.

**Contrast with the inherited approach.** Nicoletti 2024 fit per-channel-
per-cell gbar values against I-V data measured under fixed E_Ca = 60 mV.
Under physiological [Ca]_in = 50 nM (E_Ca = 134 mV per §6.5), Nicoletti's
gbars over-drive Ca by ~70%, producing runaway. Path 2 replaces fitting
with derivation: 9 channel types × 4 cells = 36 parameters become
[9 γ values + 36 TPMs + 1 C_global] = ~46 parameters of which 36 (TPMs)
are empirically measured, 9 (γ) are channel-type-specific biophysics
(not cell-specific), and 1 (C_global) is calibrated.

**Calibration footprint reduction:** 36 fits → 1 fit. Cross-cell
extrapolation becomes a methodology test rather than a fitting target.

**Validation reframing:** Nicoletti's I-V curves remain as **validation
targets**, not calibration anchors. The methodology is validated by
checking whether derived gbar values reproduce Nicoletti's published
data within tolerance. Failure to match is a substantive finding about
the methodology (refine γ, or E_translation, or aggregation rule), not
a failure of fitting (because there is no fitting to fail at the per-
cell level).

This is the substrate redesign's first major realization of
"physiological Nernst emerges from substrate dynamics; channel
parameters emerge from biology." The pattern is transferable to Layers
2-7 (Wicks 1996 graded release, Nicoletti Ca pool dynamics, peptide
release rate-coupling) — see §7.

---

## 2 · Per-channel derivation formula

Under **Path B (intensive formulation)** — see Phase 1 pushback resolution.

### 2.1 Core formula

```
gbar_intensive[channel][cell] = γ[channel] × density[channel][cell]
density[channel][cell] = TPM[channel][cell] × E_translation[channel] × C_global
```

### 2.2 Unit accounting

| symbol | units | meaning |
|---|---|---|
| γ[channel] | S/channel (= 1e-12 × pS_value) | single-channel conductance, channel-type intrinsic |
| TPM[channel][cell] | dimensionless | CeNGEN-derived mRNA abundance |
| E_translation[channel] | dimensionless | translation efficiency (= 1.0 uniformly in v1) |
| C_global | channels per (cm² · TPM unit) | global scaling constant; one value across the substrate |
| density[channel][cell] | channels/cm² | per-cell intensive channel density at the membrane |
| gbar_intensive[channel][cell] | S/cm² | per-cell intensive membrane conductance |

The cell-builder consumes `gbar_intensive` directly (existing convention).
Surface area A_cell enters via the cell's membrane equation
(`I_total = i_density × A_cell`) but is **not** in the channel
parameterization itself. This is the key Path B simplification: channel
density is the intensive biological property; surface area is a separate
geometric property of the cell.

### 2.3 Multi-gene channel aggregation — default + exception rule

Per Phase 1 pushback resolution (Item 2). The biological reality is that
most "multi-gene channels" are actually paralogous genes each forming
their own functional channel, rather than heteromeric assemblies of
distinct subunit gene products.

**Default rule:** Each gene's TPM contributes to its own channel density.
Paralogs are treated as separate channels, each with its own single-gene
TPM. This is the right interpretation when each paralog forms a
homomultimeric functional channel.

**Exception rule:** Where literature documents heteromeric assembly with
multiple required pore-forming subunits, use **min across pore-forming
subunits**:

```
TPM[heteromer][cell] = min over pore-forming subunit genes of TPM[gene][cell]
```

The biological intuition: heteromeric channels require all pore-forming
subunits for functional assembly. The least-expressed pore-former is
rate-limiting. Min captures this stoichiometric constraint without
requiring per-subunit weights.

**Auxiliary subunits** (regulators that modify kinetics or modulation but
don't gate assembly) are **ignored in density estimation**. They enter
via kinetic parameters where applicable.

### 2.4 Per-family treatment (default; Phase 2 literature scoping refines)

The following channel families are catalogued for Path 2. Each entry
specifies the default rule applied at Phase 5 derivation; Phase 2
literature scoping may refine specific entries to the exception rule.

| family | genes | default rule | notes |
|---|---|---|---|
| EGL-19 | egl-19 | single-gene | L-type Cav1 homolog; pore-forming alone |
| CCA-1 | cca-1 | single-gene | T-type Cav3 homolog |
| UNC-2 | unc-2 | single-gene | P/Q-type Cav2 homolog |
| SHL-1 | shl-1 | single-gene | Kv4 homolog (Shal) |
| SHK-1 | shk-1 | single-gene | Kv1 homolog (Shaker-related) |
| EGL-36 | egl-36 | single-gene | Kv3 family |
| EXP-2 | exp-2 | single-gene | Kv1 family, originally pharyngeal |
| UNC-103 | unc-103 | single-gene | ERG-family delayed-rectifier K |
| EGL-2 | egl-2 | single-gene | EAG-family K |
| NCA channels | nca-1, nca-2 (pore); unc-77, unc-80 (aux) | paralogs separate | NCA-1 and NCA-2 = separate channels with own TPMs; unc-77/unc-80 ignored in density |
| IRK channels | irk-1, irk-2, irk-3 | paralogs separate (default) | Verify in Phase 2 whether C. elegans IRKs form heterotetramers; if so → min-across-pore-forming |
| KQT channels | kqt-1, kqt-2, kqt-3 | paralogs separate (default) | Verify in Phase 2 whether C. elegans KQTs form heterotetramers (mammalian KCNQ family does in some contexts) |
| TWK channels | twk-* family | paralogs separate | Each twk-* forms homodimeric K2P channel |
| SLO channels | slo-1, slo-2 | separate (different channel types) | slo-1 = BK; slo-2 = SK; different Ca/Na sensitivities |

Phase 2 literature scoping output: `docs/channel_gamma_inventory.md`
includes a per-family decision column (heteromer vs paralog-separate)
with citations. Default applies until verified otherwise.

### 2.5 Failure-mode connection

If Phase 5 surfaces channel-level discrepancies > 5× between derived and
Nicoletti's reported gbar, possible causes include:

- γ wrong for that channel (refine γ in Phase 2 followup)
- TPM aggregation rule wrong (heteromer treated as paralog or vice
  versa; refine §2.4 per-family table)
- E_translation per-channel needed (channel-family-specific; v2)
- Hill function E_translation needed (linear TPM scaling inadequate; v2)

Resolution path documented in §5.

---

## 3 · Calibration protocol

### 3.0 v2 reorientation — calibration anchored against measured V_rest

**Updated 2026-05-12 per machine-code-up reorientation (`docs/layer1_design_decisions.md`
§2.9).** Path 2 v2 replaces the v1 calibration anchor (Nicoletti's
derived EGL-19/AVAL gbar value, which is non-unique per §8.6 uniqueness
audit) with **per-cell-family C_global calibrated against measured
V_rest** for each cell class:

```
C_global_AVA-class:  calibrated against measured V_rest (Liu 2020 AVAL/AVAR)
C_global_AIY-class:  calibrated against measured V_rest (published AIY data)
C_global_RIM-class:  calibrated against measured V_rest (published RIM data)
```

This is the **measurement-vs-fit audit** (§8.11) applied to calibration
itself. Targets are measurements (V_rest with experimental uncertainty),
not derived parameters (Nicoletti's gbar fits).

**Calibration procedure per cell family:**

1. With all other parameters fixed (γ from Phase 2 inventory possibly
   with v2 refinements; TPM from Phase 3 inventory; E_translation = 1.0;
   full Layer 1 substrate machinery preserved)
2. For target cell family, sweep C_global value
3. Run cell at rest; measure emergent V_rest
4. Find C_global value at which V_rest lands within published range for
   that cell family
5. Verify rest stability at that C_global (all ions stable, no runaway)
6. Document C_global value per family with measurement source and
   calibration path

If V_rest target is achievable with C_global in plausible range (1e1 to
1e7 channels/cm²/TPM), calibration succeeds for that family. If not,
surface as substantive finding (potential failure of v2; route to v3
refinement or Option γ Path 1 fallback).

**v1 single-anchor approach retained for historical reference but
superseded by v2:**

### 3.1 Reference choice — EGL-19 in AVAL (v1; SUPERSEDED by §3.0)

Pre-authorized Decision 4 (Rohit, 2026-05-12). Justification:

- **EGL-19** is the most-characterized C. elegans voltage-gated Ca
  channel; γ is reasonably constrained by both heterologous expression
  studies and homologous Cav1 (L-type) biophysics.
- **AVAL** is the most-characterized C. elegans neuron for which
  voltage-clamp data exists in the substrate's empirical anchor —
  Liu/Chen/Wang 2020 *Nat Commun* 11:5076 (Nicoletti 2024 reference [29];
  C-45 Direct in `docs/state_of_claims_2026-05-02.md`).
  - Note: Mellem 2008 PMC2697921 was previously cited as the AVA voltage
    anchor but characterizes RMD plateau dynamics, not AVA ("we never
    observed action potentials in AVA"). Corrected per the Wave 2 Mellem
    investigation (`scripts/brain/wave2/artifacts/mellem_investigation_pushback.md`).
- **AVAL's surface area is well-constrained** by Nicoletti's published
  capacitance + standard 1 μF/cm² specific Cm.
- **AVAL EGL-19 TPM is well-above CeNGEN threshold 2** (~89.5 TPM per
  §7.2 v2 CeNGEN pull) → reliable expression measurement.

### 3.2 Calibration math

```
C_global = gbar_Nicoletti_intensive[EGL-19][AVAL] / (γ[EGL-19] × TPM[EGL-19][AVAL] × E_translation[EGL-19])
```

With v1 uniform `E_translation = 1.0`, this reduces to:

```
C_global = gbar_Nicoletti[EGL-19][AVAL] / (γ_EGL19 × TPM_EGL19_AVAL)
```

Units check (Path B):
- `gbar_Nicoletti`: S/cm²
- `γ_EGL19`: S/channel (= γ_pS × 1e-12)
- `TPM_EGL19_AVAL`: dimensionless
- Result: `S/cm² / (S/channel × dimensionless) = channels/cm²` per TPM unit ✓

### 3.3 Order-of-magnitude sanity check

Inputs for verification:
- Nicoletti AVAL EGL-19: `gbar = 9.288e-6 S/cm²` (from `option_alpha_ava_cell.py`)
- AVA EGL-19 TPM (CeNGEN T2): 89.5
- γ_EGL19 reference: Cav1-family heterologous expression typically 4-20 pS;
  v1 default 10 pS (Phase 2 literature scoping refines)

```
C_global ≈ 9.288e-6 / (10e-12 × 89.5 × 1.0)
        ≈ 9.288e-6 / 8.95e-10
        ≈ 1.04e4 channels per (cm² · TPM unit)
```

This is in the biophysically plausible range (10^1 to 10^7 — see §3.4).
Implies AVAL has ~1.04e4 × 89.5 ≈ 9.3e5 EGL-19 channels per cm² of
membrane, which at AVAL's surface area (~1.12e-5 cm²) gives ~10 EGL-19
channels per cell. C. elegans neurons are small; <100 channels per cell
for any given type is biologically reasonable.

### 3.4 Sanity-check-based hard stop (not fixed numerical range)

Per Phase 1 pushback resolution (Item 3). The Phase 4 hard-stop condition
fires only on **biophysically nonsensical** C_global values:

- C_global is negative (sign error somewhere)
- C_global is infinity or NaN (numerical / division-by-zero issue)
- C_global × max TPM × max cell surface area < 1 channel/cell (predicts
  fewer than 1 channel per cell, biologically implausible for an
  expressed channel)
- C_global × min TPM × min cell surface area > 1e7 channels/cm² (exceeds
  realistic membrane channel density of ~10^7 channels/cm² assuming ~10
  nm² per channel footprint)

The order-of-magnitude expected range is ~1e1 to 1e7 channels/(cm²·TPM).
The hard stop is binary (nonsensical/sensible), not a numerical band.

### 3.5 Reference verification

By construction, derived `gbar[EGL-19][AVAL]` = Nicoletti's published
value. Phase 4 confirms this in code with explicit assertion.

---

## 4 · Validation hierarchy

### 4.0 v2 reorientation — four-tier validation under machine-code-up

**Updated 2026-05-12.** Path 2 v2 expands validation to four tiers per
`docs/layer1_design_decisions.md` §8.12 reorientation summary. Cell-level
behavior is **emergent consequence** of biophysics, not a calibration
target. Validation occurs at multiple scales of biological organization:

**Tier A — First-principles consistency**
- Mass conservation across simulation (ion totals conserved modulo
  membrane current integrals)
- Nernst self-consistency (E_X computed from current concentrations
  matches observed equilibrium behavior)
- Ion concentrations physiological at rest:
  - [K]_in: 100-140 mM
  - [Na]_in: 5-15 mM
  - [Cl]_in: 3-10 mM
  - [Ca]_in: 50-200 nM
- No runaway dynamics over 5s rest simulation

**Tier B — Cell-level measurements (consume measurements, not fits)**
- V_rest within published range per cell class (target ±5 mV)
  - AVA-class: published V_rest from Liu 2020 + Mellem 2008 envelope
  - AIY-class: published V_rest from Nicoletti 2024 underlying
    measurements
  - RIM-class: published V_rest from Nicoletti 2024 underlying
    measurements
- Recovery from current injection on biological timescales
- Ion homeostasis maintained under perturbation

**Tier C — Phenotype categories (qualitative emergence)**
- AVA-class: plateau capability under sufficient depolarizing current
  (sustained depolarization that doesn't immediately repolarize)
- AIY-class: graded response without plateau (proportional to input
  current)
- RIM-class: intermediate dynamics (some plateau capability but less
  pronounced than AVA)

**Tier D — Cross-cell consistency (biological differentiation)**
- Cells with similar gene expression (AVAL/AVAR) show similar behavior
- Cells with different gene expression (AVA-class vs AIY-class) show
  expected biological distinctions
- Phenotype distinctions emerge from differential expression, not from
  per-cell tuning

**Acceptance criteria under v2:**

v2 passes if all four cells satisfy Tier A + Tier B AND at least 2 of 4
cells satisfy Tier C AND Tier D shows expected cell-class distinctions.
Tier C partial pass acceptable because plateau capability depends on
kinetic parameters which are still inherited from Nicoletti (kinetic
audit deferred to Layer 1.5 v3 or Layer 2 work block).

v2 fails if Tier A or Tier B fails for any cell. Failure pattern
informs next refinement direction.

**v1 hierarchy retained for historical reference but superseded:**

### 4.1 Channel-level (v1; SUPERSEDED by §4.0 v2)

For each (channel, cell) where Nicoletti measured published gbar:

- Compute derived gbar = γ × density × E_translation × C_global
- Compare to Nicoletti's reported value
- Categorize:
  - **Within 2×** (good): derived methodology reproduces Nicoletti
  - **Within 5×** (acceptable): methodology approximates but not exact;
    document residual without retuning
  - **Beyond 5×** (substantive finding): investigate per §5

#### 4.1.1 Justification for 2×/5× bands — combined uncertainty grounding

The 2× / 5× bands are pragmatic thresholds reflecting combined uncertainty
from two distinct sources:

- **Single-channel γ measurement variance** (~1.5-2× spread). Literature
  γ values typically range 1.5-2× across studies for the same channel
  type, reflecting differences in expression system, ionic composition,
  temperature, and recording method. This is the lower bound on derived-
  gbar uncertainty — even with perfect TPM scaling, γ alone introduces
  this much spread.
- **Linear-TPM-scaling residual** (~3.6×). Equation-validation overnight
  work surfaced a residual of this magnitude when TPM is used as a
  linear proxy for channel density across cells. This reflects regulation
  beyond transcription: translation efficiency variation, post-
  translational modification, trafficking and membrane targeting, and
  channel-protein turnover rates differ across channel families.

Combined: ~1.5-2× (γ variance) × ~3.6× (TPM-scaling residual) ≈ ~5× total
acceptable residual under v1 methodology.

Interpretation:

- **Within 2×** — derived methodology reproduces Nicoletti precisely; both
  uncertainty sources happen to align well for that channel
- **Within 5×** — approximate match within combined uncertainties; residual
  consistent with v1 methodology limits; document as substrate uncertainty
  quantification without retuning
- **Beyond 5×** — residual exceeds combined-uncertainty envelope;
  methodology assumptions (γ, aggregation rule, E_translation, or
  C_global) likely incorrect for that channel; investigate per §5

These thresholds are **uncertainty-grounded**, not arbitrary cutoffs.
v2 methodology refinement (Hill E_translation, per-channel-family
C_global, per-channel-family E_translation) targets the 3.6× TPM-scaling
component specifically.

For (channel, cell) where Nicoletti's published I-V data exists:

- Run voltage-clamp simulation with derived channel
- Compare simulated I-V curve to digitized Nicoletti data
- Acceptance: within Nicoletti's reported SEM at tested voltages, OR
  within 10% L2 error against digitized curve

### 4.2 Cell-level (Phase 6)

For each cell (AVAL, AVAR, AIY, RIM), with derived parameters substituted
into the Wave 2 cell builder:

- **Rest stability:** [K]_in stable within ±2% over 5s; [Na]_in same;
  [Cl]_in in physiological [3, 7] mM; [Ca]_in returns near target
  (50-200 nM) after perturbation
- **V_rest:** within published range per cell (Liu 2020 for AVA-class
  cells; Nicoletti 2024 envelopes for AIY/RIM)
- **Voltage-clamp response:** phenotype categories preserved (plateau /
  graded / spiking); rise/decay timescales within published ranges

### 4.3 Cross-cell consistency (Phase 6)

Independent of Nicoletti's specific cell-level fits, the methodology
should produce biologically consistent patterns:

- Cells with higher EGL-19 TPM show proportionally larger Ca currents
- Cells with lower TWK-family TPM show proportionally weaker leak
- Pumps scale consistently (§7.2 v2 TPM scaling) — combined with derived
  channels, the system reaches stable rest across cell sizes

Cross-cell consistency is a **methodology check** that surfaces TPM-density
non-linearity (if any) directly, independent of Nicoletti's parameterization.

---

## 5 · Failure modes and what they mean

### 5.1 Channel-level discrepancy > 5× (Phase 5)

Possible causes, in order of investigation:

1. **γ wrong** for that channel — refine γ via deeper literature scoping
   (different homolog, different conditions). Phase 2 followup; document
   in `path2_channel_findings.md`.
2. **Multi-gene aggregation rule wrong** — channel treated as paralog-
   separate but actually heteromeric, or vice versa. Refine §2.4 per-
   family table; document the basis (literature evidence) for the
   correction.
3. **E_translation per-channel needed** — gene-to-protein translation
   efficiency varies systematically by channel family. Surface as v2
   refinement candidate.
4. **Hill function E_translation needed** — TPM-to-density scaling is
   non-linear. Surface as v2 refinement candidate.

### 5.2 Cross-channel methodology adequacy — three-tier triggers

After Phase 5 derivation completes, tally the percentage of (channel, cell)
combinations falling into each §4.1 band. Apply per the following tiers
(this makes pre-Phase-1 Decision 5 explicit rather than requiring in-flight
Phase 5 judgment):

#### Tier 1: <30% of (channel, cell) combinations beyond 5×

**Methodology working adequately.** Combined-uncertainty bounds (§4.1.1)
suggest most channels should land within 5×. <30% beyond is consistent
with normal variance across channel families. Action:

- Document outliers as substrate uncertainty quantification per §2.8
  epistemic labeling (the affected channels carry "biophysically derived
  with documented residual" label going forward)
- Proceed to Phase 6 cell integration with documented limitations
- Outliers may inform v2 refinement candidates but don't block Path 2 v1
  shipping

#### Tier 2: 30-50% of combinations beyond 5×

**Methodology refinement required.** Systematic residual beyond what
combined uncertainty bounds alone explain; v1 simplifications (uniform
E_translation, single global C_global, linear TPM scaling) are
inadequate. Action:

- **Pause and surface diagnostic table** to Rohit with per-channel
  residuals + suspected causes
- Decide which v2 refinement(s) to deploy:
  - Hill function for E_translation (addresses linear-TPM-scaling
    inadequacy)
  - Per-channel-family C_global (addresses uniform-global hypothesis
    violation)
  - Per-channel-family E_translation (addresses systematic translation-
    efficiency variation by family)
- v2 deployment is a separate work block; Path 2 v1 documented as
  foundational but incomplete; proceed to v2 work block before declaring
  §7.3.5 complete

#### Tier 3: >50% of combinations beyond 5× — HARD STOP

**Methodology fundamentally inadequate at the formula level**, not just
parameter refinement. The γ × TPM × E_translation × C_global framework
itself fails for the majority of channels — refinements within this
framework won't recover acceptable validation. Action:

- Write `HARD_STOP.txt` in artifacts with diagnostic data
- Surface for direction; possible outcomes:
  - Return to higher-level architectural discussion about substrate
    parameterization methodology
  - Reconsider Path 1 (refit Nicoletti under physiological Nernst) as
    fallback for v1 Layer 1
  - Restructure derivation framework (e.g., per-channel γ becomes a
    free parameter calibrated against published I-V curves, retaining
    only TPM × E_translation × surface-area as the cross-cell scaling)
- Path 2 v1 does not ship until architectural-level decision lands

### 5.3 Other failure modes (independent of tier triggers)

Channel-level discrepancy > 5× for individual channels (within Tier 1
overall):

### 5.4 Cell-level failure despite acceptable channel-level (Phase 6)

Indicates **kinetic parameter inheritance** also has implicit assumptions
(not just gbar). Possible causes:

- Channel kinetics (V_half, slopes, time constants) were fit under
  Nicoletti's E_Ca = 60 mV; under physiological E_Ca they may give
  different equilibrium voltages
- Pumps may need re-calibration under the new channel current profiles
  (substrate is more excitable than under Nicoletti's parameterization,
  or less)

Surface as substantive finding. Path 2 v1 ships channel gbar derivation;
kinetic-parameter audit defers to Layer 2 or Path 2 v2.

### 5.5 Heteromeric assumption wrong

If a channel was treated as paralog-separate but is actually heteromeric:
gbar values may be 2-4× too high (because separate paralogs were summing
into the I-V where only one heteromer exists). Surface in §4.1 channel-
level validation; refine §2.4 per-family table.

If treated as heteromer but actually paralogs: gbar values may be too low
(min-across-pore-forming under-counts). Same surfacing mechanism.

### 5.6 C_global mis-calibrated

If reference choice (EGL-19 in AVAL) doesn't represent the channel
population well, C_global is biased. Phase 4 surfaces this via biophysical
sanity check. Refinement: test alternative reference (e.g., KQT-1 in RIM,
or IRK-1 in AIY). Sensitivity analysis on reference choice is v2.

---

## 6 · Epistemic labeling (per `docs/layer1_design_decisions.md` §2.8)

Path 2 channel parameters fall into specific §2.8 categories:

| parameter | label | rationale |
|---|---|---|
| γ values | per-channel labels: empirically grounded / biophysically derived / approximation from adjacent biology | Phase 2 inventory documents per-channel source. Likely heterogeneous: some channels measured in C. elegans directly; most via heterologous expression of the C. elegans gene; rare cases via mammalian/Drosophila homolog fallback |
| TPM values | empirically grounded | CeNGEN integrated bulk + scRNA-seq; threshold 2 default per §7.2 v2 |
| E_translation | free parameter with sensitivity sweep (initially uniform = 1.0) | v1 default. v2 refinement candidate if Phase 5 surfaces systematic per-family residuals |
| A_cell | biophysically derived (from capacitance × 1 μF/cm⁻¹) | Note: under Path B, A_cell does **not** enter gbar derivation; it appears only via cell builder membrane equation |
| C_global | biophysically derived (calibrated against single reference) | One free parameter, calibrated against one reference channel-cell pair, sanity-checked against biophysical plausibility |
| Multi-gene aggregation rule (default paralog-separate; exception min-across-pore-forming) | biophysically derived (under literature-documented heteromeric vs paralogous status) | Per-family table in §2.4 documents which rule applies and why |

Channel **kinetic** parameters (V_half, slopes, time constants) inherited
from Nicoletti retain the §2.8 label "biophysically derived under
assumptions inconsistent with substrate" from `docs/layer1_design_decisions.md`
§8 until separately audited. Path 2 v1 does NOT refit kinetics; only
gbar derivation falls in scope. Kinetic audit is Layer 2 or v2.

### 6.0.1 New §2.8 category: "biophysically derived under non-unique parameter fits"

**Phase 5 finding (2026-05-12) added a second epistemic label category**
distinguishing parameter fits that may be one of multiple valid solutions
in degenerate parameter space:

| §2.8 label | meaning |
|---|---|
| empirically grounded | directly measured for these cells |
| biophysically derived | calibrated against well-defined biophysical constraint |
| **biophysically derived under non-unique parameter fits** *(NEW)* | derived from fitting to data but the fit problem is under-constrained; multiple parameter combinations produce equivalent fit quality; specific point estimates lack uncertainty quantification |
| biophysically derived under assumptions inconsistent with substrate | inherited fits whose implicit state assumptions don't match current substrate (§7.3 finding) |
| approximation from adjacent biology | mammalian/different-cell C. elegans values applied with documented limitation |
| free parameter with sensitivity sweep | bounded by validation invariance |

**Channels inherited from Nicoletti 2024 fall into BOTH the "inconsistent
substrate" category (§7.3 E_Ca finding) AND the "non-unique parameter
fits" category (§7.3.5 Phase 5 finding).** The uniqueness audit is a
distinct check from the state-variable audit:
- **State-variable audit** (§7.3): does the fit's implicit ionic state
  match the current substrate?
- **Uniqueness audit** (§7.3.5 Phase 5): does the fit have error bars,
  sensitivity analyses, or other evidence that the specific values are
  uniquely determined by the data?

Inherited fits failing EITHER audit weaken the case for using their
specific values as validation anchors. Both audits should run before
treating inherited parameters as ground truth.

### 6.1 Field-state observation from Phase 2 (2026-05-12)

Phase 2 γ literature scoping surfaced an empirical observation that informs
all subsequent phases:

**Direct C. elegans single-channel patch-clamp γ measurements do NOT exist
for any of the 9 Wave 2 channels in scope.** All Path 2 γ values derive
from mammalian homolog fallback (Cav1.2, Cav3.1, Cav2.1, Kir2.1, KCNQ
family, Kv4.2, Kv10.1, hERG) with documented cross-species transferability
assumptions. NCA has no published unitary γ in any species (literature gap;
v1 uses estimated placeholder).

**This is field state, not methodology limitation.** Single-channel
recordings in C. elegans neurons are technically rare due to small cell
size + low channel density + difficulty of patch-clamp in this organism.
The field's biophysical modeling depends heavily on mammalian homolog
transferability — including Nicoletti's macroscopic fits.

**Combined uncertainty budget validation:** Per-channel γ uncertainty
from mammalian homolog fallback (~1.5-3× measurement variance × cross-
species transferability) combined with linear-TPM-scaling residual
(~3.6× from §4.1.1) produces total combined uncertainty of **~5-10× per
channel**. This empirically validates the Phase 5 decision thresholds:

- **Within 2× band (good):** empirically rare under v1 methodology; only
  expected when both homolog γ + TPM-density assumptions happen to align
  precisely for that channel
- **Within 5× band (acceptable):** realistic central performance
  expectation under v1; the "matches Nicoletti within combined uncertainty"
  zone
- **Beyond 5× band (substantive finding):** indicates methodology
  refinement candidate (γ refinement, heteromer aggregation rule, Hill
  E_translation, per-family C_global)

Phase 5's `>30% beyond 5× → methodology refinement` tier 2 trigger
(see §5.2) was set assuming ~5× combined uncertainty. Phase 2's coverage
outcome (8/9 mammalian homolog + 1 estimated) confirms this assumption.
The tiered triggers remain calibrated.

**Implication for downstream layers:** Layer 2-7's inherited parameter
audit (§7) will encounter the same field-state limitation. Most C. elegans
biophysical parameters lack direct measurement; methodology must accept
mammalian homolog transferability as standing assumption while documenting
per-parameter uncertainty explicitly. This is not unique to Path 2;
it's the rigor mode for any C. elegans substrate redesign.

---

## 7 · Forward-looking application to Layers 2-7

### 7.0 What transfers vs what requires methodology extension

Path 2 establishes two distinct things that get conflated if not
distinguished:

1. **The methodology PATTERN** — transferable to any inherited
   parameter set:
   - Audit inherited fits → identify implicit state-variable assumptions
   - Check consistency with current substrate state
   - If inconsistent: derive from biology + minimal calibration
   - Validate against original published data as TARGETS, not anchors
   - Surface discrepancies as substantive findings rather than retuning

2. **The specific FORMULA** (`γ × TPM × E_translation × C_global`) — **gbar-
   specific**. This math derives membrane conductance density from
   per-channel single-channel conductance and per-cell mRNA abundance.

   The formula transfers directly **only to other conductance-density
   parameters**. Channel kinetics, receptor binding kinetics, release rate
   constants, and other parameter types require **different derivation math
   even when applying the same pattern**.

The forward-looking applications below note which case each falls into.

### 7.1 Where the formula transfers directly (PATTERN + FORMULA)

Future inherited parameter sets that are themselves gbar/conductance-
density parameters take Path 2's formula with only the γ-equivalent and
TPM source changing. Anticipated examples:

- **Layer 3 explicit GABA-A / GluCl receptor conductance densities** —
  if Layer 3 instantiates these as explicit channels (rather than just
  modulated via Phase G hooks), their gbar derives from γ_receptor ×
  TPM_receptor-gene × E_translation × C_global with the same formula
  structure. Reference channel/cell may need updating; calibration
  protocol is otherwise identical.
- **Layer 5+ gap junction conductances** — gap junction density derives
  from γ_innexin × TPM_innexin-gene × E_translation × C_global, with γ
  being the unitary innexin channel conductance. Same formula structure.

For these, Path 2's full methodology applies and the formula transfers
directly.

### 7.2 Where the pattern transfers but the formula doesn't (PATTERN only)

The following applications inherit Path 2's PATTERN (audit + derive-not-
inherit + validate-against-targets) but require **methodology extension** —
new derivation math specific to the parameter type.

#### 7.2a Layer 2 — Channel kinetic parameter audit (kinetic, not gbar)

Channel V_half, k, time constants inherited from Nicoletti. Each fit
assumes a specific E_X for the ion the channel carries. Under physiological
E_X (Layer 1 substrate), kinetic parameters may need re-fitting.

**Pattern applies:** audit Nicoletti's kinetic fits for implicit ionic
state assumptions; identify discrepancies; derive new kinetics or refit.

**Formula does NOT apply:** kinetic parameters don't decompose into γ ×
TPM × E_translation × C_global. They derive from underlying gating
biophysics — state-transition rates between channel conformations, energy
barriers, voltage-dependence of activation/inactivation gates.

**Methodology extension TBD:** Layer 2 work block requires deliberate
scoping of how to derive (or refit) kinetic parameters under physiological
state. Two candidate approaches: (a) refit kinetics against Nicoletti's
published I-V curves under physiological E_X (Path-1-like for kinetics);
(b) derive from explicit gating models for each channel family
(channel-biophysics-from-first-principles, much harder).

#### 7.2b Layer 3 — Wicks 1996 graded release Boltzmann parameters (kinetic-like, not gbar)

Wicks 1996 derived σ-V_half Boltzmann parameters for graded release from
Ascaris ventral cord recordings. The parameters describe the σ-shaped
release function: presynaptic V_pre → release probability. **σ and V_half
are kinetic-like parameters, not gbar-like.**

**Pattern applies:** audit Wicks' fit for implicit ionic state assumptions
(intracellular Cl, Mg, ATP, resting V under Ascaris saline); check
consistency with Layer 1 substrate state; if inconsistent, derive new
σ-V_half or refit under physiological state.

**Formula does NOT apply:** σ and V_half don't derive from γ × TPM ×
E_translation × C_global. They describe the voltage-dependence of release
machinery, not the density of channels at the membrane. Derivation would
come from underlying release biophysics — vesicle pool dynamics, calcium
sensor (synaptotagmin) Boltzmann response, presynaptic-Ca-channel-to-
vesicle coupling distance.

**Methodology extension TBD:** Layer 3+ work block requires scoping of
how to derive (or refit) σ-V_half under physiological state. This is **not**
a direct Path 2 application; setting expectations correctly downstream
prevents Layer 3 from inheriting Path 2's gbar formula inappropriately.

#### 7.2c Layer 4 — Nicoletti Ca pool dynamics (rate-constant-like, not gbar)

Nicoletti's calcium pool model (cadiff, caintra1) assumes specific buffer
kinetics + ER coupling. Parameters are rate constants for buffer binding/
unbinding + ER Ca uptake/release, not conductance densities.

**Pattern applies:** audit Nicoletti's Ca pool fit for implicit state
assumptions ([Ca]_in operating range, buffer concentrations, ER membrane
state); check consistency with Layer 1+4 substrate; if inconsistent,
derive from explicit buffer biology.

**Formula partially applies:** if Layer 4 derives buffer **densities** from
calbindin / calmodulin TPMs (gbar-like), Path 2 formula applies for the
density. But buffer **kinetics** (binding rates, dissociation rates) are
kinetic-like and require methodology extension as in 7.2a.

#### 7.2d Layer 5+ — Peptide release rate-coupling (rate-constant-like, not gbar)

Peptide release rate constants fit under specific calcium-imaging
conditions. Parameters are: peptide-vesicle-pool size, Ca-triggered
release rate, peptide-diffusion-field decay.

**Pattern applies:** audit for implicit Ca operating-range assumptions;
derive under Layer 1 substrate.

**Formula does NOT apply:** these are kinetic rates, not conductance
densities.

### 7.3 Standing methodology step — TWO audits required

For every inherited parameter set, before composing into substrate,
run **both audits**:

#### Audit A — State-variable audit (introduced §7.3 finding)

1. **What state variables does the fit implicitly assume?** Reverse-
   engineer from the original paper's methods section, control conditions,
   and reference saline composition.
2. **Are those assumptions consistent with the current substrate state?**
   Compare to Layer 1+ ionic concentrations, [ATP], Mg, pH, temperature.
3. **If inconsistent: replace or refit** under physiological state.

#### Audit B — Uniqueness audit (introduced §7.3.5 Phase 5 finding, NEW)

1. **Does the original paper report error bars / confidence intervals /
   sensitivity analyses for the fitted parameters?**
2. **Does the paper acknowledge parameter non-uniqueness or local-optimum
   issues?** (e.g., "non-uniqueness of the set of parameters")
3. **Are the fitted values point estimates from optimization without
   uncertainty quantification?**
4. **If yes to (3): the inherited values are weak validation anchors.**
   Treat as one of multiple valid solutions in degenerate parameter
   space; reframe validation against underlying measured data with SEM
   (where available) rather than against fitted point estimates.

#### Decision tree

5. **Determine parameter type:** gbar-like (conductance density) or
   kinetic-like (rates, voltage-dependence parameters, binding constants).
6. **If gbar-like AND inconsistent state OR non-unique:** apply Path 2
   formula directly with appropriate γ-equivalent + TPM source.
7. **If kinetic-like AND inconsistent state OR non-unique:** apply Path 2
   PATTERN with parameter-type-specific derivation math; treat as
   methodology extension requiring its own work block.
8. **If consistent state AND fits are well-constrained:** inherit with
   documented consistency check on both axes.
9. **Document both audit outcomes** in the layer's design decisions doc
   + `docs/substrate_redesign_roadmap.md` cross-cutting tracks.

Both audits combined are "parameter audit before integration" as a
standing methodology step — see `docs/substrate_redesign_roadmap.md`
cross-cutting tracks + `docs/layer1_design_decisions.md` §2.8 + §8.6.

#### Specific layer applications under both audits

- **Layer 1 §7.3.5 (Nicoletti channels):** Audit A failed (E_Ca=60
  inconsistent with physiological E_Ca=134); Audit B failed (no error
  bars; non-uniqueness acknowledged). Path 2 derivation deployed; v1
  validation reframed against measured I-V envelopes rather than
  Nicoletti's degenerate point estimates.
- **Layer 3+ Wicks 1996 graded release:** Audit A pending (need to verify
  Ascaris ionic state assumptions); Audit B pending (need to check
  Wicks paper for parameter uncertainty quantification). **Both audits
  must run before any WB3-equivalent reuse.**
- **Layer 4 Nicoletti Ca pool dynamics:** Audit A pending; Audit B
  pending. Same dual-audit requirement.
- **Layer 5+ peptide release rate-coupling:** Same dual-audit requirement.

The dual audit makes "parameter audit before integration" load-bearing
across multiple dimensions, not just state-variable consistency.

---

## 8 · Status

Phase 1 (this document) is the foundational deliverable. Implementation
of Path 2 derivation proceeds in Phases 2-7 per the §7.3.5 entry in
`docs/substrate_redesign_roadmap.md`.

Phase 1 acceptance criteria (per §7.3.5 Path 2 spec):
- ✓ Methodology document complete with all 7 sections
- ✓ Roadmap §7.3.5 entry updated to Path 2 scope (separate edit)
- ✓ Document is peer-readable and reference-quality
- ✓ No implementation code yet (deferred to Phases 2-6)

Phase 1 pre-flight pushback resolved 2026-05-12 (see
`docs/channel_parameter_derivation_methodology_pushback.md` for the
four items and their resolutions). All four resolutions are baked into
this document:

- **Item 1** (units): Path B intensive formulation; A not in gbar formula
- **Item 2** (multi-gene): default paralog-separate; exception min-across-
  pore-forming; per-family table in §2.4
- **Item 3** (C_global range): sanity-check-based hard stop per §3.4
- **Item 4** (AVA anchor): Liu 2020 not Mellem 2008 per §3.1

Next: Phase 2 (single-channel γ literature scoping) — see `docs/substrate_redesign_roadmap.md`
§7.3.5 entry for work block sequencing.
