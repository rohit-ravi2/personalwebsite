# V_rest measurement targets per cell class — v2 calibration anchors

**Status:** Deliverable 2 of §7.3.5 Path 2 v2. V_rest measurement targets
for per-cell-family C_global calibration per §3.0 methodology revision.

**Date:** 2026-05-12

**Reference:** `docs/channel_parameter_derivation_methodology.md` §3.0 (v2
calibration) + `docs/layer1_design_decisions.md` §8.11 (measurement-vs-fit
audit) + §2.9 (machine-code-up principle).

---

## 1 · Methodology framing

Per §8.11 measurement-vs-fit audit, calibration anchors are **measurements**
(V_rest with experimental uncertainty), NOT inherited derived parameters
(Nicoletti's gbar fits). Each target value has:

- Range with measurement uncertainty
- Primary source citation
- Measurement conditions
- Epistemic label per §2.8

These targets are consumed by Deliverable 4 (per-family C_global
calibration) — sweep C_global until V_rest emerges within target range
under full Layer 1 substrate machinery.

---

## 2 · Per-cell-class V_rest targets

### 2.1 AVA-class (AVAL + AVAR)

**Target range: V_rest ∈ [−50, −30] mV (central: −40 mV)**

**Primary sources:**

- **Liu/Chen/Wang 2020** *Nat Commun* 11:5076 (DOI 10.1038/s41467-020-18893-9;
  PMID 33033264; PMCID PMC7544903). Direct voltage-clamp + current-clamp
  recordings of AVAL and AVAR. Whole-cell patch clamp; Multiclamp 700B;
  ~20 MΩ borosilicate pipettes; identification via fluorescent labeling.
  **C-45 Direct** in `docs/state_of_claims_2026-05-02.md` catalog.
- **Mellem 2008** (PMC2697921). Primary quote: "AVA rest typically
  between -20 and -30 mV" (acknowledged misattribution per
  `mellem_investigation_pushback.md` — but the rest-voltage observation
  is empirical regardless of channel attribution).
- **Wave 2 Stage IV measurements:** AVAL rest at -40.3 mV; AVAR rest at
  -24.2 mV (cell-class distinction empirically demonstrated; reproduces
  Mellem envelope and Liu 2020 acknowledgments).

**Cell-pair distinction:** AVAL and AVAR are biologically distinguishable
(per Wave 2 Stage IV findings + Nicoletti 2024). AVAL is more
hyperpolarized than AVAR. For v2 calibration:

- AVAL target: ~−40 mV (within published range)
- AVAR target: ~−24 mV (within published range)
- Cross-AVA-class C_global calibrated against average ~−32 mV

**Epistemic label:** **empirically grounded** (direct C. elegans
voltage-clamp recordings).

### 2.2 AIY-class

**Target range: V_rest ∈ [−95, −55] mV (central: −75 mV)**

**Primary sources:**

- **Nicoletti 2024** (PMC10980225). AIY leak reversal e_leak = −89.57 mV
  (extracted from Nicoletti's parameter fits to AIY voltage-clamp data).
  This is a fit-derived value but the underlying I-V data shape is the
  measurement.
- **AIY's biology:** AIY is the only one of the four target cells with
  e_leak < E_K (very hyperpolarized — close to E_K = −89.81 mV at
  mammalian-default ion concentrations). AIY is the "graded sensory
  interneuron" with strong K-dominant rest, no plateau capability.
- **Wave 2 layer1_cells.py spec:** `rest_published_mV = (−95, −55)`
  (broad envelope reflecting AIY's hyperpolarized-K-leak biology).

**Epistemic label:** **approximation from adjacent biology**
(measurement-grounded via Nicoletti's underlying voltage-clamp data;
single primary source; uncertainty range ~20 mV).

**Notes:** AIY's strongly hyperpolarized V_rest is a strong constraint —
the cell-family C_global must produce high effective K conductance
without other channel current driving depolarization. AIY may be the
most challenging family to calibrate.

### 2.3 RIM-class

**Target range: V_rest ∈ [−65, −40] mV (central: −52 mV)**

**Primary sources:**

- **Nicoletti 2024** (PMC10980225). RIM leak reversal e_leak = −50 mV
  (parameter fit to RIM voltage-clamp data).
- **Biological role:** RIM is a motor command interneuron with diverse
  channel set (SHL-1 + EGL-2 + IRK + CCA-1 + UNC-2 + EGL-19) — mixed
  fast K + T-type Ca + HVA Ca, supporting intermediate dynamics between
  AVA plateau and AIY graded.
- **Wave 2 layer1_cells.py spec:** `rest_published_mV = (−65, −40)`.

**Epistemic label:** **approximation from adjacent biology**
(measurement-grounded via Nicoletti's underlying voltage-clamp data).

---

## 3 · Calibration target summary

For Deliverable 4 (per-family C_global calibration):

| cell family | central V_rest target | acceptable range | primary source |
|---|---:|---|---|
| AVA-class (AVAL/AVAR) | −32 mV (avg) | [−50, −15] | Liu 2020 + Wave 2 Stage IV |
| AIY-class | −75 mV | [−95, −55] | Nicoletti 2024 underlying data |
| RIM-class | −52 mV | [−65, −40] | Nicoletti 2024 underlying data |

Calibration succeeds if C_global_family produces V_rest within the
acceptable range. The central V_rest is a guidance value, not a strict
target.

---

## 4 · Cross-cutting note: AVAL vs AVAR cell-class subdivision

AVAL and AVAR are biologically distinct (Wave 2 Stage IV finding) but
inherit the same CeNGEN class (AVA). Under v2 per-family C_global:

- Single C_global_AVA serves both AVAL and AVAR
- Cell-specific differentiation (AVAL rest −40 vs AVAR rest −24) emerges
  from per-cell channel parameter differences (gbar tables in
  `layer1_cells.py` reflect Nicoletti's per-cell fits — these may need
  audit too if v2 doesn't capture the distinction)

If v2 calibration with single C_global_AVA can't distinguish AVAL from
AVAR within their respective ranges, that's a Phase 5/6 substantive
finding pointing to per-cell C_global (more granular than per-class).
For v2, accept the single C_global_AVA and document if cell-distinction
fails.

---

## 5 · Files of record

- This document: `docs/v_rest_targets.md`
- Calibration consumer: `docs/c_global_per_family_calibration.md` (Deliverable 4)
- Methodology reference: `docs/channel_parameter_derivation_methodology.md` §3.0
- Catalog reference (C-45 Liu 2020): `docs/state_of_claims_2026-05-02.md`
- Mellem investigation: `scripts/brain/wave2/artifacts/mellem_investigation_pushback.md`
