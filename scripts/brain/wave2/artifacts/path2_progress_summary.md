# §7.3.5 Path 2 — progress summary (live tracking)

**Purpose:** Master tracking file across all 7 phases of Path 2 work block.
Updated after every milestone for cross-session resumability.

**Started:** 2026-05-12
**Last updated:** 2026-05-12 (Phase 4 SHIPPED — C_global = 1.7297e4
channels/(cm²·TPM) calibrated from EGL-19/AVAL; reference verified by
construction; biophysical plausibility checks pass; per-(channel,cell)
audit surfaces 5 substantive findings: 4 AIY channels fractional + RIM
CCA-1 fractional; 27.8% combinations already beyond plausible 1-channel
floor pre-validation, approaching §5.2 Tier 2 boundary)

---

## Status by phase

| phase | name | status | deliverable | date |
|---|---|---|---|---|
| 1 | Methodology document | ✓ SHIPPED | `docs/channel_parameter_derivation_methodology.md` | 2026-05-12 |
| 2 | γ literature scoping | ✓ SHIPPED | `docs/channel_gamma_inventory.md` | 2026-05-12 |
| 3 | CeNGEN TPM extension | ✓ SHIPPED | `docs/channel_tpm_inventory.md` | 2026-05-12 |
| 3.5 | SHL-1 + nca-1 disambiguation | ✓ SHIPPED | progress summary addendum | 2026-05-12 |
| 4 | `C_global` calibration | ✓ SHIPPED | `docs/channel_calibration_protocol.md` | 2026-05-12 |
| 5 | Derivation + per-channel validation | pending | `scripts/brain/wave2/channels/derived_channel_parameters.py` + `path2_channel_validation.md` | — |
| 6 | Per-cell integration | pending | updated cell builders + `path2_cell_validation.md` | — |
| 7 | Documentation + commit + push | pending | design doc §8.6/§8.7, roadmap update, four commit groups | — |

---

## Phase 1 outcomes

**Pushback file:** `docs/channel_parameter_derivation_methodology_pushback.md`
surfaced 4 items before methodology doc write. All four resolved
(2026-05-12). Resolutions baked into methodology doc:

- **Item 1 — Unit pipeline:** Path B intensive formulation (no A in
  gbar formula). Cell builders consume `gbar_intensive` directly.
- **Item 2 — Multi-gene aggregation:** default paralog-separate; exception
  min-across-pore-forming for documented heteromers; auxiliary subunits
  ignored. Per-family table in methodology §2.4.
- **Item 3 — `C_global` plausibility:** sanity-check-based hard stop
  (negative, infinity, <1 channel/cell after scaling, >10^7 channels/cm²
  membrane saturation). Not a fixed numerical range.
- **Item 4 — AVA voltage anchor:** Liu/Chen/Wang 2020 *Nat Commun* 11:5076
  (C-45 Direct), NOT Mellem 2008 (which characterizes RMD).

**Pre-authorized decisions (Rohit, 2026-05-12) carry through all phases:**
1. Single global `C_global` for v1; per-family/cell refinement is v2
2. Literature-fallback γ with explicit epistemic labels
3. Uniform `E_translation = 1.0` for v1; per-channel-family is v2
4. EGL-19 in AVAL as calibration reference
5. Refinement triggers deferred to Phase 5 evidence

**Deliverable status:**
- Methodology document: ✓ peer-readable, reference-quality, 7 sections complete
- Roadmap §7.3.5 entry: ✓ updated to Path 2 scope
- Pushback file: ✓ written, all four items resolved
- Progress summary file: ✓ (this file)
- Checkpoint JSON: ✓ `path2_phase1_checkpoint.json`

**Time spent on Phase 1:** ~1 work block (per estimate)

**Next phase:** Phase 2 — γ literature scoping.

**Phase 2 Step 0 (added per Rohit's review note):** Per-cell channel
inventory before γ scoping, to bound Phase 2 effort to channels actually
appearing in current Layer 1 cells:

| cell | channels used |
|---|---|
| AVAL | EGL-19, IRK, (NCA with g=0; treat as no-op for v1) |
| AVAR | EGL-19, IRK, NCA, UNC-103 |
| AIY (v1) | EGL-19, KQT-1, SHL-1, NCA |
| RIM | EGL-19, SHL-1, IRK, CCA-1, UNC-2, EGL-2 |

**Union of channels in scope for Phase 2 v1:** EGL-19, IRK, NCA, UNC-103,
KQT-1, SHL-1, EGL-2, CCA-1, UNC-2 (9 channels).

**Channels NOT in Phase 2 v1 (channel modules exist but not in current
Layer 1 cells):** SHK-1, EGL-36, KQT-3, TWK family. Skip γ scoping.

**Channels DEFERRED to Phase 2 v2 (AIY full channel set):** SLO-1 isolated
(slo1iso), SLO-1+EGL-19 coupled (slo1egl19). AIY v1 in Layer 1 §7.3 is
explicitly simplified; full AIY with SLO family is v2 substrate work.

**Per-channel scoping outputs:**
- γ value (pS)
- Source citation
- Epistemic label (empirically grounded / biophysically derived /
  approximation from adjacent biology)
- Uncertainty range (if multiple sources give different values)
- Measurement conditions (temperature, ionic composition)
- Heteromer-vs-paralog decision for multi-gene families (default per
  methodology §2.4 unless literature indicates exception)

**Phase 2 Step 1 onwards:** literature scoping per scoping hierarchy
(C. elegans direct → C. elegans heterologous → mammalian/Drosophila
homolog fallback) for each of the 9 in-scope channels.

---

## Cross-phase notes

### Key references (recurring across phases)

- **Layer 1 design doc (§8):** `docs/layer1_design_decisions.md` — motivates
  Path 2 via §7.3 finding; epistemic labeling framework
- **Methodology doc:** `docs/channel_parameter_derivation_methodology.md`
  — the reference material for Path 2 (and beyond)
- **AVA voltage anchor:** Liu/Chen/Wang 2020 *Nat Commun* 11:5076
  (DOI 10.1038/s41467-020-18893-9; PMCID PMC7544903) — voltage-clamp +
  current-clamp recordings of AVAL and AVAR
- **Channel sources:** Nicoletti 2024 *PLoS ONE* (PMID equivalent
  PMC10980225) for channel kinetic params and Nicoletti's I-V validation
  targets (NOT calibration anchors under Path 2)
- **CeNGEN:** cengen.org/downloads/021821_medium_threshold2.csv (already
  cached at `/tmp/cengen/` from §7.2 v2 pull); may need re-download in
  Phase 3 if cache cleared

### Hard-stop conditions across phases

- CeNGEN data inaccessible or schema changed (Phase 3)
- Single-channel γ unavailable for >50% of channels (Phase 2)
- `C_global` biophysically nonsensical (Phase 4)
- > 50% channels beyond 5× discrepancy (Phase 5)
- Cells fail stable rest after acceptable channel-level validation (Phase 6)
- Brian2 codegen failures (any phase touching cell builders)

When a hard stop triggers: write `HARD_STOP.txt` with diagnosis, update
this summary, terminate cleanly.

### Out-of-scope reminders

- Kinetic parameters (V_half, k, τ) preserved from Nicoletti — only gbar
  derivation in Path 2
- Per-channel-family C_global / E_translation (v2 refinement if needed)
- Hill function or non-linear scaling (v2 refinement if needed)
- New channels not in current Wave 2 cell builders (Layer 3 work)
- Pumps from §7.2 v2 (already shipped and validated)
- Web page / public-facing documentation (separate work block at Layer 1
  full ship)

---

## Substantive findings log (populated as discovered)

### Phase 2 — 2026-05-12 — NCA single-channel γ literature gap

**Observation:** No published unitary single-channel conductance value for
NALCN-family channels exists in literature (explicitly stated in Belal et
al. NALCN preprint: "There is no available estimate for NALCN single-
channel conductance"). NCA is one of 9 channels in Phase 2 scope.

**Diagnosis:** NALCN's tight regulation, low open probability, and small
contribution to total cell conductance make single-channel measurement
technically very difficult. Macroscopic NALCN current is well-characterized;
unitary γ is not.

**Resolution (v1):** γ_NCA = 5 pS placeholder estimate based on (a) NALCN's
~2-5% maximal conductance voltage-insensitive behavior, (b) typical leak-
channel γ range, (c) NaV-family structural relation suggesting γ < NaV
(~20 pS) but > HERG-like (~2 pS). Epistemic label: "approximation from
adjacent biology — LITERATURE GAP."

**Phase 5 sensitivity candidate:** If derived NCA channels surface beyond
5× discrepancy with Nicoletti, γ refinement is a candidate (could test
γ = 1, 5, 20 pS as sensitivity sweep). Layer 1 v1 ships with the 5 pS
estimate; refinement explicitly deferred.

### Phase 3.5 — 2026-05-12 — AIY SHL-1 + nca-1 disambiguation (T4 + unfiltered TMM check)

**Methodology:** Per Rohit's Phase 3.5 authorization, checked T4 (stringent)
and unfiltered TMM counts (`Average_integrated_TMM_counts_lengthNormalized_111521.tsv`,
138 per-replicate columns: AVA n=6, AIY n=3, RIM n=4) for SHL-1 and nca-1.

**SHL-1 in AIY — Case (a) confirmed: T2 false negative for low-expression gene.**

| threshold | shl-1 AVA | shl-1 AIY | shl-1 RIM |
|---|---:|---:|---:|
| T4 (stringent) | 0 | 0 | 153.1 |
| T2 (medium-default) | 0 | 0 | 153.1 |
| **Unfiltered TMM mean** | **6.55** | **8.23** | **287.0** |
| per-replicate (AIY) | — | 9.83, 6.07, 8.78 | — |

AIY SHL-1 shows consistent low expression across all 3 replicates (mean 8.2,
range 6-10). This is a **real signal below T2's recommended threshold**, not
a true absence. RIM shows high expression (~287); AVA also has low signal
(~6.5) — but Wave 2 AVAL/AVAR don't use SHL-1, so AVA low-but-nonzero
SHL-1 isn't methodologically relevant.

**Interpretation:** Nicoletti's inclusion of SHL-1 in AIY is **biologically
supportable** at this low expression level. Path 2 v1 (T2-based derivation)
gives gbar = 0 for AIY SHL-1, which underestimates biology. Phase 5
implications:
- If AIY achieves stable rest without SHL-1 → low-expression channels
  matter little; T2-based v1 is acceptable; Nicoletti's SHL-1 inclusion
  was a redundant fit term
- If AIY fails rest without SHL-1 → low-expression channels DO matter;
  v1 underestimates them; refinement candidate: "use unfiltered TMM for
  channels below T2 but consistently non-zero across replicates" (a
  per-channel methodology refinement; v2 scope)

Documented as **v1 acknowledged limitation** rather than methodology
failure. Methodology doc update NOT needed (Rohit's "phenomenological
assignment" case rule doesn't apply — this is the false-negative case).

**nca-1 — not in CeNGEN dataset at all (not annotated).**

T4 and T2 both don't contain `nca-1` as a gene_name. Either gene symbol
mismatch with CeNGEN's annotation, or the gene is consistently
indistinguishable from another sequence in CeNGEN's mapping. Pragmatic
v1 resolution stands: **NCA channel uses nca-2 alone**. Consistent with
project-internal `cengen_channel_inventory.csv` mapping. If Phase 5
surfaces NCA-specific discrepancy, alternative aggregation including
unc-77 as pore-forming is a refinement candidate (unc-77's pore-vs-
auxiliary status is debated in NALCN-family literature).

**Both findings status:** Documented as v1 limitations with explicit
Phase 5 dependency. Methodology proceeds. Phase 4 (C_global calibration)
authorized.

### Phase 3 — 2026-05-12 — AIY SHL-1 zero-TPM discrepancy (FINDING)

**Observation:** Wave 2 AIY cell builder uses SHL-1 (Kv4 A-type K) with
gbar ≈ 7.59e-4 S/cm². CeNGEN T2 reports `TPM_shl-1_AIY = 0` (below
threshold 2).

**Diagnosis:** Three possible explanations:
1. SHL-1 expressed below T2 threshold but functionally present (T2 false
   negative)
2. Nicoletti's AIY parameterization uses SHL-1 phenomenologically (whole-
   cell I-V fit, not AIY-specific expression data); "AIY SHL-1" may
   capture a different K channel
3. CeNGEN AIY sampling insufficient

**Resolution:** Phase 5 tests whether AIY achieves stable rest with
derived SHL-1 gbar = 0. If yes → Path 2 rejects Nicoletti's SHL-1
inclusion as non-biology-grounded artifact of fit. If no → substantive
question about Nicoletti-vs-Path-2 reconciliation. Documented as exactly
the kind of methodology-validation finding Path 2 was designed to surface.

### Phase 3 — 2026-05-12 — nca-1 below CeNGEN T2 threshold

**Observation:** `nca-1` (NALCN pore-forming paralog) not found in CeNGEN
T2 for any neuron. WBGene00003502 (nca-1 WormBase ID) matches no T2 row.

**Resolution (v1):** NCA channel TPM = TPM_nca-2 alone for v1 derivation.
Consistent with `cengen_channel_inventory.csv` mapping (NCA channel ↔
{nca-2, unc-77}; nca-1 absent). Documented as substantive finding for
Phase 5 sensitivity analysis. If NCA discrepancy surfaces, alternative
aggregation including unc-77 as pore-forming is a candidate (unc-77's
pore-vs-auxiliary status is debated in NALCN-family literature).

### Phase 3 — 2026-05-12 — KQT-1/KQT-3 heteromer hypothesis REJECTED

**Observation:** Per Phase 2 §4.2 KQT heteromer flag check:
- AIY kqt-1 TPM = 63.4
- AIY kqt-3 TPM = 0.0
- ratio kqt-3 / kqt-1 = **0%** (well below 20% paralog-separate threshold)

**Resolution:** Paralog-separate confirmed for KQT-1 in AIY. KQT-1 alone
determines KQT density per methodology §2.4. Methodology KQT-family flag
RESOLVED.

### Phase 2 — 2026-05-12 — KQT-1/KQT-3 heteromer hypothesis flagged

**Observation:** Mammalian KCNQ2 + KCNQ3 form heterotetrameric M-current
channels (well-established). C. elegans KQT-1 groups phylogenetically
with KCNQ2-5. KQT-2 + KQT-3 heteromer hypothesized in C. elegans (Okahata
2019) but not biochemically confirmed. KQT-1's potential heteromerization
with KQT-3 in C. elegans neurons (e.g., AIY) is an open question.

**v1 treatment:** Paralog-separate (default rule) for KQT-1 with single-
gene TPM.

**Phase 3 action:** Pull both KQT-1 and KQT-3 TPMs for AIY. If both
expressed at comparable levels, **Phase 5 sensitivity analysis tests
heteromer aggregation** (min-across-pore-forming TPM as alternative).
If only KQT-1 expressed meaningfully (KQT-3 below threshold or much
lower), default paralog-separate is correct.

Findings template (to be filled by subsequent phases):

```
[Phase N] — date — finding title
  observation: <what was found>
  diagnosis: <best-guess cause>
  resolution: <within-Path-2 / v2 refinement / fundamental issue>
  links: <relevant files, log entries, validation outputs>
```

---

## Checkpoint resumability

For session resume after pause: read this file + the Phase N checkpoint
JSON + the methodology doc. The combination gives complete state for
continuing the work block.

Checkpoint files (one per phase, written at phase completion):
- `path2_phase1_checkpoint.json` (current — Phase 1 complete)
- `path2_phase2_checkpoint.json` (future)
- `path2_phase3_checkpoint.json` (future)
- etc.
