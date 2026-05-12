# §7.3.5 Path 2 — progress summary (live tracking)

**Purpose:** Master tracking file across all 7 phases of Path 2 work block.
Updated after every milestone for cross-session resumability.

**Started:** 2026-05-12
**Last updated:** 2026-05-12 (Phase 2 SHIPPED — γ inventory complete
for 9 channels; 8/9 sourced from mammalian homolog literature, 1 (NCA)
estimated due to documented literature gap; coverage 89% well above 50%
hard-stop threshold)

---

## Status by phase

| phase | name | status | deliverable | date |
|---|---|---|---|---|
| 1 | Methodology document | ✓ SHIPPED | `docs/channel_parameter_derivation_methodology.md` | 2026-05-12 |
| 2 | γ literature scoping | ✓ SHIPPED | `docs/channel_gamma_inventory.md` | 2026-05-12 |
| 3 | CeNGEN TPM extension | pending | `docs/channel_tpm_inventory.md` | — |
| 4 | `C_global` calibration | pending | `docs/channel_calibration_protocol.md` | — |
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
