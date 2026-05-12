# §7.3.5 Path 2 — progress summary (live tracking)

**Purpose:** Master tracking file across all 7 phases of Path 2 work block.
Updated after every milestone for cross-session resumability.

**Started:** 2026-05-12
**Last updated:** 2026-05-12 (Phase 1 SHIPPED)

---

## Status by phase

| phase | name | status | deliverable | date |
|---|---|---|---|---|
| 1 | Methodology document | ✓ SHIPPED | `docs/channel_parameter_derivation_methodology.md` | 2026-05-12 |
| 2 | γ literature scoping | pending | `docs/channel_gamma_inventory.md` | — |
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

**Next phase:** Phase 2 — γ literature scoping. Per-channel inventory
covering EGL-19, CCA-1, UNC-2, SHL-1, SHK-1, EGL-36, EXP-2, UNC-103,
EGL-2, NCA channels (nca-1, nca-2), IRK channels (irk-1, -2, -3),
KQT channels (kqt-1, -2, -3), TWK family (twk-*), SLO channels (slo-1,
slo-2). For each: γ value (pS), source citation, epistemic label,
uncertainty range, conditions for the measurement. Also: per-family
heteromer-vs-paralog literature scoping (refines methodology §2.4 per-
family table).

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

*No findings yet — Phase 1 is methodology document only; no implementation.*

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
