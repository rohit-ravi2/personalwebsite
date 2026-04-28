# Wave 2 cellular validation — combined work block summary

**Date:** 2026-04-26
**Scope:** Option B (AIY only) — pre-flight pushback adjudicated to AIY scope.
RIM and RMD deferred to separate work blocks.

---

## Cell verdicts

| Cell | Verdict | Channels | Layer A VC | Layer A CC |
|---|---|---|---|---|
| AVAL | **PRODUCTION_GRADE** (carried) | 4: egl19, leak, irk, nca | n/a (option α) | n/a (option α) |
| AIY | **PRODUCTION_GRADE** (this work block) | 7: egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1 | 11/11 (100%) | 10/11 (90.9%) |
| RIM | DEFERRED | 7: shl1, egl2, irk, cca1, unc2, egl19, leak | — | — |
| RMD | DEFERRED | unknown — Nicoletti 2024 has no RMD model | — | — |

---

## AIY (Option B, this work block)

**Channels (verified primary-source, 7 channels):**
[egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1]

**KQT-1 channel translation (CP1):** PRODUCTION_GRADE.
- 11/11 holds passing at zero divergence vs NEURON kqt1.mod.
- Module: `wave2/channels/kqt1.py`, follows F1/F3 pattern; 2-state m·s gating
  with double-Boltzmann sinf, no GLOBAL declarations.
- Distinct from KQT-3 (different gating; verified by mod-file diff).

**AIY cell construction (CP2):** Brian2 7-channel cell at
`wave2/option_alpha_aiy_cell.py`. Smoke test passed.
- Surf = 65.89e-8 cm², cm = 1.6 μF/cm², eleak = -89.57 mV.
- g vector matches Nicoletti's published g0 (gScm2-rescaled at AIY surf).
- Top-level comment in AIY_simulation.py mislabeling pos[6] as "irk"
  resolved: the actual code uses index [6] as shl1.gbar. Code wins.

**AIY voltage-clamp Layer A (CP3):** PRODUCTION_GRADE.
- 11/11 holds (100%), peak/ss divergences ≤ 1.13%.
- Brian2 vs NEURON AIY (Nicoletti's published canonical cell via
  NEURONReference("AIY")).

**AIY current-clamp Layer A (CP4):** PRODUCTION_GRADE.
- 10/11 sweeps (90.9%) at ≤3 mV voltage-feature tolerance for peak + plateau.
- Aggregate timepoint-level: 95.9% within 3 mV.
- Failing sweep -15 pA: 6.84 mV plateau divergence at extreme hyperpolarization
  (~-128 mV). Caused by integrator drift on KQT-1's very-slow s-gate
  (stau ≈ 186 s) over 5-second stim window. F19 standing followup; not
  blocking production grade.

**AIY verdict (CP5):** **VERDICT_AIY_PRODUCTION_GRADE**

---

## Major finding — F18 (NEURON ion_style override)

**Cells with >1 USEION ca mechanism (e.g. AIY's egl19+slo1egl19) have eca
silently overridden at NEURON runtime to the Nernst-computed value, NOT the
user-set `seg.eca = 60` mV.**

Specific values (NEURON default celsius=6.3°C, cai=5e-5 mM, cao=2 mM):
- Hand-computed Nernst eca = (RT/zF) ln(cao/cai) = 12.040 × 10.597 = 127.583 mV
- NEURON observed runtime eca = +127.590 mV (rounding from internal R, F).
- AVAL (single USEION ca, only egl19) preserves user-set eca = 60 mV. Confirmed.

**First-run divergence pattern matched this exactly:** caCALC in slo1egl19
diverged 53-77% (NEURON 585.89 μM vs Brian2 275.55 μM at +0 mV). Other K
channels (kqt1, shl1, slo1iso) matched within <5%; only the slo1egl19
nanodomain calcium formula was hit by the wrong eca.

**Fix applied:** AIY_ECA_MV = 127.59 in cell builder. Added `eca_mV` parameter
to `slo1egl19_apply_params()` (was missing — now propagates eca to slo1egl19's
nanodomain calcium formula). Both egl19 and slo1egl19 channels in the AIY
cell now use the NEURON-runtime eca explicitly.

**For future RIM work:** RIM has cca1+unc2+egl19 (three USEION ca). Same
ion_style override applies. Plan to use `eca_mV = 127.59` (or recompute at
RIM's specific cao if different) when constructing RIM Brian2 cell.

**Methodology note:** this is an upstream NEURON behavior, not a translation
defect in any specific channel module. The published Nicoletti AIY traces
are produced AT eca=127.59 mV — our published-model reproduction goal is to
match that. Setting Brian2 to 60 mV would diverge from the published model.

---

## RIM — DEFERRED to separate work block

**Reason:** RIM requires 3 new channel translations (CCA-1, EGL-2, UNC-2),
of which UNC-2 has GLOBAL declarations in its mod file. This is
Phase-β-scale work, not within the cell-validation envelope.

**Channel set (verified primary-source):**
[shl1, egl2, irk, cca1, unc2, egl19, leak]

Existing translations cover only shl1, irk, egl19, leak (4 of 7). The 3
missing channels:

- **CCA-1** (cca1.mod, 105 lines): T-type Ca, standard m²·h gating, no GLOBAL.
  Expected ~30-45 min translation effort following established F1/F3 pattern.
- **EGL-2** (egl2.mod, 85 lines): EAG-family voltage-gated K, presumed
  standard pattern. Expected ~30 min.
- **UNC-2** (unc2.mod, 132 lines): P/Q-type Ca with **GLOBAL minf, hinf,
  mtau, htau, munc2, hunc2** declarations. F2-class translation pattern.
  Per F2 catalog (caintra1 origin) the GLOBAL handling is well-established
  via per-mechanism h.<param>_<suffix> setattr. Expected ~45-60 min.

**Estimated total RIM work:** ~2-2.5 hours minimum for translations + cell
construction + Layer A validation. Comparable to Phase β shape, not Phase
F-cellular shape. Should be scoped as its own work block.

**RIM-specific quirks to expect:**
- RIM's `g` vector is already in S/cm² (verified RIM_simulation.py line 27);
  no gScm2() rescaling for channel conductances in upstream. NEURONReference
  fixed accordingly (this work block's incidental fix to neuron_reference.py
  RIM build path).
- RIM's stim.delay = 5000 ms (later than AIY/AVAL); simdur = 14000 ms (long).
- RIM has 3 USEION ca mechanisms — F18 ion_style override will apply.

---

## RMD — DEFERRED (Scope C)

**Reason:** Nicoletti 2024 has no RMD simulation script. The Mellem
investigation noted "Nicoletti 2019 had RMD model"; the 2019 PLOS ONE paper
does exist (DOI 10.1371/journal.pone.0218738) but its source code is **not**
in our local upstream `simulation/upstream/`. Reference acquisition required
before any translation can begin:

1. Locate 2019 paper's published code repository.
2. Clone locally.
3. Inspect their RMD simulation script for channels, parameters, protocol.
4. Possibly translate any 2019-only channels not in our existing set
   (kvs1, egl36 may have 2019-specific RMD parameterizations not in 2024).

This is fundamentally a different work-block shape (data acquisition + new
translations + new validation). Defer until reference acquired.

---

## Phase δ readiness

**With AVAL + AIY both production-grade:** Phase δ (network-level integration
of multiple cells via gap junctions and chemical synapses) has 2 cells
ready as building blocks. Both are interneuron-class (AVA = command
interneuron, AIY = thermosensory/integrating interneuron). This covers the
"forward locomotion + sensory integration" axis of the C. elegans
connectome at the cellular building-block level.

**Recommended Phase δ work block sequence:**
1. **Phase δ-1**: AVAL ↔ AIY synaptic coupling (bidirectional or
   feedforward). Validate against published connectome data + functional
   couplings.
2. **Phase δ-2 (after RIM)**: add RIM into the network. AVAL→RIM and
   RIM→motor outputs are documented coupling motifs.
3. **Phase δ-3+**: branch into motor neuron classes (VA5/VB6/VD5 — Nicoletti
   2024 has these models locally; same 7-channel-class pattern as AIY/RIM).

**Caveat for Phase δ:** F18 ion_style behavior must be applied consistently
across all multi-Ca cells in the network. The fix applied to AIY/slo1egl19
generalizes — any cell with >1 USEION ca needs the eca=127.59 mV (or
network-temperature-specific) setting.

---

## Standing followups (work-block-level)

- **F18 generalization:** RIM, AVAR (has unc103 + egl19 + nca → only one
  USEION ca), and any multi-Ca cell needs F18-aware eca handling.
- **F19 — slow-gate integrator drift:** Brian2 rk4 vs NEURON cnexp diverges
  over long simulations with very-slow taus (KQT-1 stau=186 s). Affects
  -15 pA AIY current-clamp at extreme hyperpolarization. Mitigation:
  consider Brian2 `exponential_euler` for slow-gate ODEs.
- **F20 (latent):** SLO1_EGL19_PARAMS["eca_mV"] default of 60 mV is
  misleading; cells where ion_style overrides should pass explicit eca_mV.
  Consider runtime warning when slo1egl19 loaded without explicit eca.
- **Translation-pattern expansion to F18:** add F18 to
  `translation_patterns.md` so future Ca-multi-mechanism cells benefit.
- **Pre-flight catch #4 documented:** propagation pattern is consistent
  with F1-F3, F12-F17 — orchestrator-side claims without primary-source
  verification, caught by agent-side reading. See
  `cellular_validation_findings.md` for full catalog.

---

## Recommendations

1. **Promote AIY to production-grade in main wave2 records.** Update
   `wave2/artifacts/option_alpha_summary.md` (or similar) to add AIY entry
   alongside AVAL.
2. **Add F18 to `translation_patterns.md` (F1-F17 catalog).** Include the
   diagnostic pattern: divergence localized to a single Ca-dependent
   channel formula (in this case slo1egl19's caCALC) is a fingerprint of
   F18.
3. **Schedule RIM work block** when bandwidth permits. Estimate 3-4 hours
   for 3 channel translations + cell + Layer A. Pre-flight verification of
   RIM's `g` vector format (already-S/cm² vs nS) and the ion_style F18
   handling should be in scope from start.
4. **Schedule RMD reference acquisition** as separate task (no engineering
   work until 2019 paper code is acquired and inspected).
5. **Phase δ readiness:** AVAL + AIY now ready as building blocks for
   network-level work. Apply F18 lesson uniformly across multi-Ca cells.

---

## Files produced this work block

- `wave2/channels/kqt1.py` — KQT-1 Brian2 translation (NEW).
- `wave2/option_alpha_aiy_cell.py` — Brian2 7-channel AIY cell (NEW).
- `wave2/run_kqt1_validation.py` — KQT-1 Layer A runner (NEW).
- `wave2/run_option_b_aiy.py` — AIY Layer A runner CP3+CP4+CP5 (NEW).
- `wave2/diagnose_aiy_divergence.py` — per-channel diagnostic (NEW).
- `wave2/diagnose_slo1egl19_states.py` — slo1egl19 internals diagnostic (NEW).
- `wave2/channels/slo1_egl19_coupled.py` — added `eca_mV` parameter to
  apply_params (MODIFIED, F18 fix).
- `wave2/neuron_reference.py` — fixed AIY+RIM build paths (MODIFIED;
  AIY_simulation_s import, RIM g vector handling).
- `wave2/artifacts/kqt1_validation_results.json` — KQT-1 VC results (NEW).
- `wave2/artifacts/option_b_aiy_results.json` — AIY full validation (NEW).
- `wave2/artifacts/cellular_validation_findings.md` — full findings + F18
  documentation (NEW).
- `wave2/artifacts/cellular_validation_summary.md` — this document (NEW).

---

*End of cellular validation summary. AIY now production-grade; RIM and RMD
remain deferred to separate work blocks.*
