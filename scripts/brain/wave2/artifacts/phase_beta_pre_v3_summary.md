# Phase β-pre v3 — Standalone Summary

**Generated:** 2026-04-26
**Author:** Phase β-pre v3 engineering session
**Scope:** Layer B verification + AVAR UNC-103 patch + citation expansion + AVAR upstream issue draft
**Canonical record:** `phase_beta_pre_validation.md` (v1+v2+v3 combined)

---

## TL;DR

- **Layer B strict verdict:** 0/4 panels pass at per-feature 5% relative tolerance — same methodological wall v1 hit at Layer C (post-hoc) and v2 hit at Layer C (fit-target).
- **Layer B substantive verdict:** voltage absolute errors **3.3-4.8 mV mean per cell**, roughly half of v2's Layer C residuals (6.8-15 mV). NEURON code IS reproducing the published Model figures within tighter bounds than the published Model fits experimental data.
- **AVAR patch verdict:** -24.25 mV resting potential after UNC-103 patch — well within target -25 ± 5 mV. Resolves v2's +11 mV bias from missing-upstream-file workaround.
- **Decision:** strict reading triggers stop-and-ask before Phase β proper. Substantive reading and the v1 → v2 → v3 methodological pattern indicate the right Phase β gate criterion is voltage-feature-only with absolute-error budget, not strict 5% per-feature relative.

---

## Layer B per-cell results

| Cell | Strict per-feature 5% | V-only abs err mean (mV) | V-only steps passing |
|------|------------------------|---------------------------|----------------------|
| AVAL | 0/7 panel fail        | 4.81                      | 3/7 (43%)            |
| AVAR (patched) | 0/7 panel fail | 4.09                      | 2/7 (29%)            |
| AIY  | 0/11 panel fail       | 3.41                      | 5/11 (45%)           |
| RIM  | 0/11 panel fail       | 3.31                      | 4/11 (36%)           |

**Overall:** 0/4 panels pass strict. Mean V abs error 3.91 mV across the four cells. 35% of all steps pass the V-only secondary diagnostic.

**Why timing features dominate the strict-criterion failure:** digitization samples each trace at ~60 timepoints; per-sample resolution is ~17 ms (Fig 1) or ~83 ms (Fig 3, 5). NEURON's internal dt = 0.025 ms produces sub-millisecond timing precision. The timing-feature divergences hit the 5% relative-with-floor metric on the small-denominator side. Voltage features (peak, plateau) — Nicoletti's actual fit objective — are within 3-5 mV per cell.

---

## Citation correction summary

Verified DOIs and roles applied:
- **Nicoletti 2019** (PLOS ONE `10.1371/journal.pone.0218738`): AWCon/RMD biophysical models — upstream paper that 2024 extends.
- **Nicoletti 2024** (PLOS ONE `10.1371/journal.pone.0298105`): 22-channel library; AVA/AIY/RIM/VA/VB/VD; primary Wave 2 import target.

Files modified by v3:
- `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md` — added "Primary upstream citations" block at top with both DOIs and roles + v3 correction note.
- `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_phase_beta_pre_prompt.md` — corrected v1-era `pcbi.1007611` reference to `pone.0218738` with role description.
- `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_phase_beta_pre_v2_prompt.md` — added v3 update note expanding both citations with roles.

Files preserved unchanged (v1 historical artifacts that contain the wrong DOI inside v1's own diagnostic narrative — historical record of how the citation error was detected):
- `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/published_traces.json`
- `~/Desktop/website/personalwebsite/scripts/brain/wave2/digitize_panels.py`

The v3 prompt itself (`phase_v_w2_phase_beta_pre_v3_prompt.md`) contains `pcbi.1007611` only as **literal grep target patterns** in command examples (the prompt is instructing this session to grep for the wrong DOI). These are intentionally preserved.

---

## AVAR UNC-103 patch summary

**File:** `~/Desktop/website/personalwebsite/scripts/brain/wave2/avar_unc103_patch.py`
**Standalone:** does not modify Nicoletti's upstream code in place; only imports from upstream.

**What it does:** mirrors `AVAL_simulation_iclamp.py` with:
- AVAR surface area `1121.79e-8 cm²`
- AVAR parameter vector `[0.0643372, 0.225225, 0.042079, 0.0493356, 0.0481669, -37, 0.751761]` (EGL19, LEAK, IRK, NCA, UNC103, ELEAK, CM)
- `gScm2` rescale at index 4 (UNC-103 included in rescaled set)
- `soma.insert("unc103")` and `seg.unc103.gbar` assignment — the channel insertion that AVAL's iclamp script omits

**Resting potential check:** -24.25 mV (mean across 7 steps). Target -25 ± 5 mV. **PASS.**

**Plateau values:** -127, -96, -62, -24, +16, +49, +80 mV across the 7 steps — closely match the published Fig 1B AVAR Model-trace plateaus we digitized (-125, -91, -58, -22, +20, +56, +85 mV).

**Confidence:** high. Channel set verified against `AVAR_simulation.py` line 27 comment; conductance verified against line 28; surface area + ELEAK + CM verified against same source; `unc103.mod` is shipped and compiled in the local mech library.

---

## AVAR upstream issue draft

**Location:** `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/avar_upstream_issue_draft.md`

**Status:** DRAFT — awaiting user authorization before filing against `github.com/ModelDBRepository/2017403`.

**Suggested title:** "AVAR_simulation.py imports AVAR_simulation_iclamp but the module file is missing from the repo head tree"

**Content:** description, impact, our workaround (the standalone patch), confidence in the workaround, suggested fix (restore the missing iclamp file from author's working tree). Drafted to be filed verbatim if user authorizes.

---

## Phase β proper readiness assessment

**Status: NOT READY without cross-session discussion** per the strict reading of the spec's Layer B verdict (multi-panel fail → STOP, urgent notify, no Phase β proper without cross-session discussion).

**Items to resolve in cross-session discussion:**

1. **Adopt a defensible Phase β gate criterion.** The 5% per-feature relative tolerance is structurally too strict at every layer measured (v1 → v2 → v3). Recommended replacement: voltage-feature-only per-step pass with absolute-error budget ≤ 3 mV per step AND > 80% of steps pass per panel. This is *tighter* than Layer B's 3-5 mV residuals against Model figures — meaningful Phase β gating without the small-denominator metric pathology. Alternative options in `phase_beta_pre_validation.md` Part 5 Issue 1 (raise floor to `0.25 × peak`; absolute waveform RMSE ≤ 5 mV per step).

2. **Decide on AVAR upstream issue filing.** Draft is ready. If filed, opens a constructive channel to the upstream maintainers; if not filed, our `avar_unc103_patch.py` remains the AVAR runtime for any AVAR Phase β validation work.

3. **Confirm Phase β proper scope and sequencing.** Per the architectural plan: cadiff/caintra1 translation, NEURONReference wrapper, EGL-19 channel translation as the first translation target, then Gate 2a evaluation against fit-target current-clamp data using the redefined tolerance.

**If discussion concludes:**
- A relaxed/redefined gate criterion is adopted: Phase β proper proceeds with the new criterion. v3's AVAR patch is the AVAR runtime until the upstream is restored.
- Strict 5% per-feature is retained: Phase β proper is structurally blocked until the upstream model is re-derived to match figures within 5% per feature, which would be a larger investigation than Wave 2 scope.

---

## Files produced by v3 (recap)

In `scripts/brain/wave2/`:
- `digitize_model_traces_v3.py` — Deliverable 1 driver
- `avar_unc103_patch.py` — Deliverable 2 patch
- `run_layer_b_validation_v3.py` — Deliverable 3 driver
- `artifacts/nicoletti_model_traces.json` — Deliverable 1 output
- `artifacts/layer_b_validation_results.json` — Deliverable 3 output
- `artifacts/avar_upstream_issue_draft.md` — Deliverable 5
- `artifacts/phase_beta_pre_validation.md` (updated through v3) — Deliverable 6
- `phase_alpha_report.md` (Phase β-pre addendum updated) — Deliverable 6
- `artifacts/phase_beta_pre_v3_summary.md` — Deliverable 7 (this document)

Citation corrections applied to:
- `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md`
- `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_phase_beta_pre_prompt.md`
- `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_phase_beta_pre_v2_prompt.md`
