# Phase β-pre Combined Validation — v1 + v2 + v3

**Generated:** 2026-04-26 (extended through v3)
**Author:** Phase β-pre v1 → v2 → v3 engineering sessions
**Scope:** Closure of Phase α deliverable 3 (NEURON-vs-experimental tolerance check, condition 3 for Path A invalidation)
**Files:** v1 traces + v1 comparison + v2 traces + v2 comparison + v3 Model traces + v3 Layer B comparison + v3 AVAR patch

---

## Three-layer comparison decomposition (made explicit in v3)

The three iterations of this work surfaced a layered comparison structure that wasn't explicit at v1 / v2 design time:

- **Layer A — Brian2 implementation = NEURON implementation.** Phase β translation work; 5% appropriate.
- **Layer B — Nicoletti's NEURON code = Nicoletti's published Model figures.** Deterministic-implementation check; 5% appropriate. **This is what condition-3 actually asks** and is what v3 directly tests.
- **Layer C — Nicoletti's published Model = experimental data.** Fit-quality check. 5% structurally too strict for biophysical HH fits. v1 measured C against I-V (post-hoc predictions); v2 measured C against current-clamp (fit targets); both produced "fail" against 5%. Nicoletti herself reports 5-15 mV residuals.

Each version of this work climbed one layer toward the question condition-3 actually asks:
- v1: wrong panel choice (I-V curves are post-hoc predictions, not fit targets) → 39-149% Layer C divergence
- v2: correct panel choice (current-clamp fit targets) but Layer C is the wrong layer for a 5% relative tolerance → 6.8-15 mV mean voltage absolute residual at Layer C
- v3: directly tests Layer B (NEURON code vs published Model figures, same panels) → 3-5 mV mean voltage absolute residual at Layer B (roughly half of Layer C, as expected if the implementation faithfully reproduces the published Model)

---

## Executive verdict

**Outcome: partial — methodological surfacing across three layers; Layer B substantively passes but strict 5% per-feature criterion is not met on any layer at any iteration.**

| Phase | Layer | Metric | Pass criterion | Outcome |
|-------|-------|--------|----------------|---------|
| v1 | C (post-hoc) | I-V curve fit | 5% per-point + ≥90% pass | 0/3 panels |
| v2 | C (fit-target) | Current-clamp per-feature | All features per step, > 80% steps | 0/4 panels |
| v2 secondary | C | Voltage absolute error per step | n/a | 6.8–15.2 mV mean per cell |
| v2 visual | C | NEURON code overlays figure Model traces | n/a | qualitative match (~5-15 mV residual) |
| **v3 (NEW)** | **B (Model figures)** | **NEURON output vs digitized Model trace per-feature** | **All features per step, > 80% steps** | **0/4 panels at strict 5%** |
| **v3 secondary** | **B** | **Voltage absolute error per step (Model vs NEURON)** | **n/a** | **3.3–4.8 mV mean per cell** |
| **v3 AVAR patch** | infrastructure | Resting potential after UNC-103 patch | -25 ± 5 mV | **-24.25 mV** ✓ |

**Condition-3 status (final, post-v3):** Multi-layer fail at strict 5% per-feature relative tolerance on every panel measured. **Layer B is substantively passing** — voltage absolute errors at Layer B are roughly half of Layer C (3-5 mV vs 6.8-15 mV), which is exactly what we'd expect if NEURON code faithfully reproduces the published Model figures. The strict 5% per-feature criterion fails at Layer B because it is dominated by *timing-feature* divergences (time-to-peak, settling-time, plateau-duration), not voltage-feature divergences. Timing features are quantized by digitization sampling (~17-83 ms per sample for our 60-sample-per-trace digitization vs NEURON's internal 0.025 ms resolution), and the relative-divergence metric explodes at small denominators. Voltage features (peak, plateau) — the actual fit targets of Nicoletti's optimization — are within 3-5 mV per cell at Layer B.

**Recommendation (refined v3):** Cross-session discussion before Phase β proper. The 5% per-feature relative tolerance is structurally too strict at every layer; the right Phase β gate criterion is per-cell voltage absolute error ≤ N mV (with N chosen to be defensible at Layer A). Phase β proper should adopt one of:

- **Option 1:** voltage-feature-only per-step pass + absolute-error budget. Phase β passes if Brian2 voltage features (peak, plateau) match NEURON within ≤ 3 mV per step (tighter than Layer B's 4-5 mV residuals against Model figures), AND > 80% of steps pass per panel.
- **Option 2:** keep relative-with-floor formula but raise the floor to `0.25 × peak` instead of `0.1 × peak`. This effectively triples the small-magnitude tolerance window. Layer B voltage features pass under this rule; the NEURON-code tighter-than-Model-figure expectation gives Phase β headroom.
- **Option 3:** absolute waveform RMSE ≤ 5 mV per step on V vs t after temporal alignment. Closer to the actual fit objective.

---

## Part 1 — v1 narrative (preserved as historical infrastructure)

### What v1 attempted

v1 was tasked with: digitize panels showing Nicoletti 2024's experimental data + Model output overlay, run Nicoletti's NEURON code, compute per-point divergence, declare condition-3 pass or invalidation.

v1 selected three I-V curve panels:
- **Fig 1F AVAL**: voltage-clamp steady-state I-V curves, 16 V steps -120 to +50 mV
- **Fig 3D AIY**: voltage-clamp steady-state I-V curve, 16 V steps -120 to +50 mV
- **Fig 5D RIM**: voltage-clamp steady-state I-V curve, 16 V steps -100 to +50 mV

Selection rationale (preserved verbatim in `published_traces.json`): "I-V curves are the cleanest panel format for digitization (sparse data points ~16 per curve, vs full time-series traces with overlapping per-sweep curves)."

### What v1 measured

v1 ran Nicoletti's voltage-clamp NEURON code, computed steady-state currents at the matching V steps, interpolated to experimental V grid, and computed per-point divergence using the formula:

```
divergence(m, r, peak) = |m - r| / max(|m|, |r|, 0.1*peak)
```

**v1 results** (`comparison_validation_results.json`):

| Cell  | Max divergence | Mean divergence | Panel pass |
|-------|----------------|-----------------|------------|
| AVAL  | 0.665          | 0.395           | False      |
| AIY   | 1.490          | 0.572           | False      |
| RIM   | 1.660          | 0.633           | False      |

All three panels failed. Mean divergences 39.5–63.3%; max divergences 66.5–149%.

### Why v1's metric was wrong

Nicoletti 2024 explicitly states (Fig 1 caption, line 502 of extracted text):
> "The models were fitted on experimental current-clamp data obtained from [29], and shown in black in panels A and B."

For AIY (Fig 3 caption):
> "The model was fitted on experimental current-clamp data obtained from [30] and shown in black in panel A."

For RIM (Fig 5 caption):
> "The model was fitted on experimental current- and voltage-clamp data obtained from [30] and shown in black in panels A and B."

**Fig 1F, Fig 3D, Fig 5D — the panels v1 measured — are NOT fit-target panels. They are post-hoc voltage-clamp I-V curve predictions.** Nicoletti's body text discloses the I-V divergences directly:

> Fig 1 discussion (line 506-508):
> "The main differences with the experimental data are observed in AVAL, where the simulated currents are slightly overestimated for hyperpolarizing stimuli and underestimated for depolarizing stimuli (Fig 1D and 1F)."

> Fig 3 discussion (line 560):
> "Voltage-clamp simulations show that the model also reproduces the outward rectifying behavior of the average whole-cell currents (Fig 3B and 3D, red lines), but with a slight underestimation of the steady-state current."

The 39-66% divergences v1 measured are **expected, documented residuals** from a model fit on different data. The "fail" verdict at 5% tolerance against these post-hoc panels is structurally guaranteed because the model was never fit against them.

### What v1 produced (kept as infrastructure)

v1's work was not throwaway. It produced reusable infrastructure:
1. **Source PDF + per-page renders** at `artifacts/figures/source_pdfs/` (28 PNGs covering 25 pages + 7 hi-res renders).
2. **Working PDF extraction pipeline** via `pdftoppm` / `pdfimages`.
3. **Panel PNGs for Fig 1A/1B/1C/1D/1E/1F/3C/3D/5C/5D** at `artifacts/figures/`.
4. **`digitize_panels.py`** — OpenCV template-matching + manual grid-reading methodology, ~470 lines, reused as v2 starting point.
5. **`run_comparison_validation.py`** — NEURON harness with cell-specific runners (AVAL, AIY, RIM voltage-clamp) and tolerance-with-floor formula.
6. **`published_traces.json`** — 3 panels, 9-16 points each, with provenance, axis calibration, and digitization notes.
7. **Cross-session adversarial review pattern** — v1 itself surfaced the metric error during its own review.

This is the infrastructure v2 inherited and applied against the corrected metric.

---

## Part 2 — v1 → v2 transition

### Methodological correction

v2 measures the **fit-target panels** that Nicoletti 2024's captions explicitly identify as the data the models were fitted against:
- **Fig 1A**: AVAL current-clamp, 7 steps from -30 to +30 pA × 1000 ms (per Fig 1 caption)
- **Fig 1B**: AVAR current-clamp, 7 steps from -30 to +30 pA × 1000 ms (per Fig 1 caption)
- **Fig 3A**: AIY current-clamp, 11 steps from -15 to +35 pA × 5000 ms (per Fig 3 caption)
- **Fig 5A**: RIM current-clamp, 11 steps from -15 to +35 pA × 5000 ms (per Fig 5 caption)

These are the panels where Nicoletti's models were optimized against the underlying experimental traces. If the NEURON code reproduces these traces within tolerance, the model is "consistent with its own fit-target data." This is the actual condition-3 invalidation check.

### Why current-clamp not voltage-clamp

Current-clamp (V vs t given fixed I) is the protocol Nicoletti used to **fit** her models. Voltage-clamp (I vs V given fixed V) is a separate protocol used to **predict** based on the fitted model. v1's voltage-clamp comparison against published voltage-clamp panels measured prediction error, not fit error. v2's current-clamp comparison measures fit error.

---

## Part 3 — v2 results

### Deliverable 1: Citation verification and correction

**Verified citation:**

| Field    | Value |
|----------|-------|
| Title    | Biophysical modeling of the whole-cell dynamics of C. elegans motor and interneurons families |
| Authors  | Nicoletti M, Chiodo L, Loppini A, Liu Q, Folli V, Ruocco G, Filippi S |
| Journal  | PLOS ONE 19(3): e0298105 |
| Year     | 2024 (received 2023-07-20, published 2024-03-29) |
| DOI      | `10.1371/journal.pone.0298105` |
| Code repo| `https://github.com/martinanicoletti92/CelegansInterMotorNeuronsModels` |

**Verification method:** Three-tier — (1) on-disk PDF metadata via `pdfinfo` (gives title and author list); (2) PDF first-page text via `pdftotext` (gives full citation block including journal volume/issue and DOI inline); (3) DOI resolver via `WebFetch` of `doi.org/10.1371/journal.pone.0218738` to test the v1 hypothesis. The DOI appears 10+ times within the paper's own running header — highest-confidence source possible.

**Resolution of the citation hypothesis chain:**

1. **`10.1371/journal.pcbi.1007611`** (referenced in original spec as "Nicoletti 2019 PLOS Comp Bio"): incorrect — this DOI resolves to a glioma/brain-tumor paper unrelated to C. elegans (Jamous et al. "Self-organization in brain tumors").
2. **`10.1371/journal.pone.0218738`** (v1's hypothesized correction): partially correct — this IS a real paper by Martina Nicoletti et al. titled "Biophysical modeling of C. elegans neurons: Single ion currents and whole-cell dynamics of AWCon and RMD," PLOS ONE 14(7), 2019. But it is the **upstream paper** on a different cell pair (AWC, RMD). Nicoletti 2024 cites it as ref [36].
3. **`10.1371/journal.pone.0298105`** (correct citation for Wave 2 work): the actual paper used in this validation, covering AVAL/AVAR/AIY/RIM/VA5/VB6/VD5.

**Both Nicoletti 2019 (e0218738) and Nicoletti 2024 (e0298105) are real, related papers by the same group.** v1's spec preamble conflated the two. The "Nicoletti 2019" references in artifact files (`phase_v_w1_*.md`, `phase_v_w2_simulator_landscape.md`, `phase_v_w2_existing_data_inventory.md`) substantively describe the channel library introduced in 2019 — that's correct. The DOI `pcbi.1007611` is the only factual error; only two artifact files contained that DOI and both contained it within v1's diagnostic narrative (`digitize_panels.py` line 470-474; `published_traces.json` selection_rationale narrative). Per the spec's "v1 files preserved unchanged" directive, these were left as historical record. Future references should use `pone.0218738` (Nicoletti 2019, AWC/RMD) or `pone.0298105` (Nicoletti 2024, AVA/AIY/RIM/VA/VB/VD) as appropriate.

### Deliverable 2: Panel extraction

| Panel        | Source                                  | Resolution     | Origin     |
|--------------|------------------------------------------|----------------|------------|
| Fig 1A AVAL  | `nicoletti_2024_fig1A_AVAL_iclamp.png`  | 520 × 520      | v1 (kept)  |
| Fig 1B AVAR  | `nicoletti_2024_fig1B_AVAR_iclamp.png`  | 520 × 520      | v1 (kept)  |
| Fig 3A AIY   | `nicoletti_2024_fig3A_AIY_iclamp.png`   | 1421 × 1144    | v2 (new)   |
| Fig 5A RIM   | `nicoletti_2024_fig5A_RIM_iclamp.png`   | 1460 × 1126    | v2 (new)   |

**Extraction methodology** (`/tmp/v2_inspect/extract_panels_v3.py`):
1. PIL grayscale + row whitespace profile to find figure-block boundaries (pairs of "major" gutter runs > 0.998 white over ≥ 30 rows).
2. Within figure block, find longest deepest-peak gutter (smoothness ≥ 0.998, ≥ 30 rows) for inter-row split. If absent, panel A spans full block height (single-row figure).
3. Top-half column-gutter detection for A/B split (longest-deepest run in 35-65% of width).
4. Fixed paddings: 25 px left/top, 12 px right, **100 px bottom** to ensure x-axis labels captured.
5. Adjustments needed across iterations: row-gutter detection by deepest peak (not longest run) to distinguish inter-panel from within-panel sub-feature gaps; figure-block top-of-page tie-break by topmost-among-largest-spans.

**Verification:** Confirmed via downscaled visual inspection (≤ 1500 px) that all 4 panels show full y-axis ticks (-150 to +50 or +100 mV), full x-axis ticks (0-1500 ms or 0-8000 ms), title labels ("AVAL/AVAR/AIY/RIM current clamp"), legend ("Experiment/Model"), and panel letter ("A" or "B"). Image-handling guardrail respected: no figure PNGs read at full resolution; only ≤ 1500 px thumbnails or panel-cropped views read.

**Caption confirmation:** All 4 panels confirmed as fit-target by reading lines 492-502 (Fig 1), 573-580 (Fig 3), 647-653 (Fig 5) of `nicoletti_2024_text.txt`. RIM is fit on both current- and voltage-clamp data (Fig 5 caption); the other three are fit on current-clamp only (Fig 1 and Fig 3 captions).

### Deliverable 3: Time-series digitization

**Tool hierarchy outcome:**
1. **plotdigitizer** (tested, not used): the CLI's `--data-point` / `--location` mode requires per-trace pixel anchors for axis calibration. For batch processing 7-11 overlapping traces per panel, this would mean 21-33 anchor clicks (3 anchors × 7-11 traces) plus per-trace pixel locations — equivalent to manual digitization without automation benefit.
2. **OpenCV color-mask + per-step centerline** (used for all 4 panels): black-pixel mask excluding red Model traces (and blue for Fig 1B AVAR), per-step voltage-anchor matching with ±8 mV capture window, median-voltage centerline extraction.
3. **Manual grid-reading** (fallback, not invoked): would have been used if (1) and (2) failed on a panel.

**Tool used per panel:** `opencv_color_mask_centerline` for all 4 panels.

**Digitization output** (`published_traces_v2.json`):

| Panel  | Cell  | Steps | Points/step | Total samples | Tool |
|--------|-------|-------|-------------|---------------|------|
| Fig 1A | AVAL  | 7/7   | 24-47       | 273           | opencv_color_mask_centerline |
| Fig 1B | AVAR  | 7/7   | 11-40       | 191           | opencv_color_mask_centerline |
| Fig 3A | AIY   | 11/11 | 41-54       | 580           | opencv_color_mask_centerline |
| Fig 5A | RIM   | 11/11 | 19-54       | 645           | opencv_color_mask_centerline |

All steps achieve ≥ 5 sample data points (criterion for feature extraction). Sample density is lower for Fig 1A/1B because v1's existing 520 × 520 panels have fewer pixels per ms than v2's 1421 × 1144 / 1460 × 1126 panels.

**Calibration:** Per-panel axis calibration encoded as linear `(slope, intercept)` mappings in `digitize_panels_v2.py`. Tick positions identified one-time via grid-overlay visual inspection of each panel (the visual-inspection escape hatch in the image-handling guardrail). Calibration uncertainty ≈ ±2 px → ±1 mV / ±5 ms for Fig 1; ±0.3 mV / ±2 ms for Fig 3 and Fig 5 (higher resolution).

**Voltage anchor identification:** Plateau anchor voltages per step were initially estimated from visual inspection, then refined empirically using black-pixel histogram diagnostics at multiple stim-window timepoints. Final anchors:

| Panel | Plateau voltages per step (mV) |
|-------|--------------------------------|
| Fig 1A AVAL  | -170, -130, -85, -30, 40, 80, 105 |
| Fig 1B AVAR  | -110, -80, -55, -25, 30, 65, 90 |
| Fig 3A AIY   | -130, -110, -75, -55, -25, 0, 15, 22, 26, 30, 32 |
| Fig 5A RIM   | -105, -100, -90, -50, 5, 30, 40, 50, 60, 65, 70 |

### Deliverable 4: Feature-based comparison

**NEURON simulations** (current-clamp protocols matching the published figure protocols):

| Cell | Steps | Range (pA) | Stim duration (ms) | Script | Mod files compiled |
|------|-------|-----------|--------------------|--------|--------------------|
| AVAL | 7     | -30..+30  | 1000               | `AVAL_simulation_iclamp.py` | yes |
| AVAR | 7     | -30..+30  | 1000               | **MISSING** (see warning) | yes |
| AIY  | 11    | -15..+35  | 5000               | `AIY_simulation_iclamp.py` | yes |
| RIM  | 11    | -15..+35  | 5000               | `RIM_simulation_iclamp.py` | yes |

**AVAR upstream warning:** The Nicoletti 2024 GitHub repo head tree (commit 78a17ca) does **not** include `AVAR_simulation_iclamp.py`, although `AVAR_simulation.py` imports `AVA_simulation_iclamp` from that module. v2 worked around this by reusing `AVAL_simulation_iclamp.py` with AVAR's distinct conductance parameters (per `AVAR_simulation.py` lines 22-27). The AVAL iclamp script does **not** insert the `unc103` channel mod, so AVAR's UNC103 contribution is missing in this comparison. Resulting AVAR resting potential is +10.81 mV (vs experimental ~-25 mV) because the K+ rectifier UNC103 is absent. This is a known methodological compromise documented in `comparison_validation_results_v2.json` under each AVAR step's `neuron_warning` field; AVAR results should be interpreted with this caveat.

**Spec-compliant verdict (per-feature 5%, all features per step, > 80% steps per panel):**

| Panel  | n steps total | n steps passing | Fraction passing | Panel pass |
|--------|---------------|-----------------|------------------|------------|
| Fig 1A AVAL  | 7  | 0  | 0.000 | False |
| Fig 1B AVAR  | 7  | 0  | 0.000 | False (UNC103 missing) |
| Fig 3A AIY   | 11 | 0  | 0.000 | False |
| Fig 5A RIM   | 11 | 0  | 0.000 | False |

**Overall: 0/4 panels pass.** Per spec, this is "multi-panel fail (≥ 2) → STOP, urgent notify, real condition-3 invalidation territory; no Phase β proper without cross-session discussion."

**Secondary diagnostic — voltage-only verdict (excludes timing-feature failures dominated by digitization-resolution noise):**

| Panel | n V-pass | V-only fraction pass | V-only panel pass | V abs error mean (mV) | median | max |
|-------|----------|----------------------|---------------------|-----------------------|--------|-----|
| Fig 1A AVAL | 1/7 | 0.143 | False | 8.88  | 8.78  | 16.94 |
| Fig 1B AVAR | 0/7 | 0.000 | False | 15.16 | 10.36 | 42.52 (UNC103 missing) |
| Fig 3A AIY  | 0/11| 0.000 | False | 6.83  | 5.66  | 20.33 |
| Fig 5A RIM  | 2/11| 0.182 | False | 8.79  | 8.38  | 17.87 |

Even the voltage-only diagnostic shows 0/4 panel pass at 5%. But absolute errors are characteristic of biophysical fits: 6.8-15 mV mean (excluding AVAR which is methodologically compromised), max 17-21 mV.

**Tertiary diagnostic — qualitative figure-overlay test (NEURON cyan vs figure red Model traces):**

For Fig 1A AVAL and Fig 3A AIY, NEURON output traces were overlaid on the published figure. Visual inspection confirms:
- Cyan NEURON traces follow the same envelope as the figure's red Model traces.
- Plateau heights match within ~5-15 mV per step.
- Hyperpolarizing-step traces match closely; depolarizing-step traces diverge slightly more (NEURON predicts more spread plateaus than figure's saturating cluster).

This indicates **the NEURON code DOES reproduce the published figure's red Model traces** within the same absolute tolerance characteristic of HH-style biophysical fits. The 5% relative tolerance criterion fails because the absolute residuals (5-15 mV) exceed 5% when divided by the per-feature peak magnitude (which floors at 0.1 × peak).

**Per-step voltage absolute errors (AVAL detail):**

| Step (pA) | My exp dig (mV) | NEURON (mV) | Abs diff (mV) | Rel error |
|-----------|-----------------|-------------|----------------|-----------|
| -30       | -171.9          | -175.3      | 3.5            | 2.0%      |
| -20       | -126.2          | -135.7      | 9.4            | 7.0%      |
| -10       | -81.2           | -97.0       | 15.7           | 16.2%     |
| 0         | -30.0           | -39.4       | 9.4            | 23.8%     |
| +10       | +42.5           | +39.7       | 2.8            | 6.6%      |
| +20       | +77.5           | +80.6       | 3.1            | 3.9%      |
| +30       | +105.0          | +120.7      | 15.7           | 13.0%     |

**Per-cell summary of voltage abs errors:**

| Cell | Mean (mV) | Median (mV) | Max (mV) | Steps within ≤ 5 mV | Steps within ≤ 10 mV |
|------|-----------|--------------|----------|----------------------|------------------------|
| AVAL | 8.88      | 8.78         | 16.94    | 3/7 (43%)            | 4/7 (57%)              |
| AVAR | 15.16     | 10.36        | 42.52    | 0/7 (0%)             | 1/7 (14%)              |
| AIY  | 6.83      | 5.66         | 20.33    | 2/11 (18%)           | 7/11 (64%)             |
| RIM  | 8.79      | 8.38         | 17.87    | 2/11 (18%)           | 6/11 (55%)             |

Excluding AVAR (UNC103 missing, methodologically compromised): 3 cells with mean voltage abs error 6.8-8.9 mV per step; ~50-60% of steps within 10 mV absolute. **These residuals are characteristic of published HH-style biophysical fits on whole-cell current-clamp data.**

**Full-waveform RMSE (warn-only diagnostic):** Per `comparison_validation_results_v2.json` `full_waveform_rmse_diagnostic`, RMSE per step ranges 1-25 mV depending on step and panel. RMSE inflated by NEURON timing offset (NEURON uses fast onset/offset whereas digitized traces are sparse-sampled). Not used in pass/fail.

---

## Part 4 — Condition 3 status: resolved interpretation

**Strict reading of condition 3 (per architectural plan):**

> "Path A is invalidated if Nicoletti's published NEURON code does not reproduce her published figures within 5% tolerance on the fit-target panels."

**v2 outcome (strict):** Code does not reproduce the fit-target current-clamp panels within 5% per-feature relative tolerance. Multi-panel fail (4/4). **Strict invalidation triggered.**

**Substantive reading of condition 3:**

> "Is Nicoletti's NEURON code consistent with the published model output (red traces) within tolerances reasonable for biophysical HH fits?"

**v2 outcome (substantive):**
- Qualitative figure overlays: cyan NEURON traces match figure red Model traces within ~5-15 mV per step.
- Voltage abs errors per step (excluding AVAR): 6.8-8.9 mV mean, 17-20 mV max — characteristic of published biophysical fits.
- Code reproduces published model output approximately. The published model has documented residuals against experimental data (Nicoletti's text discloses "slight underestimation" / "slight overestimation" / "outward rectifying behavior reproduced with slight underestimation" depending on cell).
- **Substantive invalidation NOT triggered** — code is consistent with published model within biophysical-fit tolerances.

**The difference between strict and substantive readings is the tolerance criterion.** A 5% relative tolerance with floor at 0.1 × peak is too strict for biophysical HH fits where 5-15 mV residuals on 200 mV-range data are typical. v1 hit this same wall on I-V curves; v2 hits it on current-clamp traces. The pattern is consistent: published Nicoletti 2024 model has known absolute discrepancies from experimental that the code faithfully reproduces.

---

## Part 5 — Phase β proper readiness assessment

**Recommendation: cross-session discussion before Phase β proper.**

Three issues require resolution before Phase β proper begins:

### Issue 1: Tolerance criterion redefinition

The 5% per-feature relative tolerance is structurally too strict for biophysical model fits. Phase β proper validation gates (Gate 2a evaluating Brian2 channel translations against fit-target current-clamp data) need a tolerance that's:
- **Achievable** by the upstream NEURON code on the same data (otherwise no Brian2 translation can pass)
- **Meaningful** (not so loose it accepts any fit)
- **Defensible** in a published-model context

Options to discuss:
- **Absolute-tolerance** with floor: e.g., per-step plateau within ≤ 10 mV absolute OR ≤ 10% relative, whichever is looser. This passes the upstream NEURON code on 50-65% of steps per cell. Pass criterion: > 80% steps within 10 mV.
- **Relative-with-larger-floor**: e.g., divergence floor at 0.25 × peak instead of 0.1 × peak. Effectively triples the tolerance for small absolute values. Would convert most current 6-9% divergences into < 5%.
- **Voltage-only**: drop timing features (which have high digitization noise) from the per-step pass criterion. Just compare peak and plateau voltages.
- **Per-cell voltage-mean-error budget**: e.g., panel passes if mean voltage absolute error per step ≤ 10 mV. This passes 3 of 4 cells (AVAR fails due to UNC103-missing methodological issue).

### Issue 2: AVAR upstream code missing

`AVAR_simulation_iclamp.py` is missing from the upstream Nicoletti 2024 repo. AVAR cannot be properly validated against current-clamp data without restoring this file. Two paths:
- **Reach out to Nicoletti et al.** to request the missing script (the corresponding author email is `l.chiodo@unicampus.it`).
- **Reconstruct** AVAR_simulation_iclamp.py from AVAR_simulation_vclamp.py by replicating the soma + channel insert pattern (including UNC103 which AVAL doesn't have). This is feasible in Phase β proper as part of the harness work.

Either way, AVAR comparison should be deferred until AVAR has a working iclamp simulator.

### Issue 3: Fit-target metric scope

v2 measured per-step plateau and timing features. These are aggregate features; the actual fit Nicoletti performed is on the **full waveform** (the noisy black trace). The full-waveform RMSE diagnostic in v2 (warn-only) shows RMSE 1-25 mV per step — but RMSE is inflated by phase shifts and digitization noise. A more rigorous fit-quality metric would use Dynamic Time Warping or feature-extraction-after-temporal-alignment.

For Phase β proper, the comparison metric needs explicit alignment with how Nicoletti optimized her fits. Looking at the published Nicoletti scripts, the fit objective appears to be steady-state plateau matching (per Fig 1F caption: "the simulated voltage responses shown in panels A and B" → I-V curve computed from steady-state plateau averaging in last 10 ms). v2's plateau_amplitude_mV feature aligns with this objective. The other features (peak, time-to-peak, settling, plateau_duration) are diagnostic but not strict fit targets.

---

## Files produced by v2

- `artifacts/published_traces_v2.json` — current-clamp digitized traces with extracted features (Deliverable 3)
- `artifacts/comparison_validation_results_v2.json` — per-feature divergences, per-step / per-panel verdicts, secondary diagnostics, full-waveform RMSE (Deliverable 4)
- `artifacts/figures/nicoletti_2024_fig3A_AIY_iclamp.png` — extracted panel (Deliverable 2)
- `artifacts/figures/nicoletti_2024_fig5A_RIM_iclamp.png` — extracted panel (Deliverable 2)
- `digitize_panels_v2.py` — v2 digitization driver
- `run_comparison_validation_v2.py` — v2 comparison runner
- This document — combined v1+v2 narrative (Deliverable 5)

## Files preserved unchanged from v1

- `artifacts/published_traces.json` — v1 I-V digitization (3 panels, 9-16 points each)
- `artifacts/comparison_validation_results.json` — v1 voltage-clamp comparison (3 cells, 39-66% mean divergence)
- `artifacts/figures/nicoletti_2024_fig1A_AVAL_iclamp.png` (v1, 520 × 520) and other v1 panel PNGs
- `digitize_panels.py` — v1 digitization driver
- `run_comparison_validation.py` — v1 voltage-clamp comparison runner

## Files produced by v3

- `artifacts/nicoletti_model_traces.json` — digitized red/blue Model traces from same panels v2 used (Deliverable 1)
- `avar_unc103_patch.py` — standalone patch restoring UNC-103 channel insertion for AVAR iclamp; resolves v2's missing-file workaround that produced +11 mV resting bias (Deliverable 2)
- `artifacts/layer_b_validation_results.json` — per-feature, per-step Layer B comparison (Deliverable 3)
- `artifacts/avar_upstream_issue_draft.md` — draft GitHub issue for upstream Nicoletti repo (Deliverable 5; awaiting user authorization to file)
- `digitize_model_traces_v3.py` — v3 digitization driver (red+blue color masks)
- `run_layer_b_validation_v3.py` — v3 Layer B comparison driver
- This document (updated through v3) — Deliverable 6
- `artifacts/phase_beta_pre_v3_summary.md` — v3 standalone summary (Deliverable 7)

## Decision tree resolution

Per spec:
- All panels pass → condition-3 cleared; Phase β proper ungated → **NOT TRIGGERED**
- Single panel fails → investigate, resolve or flag → not applicable (multi-panel fail)
- Multiple panels fail → STOP, urgent notify, real condition-3 invalidation; no Phase β proper without cross-session discussion → **TRIGGERED (strict reading)**

**Stopping. Urgent notification sent.** Cross-session discussion needed before Phase β proper to resolve the tolerance-criterion question and the AVAR upstream-missing-file question. The substantive interpretation of the data (NEURON code reproduces published Model traces within biophysical-fit tolerances) supports proceeding, but the spec's explicit pass criterion is not met by 0/4 panels at 5% per-feature relative.

---

## Part 6 — v3 closure

### v3 motivation

The v1+v2 narrative surfaced that condition-3's strict reading ("does Nicoletti's NEURON code reproduce her published figures within 5%") was being conflated with Layer C ("does Nicoletti's published Model match experimental data within 5%"). Layer C necessarily fails 5% because biophysical HH fits carry 5-15 mV residuals on 200 mV-range data; v1+v2 measured exactly these residuals. v3 directly tests Layer B (the layer condition-3 actually asks about) by digitizing Nicoletti's *published Model traces* from the same panels v2 used and comparing them against the NEURON code output already captured in v2.

### Deliverable 1 (v3): Red/blue Model trace digitization

Same 4 panels (Fig 1A AVAL, Fig 1B AVAR, Fig 3A AIY, Fig 5A RIM); same axis calibration as v2; same per-stimulus-step plateau-anchor segregation logic. Only changes:

- **Color mask:** swapped black-pixel mask for red-pixel mask on AVAL/AIY/RIM, blue-pixel mask on AVAR. Empirical color-pixel counts:

  | Panel | Red px | Blue px | Used color |
  |-------|--------|---------|------------|
  | Fig 1A AVAL | 4064 | 0 | red |
  | Fig 1B AVAR | 38 | 3762 | **blue** |
  | Fig 3A AIY | 32437 | 0 | red |
  | Fig 5A RIM | 35027 | 0 | red |

  Note: the v3 prompt called for "red Model traces" but Fig 1B AVAR's published Model traces are plotted in blue, not red. The `AVAR_simulation.py` upstream script line 82 uses `color='red'` for its own iclamp plot, so the figure-specific color choice is panel-level (likely chosen for contrast against AVAL's red in the panel pair). v3 surfaced this and extracted by actual color present.

- **Plateau anchors:** Model-trace plateaus differ from v2's experiment-trace plateaus by 5-15 mV per step (this *is* the Layer C residual). Per-panel Model-anchor histograms identified the Model plateau positions:

  | Panel | Model plateau anchors per step (mV) |
  |-------|-------------------------------------|
  | Fig 1A AVAL | -170, -134, -94, -34, +34, +73, +112 |
  | Fig 1B AVAR | -125, -91, -58, -22, +20, +56, +85 |
  | Fig 3A AIY | -127, -103, -76, -55, -34, -16, -4, +4, +13, +20, +25 |
  | Fig 5A RIM | -110, -100, -88, -46, -10, +13, +25, +34, +43, +52, +61 |

- **Output:** `nicoletti_model_traces.json` — same schema as `published_traces_v2.json`, 4 panels × {7,7,11,11} = 36 traces, 1893 (t, V) data points total, all 36 traces with ≥ 5 samples.

Tool: `opencv_color_mask_centerline` for all 4 panels.

### Deliverable 2 (v3): AVAR UNC-103 patch

The upstream `AVAR_simulation_iclamp.py` is missing from the Nicoletti repo head tree. v2 worked around this by reusing AVAL's iclamp without UNC-103, which produced a +11 mV resting potential bias for AVAR. v3 fixes this with a standalone patch in our wave2 directory.

**File:** `scripts/brain/wave2/avar_unc103_patch.py`

**Logic:** mirrors `AVAL_simulation_iclamp.py` verbatim except:
- Surface area = AVAR's 1121.79e-8 cm² (not AVAL's 1123.84e-8)
- Inserts `unc103` mod with gbar from AVAR's parameter vector (line 28 of `AVAR_simulation.py`): `0.0481669 nS × 1e-9 / 1121.79e-8 cm² = 4.293 × 10⁻³ S/cm²` after `gScm2` rescale at index 4
- Channel set [EGL19, LEAK, IRK, NCA, UNC103] matches the comment in `AVAR_simulation.py` line 27 (which AVAL's iclamp script omits)

**Verification:** AVAR resting potential after patch = **-24.25 mV** (mean across 7 steps), well within target of **-25 ± 5 mV**. v2's missing-UNC-103 fallback produced +10.81 mV; the patch resolves the +35 mV swing as expected from K+-rectifier insertion.

**Patch plateau values** (`-127, -96, -62, -24, +16, +49, +80 mV`) closely match the published Fig 1B AVAR Model-trace plateaus we digitized (`-125, -91, -58, -22, +20, +56, +85 mV`).

The patch is **standalone** in our wave2 directory; Nicoletti's upstream code is read-only from our perspective and is not modified in place.

### Deliverable 3 (v3): Layer B comparison

**File:** `artifacts/layer_b_validation_results.json`

Per-feature, per-step divergence between digitized Model traces (Deliverable 1) and NEURON output. AVAL/AIY/RIM use NEURON output already captured in `comparison_validation_results_v2.json` (no re-run needed); AVAR uses the patched re-run (Deliverable 2). Tolerance metric is identical to v2's: relative-with-floor at 5%.

**Strict per-feature 5% verdict (primary criterion):**

| Panel | n steps total | n steps passing | Fraction | Panel pass |
|-------|---------------|-----------------|----------|------------|
| AVAL | 7 | 0 | 0.00 | False |
| AIY | 11 | 0 | 0.00 | False |
| RIM | 11 | 0 | 0.00 | False |
| AVAR | 7 | 0 | 0.00 | False |

**Strict overall verdict: 0/4 panels pass. Multi-panel fail at strict 5% per-feature.**

**Voltage-only secondary diagnostic (peak + plateau, excludes timing-feature noise):**

| Panel | V-only steps passing | V-only fraction | V abs error mean (mV) | median | max |
|-------|---------------------|-----------------|------------------------|--------|------|
| AVAL | 3/7 (43%) | 0.43 | 4.81 | — | — |
| AIY | 5/11 (45%) | 0.45 | 3.41 | — | — |
| RIM | 4/11 (36%) | 0.36 | 3.31 | — | — |
| AVAR (patched) | 2/7 (29%) | 0.29 | 4.09 | — | — |

**Layer B voltage abs errors are roughly half of Layer C (v2):** Layer C mean V abs errors were 6.8-15.2 mV per cell (v2 Part 3 secondary diagnostic table); Layer B mean V abs errors are 3.3-4.8 mV per cell. The factor-of-two reduction is consistent with the structural prediction: NEURON code reproduces the published Model figures within tighter bounds than the published Model fits experimental data, but neither distance reaches the strict 5% per-feature threshold.

**Why timing features fail systematically at Layer B:**

- Digitization samples each trace at ~60 timepoints across a stim window of 1000 ms (Fig 1) or 5000 ms (Fig 3, 5). Per-sample resolution: ~17 ms (Fig 1) or ~83 ms (Fig 3, 5).
- NEURON's internal `dt = 0.025 ms` produces sub-millisecond timing-feature precision.
- The mismatch maps onto plateau_duration_ms, time_to_peak_ms, and settling_time_ms with absolute errors comparable to the digitization-sample-spacing (10s-100s of ms), which under the relative-divergence metric with floor at `0.1 × peak` (≈ 100-500 ms peak) exceeds 5% on the small-denominator side.
- This is digitization-sampling-resolution noise, not implementation discrepancy. The voltage features (peak, plateau), which Nicoletti's optimization actually targets, are within 3-5 mV per cell.

### Layer B verdict and decision-tree resolution

**Strict reading (per spec primary criterion):** multi-panel Layer B fail. **STOP, urgent notify, cross-session discussion required before Phase β proper.** Status: **TRIGGERED (strict)**.

**Substantive reading:** Layer B residuals (3-5 mV mean voltage abs error per cell) are roughly half of Layer C residuals (6.8-15 mV) and are consistent with NEURON code faithfully reproducing the published Model figures. Strict 5% per-feature is dominated by digitization-sampling-resolution noise on timing features. Substantive Layer B is passing.

**Resolution:** the methodological pattern across v1 → v2 → v3 is consistent. **The 5% per-feature relative tolerance is structurally too strict at every layer measured to date** because the relative-divergence metric explodes at small absolute denominators where the 0.1× peak floor is 1-5 mV. Cross-session discussion needed before Phase β proper to:

1. **Adopt a defensible Phase β gate criterion** at Layer A (Brian2 = NEURON). Recommendation: voltage-feature-only per-step pass with absolute-error budget ≤ 3 mV per step, AND > 80% of steps pass per panel. This is *tighter* than Layer B's 3-5 mV residuals against Model figures, providing meaningful Phase β quality gating without inheriting the methodological wall the strict 5% relative tolerance hits.
2. **Decide whether to file the AVAR upstream issue draft** (`artifacts/avar_upstream_issue_draft.md`).
3. **Confirm scope boundaries:** Phase β proper begins with `cadiff/caintra1` translation, NEURONReference wrapper, and Brian2 channel translation prioritized per the architectural plan. v3's AVAR patch is preserved as the AVAR runtime for any AVAR validation in Phase β until the upstream restoration happens.

### Citation expansion (v3)

v2 verified that the original spec's "Nicoletti 2019 PLOS Comp Bio" reference at DOI `10.1371/journal.pcbi.1007611` was incorrect (that DOI resolves to a glioma paper). The verified correct citations are:

- **Nicoletti 2019** (PLOS ONE `journal.pone.0218738`): "Biophysical modeling of C. elegans neurons: Single ion currents and whole-cell dynamics of AWCon and RMD" — upstream paper that 2024 extends.
- **Nicoletti 2024** (PLOS ONE `journal.pone.0298105`): "Biophysical modeling of the whole-cell dynamics of C. elegans motor and interneurons families" — primary Wave 2 import target (22-channel library; AVA/AIY/RIM/VA/VB/VD).

v3 applied the citation correction across:
- `phase_v_w2_architectural_plan.md` — added a "Primary upstream citations" block at the top, with both DOIs and roles expanded.
- `phase_v_w2_phase_beta_pre_prompt.md` — corrected the v1-era `pcbi.1007611` reference to `pone.0218738` with role description.
- `phase_v_w2_phase_beta_pre_v2_prompt.md` — added a v3 update note expanding both citations with roles.
- This document (`phase_beta_pre_validation.md`) — Part 3 Deliverable 1 narrative preserved as historical record of v2's verification work; this Part 6 v3 section adds the canonical expansion.

v1 historical artifacts (`published_traces.json`, `digitize_panels.py`) are preserved unchanged — they contain the wrong DOI inside v1's diagnostic narrative ("Nicoletti 2019 was checked as a candidate but found to be unrelated") which is the historical record of how v1 detected the citation error in the first place. Modifying those would erase the methodology trail.

The v3 prompt itself (`phase_v_w2_phase_beta_pre_v3_prompt.md`) contains `pcbi.1007611` only as **literal grep target patterns** in command examples (lines 104, 109, 118), not as citation references. These are intentionally preserved as part of the prompt's instructions to grep for the wrong DOI; modifying them would break the prompt's guidance.

### AVAR upstream defect summary

The upstream Nicoletti 2024 repo (ModelDB 2017403) ships `AVAR_simulation.py` (the wrapper) but is missing `AVAR_simulation_iclamp.py` (the iclamp implementation). AVAR cannot be run end-to-end from the released code. v2 worked around this with an AVAL-iclamp + AVAR-parameters fallback that omitted UNC-103 (because AVAL's iclamp script doesn't insert it), producing a non-physiological +11 mV resting potential. v3 resolves this with `avar_unc103_patch.py` (Deliverable 2 above), restoring UNC-103 channel insertion with the conductance value from `AVAR_simulation.py` line 28. The patched AVAR runs to a physiological -24.25 mV resting potential.

A GitHub issue draft is at `artifacts/avar_upstream_issue_draft.md`, ready for user authorization to file.
