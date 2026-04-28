# Phase β-pre v2 — Fit-target ground-truth verification

You are the engineering session executing **Phase β-pre v2** of Wave 2 of the C. elegans biophysical simulator project. This is a focused continuation work block that corrects v1's metric error and properly closes Phase α deliverable 3.

You have full user permission for package installs, file creation, and PDF/figure access for this phase.

---

## HARD CONSTRAINT — image handling guardrail (read first, do not skip)

**v1 terminated on the agent's output return path hitting a 2000px image dimension limit.** The cause was figure PNGs loaded into the agent's multimodal context. v2's time-series work is more image-intensive than v1 — multiple overlapping curves per panel, finer-grained inspection, more iterations. Without an explicit guardrail, v2 hits the same wall.

**Hard constraint, not preference:**

1. **Process figures programmatically only.** Use `pdftoppm` / `pdfimages` to extract figures from PDFs to disk. Use `OpenCV` (cv2), `PIL` (Pillow), `plotdigitizer`, and `numpy` to read images **from disk** and output **JSON or numerical results**. Never rely on multimodal vision for the primary digitization workflow.

2. **Do NOT use the `Read` tool on figure PNGs.** The Read tool loads images into multimodal context. Even small panel extracts can be > 1500px on a side and trigger the dimension limit. Scripts loading images via `cv2.imread()` or `PIL.Image.open()` are fine — those don't enter the agent's context.

3. **Visual inspection escape hatch (use sparingly):** if a specific debugging step genuinely requires the agent to see a figure (e.g., diagnosing why automated detection failed on a panel), downscale to ≤ 1500px on the longest side via PIL first:

   ```python
   from PIL import Image
   img = Image.open("figures/source.png")
   img.thumbnail((1500, 1500))
   img.save("figures/_inspect/source_small.png")
   ```

   Then `Read` the downscaled version. Default mode is programmatic; visual inspection is exception, not rule.

4. **The Phase α subagent in v1 hit this limit during digitization iteration. Avoid the same path:** when `plotdigitizer` or your custom OpenCV extraction produces a result, write the JSON output directly. Inspect numerical outputs (per-pixel coordinates, detected features, divergence metrics) rather than rendered figures wherever possible.

This constraint is load-bearing for v2 completion. Hitting the limit again would terminate the session before deliverables land.

---

## Strategic context (read after the guardrail)

Phase β-pre v1 ran end-to-end and produced:

- 14 panel PNGs extracted from Nicoletti 2024 PDF (in `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/`)
- `published_traces.json` digitizing 3 I-V curve panels (Fig 1F AVAL, Fig 3D AIY, Fig 5D RIM)
- `comparison_validation_results.json` showing 3-panel fail (39%, 57%, 63% mean divergences)
- Working `digitize_panels.py` infrastructure (OpenCV template-matching + manual grid-reading methodology)

**v1's verdict was empirically correct against the 5% tolerance, but applied the wrong metric.** Nicoletti 2024's text (line 502 of extracted text):

> "The models were fitted on experimental current-clamp data obtained from [29], and shown in black in panels A and B."

For AIY (line 580):

> "The model was fitted on experimental current-clamp data obtained from [30] and shown in black in panel A."

**Fig 1A/1B (AVAL/AVAR), Fig 3A (AIY), Fig 5A (RIM) are the fit-target panels.** Fig 1F/3D/5D (I-V curves) are post-hoc predictions. Nicoletti's body text explicitly discloses I-V divergences:

> "The main differences with the experimental data are observed in AVAL, where the simulated currents are slightly overestimated for hyperpolarizing stimuli and underestimated for depolarizing stimuli (Fig 1D and 1F)."

> "Voltage-clamp simulations show that the model also reproduces the outward rectifying behavior of the average whole-cell currents (Fig 3B and 3D, red lines), but with a slight underestimation of the steady-state current."

v1's NEURON output reproduces these documented divergences — i.e., **Nicoletti's NEURON code is consistent with her published model**. v2 tests whether her NEURON code reproduces the **fit-target current-clamp data** within tolerance — the actual condition-3 invalidation check.

**Read before starting any substantive work:**

1. `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md` — especially "What would invalidate Path A?" subsection (condition 3)
2. `~/Desktop/website/personalwebsite/scripts/brain/wave2/phase_alpha_report.md` — Phase α completion report
3. `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/published_traces.json` — v1's digitization output (selection rationale + I-V data)
4. `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/comparison_validation_results.json` — v1's comparison (note this is fail against the wrong metric)
5. `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/source_pdfs/nicoletti_2024_text.txt` — extracted paper text including all Fig 1/3/5 captions and methods
6. `~/Desktop/website/personalwebsite/scripts/brain/wave2/digitize_panels.py` — v1's digitization infrastructure (reusable for v2)

---

## Five deliverables

### Deliverable 1: Citation verification and correction

Verify the actual Nicoletti 2019 C. elegans paper. v1 surfaced that "Nicoletti 2019 PLOS Comp Bio (DOI 10.1371/journal.pcbi.1007611)" is incorrect — that DOI points to a glioma paper. v1 hypothesized the actual paper is at PLOS ONE (DOI `journal.pone.0218738`). **v3 update:** v2 confirmed the hypothesis. The correct citations are: **Nicoletti 2019** (PLOS ONE `journal.pone.0218738`, AWCon/RMD biophysical models — upstream paper that 2024 extends) and **Nicoletti 2024** (PLOS ONE `journal.pone.0298105`, 22-channel library covering AVA/AIY/RIM/VA/VB/VD — primary Wave 2 import target).

**Sequence:**

1. **Verify hypothesis first.** Fetch metadata for `10.1371/journal.pone.0218738` via DOI resolver / journal site / `requests` to PLOS API. Confirm: title contains "C. elegans" and is by Nicoletti et al. If hypothesis fails, search PubMed / Google Scholar for "Nicoletti 2019 C. elegans biophysical model" until the correct paper is identified. Do not proceed to step 2 until verified.

2. **Grep all artifacts for the incorrect citation.** Find all files referencing "Nicoletti 2019" or the incorrect DOI:

   ```bash
   grep -rn "Nicoletti 2019\|10.1371/journal.pcbi.1007611\|pcbi.1007611" \
     /home/rohit/Desktop/website/personalwebsite/scripts/brain/artifacts/ \
     /home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/
   ```

   Report the file list before any updates.

3. **Update each file consistently** with the verified correct DOI. Include the verified paper title, journal, year, and DOI as a footnote or inline citation per file's existing convention.

**Acceptance:** verified DOI documented in v2 validation report; all incorrect references in artifact files corrected; grep after update returns zero hits for the incorrect DOI.

### Deliverable 2: Current-clamp panel extraction

Source PDF already on disk: `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/source_pdfs/nicoletti_2024_plosone.pdf`. v1 extracted Fig 1A and 1B PNGs already (`nicoletti_2024_fig1A_AVAL_iclamp.png`, `nicoletti_2024_fig1B_AVAR_iclamp.png`).

Extract additionally:

- Fig 3A (AIY current-clamp, fit-target panel)
- Fig 5A (RIM current-clamp — verify caption confirms it as fit-target by reading the extracted text file)

Use `pdftoppm` or `pdfimages` for extraction. Save panel crops to `figures/`. **Do NOT Read the extracted PNGs into agent context.** Inspect via dimension checks programmatically (`PIL.Image.open(path).size`).

**Acceptance:** four panel PNGs on disk (Fig 1A AVAL, Fig 1B AVAR, Fig 3A AIY, Fig 5A RIM), all current-clamp time-series.

### Deliverable 3: Time-series digitization

For each of the four panels, extract experimental black trace(s) at protocol-specified timepoints. Output to `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/published_traces_v2.json` (separate file from v1's I-V data; v1's file is preserved as historical record).

**Tool hierarchy (try in order, document which worked per panel):**

1. **`plotdigitizer`** — try first since infrastructure from v1 exists. May not handle multiple overlapping curves well; if it fails on a panel, escalate to step 2.
2. **Custom OpenCV color-mask extraction** — mask black pixels in the trace region, find centerline per protocol-aligned x position, sample at protocol timepoints. v1 built similar infrastructure; reuse `digitize_panels.py` patterns.
3. **Manual grid-reading** — overlay axis-tick gridlines, sample visible points. v1 used this for AIY/RIM I-V curves; same methodology applies to traces with sparse sampling (e.g., 50-100 timepoints per trace).

If all three fail on a specific panel, **STOP and surface to user** — don't escalate to heavier tools (WebPlotDigitizer CLI install) without cross-session discussion.

Document tool used per panel in the JSON output.

**Per-panel JSON structure (extends v1's format):**

```json
{
  "id": "nicoletti_2024_fig1A_AVAL",
  "shows": "experimental_overlay",
  "cell": "AVAL",
  "protocol": "current_clamp",
  "protocol_detail": "7 current steps from -30 pA to +30 pA, duration 1000 ms (Fig 1 caption)",
  "fit_target": true,
  "experimental_data_origin": "Liu et al. 2018, ref [29] in Nicoletti 2024 (Mellem 2008 lineage)",
  "x_axis": {"label": "Time", "units": "ms", "scale": "linear"},
  "y_axis": {"label": "Voltage", "units": "mV", "scale": "linear"},
  "traces": [
    {
      "stimulus_pA": -30.0,
      "data": [{"t": 0, "v": -45.2}, {"t": 100, "v": -78.5}, ...],
      "n_points": 100,
      "tool": "opencv_color_mask"
    },
    ...
  ],
  "extracted_features": {
    "peak_voltage_mV_per_step": {-30: -85.3, -20: -78.1, ..., 30: -10.4},
    "plateau_amplitude_mV_per_step": {...},
    "plateau_duration_ms_per_step": {...},
    "time_to_peak_ms_per_step": {...},
    "settling_time_ms_per_step": {...}
  },
  "digitization_notes": "..."
}
```

**Acceptance:** four panels digitized, traces saved with extracted features computed, tool used documented per panel.

### Deliverable 4: Feature-based comparison validation

Run Nicoletti's NEURON code under matching current-clamp protocols. For each cell:

- Match the protocol from her paper (AVAL: 7 steps -30 to +30 pA × 1000 ms; AIY: 11 steps -15 to +35 pA × 5000 ms; RIM: per Fig 5 caption — read from extracted text)
- Capture NEURON voltage trace at protocol-aligned timepoints
- Extract the same features from NEURON output as from digitized data

**Comparison metric:**

Per-feature divergence using v1's relative-tolerance-with-floor formula:

```python
def feature_divergence(measured, reference, peak):
    return abs(measured - reference) / max(abs(measured), abs(reference), 0.1 * peak)
```

Per-feature pass: divergence ≤ 0.05 (5% relative). Per-step pass: all features pass for that current step. Per-panel pass: > 80% of steps pass (looser than v1's per-point ≥ 90% — feature-based has fewer comparison points so 80% is reasonable; document the choice). Per-cell pass: panel passes.

**Secondary diagnostic (warn-only, not gate):**

Compute full-waveform RMSE between NEURON and digitized traces. Report alongside but don't fail on it — phase shifts and digitization noise inflate full-waveform RMSE even when dynamics match. Useful as diagnostic ("RMSE high but features pass" → dynamics correct, traces visually overlap with phase drift).

**Decision tree:**

- All panels pass (extracted features within 5% across cells) → condition-3 properly cleared; Phase β proper ungated
- Single panel fails → investigate (digitization error? protocol mismatch? Nicoletti's NEURON uses different parameters than published?); resolve or flag in v2 doc
- Multiple panels fail (≥ 2) → STOP, urgent notify, real condition-3 invalidation surfaced; no Phase β proper without cross-session discussion

**Acceptance:** per-feature comparison computed for all four panels; per-step / per-panel / per-cell verdicts documented; full-waveform RMSE reported as diagnostic.

### Deliverable 5: Combined v1+v2 validation document + Phase α addendum

**Single combined document** at `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/phase_beta_pre_validation.md`. Structure:

1. **Executive verdict** — pass / fail / partial; condition-3 status (cleared / invalidated / partial)
2. **v1 narrative** — what was attempted, what was measured, why the metric was wrong (with Nicoletti text quotes), what infrastructure was produced (PDFs extracted, plotdigitizer infrastructure, NEURON harness validation, methodology for figure inspection)
3. **v1 → v2 transition** — methodological correction surfaced, fit-target panels identified, why current-clamp not voltage-clamp is the right metric
4. **v2 results** — citation correction details, panel extraction, digitization tool used per panel, feature-extraction methodology, per-feature comparison results, per-cell verdicts
5. **Condition-3 status** — resolved interpretation (cleared if v2 passes; genuine invalidation territory if v2 fails on multiple panels)
6. **Phase β proper readiness assessment** — does v2 outcome support proceeding? Or are there remaining methodological items to settle?

**v1 work credited as infrastructure-establishing**, not throwaway. Honest framing: v1 produced reusable infrastructure (PDFs, plotdigitizer, NEURON harness, figure-inspection methodology) AND caught its own methodology error. v2 applies corrected methodology against the established infrastructure. The cross-session adversarial review pattern caught its own error in real time — that's a methodology success, not a wasted phase.

**Phase α report addendum:** add a section to `~/Desktop/website/personalwebsite/scripts/brain/wave2/phase_alpha_report.md` titled "Phase β-pre addendum — deliverable 3 closure" referencing the combined validation doc. Document:

- v1 partial closure (deterministic-self-consistency only)
- v2 fit-target closure (per outcome)
- Final deliverable 3 status

**Acceptance:** combined doc written; Phase α addendum added; both reference each other consistently.

---

## Methodology continuity

Maintain the pre-flight pushback discipline:

- **Pre-flight:** your VERY FIRST OUTPUT before any tool calls beyond reading the spec files should be a brief plan summary covering: (a) confirmation you've read the image-handling guardrail and how you'll comply, (b) approach to citation verification (which DOI resolver/API), (c) any concerns or questions before execution. Only after that pre-flight step should you begin work.

- **Mid-flight:** if citation verification fails (the hypothesized DOI doesn't point to a Nicoletti C. elegans paper), surface immediately. If panel extraction surfaces unexpected structure (Fig 5A isn't actually current-clamp, or Nicoletti's RIM model wasn't fit on current-clamp), surface immediately.

- **Stop-and-ask:** if all three digitization tools fail on any panel, if comparison surfaces > 5% divergence on multiple panels (real condition-3 invalidation territory), if any deliverable's acceptance criteria can't be met, stop and surface.

- **Empirical grounding:** trust what you observe in Nicoletti's actual paper and code over what the spec describes. The spec is informed by v1 findings but the source of truth is the paper.

- **No time estimates** in the report or notifications.

---

## Notifications

Use `~/bin/notify` for milestones:

- v2 started — one-line plan summary, including image-handling-guardrail acknowledgment
- Citation verified — one-line correct DOI summary
- Citation correction applied to N files — one-line summary
- Panel extraction complete (4 panels on disk) — one-line summary
- Digitization complete (per-panel tool used) — one-line summary
- Feature extraction complete — one-line summary
- Comparison validation complete — pass/fail per cell + overall verdict
- v2 complete — one-line result (pass: condition-3 cleared, Phase β ungated; fail: real invalidation surfaced)
- Blocked / urgent — use `urgent` priority

Keep messages under ~150 chars.

---

## Output format

Files produced by v2:

- `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/published_traces_v2.json` — current-clamp digitized traces with extracted features (deliverable 3)
- `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/comparison_validation_results_v2.json` — per-feature divergences and per-cell verdicts (deliverable 4)
- `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/phase_beta_pre_validation.md` — combined v1+v2 narrative (deliverable 5)
- `~/Desktop/website/personalwebsite/scripts/brain/wave2/phase_alpha_report.md` — addendum added (deliverable 5)
- Citation corrections applied across artifact files (deliverable 1)
- Panel PNGs in `figures/` directory (deliverable 2)

v1 files preserved unchanged: `published_traces.json`, `comparison_validation_results.json`, existing figure PNGs.

---

## Scope discipline — what is NOT in v2

Do NOT do any of these:

- **Translating any of Nicoletti's channels to Brian2** — Phase β proper
- **Building additional harness infrastructure beyond what v1 produced** — v2 reuses v1's `digitize_panels.py` patterns and Phase α's harness code
- **Investigating Nicoletti's code beyond verifying current-clamp reproduction** — if her code reproduces fit-target data, the work is done
- **License verification on Nicoletti** — production-prep gate
- **Cadiff + caintra1 translation** — Phase β proper
- **Re-doing v1's I-V curve digitization** — preserved as historical record; not the right metric but the work stands
- **Brian2 implementation of any kind** — Phase β proper

If you find yourself doing any of the above, stop and surface to user.

---

## Reference assets on disk (from v1)

- Source PDF: `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/source_pdfs/nicoletti_2024_plosone.pdf`
- Extracted paper text: `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/source_pdfs/nicoletti_2024_text.txt`
- Per-page PDF rasters: `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/source_pdfs/nicoletti_2024_p-*.png` (DO NOT READ via Read tool — process programmatically only)
- v1 panel PNGs: `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/nicoletti_2024_fig*.png`
- v1 digitization script: `~/Desktop/website/personalwebsite/scripts/brain/wave2/digitize_panels.py`
- v1 NEURON harness: `~/Desktop/website/personalwebsite/scripts/brain/wave2/run_comparison_validation.py`
- Nicoletti 2024 NEURON code: `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`
- Isolated venv: `~/venvs/wave2-neuron/` (NEURON, Brian2, plotdigitizer, OpenCV, PIL, numpy, scipy, matplotlib already installed)

---

## Final instructions

Execute v2 end-to-end as a single session. Begin with pre-flight: read the spec, confirm image-handling guardrail compliance, plan citation verification approach, surface any concerns. Only after pre-flight should you start substantive work.

If v2 clears (pass verdict): condition-3 properly cleared; Phase β proper begins as separate work block with revised priority list (cadiff/caintra1 first, then NEURONReference wrapper, then EGL-19, then Gate 2a evaluation against fit-target current-clamp data).

If v2 fails (real condition-3 invalidation): cross-session discussion before any further Wave 2 work.

The other cross-session adversarial review sessions remain idle until v2 completes. Your output is what they review.

**Begin with pre-flight pushback. Acknowledge the image-handling guardrail explicitly in your first response.**
