# Phase β-pre — Ground-truth verification work block

You are the engineering session executing **Phase β-pre** of Wave 2 of the C. elegans biophysical simulator project. This is a focused continuation work block that closes Phase α's deliverable 3 properly before Phase β proper begins.

You have full user permission for package installs, file creation, and PDF/figure access for this phase.

---

## Strategic context (read first)

Phase α (the prior work block) delivered six items including running Nicoletti 2024's NEURON code end-to-end on AVAL/AIY/RIM with bit-exact determinism between consecutive runs. **However:** the spec asked for "≥ 2-3 traces reproduce within 1% against published-figure values." Nicoletti's repo doesn't ship her published numerical traces — only the protocol scripts. Phase α operationalized as deterministic-self-consistency, which validates that her code runs and is reproducible but does NOT validate that her code reproduces her *published* numerical claims.

This is a real gap. Phase α didn't fully clear the **condition-3 invalidation check** ("Nicoletti's models don't reproduce Mellem 2008 cellular targets in NEURON either" → if running her code locally produces voltage-clamp traces that don't match published experimental values, the translation effort is moot).

Phase β-pre closes this gap by digitizing published experimental traces and comparing them against Nicoletti's NEURON output under matching protocols.

**Read before starting any substantive work:**

1. `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md` — especially "What would invalidate Path A?" subsection. Conditions 3 and 6 are particularly relevant here.
2. `~/Desktop/website/personalwebsite/scripts/brain/wave2/phase_alpha_report.md` — Phase α completion report including the "interpretation note" on deliverable 3 partial closure.
3. `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_phase_alpha_prompt.md` — to understand what Phase α was supposed to produce vs what it actually produced.

---

## Critical methodology distinction — experimental vs simulation panels

This refinement is the most load-bearing aspect of Phase β-pre. Read carefully.

The Phase β-pre validation logic is: digitize published traces → run Nicoletti's NEURON code under matching protocols → compare. **The diagnostic value depends entirely on what the digitized panel actually shows.**

- **Experimental data panels** (e.g., Mellem 2008 voltage-clamp recording from real AVA neurons; Nicoletti's own electrophysiology if she did any; experimental traces overlaid with simulation comparison) — these are true external ground truth. Comparing Nicoletti's NEURON code to digitized experimental data is the real condition-3 invalidation check.

- **Simulation-output-only panels** (just Nicoletti's NEURON model running, with no experimental overlay) — digitizing these and comparing to running her code is **self-validation**. Her code matching her own simulation figure tells us nothing beyond what Phase α already established (bit-exact determinism between consecutive runs). This does NOT close deliverable 3.

**Selection criterion (load-bearing):** panels must show experimental data, OR experimental data overlaid with simulation. Pure simulation-output panels do not constitute ground-truth validation and must be excluded from the selection.

**Fallback path:** if Nicoletti 2019/2024 don't contain sufficient experimental-data panels for the cells in our 7-channel essential set (EGL-19, SLO-1+EGL-19 coupled, SLO-1 isolated, SHK-1, SHL-1, NCA, KQT-3 contexts; cells AVAL, AIY, RIM, AVAR, AWC, RMD, VA5, VB6, VD5), digitize **Mellem 2008's voltage-clamp traces directly** for AVA specifically. Mellem 2008 is the upstream experimental source Nicoletti's AVA model claims to reproduce; digitizing her figures gives unambiguous experimental ground truth.

**Both paths are acceptable, with priorities:**

- **Preferred:** Nicoletti experimental or experimental-overlay panels (they are the model's stated reference targets, so matching them is what Path A's claims rest on)
- **Supplement:** Mellem 2008 voltage-clamp traces for AVA — this strengthens Gate 2b downstream validation regardless of whether Nicoletti's panels are sufficient
- **Fallback:** Mellem 2008 alone if Nicoletti's papers don't cover the cells we need

---

## Four-step deliverables

### Deliverable 1: Figure inspection and selection rationale

Inspect Nicoletti 2019 (PLOS Computational Biology 2019) and Nicoletti 2024 (PLOS ONE 2024 / ModelDB 2017403) papers. Source PDFs:

- Nicoletti 2019: `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0218738` (PLOS ONE 14(7): e0218738). Open-access (CC BY). AWCon/RMD biophysical models — upstream paper that 2024 extends. **NOTE (corrected v3):** earlier versions of this prompt referenced DOI `10.1371/journal.pcbi.1007611` as "Nicoletti 2019 PLOS Comp Bio" — that DOI resolves to a glioma paper unrelated to C. elegans. The correct paper is at PLOS ONE under `pone.0218738`.
- Nicoletti 2024: `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298105` (PLOS ONE 19(3): e0298105). Open-access. 22-channel library; AVA/AIY/RIM/VA/VB/VD cells; primary Wave 2 import target.
- Mellem 2008 (J Neurophysiol): `https://journals.physiology.org/doi/10.1152/jn.00071.2008` — likely accessible via ResearchGate, PubMed Central, or institutional access. If not freely accessible, surface and we'll discuss.

Download the PDFs. If the Nicoletti repo at `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/` already contains PDFs, use those. Save downloaded PDFs to `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/source_pdfs/`.

For each candidate panel, document:

- Panel identifier (paper, figure number, panel letter)
- What the panel shows (caption text + your interpretation)
- **Critical: is this experimental data, simulation output, or experimental-overlay-with-simulation?** This is the load-bearing classification.
- Which cell(s) the panel covers (must be in our 7-channel essential set or relevant for Path A's stated reference targets)
- Axis labels, units, log/linear, time/voltage/current ranges
- Resolution / digitization feasibility (panels < 200×200 px or with crowded curves are harder)

**Select 2-3 panels** that meet:

1. Panel shows experimental data or experimental-overlay (load-bearing — pure simulation panels excluded)
2. Panel covers a cell in our 7-channel essential set (or covers AVA via Mellem 2008 fallback)
3. Panel has sufficient resolution for digitization

Rationale for the selection: explicit reasoning per panel why this one, not another. Document in deliverable 4.

**If no experimental panels are available across Nicoletti 2019, 2024, AND Mellem 2008:** STOP and surface to user. This is a load-bearing finding — it would mean the published literature doesn't contain the ground-truth references Path A's validation strategy assumes.

### Deliverable 2: Digitization

Use `plotdigitizer` (Python CLI, install via pip) as the default tool:

```bash
pip install plotdigitizer
```

If plotdigitizer fails on a specific panel (e.g., log axes that confuse it, multiple curves overlaid, low resolution causing axis-detection failure), **surface as a pre-flight finding** and we'll discuss alternatives (custom OpenCV/PIL extraction, web-based tool fallback). Don't silently switch tools.

For each selected panel:

- Extract figure from source PDF (use `pdftoppm` or `pdfimages` to extract page images, then crop to panel)
- Save extracted panel image as PNG to `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/[paper]_fig[N][panel].png`
- Run plotdigitizer with explicit axis calibration (mark known axis points)
- Save digitized output as JSON

Output JSON file: `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/published_traces.json` (name chosen to be content-neutral — works whether Nicoletti or Mellem dominates the selection).

JSON structure (per panel):

```json
{
  "panels": [
    {
      "id": "nicoletti_2024_fig3a",
      "source": {
        "paper": "Nicoletti et al. 2024",
        "figure": "3",
        "panel": "A",
        "doi": "10.1371/journal.pone.0298105"
      },
      "shows": "experimental | overlay | simulation_only",
      "cell": "AVAL",
      "protocol": "voltage_clamp",
      "x_axis": {"label": "Holding potential", "units": "mV", "scale": "linear"},
      "y_axis": {"label": "Current", "units": "pA", "scale": "linear"},
      "data": [{"x": -80, "y": 0.1}, ...],
      "digitization_notes": "calibrated against axis tick marks at -80, -40, 0, 40 mV; estimated digitization error ±2 mV on x, ±5% on y due to manual click placement"
    }
  ]
}
```

**Document digitization process explicitly:**

- Tool used and version
- Calibration steps (which axis points marked, tick spacing)
- Manual error sources (axis log/linear, panel resolution, click precision)
- Estimated digitization error per panel

Honest documentation supports reasonable tolerance interpretation downstream.

### Deliverable 3: Comparison validation

Run Nicoletti's NEURON code under matching protocols for the digitized panels.

If the digitized panel is Mellem 2008 (fallback path): Nicoletti's AVA simulation script under matching voltage-clamp/current-clamp protocol approximates Mellem's recording conditions. Compare Nicoletti's NEURON output to Mellem's digitized values.

If the digitized panel is Nicoletti's own experimental-overlay panel: run her simulation under the protocol she ran (parameters in her scripts), and compare her NEURON output to the digitized experimental component of the panel.

**Tolerance metric (refinement #2):**

```python
def divergence(measured, reference, peak):
    """Relative tolerance with absolute floor.
    
    For values > 10% of peak: 5% relative tolerance
    For values < 10% of peak: absolute tolerance ≤ 5% of peak
    
    Equivalent: max(|measured - reference| / max(|measured|, |reference|, 0.1 * peak))
    """
    return abs(measured - reference) / max(abs(measured), abs(reference), 0.1 * peak)
```

Where `peak = max(|values|)` across the IV curve or trace.

Per-point divergence ≤ 5% (interpreted via the formula above) constitutes a passing data point. Per-panel pass: > 90% of points pass per-point criterion AND no single point exceeds 15% divergence (catches outliers that average-pass would mask).

Document tolerance interpretation explicitly in the validation report so Phase β translation work uses the same metric.

**Outputs of comparison:**

- Per-panel: pass/fail with divergence statistics (mean, max, % points passing)
- Overall: aggregate verdict per cell

### Deliverable 4: Documentation

**Files to produce:**

- `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/published_traces.json` — digitized values with metadata (per-panel structure shown above)
- `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/figures/` — preserved source figure PNGs and source PDFs
- `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/phase_beta_pre_validation.md` — comparison results: per-panel divergence, pass/fail per panel, overall verdict, panel selection rationale, digitization process documentation
- Update to `~/Desktop/website/personalwebsite/scripts/brain/wave2/phase_alpha_report.md` — add a "Phase β-pre addendum" section noting deliverable 3 closure with reference to `published_traces.json` as the external numerical reference Phase β will validate translations against

---

## Decision tree

**If overall verdict = pass** (NEURON output matches digitized experimental values within tolerance across all selected panels):

- Phase α deliverable 3 closure formally documented
- Phase β proper is ungated
- Notify user with summary; Phase β proper begins as separate work block

**If overall verdict = fail** on a single panel:

- Investigate the specific panel: digitization error larger than expected? Protocol mismatch (Nicoletti's code runs different conditions than the figure shows)? Parameter setup we're missing?
- If investigation resolves the divergence, document the resolution and re-evaluate
- If unresolvable, flag the panel and run additional panel(s) to assess whether the failure is panel-specific or systematic

**If overall verdict = fail** on multiple panels (≥ 2):

- **STOP and surface to user via notify (urgent priority).** This is condition-3 invalidation territory.
- Possible causes: Nicoletti's published code differs from her published model, environment-specific issues (NEURON version differences, dependency drift), parameter setup we're missing, the digitization error is dominating
- Do NOT proceed to Phase β proper without cross-session discussion of the finding

---

## Methodology continuity

Maintain the pre-flight pushback discipline:

- Pre-flight: your VERY FIRST OUTPUT before any tool calls beyond reading the spec files should be a brief plan summary covering: (a) where you'll source the PDFs, (b) initial assessment of `plotdigitizer` availability via pip, (c) any concerns or questions before execution. Only after that pre-flight step should you begin downloading PDFs.

- Mid-flight: if during figure inspection you find that Nicoletti's papers don't actually contain experimental panels (everything is simulation output), surface immediately. Do not proceed silently to a fallback that wasn't pre-discussed. This is the load-bearing finding the methodology is designed to surface.

- Stop-and-ask: if `plotdigitizer` fails on multiple panels, if a paper PDF isn't accessible, if Mellem 2008 isn't accessible either, stop and surface. Don't build custom tools or workaround infrastructure without cross-session discussion.

- Empirical grounding: trust what you observe in the actual papers over what the spec describes. The spec is informed by Phase α's findings but the source of truth is the published figures themselves.

- No time estimates in the report or notifications.

---

## Notifications

Use `~/bin/notify` for milestones:

- Phase β-pre started — one-line plan summary
- Figure inspection complete (panel selection done) — list selected panels in 1-2 lines
- Digitization complete — per-panel tool/quality summary in 1-2 lines
- Comparison validation complete — pass/fail per panel + overall verdict in 1-2 lines
- Phase β-pre complete — one-line result (pass: Phase β proper ungated; fail: condition-3 invalidation surfaced)
- Blocked / urgent — use `urgent` priority

Keep messages under ~150 chars.

---

## Scope discipline — what is NOT in Phase β-pre

Do NOT do any of these:

- **Translating any of Nicoletti's channels to Brian2** — Phase β proper
- **Building additional harness infrastructure beyond what Phase α produced** — voltage_clamp_harness.py and plateau_harness.py are infrastructure for Phase β, not for β-pre
- **Investigating Nicoletti's code beyond verifying it reproduces published values** — if her code does match published values, the work is done; no deeper code archaeology needed in this phase
- **License verification on Nicoletti** — production-prep gate, not Phase β-pre
- **Cadiff + caintra1 translation** — Phase β proper, surfaced by Phase α as the new first translation step
- **Brian2 implementation of any kind** — Phase β proper

If you find yourself doing any of the above, stop and surface to user. The scope was set deliberately.

---

## Reference packages on disk (already cloned, from Phase α)

- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/` — Nicoletti 2024 (24 mod files + 9 simulation scripts). Phase α verified all compile and AVAL/AIY/RIM run end-to-end. Check for any PDFs or supplementary materials in this directory before fetching from journal sites.
- `~/Desktop/website/personalwebsite/scripts/brain/wave2/` — Phase α deliverables. `voltage_clamp_harness.py` and `plateau_harness.py` exist; do not modify in this phase.
- `~/venvs/wave2-neuron/` — isolated venv with NEURON + Brian2 + numpy + scipy + matplotlib. Use this venv. Install `plotdigitizer` and any digitization dependencies into this venv.

---

## Final instructions

Execute Phase β-pre end-to-end as a single session. Begin with pre-flight: read the three spec files, locate source PDFs (on disk or surface that fetch is needed), assess plotdigitizer availability, and produce a one-paragraph plan summary in your first response before doing any substantive work.

Surface findings mid-flight. Stop-and-ask if anything's unclear, especially the load-bearing experimental-vs-simulation classification during figure inspection.

If Phase β-pre clears (pass verdict), Phase β proper is ungated with revised priority list:

1. cadiff.mod + caintra1.mod translation (Ca-pool prerequisite, surfaced by Phase α)
2. NEURONReference wrapper class for harness extension
3. EGL-19 translation as first channel
4. Gate 2a evaluation on EGL-19 against both wrapped NEURON reference and digitized experimental values

If Phase β-pre fails (condition-3 invalidation): cross-session discussion before any further Wave 2 work.

The other cross-session adversarial review sessions remain idle until Phase β-pre completes — your output is what they review.

Begin with pre-flight pushback.
