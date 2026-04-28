# Phase β-pre v3 — Layer B verification + citation/AVAR cleanup

You are the engineering session executing **Phase β-pre v3** of Wave 2 of the C. elegans biophysical simulator project. This is the final closure work block for Phase α deliverable 3.

You have full user permission for package installs, file creation, file edits, and PDF/figure access for this phase. Permission does NOT extend to filing the AVAR upstream issue directly — that's drafted only.

---

## HARD CONSTRAINT — image handling guardrail (carry forward from v2)

v1 terminated on a 2000px image dimension limit. v2 cleared this constraint by processing figures programmatically only. **v3 maintains the same guardrail.** It worked once; do not deviate.

- Process figures programmatically only via OpenCV / PIL / plotdigitizer
- Do NOT use the `Read` tool on figure PNGs
- If visual inspection is genuinely needed for debugging, downscale to ≤ 1500px via PIL first, save downscaled version, then Read it. Default mode is programmatic.

---

## Strategic context (read first)

Phases β-pre v1 and v2 surfaced a layered comparison decomposition that wasn't explicit until now:

- **Layer A — Brian2 implementation = NEURON implementation.** Phase β translation work; 5% appropriate.
- **Layer B — Nicoletti's NEURON code = Nicoletti's published model figures.** Deterministic-implementation check; 5% appropriate. **This is what condition-3 actually asks** and is what v3 directly tests.
- **Layer C — Nicoletti's published model = experimental data.** Fit-quality check. 5% structurally too strict for biophysical HH fits. v1 measured C against I-V (post-hoc predictions); v2 measured C against current-clamp (fit targets); both produced "fail" against 5%. Nicoletti herself reports 5-15 mV residuals.

v3 directly tests Layer B by digitizing the **red Model traces** from the same panels v2 already extracted black experimental traces from, and comparing them against Nicoletti's NEURON code output already captured in v2's results.

**Layer B verdict structure:**

- Pass at 5% across all panels → condition-3 cleared cleanly; Phase β proper proceeds
- Multi-panel fail at 5% → real finding (her published code differs from her published figures); surface for cross-session discussion before any Phase β work
- Borderline (some pass, some fail) → document carefully; surface for discussion

---

## Read before substantive work

1. `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md` — project commitment doc; especially condition-3
2. `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/phase_beta_pre_validation.md` — v1+v2 combined doc
3. `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/published_traces_v2.json` — v2 black-trace digitization (reference for format + per-panel structure)
4. `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/comparison_validation_results_v2.json` — NEURON output captured per-step in v2; reused by v3 (no need to re-run NEURON)
5. `~/Desktop/website/personalwebsite/scripts/brain/wave2/digitize_panels_v2.py` — v2 OpenCV color-mask infrastructure; v3 builds on this
6. `~/Desktop/website/personalwebsite/scripts/brain/wave2/run_comparison_validation_v2.py` — v2 NEURON comparison runner; reused for AVAR patched run
7. `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/AVAR_simulation.py` — read this to extract AVAR's UNC103 conductance value for the patch

---

## Seven deliverables

### Deliverable 1: Red Model trace digitization

For each of the four panels (Fig 1A AVAL, Fig 1B AVAR, Fig 3A AIY, Fig 5A RIM), extract the **red Model traces** using v2's OpenCV color-mask infrastructure with the color filter changed from black → red.

Output to `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/nicoletti_model_traces.json` with the same structure as v2's `published_traces_v2.json` (per-panel metadata + per-stimulus-step traces + extracted features).

Per-trace separation logic from v2 carries over (v2 successfully separated 36 black traces across 4 panels). Same logic should work for red traces.

**Acceptance:** four panels digitized with red Model traces + extracted features (peak voltage, plateau amplitude, plateau duration, time-to-peak, settling time) per stimulus step. Tool used per panel documented.

### Deliverable 2: AVAR patch + clean comparison

Create `~/Desktop/website/personalwebsite/scripts/brain/wave2/avar_unc103_patch.py` — a **standalone file** in the project's wave2 directory. The patch must NOT modify Nicoletti's upstream code in place; it adapts AVAL's iclamp script to insert UNC103 with AVAR's conductance value extracted from `AVAR_simulation.py`.

Patch contents:
- Function or wrapper that takes AVAL's iclamp simulation function and adds `h.insert("unc103")` with the conductance setting
- Conductance value extracted from `AVAR_simulation.py` (read the file, locate the UNC103 conductance assignment)
- Documentation header: what the upstream defect is, what the patch does, why this approximates the upstream script's intended behavior, what confidence level we have

Re-run AVAR's current-clamp simulation using the patch. Capture the patched NEURON output. Compare against red Model trace from Fig 1B.

**Acceptance:** patch file written; AVAR re-run with UNC103; AVAR comparison produces reasonable resting potential (-25 mV target ± 5 mV) rather than the +11 mV bias from v2.

### Deliverable 3: Layer B comparison validation

For each of the four panels, compute per-feature divergence between digitized red Model traces (Deliverable 1) and NEURON output:

- AVAL/AIY/RIM: NEURON output already captured in `comparison_validation_results_v2.json`; reuse directly
- AVAR: NEURON output from Deliverable 2's patched re-run

Per-feature 5% relative tolerance with v2's relative-tolerance-with-floor formula:

```python
def feature_divergence(measured, reference, peak):
    return abs(measured - reference) / max(abs(measured), abs(reference), 0.1 * peak)
```

Per-step pass: all features within 5%. Per-panel pass: > 80% of steps pass per-step criterion. Per-cell pass: panel passes.

Output to `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/layer_b_validation_results.json` with per-feature, per-step, per-panel, per-cell results.

**Acceptance:** four-cell Layer B comparison computed; verdict per cell; overall Layer B verdict.

### Deliverable 4: Citation correction (proper expansion)

Verified DOIs from v2:
- **Nicoletti 2019** (PLOS ONE `journal.pone.0218738`): AWCon/RMD biophysical models, upstream paper that 2024 extends
- **Nicoletti 2024** (PLOS ONE `journal.pone.0298105`): primary 22-channel library; AVA/AIY/RIM/VA/VB/VD cells; principal Wave 2 import target

Sequence:

1. Grep all artifacts for incorrect or incomplete Nicoletti citations:
   ```bash
   grep -rn "Nicoletti\|10.1371/journal.pcbi.1007611\|pcbi.1007611\|pone.0218738\|pone.0298105" \
     /home/rohit/Desktop/website/personalwebsite/scripts/brain/artifacts/ \
     /home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/
   ```

2. For each file with the incorrect DOI (`pcbi.1007611`), replace with the correct DOI per context:
   - References to "Nicoletti 2019" should resolve to `pone.0218738` (AWC/RMD)
   - References to "Nicoletti 2024" or to the 22-channel library should resolve to `pone.0298105` (AVA/AIY/RIM/VA/VB/VD)
   - References that are ambiguous about which paper should be expanded to cite both with their respective roles

3. Update the **architectural plan** specifically with both citations expanded — it's the project-level commitment doc and Wave 2 references "Nicoletti's 22 channels" which is the 2024 paper specifically.

4. Confirm zero hits for the incorrect DOI after updates:
   ```bash
   grep -rn "10.1371/journal.pcbi.1007611\|pcbi.1007611" /home/rohit/Desktop/website/personalwebsite/scripts/brain/
   ```

**Acceptance:** all incorrect DOI references corrected; architectural plan + relevant prompts cite both Nicoletti papers with their respective roles; grep confirms zero hits for the incorrect DOI.

### Deliverable 5: AVAR upstream issue draft

Draft text suitable for filing as a GitHub issue against `github.com/ModelDBRepository/2017403`. Save to `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/avar_upstream_issue_draft.md`.

Draft should include:

- **Title:** concise, informative (e.g., "Missing `AVAR_simulation_iclamp.py` script imported by `AVAR_simulation.py`")
- **Description:** what's missing, what error occurs when AVAR is run end-to-end, evidence (file listing, import line)
- **Impact:** what AVAR runs cannot do without it (UNC103 not inserted via AVAL's iclamp script; resting potential biased)
- **Workaround we used:** reference our `avar_unc103_patch.py`; describe the patch's logic
- **Confidence level:** verified that patched run produces correct resting potential; cannot verify against missing reference; suggest upstream restore of the file
- **Suggested fix:** restore `AVAR_simulation_iclamp.py` to the repo with appropriate UNC103 insertion

**Do NOT file the issue.** Surface the draft for user review and authorization. The draft lives in `artifacts/` for user review.

**Acceptance:** issue draft written, complete, ready for user authorization.

### Deliverable 6: Combined documentation update

Update `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/phase_beta_pre_validation.md` to reflect v3's findings. Narrative arc:

- v1 surfaced wrong-metric issue (compared against I-V curves which are post-hoc predictions, not fit targets)
- v2 applied fit-target metric (current-clamp) and surfaced second wrong-metric issue (5% per-feature is structurally too strict for Layer C biophysical-fit residuals; Nicoletti herself reports 5-15 mV)
- v3 applied direct Layer B test (red Model trace vs NEURON output, deterministic-implementation territory where 5% is appropriate)
- [verdict per Deliverable 3 results]

Three-layer decomposition made explicit. v1 and v2 work credited as infrastructure-establishing AND methodologically self-correcting (each iteration tightened the methodology).

Add a section documenting:
- AVAR upstream defect + patch
- Citation expansion (both Nicoletti DOIs with roles)

Update `~/Desktop/website/personalwebsite/scripts/brain/wave2/phase_alpha_report.md` Phase β-pre addendum with v3's deliverable 3 closure verdict.

**Acceptance:** both docs updated; cross-references consistent; three-layer narrative explicit.

### Deliverable 7: v3 standalone summary

Brief standalone v3 summary at `~/Desktop/website/personalwebsite/scripts/brain/wave2/artifacts/phase_beta_pre_v3_summary.md`:

- Layer B verdict per cell + overall
- Citation correction summary (files modified, before/after)
- AVAR patch summary (what, where, confidence)
- AVAR upstream issue draft location
- Phase β proper readiness assessment

This is for quick reference; the combined doc (Deliverable 6) is the canonical record.

**Acceptance:** summary written, links to canonical artifacts, ≤ 200 lines.

---

## Methodology continuity

- **Pre-flight:** your VERY FIRST OUTPUT before any tool calls beyond reading spec files should be a brief plan summary covering: (a) confirmation you've read the image-handling guardrail, (b) approach to red-trace separation per stimulus step (reuse v2's per-trace logic), (c) approach to AVAR patch (read AVAR_simulation.py for UNC103 conductance), (d) any concerns. Only after that pre-flight step should you begin work.

- **Mid-flight surfacing:** if red-trace digitization reveals trace separation issues that v2's logic doesn't handle, surface immediately. If AVAR patched run still produces non-physiological resting potential, surface immediately (suggests the patch is incomplete, not just UNC103). If Layer B comparison fails on multiple panels at 5%, that's a real finding — STOP and urgent notify; cross-session discussion required before Phase β proper.

- **Stop-and-ask:** any unexpected complication during pre-flight, multi-panel Layer B fail, or any deliverable that can't meet acceptance criteria.

- **Empirical grounding:** trust the data. v2's NEURON output is captured; reuse it. v2's digitization infrastructure works; reuse it. v3's marginal cost is small.

- **No time estimates** in reports or notifications.

---

## Notifications

Use `~/bin/notify` for milestones:

- v3 started — one-line plan with image-guardrail acknowledgment
- Red-trace digitization complete — one-line per-panel tool used
- AVAR patch + comparison complete — one-line resting-potential check
- Layer B comparison complete — one-line per-cell verdicts
- Citation correction applied to N files — one-line summary
- AVAR issue draft complete — one-line confirmation
- Documentation updates complete
- v3 complete — one-line result (pass: condition-3 cleared, Phase β ungated; fail: real Layer B finding surfaced)
- Urgent for any blocked state, multi-panel Layer B fail, or condition-3 invalidation

Keep messages < 150 chars.

---

## Output format (recap)

Files produced by v3 (under `scripts/brain/wave2/`):

- `artifacts/nicoletti_model_traces.json` — Deliverable 1
- `avar_unc103_patch.py` — Deliverable 2
- `artifacts/layer_b_validation_results.json` — Deliverable 3
- (citation corrections applied across artifact files) — Deliverable 4
- `artifacts/avar_upstream_issue_draft.md` — Deliverable 5
- `artifacts/phase_beta_pre_validation.md` (updated) + `phase_alpha_report.md` (updated addendum) — Deliverable 6
- `artifacts/phase_beta_pre_v3_summary.md` — Deliverable 7

v1, v2 outputs preserved unchanged as historical record (except where the citation correction touches their prompts in `artifacts/`, which is in scope for Deliverable 4).

---

## Scope discipline — what is NOT in v3

- **Channel translation to Brian2** — Phase β proper
- **Re-running NEURON for AVAL/AIY/RIM** — already captured in v2; reuse
- **License verification** — publication-prep gate
- **Filing the AVAR upstream issue** — drafted only; user authorizes
- **cadiff/caintra1 translation** — Phase β proper
- **Brian2 implementation of any kind** — Phase β proper
- **Re-doing v1/v2 work** — preserved as historical; v3 is additive

If you find yourself doing any of the above, stop and surface to user.

---

## Final instructions

Execute v3 end-to-end as a single session. Begin with pre-flight pushback (acknowledge image guardrail, plan red-trace separation, plan AVAR patch). Then deliverables 1-7 in order, with stop-and-ask on multi-panel Layer B fail.

If v3 clears (Layer B pass): condition-3 properly cleared at the layer condition-3 actually asks about. Phase β proper begins as separate work block with revised priority list (cadiff/caintra1 → NEURONReference wrapper → EGL-19 → Gate 2a evaluation against fit-target current-clamp data).

If v3 fails (Layer B fail on multiple panels): real condition-3 invalidation territory; cross-session discussion before any Phase β work.

The other cross-session adversarial review sessions remain idle until v3 completes.

**Begin with pre-flight pushback. Acknowledge the image-handling guardrail explicitly in your first response.**
