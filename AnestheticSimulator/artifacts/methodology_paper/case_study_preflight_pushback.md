# Case study 5 — Pre-flight pushback as systematic methodology

**Project:** AnestheticSimulator / Wave P pharmacology pipeline + broader *C. elegans* digital pharmacology project
**Date pattern formalized:** progressive, ~2026-04-15 to 2026-04-27
**Methodology pattern:** structured pre-execution verification before launching computational work blocks

---

## The pattern

Pre-flight pushback is a systematic methodology applied before launching any non-trivial computational work block. It consists of explicit verification of:

1. **Citation hygiene** — every claim that grounds the work block in a primary source has its citation verified (paper exists, PMID matches, paper actually supports the claimed direction).
2. **Parameter inventory** — every tunable parameter in the model is enumerated, classified as literature-derived vs hand-set, and surface-area for "tuning to target" is identified.
3. **Sensitivity envelope** — for any quantitative prediction the work block will produce, identify the input dependencies and ask "what range of inputs produces the predicted output? is the prediction sensitive to the inputs that are supposed to drive it?"
4. **Boundary tests** — explicit identification of what the model claims to do AND what it claims not to do; what experiments would falsify each.
5. **Honest verdict scaffolding** — the verdict structure ("PASS within tolerance band," "FAIL outside range," "DEFERRED pending wet-lab," "STRUCTURALLY UNCALIBRATED") is written before execution, not after.

The output is an `OVERNIGHT_PUSHBACK.md` or pre-flight document that either (a) clears for launch or (b) flags concerns that pause the work block until human input.

## What pre-flight pushback has caught (cumulative log across project history)

### Citation issues (≥ 37 caught across all work blocks)

Specific to Wave P rigor pass:

- **Crowder 1996 PMID 8855256** → wrong; real is **8873562** *Anesthesiology* 85(4):901-12
- **Morgan & Sedensky 1995 PMID 7549290** → wrong year + wrong PMID; real is **1994 PMID 7943840** *Anesthesiology* 81(4):888-98
- **Sedensky 1992 PMID 1346264** → wrong; real is Sedensky & Meneely **1987 PMID 3576211** *Genetics* 116(3):417-26
- **van Swinderen 1999 (cited for unc-13)** → domain mis-cited; the paper is about unc-64 SNARE, not unc-13. For unc-13 use Nguyen 1995 PMID 7647836
- **Sedensky 2001 PMID 11756669 (twk-18)** → fabricated; real source is Singaram 2011 PMID 22137475
- **Boddington 2017 (propofol *C. elegans*)** → fabricated; closest real anchor is Heuer 2014 PMID 24501356 (channel-level oocyte IC50, NOT whole-animal)
- **30/32 UniProt IDs in initial target CSV** → wrong; corrected during Phase A re-fetch

Earlier in the project (across C-Elegans work blocks):

- **Mellem 2008 misattribution** → caught before launching a 3-week morphology fork that would have built on the wrong attribution
- **Rogers 2003 → Cohen 2009 correction** for FLP-13 anchor identification
- **Gao-Hobert vs Mellem 2008 correction** for AVA voltage-scale finding

### Parameter-locking errors (1 major)

- **Phase F structural parameter-lock** (CP1) — block_factor cancels in d_WT/d_g1 ratio; the apparent "5/6 PASS" was tuning to target. Caught via sensitivity sweep of preregistered model inputs.

### Direction inversions (1 confirmed; ≥ 2 candidates flagged earlier)

- **twk-18 GoF direction** — original "RESISTANT" was inverted; real biology per Singaram 2011 says HYPERSENSITIVE. Caught via mechanism-trace-vs-empirical-direction check.

### Conflation errors

- **Kd vs EC50 conflation** (CP4) — original calibration table labeled functional EC50 entries as "Kd," producing a systematic +0.40 log_err allosteric bias that was misread as pipeline error. Caught via directness-tier audit during pre-flight.

### Out-of-scope ambitions

- **Wet-lab proposals filtered out** — multiple times, pre-flight identified that a proposed "test" required wet-lab work that's outside Wave P's $0-spend computational scope. Flagged for descope rather than fabrication.

### Mid-flight saturation collapse

- **Phase E `CLINICAL_EFFECTIVE_OCCUPANCY=0.30` mid-flight tune** — caught as a parameter-tuning concern during pre-flight; CP2 sensitivity sweep then validated the choice across [0.10, 0.30] range, lifting verdict from "post-hoc fit" to "robust within sensitivity envelope."

## Why pre-flight pushback works

Three structural reasons:

### 1. Cheap relative to consequences

Each pre-flight item takes minutes to verify. The cost of skipping pre-flight and discovering an error after a multi-hour or multi-day work block is much higher: re-running pipelines, retracting documentation, propagating corrections through dependent claims.

For Wave P specifically, the rigor pass that caught Phase F parameter-lock + twk-18 inversion + Kd/EC50 conflation was ~1 day of focused work. Without pre-flight, those errors would have shipped to the methodology paper draft and been caught (if at all) at peer review — at much higher reputational and time cost.

### 2. Verification is asymmetric — false positives are cheap, false negatives are expensive

Pre-flight items are designed to be over-skeptical. If pre-flight raises a concern that turns out to be unfounded, the cost is one extra hour of investigation. If pre-flight misses a concern that turns out to be founded, the cost is significant downstream rework.

The asymmetry is especially favorable for AI-assisted work: AI-generated content has well-documented failure modes (hallucinated citations, paraphrase drift, direction inversions). Pre-flight is calibrated to those failure modes specifically.

### 3. Pause-with-documentation is structurally preferable to push-through

When pre-flight surfaces a concern, the workflow is to PAUSE and document the concern, not to push through and resolve it on the fly. Pause-with-documentation produces a structured artifact that can be reviewed asynchronously; push-through-and-resolve produces a faster but less reviewable result.

For overnight runs, this is critical — pause states are useful even if no further work happens overnight. The next morning, the human can review the pause state and decide whether to proceed, modify the plan, or escalate.

## Specific pre-flight templates that have worked

### Citation pre-flight (1-3 minutes per cite)

1. Look up the cited PMID directly (PubMed, Google Scholar, NCBI).
2. Verify the paper exists at the cited identifier.
3. Read the abstract for the direction of the reported finding.
4. Confirm the directing of the reported finding matches the direction claimed by the citing document.

If any of (1)-(4) fail, flag for documentation and don't proceed.

### Parameter inventory pre-flight (5-15 minutes per model)

1. Enumerate every parameter the model uses.
2. For each: literature-derived (cite the source), normalized/reference (set to 1.0 by convention), or hand-set (admit it).
3. Count "effectively tunable" parameters — those that meaningfully affect the prediction.
4. For each tunable parameter, ask: was this set BEFORE running the model against the validation target, or AFTER?
5. Parameters set after running against the target are tuning-to-target. Flag for sensitivity sweep.

### Sensitivity sweep pre-flight (15-60 minutes for non-trivial models)

1. Identify the input that's supposed to drive the prediction (e.g., block_factor for Phase F).
2. Identify a plausible range (literature-supported or chosen-conservatively).
3. Run the model across that range, holding tunable parameters fixed.
4. Plot/print the prediction across the input range.
5. If prediction varies meaningfully → input genuinely drives output → claim is potentially valid.
6. If prediction is constant or near-constant → input does not drive output → parameter-lock; claim rests on tuning.

### Boundary test pre-flight (30-60 minutes per validation claim)

1. For each claim "model passes test X," explicitly identify the contrapositive: what experiment would falsify the claim?
2. Run the contrapositive test if possible.
3. Document both the claim AND the boundary in the validation table.

For Wave P: the multi-target discriminative claim (Stage 5) was tested with easy negative controls (benzene, methanol). The contrapositive boundary test (Eger non-immobilizers) was added as CP3 + CP7 and produced FAIL. The validation table now reports both: PASS on easy discrimination, FAIL on hard discrimination.

## Methodology lesson

**Surface 1 (project-specific):** the project's accumulated catch-list (>37 citation issues + parameter-lock + direction-inversion + Kd/EC50 conflation + saturation issues) is a representative sample of the failure modes that AI-assisted scientific work introduces. Pre-flight pushback is the project's primary defense.

**Surface 2 (general):** AI-assisted scientific work benefits from systematic, structured pre-execution verification that targets known AI failure modes (hallucination, paraphrase drift, direction inversion, calibration overfitting). The pre-flight templates above are specific enough to be checklists, generic enough to apply across domains.

**Surface 3 (broader):** scientific methodology in the era of AI assistance needs to make explicit what the AI tools are good at (computation, search, drafting) and what they're bad at (uncited claims, direction nuance, deep verification). Pre-flight pushback is one workable structure for this division of labor.

## Generalization

The pre-flight pushback methodology has been applied to:

- Wave P pharmacology (this project) — multiple work blocks
- *C. elegans* digital simulator (parent project) — Phase 0 + overnight runs
- Vivekananda corpus synthesis (separate project) — ensures axiomatic statements are located in their proper context

The recurring observation: pre-flight pushback adds ~10-30% wall-clock overhead to a work block but reduces post-execution rework by ~50-90% in cases where it surfaces an issue. The cost-benefit is strongly favorable when the work involves citation chains, parameter-tuning surfaces, or direction-sensitive biological claims.

## What pre-flight pushback is NOT

To be precise about scope:

- It is NOT a guarantee of correctness. Errors can survive pre-flight if the verification template doesn't target them. The project still finds errors at execution time and during review.
- It is NOT a substitute for peer review. Pre-flight catches the most common AI-failure-mode errors; substantive scientific review is still needed.
- It is NOT primarily about caution. The point is to keep work blocks on the right scientific path, not to slow them down. Cleared pre-flight should fast-track execution; failed pre-flight should pause early.
- It is NOT generic skepticism. Each pre-flight item is targeted to a known failure mode, with a specific verification step.

## Reference artifacts

- `artifacts/calibration/rigor_tightening_pushback.md` — example pre-flight document (CP1-CP8 work block)
- `artifacts/calibration/calibration_pushback.md` — earlier pre-flight document (calibration work block)
- `artifacts/calibration/cp6_anchor_classification.md` — reframing after pre-flight surfaced direction-inversion
- `src/preflight_phase_f_saturation.py` — canonical sensitivity-sweep template
- `src/calibration_cp4_strict_kd_subset.py` — directness-tier template

## Methodology paper claim

The protective value of structured pre-flight pushback for AI-assisted scientific work is empirically documented across this project's history. The case studies in this collection (Phase F parameter-lock, Kd/EC50 conflation, Eger boundary, twk-18 inversion, this pre-flight framework) are concrete instances of the pattern. A methodology paper organized around these case studies would argue that:

> Systematic pre-flight pushback is a cost-effective methodology for AI-assisted computational research, particularly for work that grounds claims in primary sources, depends on parameter calibration, or makes direction-sensitive biological predictions. The methodology surfaces failure modes that are otherwise difficult to detect, including citation fabrication, parameter-locking, direction inversion, and ground-truth conflation. The cost of running pre-flight is low relative to the cost of correcting errors at execution time or peer review.

This is the umbrella thesis the four prior case studies support empirically.
