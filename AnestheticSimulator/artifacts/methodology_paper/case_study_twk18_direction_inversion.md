# Case study 4 — twk-18 direction inversion in literature anchor verification

**Project:** AnestheticSimulator / Wave P pharmacology pipeline
**Date diagnosed:** 2026-04-26 (pre-flight pushback for rigor work block)
**Methodology pattern:** primary-source verification with explicit direction check, even when citation appears to support the claim

---

## Finding

Wave P's Phase H validation table included Anchor 6: *"Sedensky 2001 PMID 11756669, twk-18(cn110) gain-of-function confers halothane RESISTANCE."* The anchor was originally classified DEFERRED because Wave P's K2P channel target (KCNK2/TREK-1) lacked an in-pipeline behavioral prediction; the genetics anchor was held aside for future Phase G testing.

Pre-flight verification of the anchor before the rigor work block surfaced two errors:

1. **The cited PMID does not exist as a twk-18 paper.** Sedensky 2001 PMID 11756669 corresponds to a different topic. Real Sedensky 2001 papers are about stomatins (PMID 11287347) or GAS-1 Complex I (PMID 11278828) — neither addresses twk-18 or halothane. The citation was fabricated.

2. **The biological direction is inverted.** Looking up the actual K2P-halothane literature in *C. elegans*:

   - **Singaram et al. 2011** PMID 22137475 *Curr Biol* 21(24):2070-6, "TWK-18, a TASK-Like K+ Channel, Modulates Sensitivity to Halothane in C. elegans"
   - K2P **gain-of-function** (unc-92(n200)) → halothane EC50 1.43% atm → **HYPERSENSITIVE** (lower EC50)
   - K2P **loss-of-function** (sup-9(n180)) → halothane EC50 3.35% atm → **MODESTLY RESISTANT** (higher EC50)
   - WT halothane EC50 in matched conditions: 3.08% atm

   The direction reported in the original Wave P preregistration was **inverted** — GoF was claimed to be RESISTANT; real biology says GoF is HYPERSENSITIVE.

## Mechanistic interpretation of the corrected biology

Halothane potentiates TREK-1/TASK family K2P channels at the molecular level (Patel & Honoré 1999 PMID 10321245 — confirmed by Wave P calibration: KCNK2/halothane Vina-Kd 702 µM vs Patel & Honoré EC50 700 µM, log_err = +0.001). At the network level:

```
Halothane → potentiates K2P → increased K+ leak → membrane hyperpolarization
            → reduced neuronal excitability → reduced locomotion → behavioral immobilization
```

A K2P gain-of-function mutation (TWK-18 cn110 GoF) pre-opens the channel further at baseline. Halothane's potentiation is then ADDITIONAL inhibition on top of an already-elevated K leak. The mutant immobilizes at lower halothane concentration → **HYPERSENSITIVE**.

A K2P loss-of-function (sup-9 LoF) removes a tonic inhibitory contribution. Halothane's K2P potentiation has less effective machinery to act on. The mutant requires more halothane to reach the immobilization threshold → **RESISTANT**.

This is consistent with Singaram's reported phenotypes and with the molecular Wave P calibration showing halothane potentiation of K2P with µM EC50.

## How the issue was caught

The pre-flight pushback workflow before the rigor work block included an item: **"primary-source verification of all DEFERRED anchors before treating them as future-validation targets."** This workflow has caught multiple citation issues earlier in the project (including 6 wrong PMIDs and 4 fabricated citations across the original WAVE_P_PHASE_H_VALIDATION table).

For Anchor 6, the pre-flight steps were:
1. Look up cited PMID 11756669 → returns a paper unrelated to twk-18 → **fabricated cite**
2. Search for actual twk-18 + halothane primary literature → finds Singaram 2011 PMID 22137475 → **real source**
3. Check the reported direction in the real source → GoF is HYPERSENSITIVE, not RESISTANT → **direction inversion**

Without step 3, simply finding a real source might have been treated as "anchor rescued — replace fabricated PMID with real PMID, declare done." The direction-check step caught a deeper error: the original anchor's biological claim was wrong, regardless of which paper supported it.

This is the load-bearing methodological observation: **citation verification has two parts — (a) does the cited paper exist? and (b) does the cited paper support the direction claimed?** Both must pass.

## Methodology lesson

**Surface 1 (Wave P-specific):** Anchor 6 was lifted from DEFERRED to STRUCTURALLY_GROUNDED_AWAITING_WETLAB with corrected biology:
> "K2P gain-of-function (unc-92(n200) per Singaram 2011 PMID 22137475) confers halothane HYPERSENSITIVITY (EC50 1.43% atm vs WT 3.08% atm). Wave P's KCNK2/halothane binding-side prediction is VERIFIED (log_err 0.001) and structurally consistent with the corrected genetic phenotype via Phase G (predicted: increased K+ leak under halothane × pre-opened K2P → enhanced hyperpolarization → hypersensitivity)."

The CP6 four-category classification frame separates the binding-side claim (VERIFIED) from the genetic-phenotype claim (STRUCTURALLY_GROUNDED_AWAITING_WETLAB pending Phase G).

**Surface 2 (general):** when verifying that a paper supports a claim, **read the abstract and check the reported direction explicitly.** Don't rely on the citing document's interpretation. Citation reuse + direction-paraphrasing introduces inversions especially when:
- The phenotype involves mutant-vs-WT comparison (which way does the mutation push the phenotype?)
- The mechanism involves a chain of effects (does the upstream perturbation produce up-regulation or down-regulation of the downstream readout?)
- The literature uses inconsistent vocabulary (e.g., "gain-of-function" sometimes means "increased activity" and sometimes "increased response to ligand")

**Surface 3 (broader):** AI-assisted scientific writing has well-documented failure modes around citation hallucination and direction-inversion. Methodology that surfaces these errors during the writing process — not just at peer review — is critical. Pre-flight pushback that asks "what direction does this paper actually report?" is cheap to run and catches the failure mode reliably.

## Why direction-inversions are particularly insidious

Three properties make direction-inversions hard to catch in standard review:

1. **The numerical magnitude may still pattern-match.** "K2P-gf has 2-fold differential halothane sensitivity" is roughly consistent with both "RESISTANT" and "HYPERSENSITIVE" interpretations. Reviewers reading the validation table see "2-fold differential" and check the numerical claim, missing that the direction is wrong.

2. **The mechanism may sound plausible in either direction.** Without thinking through the molecular pathway (halothane potentiates K2P → increased leak → hyperpolarization → reduced excitability → reduced locomotion), one could plausibly argue either:
   - "GoF → more channel → more leak → resistance to halothane potentiation" (wrong reasoning that gives wrong direction)
   - "GoF → more channel → halothane has more substrate to act on → hypersensitivity" (correct reasoning that gives correct direction)

3. **The cited paper may be obscure or behind paywall.** The original Wave P preregistration cited a fabricated PMID. Even if a reviewer tried to verify, the obviously-wrong cite would produce a 404 or unrelated paper, prompting "citation needs update" rather than "direction is wrong."

The protective methodology: **after looking up the real paper, write a one-sentence mechanism trace and verify the direction predicted by the trace matches the direction reported in the paper.** This was done for the corrected Anchor 6 above; the trace and the empirical direction agree.

## Generalization

Direction-inversion errors appear in:

- **Mendelian randomization studies** where allele direction is mis-coded
- **CRISPR screen interpretation** where loss-of-function vs gain-of-function readouts are flipped
- **Pharmacology** where agonist vs inverse agonist activity is misread
- **Genome-wide association studies** where odds ratios are reported in the wrong direction
- **Network analyses** where edge polarity is mis-assigned

The recurring failure pattern: writers inherit an interpretation from a previous paper or summary that quoted the original incorrectly, and the inversion propagates through citations.

The protective methodology pattern is consistent across these domains:
1. Locate the primary source.
2. Read the explicit direction reported (often in the abstract or first results sentence).
3. Trace the mechanism from intervention to readout, predicting direction from first principles.
4. Verify (2) and (3) agree.

If (2) and (3) disagree, **the citing document's direction is suspect**, regardless of whether the citation appears authoritative.

## Reference artifacts

- `artifacts/calibration/cp6_anchor_classification.md` — full corrected anchor table with twk-18 reframe
- `artifacts/calibration/rigor_tightening_pushback.md` — original pre-flight pushback document where the inversion was first surfaced
- Singaram et al. 2011 *Curr Biol* 21(24):2070-6 PMID 22137475 — corrected primary source
- Patel & Honoré 1999 *Nat Neurosci* 2(5):422-6 PMID 10321245 — molecular mechanism (halothane potentiates K2P)
