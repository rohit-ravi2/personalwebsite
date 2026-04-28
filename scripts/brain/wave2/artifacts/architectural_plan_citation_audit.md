# Architectural Plan citation audit

**Date:** 2026-04-26
**Scope:** verify the load-bearing biological citations in `phase_v_w2_architectural_plan.md` against primary sources.
**Method:** read Nicoletti 2024's full references list from the on-disk extracted text + cross-check with architectural plan claims.

---

## Methodology lesson surfaced before audit deployed

The audit prompt initially proposed verifying a per-cell target table with citations: AVE/Wang 2020, AVB/Kawano 2011, PVC/Faumont 2011, RIS/Turek 2016, DVA/Li 2006, with specific numerical targets (15/400 mV/ms, 18/500, 14/350, 18/700, 16/400).

**Pre-flight grep of the architectural plan: none of these citations exist there.** They were constructed from memory (the user's recollection of an earlier project conversation context that mentioned a Mellem-target table in passing) rather than verified against the canonical document.

This is the same propagation pattern as Mellem 2008 ("20 mV / 600 ms in AVA") and Nicoletti 2019 PCBI (glioma paper) — citation attached to claim without primary-source verification. Caught by pre-flight pushback before deployment, exactly as the audit was designed to prevent.

The pattern works in both directions: user-side fabrication caught by agent pre-flight pushback, agent-side fabrication caught by user/cross-session review. Worth noting for the methodology paper.

The audit reduced to verifying the citations that actually appear in the canonical plan + the residual Liu reference [29] flagged by the Mellem investigation.

---

## Audit results per citation

### 1. Reference [29] in Nicoletti 2024 (the AVA experimental data source)

**Status: VERIFIED + v1 digitization JSON misattribution exposed**

**Actual citation (Nicoletti 2024 reference list, line 1127-1129):**

> Liu P., Chen B., and Wang Z.-W., "GABAergic motor neurons bias locomotor decision-making in C. elegans," *Nat Commun*, vol. 11, no. 1, p. 5076, **2020**. DOI: 10.1038/s41467-020-18893-9. PMID: 33033264.

**Cross-check against v1 digitization JSON (`published_traces.json`):**

> "experimental_data_origin: Liu P, Chen B, Wang Z-W. **2018**. **Postsynaptic current bursts instruct action potential firing at a graded synapse**. Patch-clamp recordings on AVAL/AVAR neurons, ref [29] in Nicoletti 2024."

**Discrepancy:** v1 JSON had wrong year (2018 vs actual 2020) AND wrong title ("Postsynaptic current bursts..." vs actual "GABAergic motor neurons bias locomotor decision-making..."). The v1 digitization agent fabricated this citation — likely guessed from search results without primary-source verification. The actual Nicoletti reference is the 2020 Nat Commun paper.

**Does the actual reference characterize AVA?**

The cited paper is about GABAergic motor neuron influence on locomotor decision-making. AVA is downstream of these GABAergic D-MNs. Nicoletti 2024 uses [29] for both:
- D-MN→AVA integration context (line 113: "D-MNs are integrated [29]")
- Patch-clamp recordings on AVAL/AVAR (line 460: "Experimental whole-cell recording performed by Liu et al.[29] shows that AVAL and AVAR neurons have similar behavior...")

So the paper does include AVA recordings, but as part of a study of GABAergic motor neuron output, not as a dedicated AVA characterization. Critically, the **acknowledgments (line 1009)** state:

> "We acknowledge prof. Zhao-Wen Wang and prof. Ping Liu for providing **raw electrophysiological recordings of AVAL and AVAR neurons**."

This means Nicoletti's AVA model was fit against **raw recordings provided directly by Wang and Liu**, not necessarily fully published in [29]. The publication ([29]) used these recordings in service of a different question (GABAergic D-MN function); the raw data was shared informally with Nicoletti for modeling purposes.

**Implication:** Nicoletti's AVA fit is anchored on data that's only partially in the public literature. The detailed voltage-clamp + current-clamp recordings she fits against are at least partially raw-data-sharing rather than published. This is a limitation worth documenting for paper 3.

### 2. Reference [30] in Nicoletti 2024 (the AIY data source)

**Status: VERIFIED + ambiguity flag**

**Actual citation (Nicoletti 2024 line 1130-1131):**

> Liu Q., Kidd P. B., Dobosiewicz M., and Bargmann C. I., "**C. elegans AWA olfactory neurons fire calcium-mediated all-or-none action potentials**," *Cell*, vol. 175, no. 1, pp. 57-70, 2018.

**Cross-check against Nicoletti 2024 usage:**

Nicoletti 2024 line 545 uses [30] for AIY: "Electrophysiological recordings by Liu et al. [30] showed that AIY neurons do not originate regenerative responses..."

**Ambiguity:** the paper's primary subject per its title is **AWA**, not AIY. Q. Liu is a co-author of Nicoletti 2024 (and corresponding author of this AWA paper). The most likely explanation is that Liu et al. 2018 contained AIY recordings as supplementary/comparison data alongside the primary AWA characterization, OR that the AIY recordings are similarly raw-data-shared (parallel pattern to ref [29]).

The Mellem investigation flagged this ambiguity. Without reading the Liu/Kidd/Dobosiewicz/Bargmann 2018 paper directly (Cell, paywalled but likely accessible via institutional means), can't fully verify whether AIY data is in the published paper or shared raw.

**Implication:** AIY model anchored on a paper whose primary subject is AWA. Worth documenting; AIY-specific data provenance is at least partially informal.

### 3. Reference [31] in Nicoletti 2024 (the Mellem 2008 reference)

**Status: VERIFIED, used correctly by Nicoletti, MISATTRIBUTED in our architectural plan**

**Actual citation (line 1132+, truncated in extracted text):**

> Mellem J. E., Brockie P. J., Madsen D. M., and Maricq A., "Action potentials contribute to neuronal sig[naling in C. elegans]," [Nat Neurosci 2008]

This matches the Mellem investigation's primary-source verification: Mellem 2008 *Nat Neurosci* 11:865-867. DOI 10.1038/nn.2131. PMC2697921.

Mellem 2008 characterizes RMD plateau, not AVA. Direct quote from Mellem investigation: "we never observed action potentials in AVA (n=10)."

**Architectural plan misattribution:** the plan cites "Mellem 2008 plateau (20 mV / 600 ms)" as Gate 2b's biological target for **AVA**. This is wrong. The Mellem investigation already established this (`mellem_investigation_pushback.md`).

### 4. Wang 2001 reference for SHK-1 (architectural plan line 94)

**Status: MISATTRIBUTED in architectural plan**

**Architectural plan claim (line 94):** "**SHK-1** (Kv1 delayed rectifier; Wang 2001 rich worm-specific data)."

**Actual Wang 2001 paper (Nicoletti 2024 reference [60], line 1231-1233):**

> Wang Z.-W., Saifee O., Nonet M. L., and Salkoff L., "**SLO-1 potassium channels** control quantal content of neurotransmitter release at the C. elegans neuromuscular junction," *Neuron*, vol. 32, pp. 867-881, 2001. DOI: 10.1016/s0896-6273(01)00522-0. PMID 11738032.

This is about **SLO-1** at the **NMJ**, not SHK-1 in any neuron. The architectural plan's attribution of SHK-1 worm-specific data to Wang 2001 is misattributed.

**What Nicoletti 2024 actually cites for SHK-1 (and SHL-1):**

Line 215-216: "We also provide a new model for both SHL-1 and SHK-1 currents, relying on the experimental data from [28, 30, 44, 45]"

- **[28]** Dobosiewicz M., Liu Q., Bargmann C. I., "Reliability of an interneuron response depends on an integrated sensory state," *Elife* 2019. PMID 31718773.
- **[30]** Liu/Kidd/Dobosiewicz/Bargmann 2018 *Cell* (AWA paper) — same as AIY source
- **[44]** Wei A. D., Butler A. G., Salkoff L. B., "KCNQ-like potassium channels in C. elegans: Conserved properties and modulation," *J Biol Chem* 2005.
- **[45]** Gu Y. et al., "Alternative Splicing Regulates Kv3.1 Polarized Targeting to Adjust Maximal Spiking Frequency," *J Biol Chem* 2012. PMID 22105078.

None of these are Wang 2001. Most plausible reconstruction: the architectural plan author was thinking of Wang's broader body of work on C. elegans potassium channels and casually attributed SHK-1 to "Wang 2001" without verifying that the 2001 Wang paper was actually about SHK-1 (it was about SLO-1).

**Implication:** the architectural plan should either drop the Wang 2001 attribution for SHK-1 or replace it with the actual sources Nicoletti uses ([28, 30, 44, 45]). Wei et al. 2005 and Gu et al. 2012 are the most channel-specific.

### 5. The "20 mV / 600 ms" propagation chain

**Status: SOURCE NOT FOUND IN NICOLETTI 2024 PROTOCOLS**

**Nicoletti 2024's actual AVA protocols (Fig 1 caption, line 491-503):**

- **Current-clamp:** 7 current steps from -30 to +30 pA, **duration 1000 ms** (line 493)
- **Voltage-clamp:** 16 voltage steps from -120 to +50 mV, **duration 500 ms** (line 498)

Neither matches "600 ms" — Nicoletti's CC is 1000 ms, VC is 500 ms.

**Nicoletti 2024's actual AVA characterization (per body text):**

- Line 460-462: "AVAL and AVAR neurons have similar behavior both in voltage- and current clamp recordings... and have a depolarized resting potential" (~−30 mV per Mellem)
- AVA model in Nicoletti uses 5 currents only: NCA, EGL19, IRK, UNC103, LEAK (per line 470)
- I-V curves are linear (per Mellem investigation finding)

**The "20 mV / 600 ms" target appears nowhere in Nicoletti 2024.**

**Most plausible reconstruction:**

- "20 mV" approximates AVAL's typical depolarization range under +30 pA current injection (resting ~−30 mV → depolarized ~−10 mV ≈ 20 mV swing)
- "600 ms" is a misremembering or interpolation between Nicoletti's 500 ms VC and 1000 ms CC durations, OR a misremembered Mellem 2008 RMD plateau value (Mellem RMD plateau may be around 600 ms; would need to verify)

**Conclusion:** the "20 mV / 600 ms in AVA" target is not anchored to any primary source. It's a fabricated number-pair with a plausible but unverified provenance. The Mellem investigation already established this; this audit confirms it from Nicoletti 2024's protocol durations.

---

## Summary verdict

| Citation | Status | Resolution |
|---|---|---|
| Mellem 2008 → AVA plateau (20/600) | MISATTRIBUTED | Drop. Mellem characterizes RMD, not AVA. Re-ground per Mellem investigation's path α (Nicoletti's actual AVAL phenotype). |
| Nicoletti 2019 → AWCon/RMD | VERIFIED (post-v3 fix) | Already corrected in plan v3 citation block |
| Nicoletti 2024 → 22-channel library | VERIFIED (post-v3 fix) | Already corrected in plan v3 citation block |
| Liu et al. [29] (AVA experimental source) | VERIFIED — Liu, Chen, Wang **2020** Nat Commun "GABAergic motor neurons bias locomotor decision-making" | v1 digitization JSON had wrong year (2018) and wrong title; correct to 2020. Acknowledgments note raw recordings shared by Wang/Liu labs — partial informal data provenance. |
| Liu et al. [30] (AIY experimental source) | VERIFIED but ambiguous | Liu/Kidd/Dobosiewicz/Bargmann 2018 *Cell*. Title is about AWA; AIY data presumably supplementary or raw-shared. Worth documenting limitation. |
| Wang 2001 → SHK-1 worm data | MISATTRIBUTED | Wang 2001 [60] is the SLO-1 NMJ paper, not SHK-1. Replace SHK-1 citation with [28, 30, 44, 45] per Nicoletti 2024 (Dobosiewicz 2019, Liu 2018, Wei 2005, Gu 2012). |
| "20 mV / 600 ms" numerical values | NO PRIMARY SOURCE FOUND | Reconstruction: ~20 mV is AVAL depolarization range under +30 pA; 600 ms is unverified (between Nicoletti's 500/1000 ms protocols, or misremembered Mellem RMD value). Drop the specific target; re-ground to Nicoletti's published AVA phenotype description. |

**Three of seven load-bearing biological citation claims in the architectural plan are misattributed.** Plus the v1 digitization JSON's Liu reference was independently misattributed.

---

## Recommendations for architectural plan updates

1. **Remove or correct Mellem 2008 → AVA plateau claim** in lines 112, 177, 183, 275, 281, 294. Replace with re-grounding per Mellem investigation path α (Nicoletti's actual AVAL phenotype: linear I-V, RC-like passive responses, sustained-during-stimulus plateau in 4-channel cell).

2. **Correct Wang 2001 → SHK-1 attribution** in line 94. Replace with Wei 2005 (KCNQ-like K channels in C. elegans, JBC) as the primary worm-specific reference, plus Gu 2012 (Kv3.1 splicing, JBC) for the Kv-family kinetics framework, plus the Bargmann lab papers (Dobosiewicz 2019 Elife, Liu 2018 Cell) for in-vivo AIY/AWA context.

3. **Document the "20 mV / 600 ms" reconstruction as fabricated** — the specific number pair has no primary-source anchor. If a quantitative AVA target is needed, derive it from Nicoletti 2024's actual published protocols (1000 ms CC, 500 ms VC) and AVAL's measured depolarization range under +30 pA current injection.

4. **Document the data-provenance limitation:** Nicoletti's AVA model fits against raw recordings shared by Wang/Liu labs (acknowledgments line 1009), not fully against published data. AIY model fits against [30] which is primarily an AWA paper. These are limitations Wave 2 inherits and paper 3 should disclose.

5. **Standing procedure:** verify all biological citations against primary sources before adoption. Both today's audit findings (Mellem 2008 misattribution, Wang 2001 SHK-1 misattribution, Liu 2018 → 2020 misattribution) followed the same pattern: citation attached to claim without primary-source check, propagated through downstream artifacts. The cross-session adversarial review pattern catches these but the catch is downstream of damage propagation. Front-loading verification reduces propagation depth.

---

## Implications for Wave 2 trajectory

**The Mellem 2008 → AVA misattribution invalidates the framing of Phase F 2b's failure as "biological insufficiency."** The morphology fork commitment (3-4 weeks) was justified against a phantom target.

**Recommended Wave 2 re-grounding (path α from Mellem investigation):**

- Drop "Mellem 2008 plateau in AVA" as Gate 2b biological target
- Re-ground to Nicoletti's actual AVAL phenotype: linear I-V, RC-like passive responses, plateau-during-stimulus only, 4-channel cell (NCA + EGL19 + IRK + LEAK + UNC103)
- The 7-channel essential set translation work remains valuable (Wave 2 channel library is built and validated per-channel) but the 7 channels aren't AVA's actual channel set per Nicoletti
- Channels translated for AIY/RIM/etc. (SLO-1, SHK-1, SHL-1, KQT-3) remain valid for those cells
- IRK and UNC-103 are missing from our essential set — they're in Nicoletti's AVA. Adding these would complete the AVA-specific channel set

**Subsequent Wave 2 work blocks (per re-grounded path):**

1. Translate IRK + UNC-103 (the AVA channels we don't have yet)
2. Re-run Phase F Component 2b against Nicoletti's actual AVAL phenotype with 4-channel cell (NCA + EGL19 + IRK + LEAK; UNC-103 if validated)
3. If 4-channel AVA matches Nicoletti's published characterization (linear I-V, sustained-during-stimulus), Phase γ Gate 2 is cleared with revised target
4. No morphology fork needed for this re-grounded scope
5. Network integration (Phase δ) proceeds

**Paper 3 implications:**

Manuscript should disclose:
- The condition-6 framing in the original architectural plan was based on a misattributed biological target (Mellem 2008 in AVA)
- Re-grounding to Nicoletti's actual published phenotype is the principled path forward
- Wave 2's contribution: translation infrastructure + per-channel validation + Nicoletti-replica AVA cell-level model
- Limitations: data provenance partially informal (Wang/Liu raw recordings); AIY data ambiguity; full Mellem-style plateau dynamics in AVA *as a biological phenomenon* may or may not exist (Mellem's own data says it doesn't)
- Cross-session adversarial review pattern caught three citation propagation errors during Wave 2 development; documenting this as a methodology contribution

---

## Note on the methodology pattern

This audit caught three substantive errors, two on user-side (architectural plan misattributions: Mellem 2008 → AVA, Wang 2001 → SHK-1) and one on agent-side (v1 digitization JSON: Liu 2018 vs actual 2020). Plus the user's own pre-flight pushback caught a fourth: the audit prompt itself was constructed around fabricated per-cell citations.

The pattern: **citation propagation errors compound silently across artifacts unless front-loaded verification breaks the chain.** Today's audit demonstrates the value of cross-session adversarial review at the citation level, not just at the empirical/computational level.

For paper 4 (methodology paper), this is a concrete worked example of how the pattern produces decision-grade output through iterative correction. The Wave 2 trajectory was about to commit 3-4 weeks to a morphology fork against a misattributed target. The cross-session pattern caught it. The avoided cost (3-4 weeks) is a substantial validation of the methodology investment.
