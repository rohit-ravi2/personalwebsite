# Pre-flight pushback — rigor-tightening work block

**Author:** Wave P session 2 (post Phase E/F/H 5/5 PASS)
**Status:** PAUSED for cross-session review
**Date:** 2026-04-27

---

## Summary

Five pre-flight verification items requested. All five surface concerns that warrant cross-session review before launching CP1-CP8 as designed. The most consequential: **Phase F is fully parameter-locked**, not biologically informative — sensitivity sweep confirms predicted ratio is determined entirely by `GAS1_COMPLEX_I_FACTOR`, with the anesthetic's input block_factor having no effect. The "5/6 PASS" verdict for Phase F was tuning to target. Two of the three deferred anchors (twk-18, propofol) cannot be rescued — must be REMOVED rather than DEFERRED. One deferred anchor inverts direction (K2P gain-of-function is hypersensitive, not resistant).

---

## Pre-flight item 1 — Parameter inventory

**Phase F tunable parameters (full inventory):**

| # | Parameter | Value | Status | Notes |
|---|---|---|---|---|
| 1 | K_COMPLEX_I_WT | 1.0 | Reference (normalized) | — |
| 2 | K_COMPLEX_II | 0.3 | Literature-defensible | Approx 30% Complex II contribution to ATP production |
| 3 | K_COMPLEX_V | 1.0 | Reference (coupling factor) | — |
| 4 | K_BASE_CONSUMPTION | 1.3 | **CALIBRATED** | Set so WT ATP_ss = 1.0; one degree of freedom |
| 5 | ATP_SS_REFERENCE | 1.0 | Reference | — |
| 6 | K_ATP_HALF | 0.05 | **HAND-TUNED** | "Soft threshold for K-ATP opening" |
| 7 | G_K_ATP_MAX | 2.0 | **HAND-TUNED** | "Strong shift potential" |
| 8 | E_K | -90 mV | Literature | Standard K reversal |
| 9 | V_REST_BASELINE | -60 mV | Literature for C. elegans neurons | — |
| 10 | G_TOTAL_OTHER | 1.0 | Reference (normalized) | — |
| 11 | V_SHIFT_IMMOBILIZATION | 5.0 mV | **HAND-CHOSEN** | Behavioral threshold |
| 12 | GAS1_COMPLEX_I_FACTOR | 0.4 | **CALIBRATED to Morgan target** | Originally 0.5, lowered to 0.4 to land in 2-3× band |

**Effectively-tunable: 4 parameters (#4, #6, #7, #11, #12 — actually 5).** STATUS log called out 3 — undercounted. Parameters #6, #7, #11 are completely hand-set with no independent literature derivation. Parameter #12 was explicitly tuned to hit the Morgan target band.

**Phase E tunable parameters:**

| # | Parameter | Value | Status |
|---|---|---|---|
| 1 | CA_REST_uM | 0.1 | Literature |
| 2 | CA_PEAK_uM | 5.0 | Literature |
| 3 | CA_PEAK_DURATION_MS | 1.0 | Literature |
| 4 | AP_FREQUENCY_HZ | 10.0 | Chosen for evoked-release scenario |
| 5 | N_RELEASE_SITES | 5 | Chosen (could be 3-10) |
| 6 | N_CA_COOPERATIVITY_WT | 3.5 | Stewart 2000 literature |
| 7 | K_ON_PER_uM_PER_MS | 0.0001 | **CALIBRATED** to give realistic spontaneous + evoked rates |
| 8 | K_FUSE_PER_MS | 1.0 | Chosen |
| 9 | K_RECYCLE_PER_MS | 0.005 | Literature ~200 ms recovery |
| 10 | CLINICAL_EFFECTIVE_OCCUPANCY | 0.30 | **HAND-TUNED MID-FLIGHT** to fix saturation |

**Effectively-tunable: 4 parameters (#4, #5, #7, #10).** Most consequential is #10 (added mid-flight specifically to make halothane fold-change land in Stewart band).

---

## Pre-flight item 2 — Phase F sensitivity sweep (saturation hypothesis)

Sensitivity sweep produced via `src/preflight_phase_f_saturation.py`.

### Sweep 1 — block_factor varied across [0.05, 0.95]; gas1_factor=0.4 fixed

| block_factor | block % | WT dose | gas-1 dose | predicted ratio |
|---|---|---|---|---|
| 0.05 | 95% | 0.76 | 0.31 | **2.452** |
| 0.10 | 90% | 0.80 | 0.32 | **2.500** |
| 0.20 | 80% | 0.90 | 0.36 | **2.500** |
| 0.30 | 70% | 1.03 | 0.42 | **2.452** |
| 0.40 | 60% | 1.20 | 0.48 | **2.500** |
| 0.50 | 50% | 1.43 | 0.58 | **2.466** |
| 0.60 | 40% | 1.79 | 0.72 | **2.486** |
| 0.70 | 30% | 2.39 | 0.96 | **2.490** |
| 0.85 | 15% | 4.77 | 1.92 | **2.484** |

Across **17× variation** in block_factor (0.05 → 0.85), the ratio varies by **0.05** (range 2.452-2.500 — essentially constant).

### Sweep 2 — gas1_factor varied; block_factor=0.706 (halothane) fixed

| gas1_factor | WT dose | gas-1 dose | ratio |
|---|---|---|---|
| 0.30 | 2.44 | 0.18 | 13.556 |
| 0.40 | 2.44 | 0.98 | 2.490 |
| 0.50 | 2.44 | 1.47 | 1.660 |
| 0.60 | 2.44 | 1.79 | 1.363 |
| 0.70 | 2.44 | 2.02 | 1.208 |
| 0.80 | 2.44 | 2.19 | 1.114 |

The ratio varies dramatically (13.6 → 1.1) as gas1_factor varies across the Kayser literature range (0.3-0.7) and beyond. **gas1_factor=0.4 is uniquely positioned in Morgan's 2-3× band.**

### Joint sweep — block_factor × gas1_factor

| gas1\block | 0.10 | 0.30 | 0.50 | 0.706 | 0.85 |
|---|---|---|---|---|---|
| 0.30 | 13.33 | 12.88 | 14.30 | 13.56 | 14.03 |
| 0.40 | **2.500** | **2.452** | **2.466** | **2.490** | **2.484** |
| 0.50 | 1.667 | 1.661 | 1.663 | 1.660 | 1.662 |
| 0.60 | 1.356 | 1.373 | 1.362 | 1.363 | 1.363 |

**Verdict: Phase F is parameter-locked.** The predicted gas-1/WT ratio is determined almost entirely by GAS1_COMPLEX_I_FACTOR, with block_factor (the anesthetic-specific input from wave2_overlay.json) contributing < 0.05 variation. The clustered 2.48-2.49× across all 5 anesthetics is not biological signal — it's the output of `f(GAS1_COMPLEX_I_FACTOR=0.4)` independent of input.

**Mathematical explanation:** The model's WT_dose and gas-1_dose both scale with the same function of block_factor. When taking the ratio, block_factor effectively cancels out, leaving the ratio dependent only on the relative ATP availability (1.0 vs 0.4) which is set by GAS1_COMPLEX_I_FACTOR.

**Implication: Phase H anchor #1 (gas-1 hypersensitivity 2-3× PASS) is parameter-tuned, not preregistered-passed.** A Phase F that genuinely tested the multi-target framing would show different ratios for different anesthetics based on their Complex I engagement; this one doesn't.

---

## Pre-flight item 3 — Phase E CLINICAL_EFFECTIVE_OCCUPANCY justification

**The factor 0.30 was added mid-flight after Phase E first-pass produced saturation (release-p → 0).** No independent literature derivation exists. The chain of reasoning was:

1. wave2_overlay.json `n_Ca_delta` for halothane → UNC-64 = -1.45 (computed at K_p-amplified saturating occupancy ~0.97)
2. Applied to Markov model → release-p drops to 0 (full block)
3. Stewart 2000 reports halothane reduces release-p by 30-50% at clinical concentrations, NOT 100%
4. Mid-flight fix: scale n_delta by 0.30 to bring fold-change to 0.333 = within Stewart band

**The 0.30 has no derivation from Stewart's actual concentration data.** Without sensitivity analysis, this is a fitted parameter that "fits Stewart band by construction" rather than "predicts Stewart band."

**Required rigor work:**
- Derive CLINICAL_EFFECTIVE_OCCUPANCY from Stewart 2000's actual halothane concentration (1 MAC ≈ 280 µM aqueous) divided by saturating-K_p effective concentration (250 × 280 = 70,000 µM) and Kd_SNARE (need to extract Stewart's actual data)
- OR run sensitivity sweep showing Stewart band is reproduced across plausible CLINICAL_EFFECTIVE_OCCUPANCY range (e.g., 0.15-0.50)
- If only 0.30 specifically reproduces Stewart, Phase E is brittle and the pass is post-hoc fitting

---

## Pre-flight item 4 — Phase H preregistration verification

Preregistration document exists at `preregistration/phase_h_empirical_validation.md` and predates execution. Content includes:

- 8 anchors with tolerance bands ("within 2×" or "within 50%")
- Pass threshold ≥ 4/8 anchors within tolerance
- Halt rule: 0-1 pass → full pivot; 2-3 pass → partial; 4+ pass → success

**Preregistered tolerance bands (good methodology):**
- Anchor 1 (WT halothane EC50): within 2×
- Anchor 3 (gas-1 ratio): within 50% (so 1.5×-4.5× range)
- Anchor 7 (unc-13 hypersensitivity): within 50%
- Anchor 8 (propofol µM): within 10× (order of magnitude)

**However, the preregistration document uses citations now confirmed corrupt:**

| Anchor | Cited as | Status |
|---|---|---|
| 1 | Crowder 1996 PMID **8855256** | WRONG — real is **8873562** Anesthesiology |
| 3 | Morgan & Sedensky **1995** PMID **7549290** | WRONG — real is **1994** PMID **7943840** |
| 4 | Sedensky **1992** PMID **1346264** | WRONG — real is Sedensky & Meneely **1987** PMID **3576211** |
| 6 | Sedensky 2001 PMID **11756669** (twk-18 cn110 RESISTANT) | FABRICATED PMID + DIRECTION INVERTED (see item 5 below) |
| 7 | van Swinderen 1999 (unc-13) | DOMAIN MIS-CITED (paper is about unc-64) |
| 8 | Boddington 2017 (PMID lookup needed) | FABRICATED — no real source |

The numerical phenotype claims (gas-1 2-3× hypersensitive, halothane EC50 ~3% atm, etc.) are mostly correct as biological consensus, even though the citations supporting them are wrong. So preregistered tolerance bands ARE valid, but the supporting citation chain has 6/8 errors.

**For rigor purposes:** treat preregistration as VALID for tolerance bands but DEMANDING citation cleanup before claiming primary-source backing. The 5/5 PASS verdict is technically against preregistered bands, but the bands themselves rest on a corrupted citation map.

---

## Pre-flight item 5 — DEFERRED anchor verification

### Anchor 6 — twk-18(cn110) halothane resistance

Preregistration claim: "Sedensky 2001 PMID 11756669, twk-18(cn110) gain-of-function confers halothane RESISTANCE."

**Verification result: Citation FABRICATED + biological direction INVERTED.**

- Sedensky 2001 PMID 11756669 does not exist as a twk-18 paper. The real Sedensky 2001 papers are about stomatins (PMID 11287347) or GAS-1 Complex I (PMID 11278828) — neither addresses twk-18 or halothane.
- The actual *twk-18* characterization is Kunkel et al. 2000 *J Neurosci* 20(20):7517-7524 PMID **11027209** — but this paper does NOT report halothane phenotypes.
- Real K2P-halothane data in *C. elegans* comes from Singaram et al. (verified PMID lookup needed): K2P loss-of-function (sup-9(n180)) causes modest halothane RESISTANCE (EC50 3.35% vs WT 3.08%, p<0.006). K2P gain-of-function (unc-92(n200)) causes halothane HYPERSENSITIVITY (EC50 1.43%).

**The Wave P preregistration claim has the direction INVERTED.** "twk-18(cn110)gf → halothane resistance" should be "K2P gain-of-function → halothane hypersensitivity." The biology is real; the predicted direction was wrong.

**Implication:** Anchor 6 cannot be rescued by simply finding the correct paper. The directional claim itself was wrong. Either:
1. Restate the anchor as "K2P-gf → halothane hypersensitivity" using sup-9/unc-92 data and re-evaluate Wave P predictions, OR
2. Remove anchor 6 entirely from validation table

### Anchor 8 — propofol C. elegans EC50

Preregistration claim: "Boddington 2017, propofol immobilization in *C. elegans* in µM range."

**Verification result: NO PRIMARY SOURCE EXISTS for propofol-induced whole-animal immobilization EC50 in *C. elegans*.**

Searched extensively. The closest matches:
- **Heuer 2014** PMID 24501356 — propofol on recombinant *Haemonchus*/C. elegans GluCl in *Xenopus* oocytes; channel-level **IC50 = 252 ± 48 µM**. NOT whole-animal immobilization.
- **Zhang 2022** PMC9804065 — propofol learning/memory deficits in *C. elegans*, no immobilization EC50 reported.
- **Awal 2018** PMID 30004907 — multineuronal imaging under isoflurane (NOT propofol).

**Implication:** Wave P propofol predictions throughout the pipeline are tested against MAMMALIAN clinical EC50 (1-5 µM aqueous range) as a stand-in. Anchor 8 cannot be rescued and must be REMOVED from validation table.

### Anchor 10 — NCA-1/UNC-80 structures

Lu 2007 was confirmed to be the NALCN identification paper (Cell), not a binding study. AlphaFold DB does not have entries for the C. elegans NCA-1 (Q6Q762) or UNC-80 (Q9XV66) accessions.

**Verification result:** No structures available for these mass auxiliary subunits without ColabFold T4 fallback (deferred per R14 mitigation).

NCA-1/UNC-80 are functionally important — they define the canonical *C. elegans* halothane-resistance class (Sedensky & Meneely 1987 PMID 3576211 unc-79/unc-80). But:
- Wave P pipeline cannot dock against them without structures
- The unc-79/unc-80 anchor (anchor 4) requires network simulation against the WAVE 2 brain to evaluate (Phase G, not Phase B/C/D)

**Implication:** Anchor 10 stays DEFERRED (not REMOVED) — the genetics anchor (anchor 4) is what's load-bearing, not the structural anchor.

---

## Updated post-rigor verdict landscape

After applying these pre-flight findings, the realistic Phase H verdict count becomes:

| Anchor | Original verdict | Post-rigor verdict | Why |
|---|---|---|---|
| 1 (WT halothane EC50) | PASS_PENDING | PENDING (Phase G) | Requires network sim |
| 2 (WT iso EC50) | PASS_PENDING | PENDING (Phase G) | Requires network sim |
| 3 (gas-1 hypersensitivity) | PASS (5/6 anesthetics 2.48-2.49×) | **PASS_PARAMETER_LOCKED** | Sensitivity sweep shows ratio determined by GAS1_COMPLEX_I_FACTOR alone; output is f(0.4)≈2.48 regardless of input. The "5/6 PASS" was tuning to target. Real Phase F has ~zero biological information from anesthetic side. Honest verdict: pipeline shows the right ratio for the right reason in a one-parameter model, but cannot distinguish anesthetics. |
| 3-bis (multi-target framing) | PASS via discriminative gap 28 | **PASS_ROBUST** | No tunable parameters in Stage 5; load-bearing test |
| 4 (unc-79 resistance) | PENDING | PENDING (Phase G) | Requires network sim |
| 5 (unc-80 ~unc-79) | PENDING | PENDING (Phase G) | Requires network sim |
| 6 (twk-18 cn110 RESISTANT) | DEFERRED | **REMOVED + DIRECTION INVERTED** | Real biology says K2P-gf is hypersensitive; original anchor mis-stated |
| 7 (unc-13 hypersensitivity) | PENDING | PENDING (Phase G); citation needs new anchor | Original cite inverted (1999 paper is unc-64) |
| 8 (propofol µM EC50) | DEFERRED | **REMOVED** | No primary source; mammalian extrapolation only |
| 2-bis (SNARE release reduction) | PASS (halothane 0.333 in 0.3-0.7 band) | **PASS_PARAMETER_TUNED** | CLINICAL_EFFECTIVE_OCCUPANCY=0.30 chosen mid-flight to fit Stewart band; needs sensitivity sweep |
| 4-bis (rank correlation) | PASS (93%) | **PASS_ROBUST** | No tunable parameters in Stage 6 |
| 5-bis (Vina-Kd within 10×) | PASS (75%) | **PASS_ROBUST** (preregistered ≥50% within 10×) | Standard pipeline; tolerance band met cleanly |

**Robust passes (no tunable parameters): 3** — Stage 5 discriminative, Stage 6 rank correlation, Stage 4 calibration within tolerance
**Parameter-tuned passes (depend on hand-set values): 2** — Phase F gas-1, Phase E SNARE
**Removed (no primary source / direction inverted): 2** — propofol, twk-18
**Pending (require Phase G network sim): 5** — WT halothane EC50, WT iso EC50, unc-79 resistance, unc-80 resistance, unc-13 hypersensitivity

**Honest count: 3 robust + 2 parameter-tuned + 2 removed + 5 pending = 12 anchors total, 3-5 robust passes depending on whether parameter-tuned counts.**

This is a SMALLER but more defensible claim than "5/5 PASS."

---

## Proposed adjustments to CP1-CP8

The original 8-checkpoint plan covers most of these issues. Specific reinforcements:

### CP1 (Phase F sensitivity)
- ALREADY DONE in this pre-flight as the saturation diagnostic.
- Save the sensitivity table to `artifacts/calibration/phase_f_sensitivity.csv` formally.
- **Honest verdict: Phase F is parameter-locked.** Either reformulate the model (e.g., make WT_dose absolute and gas-1_dose relative to a fixed behavioral threshold) so block_factor doesn't cancel, OR downgrade Phase F to "demonstrates the qualitative direction of gas-1 hypersensitivity but cannot distinguish anesthetics."

### CP2 (Phase E sensitivity)
- Run the proposed sweep on CLINICAL_EFFECTIVE_OCCUPANCY [0.10, 0.20, 0.30, 0.40, 0.50, 0.70].
- Report which values produce halothane fold-change in Stewart 0.3-0.7 band.
- If band is reproduced across [0.20, 0.40], Phase E is robust. If only 0.30 specifically, Phase E is fitted.

### CP3 (DCE diagnostic)
- Original prompt's CP3. Verify Eger 2001 cis-DCE EC50.

### CP4-CP5 (strict-Kd subset)
- Search for true radioligand binding Kd values. Most likely sources:
  - Hall 1994 propofol photoaffinity Kd
  - Husain 2003 etomidate photoaffinity Kd
  - Forman 1996 nAChR halothane (verified existing)
  - Davies 1988 ketamine NMDA radioligand displacement
- Build strict-Kd subset; recalibrate.

### CP6 (DEFERRED resolution)
- twk-18: replace anchor with sup-9(lf) or unc-92(gf) per Singaram et al. corrected biology.
- propofol: REMOVE from validation table; document why.
- NCA-1/UNC-80 structures: stay DEFERRED for ColabFold; not on critical path.

### CP7 (allosteric correction + halogenated stratification)
- Per original prompt.

### CP8 (rock-solid verdict)
- Use the realistic post-rigor verdict count above (3 robust + 2 parameter-tuned + 2 removed + 5 pending).

---

## What I'm asking for

Cross-session review of these pre-flight findings before launching CP1-CP8. Three load-bearing decisions:

1. **Phase F as parameter-locked:** confirm or reject the sensitivity-sweep diagnosis. If accepted, anchor #3 PASS verdict is honestly downgraded to PASS_PARAMETER_LOCKED. If rejected, what additional sensitivity test is needed?

2. **Anchors 6, 8 removal vs DEFERRED:** anchor 6 (twk-18) and anchor 8 (propofol) cannot be rescued — should they be REMOVED entirely from validation table, or restated with corrected biology where possible (anchor 6 → sup-9/unc-92), or kept as DEFERRED with documentation?

3. **Honest verdict counting:** the post-rigor count is 3 robust + 2 parameter-tuned + 2 removed + 5 pending. Should the 2 parameter-tuned count as "PASS" or "PASS_TUNED" or "INCONCLUSIVE_TUNED"? This determines whether the headline becomes "3/5 robust passes (60%)" or "5/5 verifiable passes (100%, with 2 caveats)."

If adjustments accepted, the modified CP1-CP8 proceeds with these findings already documented. If rejected, will run as written.

Marker file at `artifacts/calibration/PAUSED_FOR_REVIEW.txt` will be created.
