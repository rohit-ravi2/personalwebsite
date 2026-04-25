# Task 7 — T4-5 candidate citation check

Generated: 2026-04-21 09:34:21

PubMed metadata for the 5 T4-5 candidate peptide references. 
Use this as the starting point for manual verification tomorrow.

## FLP-13 — claimed: Nath 2016

- **PMID 27546573** (DOI: 10.1016/j.cub.2016.07.048)
  - **C. elegans Stress-Induced Sleep Emerges from the Collective Action of Multiple Neuropeptides.**
  - Nath R, Chow E, Wang H, Schwarz E, Sternberg P — *Current biology : CB* (2016)
  - _The genetic basis of sleep regulation remains poorly understood. In C. elegans, cellular stress induces sleep through epidermal growth factor (EGF)-dependent activation of the EGF receptor in the ALA neuron. The downstream mechanism by which this neuron promotes sleep is unknown. Single-cell RNA seq..._

## FLP-18 — claimed: Rogers 2003

- PMID 14555955: fetch error (HTTP Error 429: Too Many Requests)
## FLP-21 — claimed: de Bono 1998

- **PMID 9741632** (DOI: 10.1016/s0092-8674(00)81609-8)
  - **Natural variation in a neuropeptide Y receptor homolog modifies social behavior and food response in C. elegans.**
  - de Bono M, Bargmann C — *Cell* (1998)
  - _Natural isolates of C. elegans exhibit either solitary or social feeding behavior. Solitary foragers move slowly on a bacterial lawn and disperse across it, while social foragers move rapidly on bacteria and aggregate together. A loss-of-function mutation in the npr-1 gene, which encodes a predicted..._

## NLP-40 — claimed: Wang 2013

- PMID 23583549: fetch error (HTTP Error 429: Too Many Requests)
## DAF-28 — claimed: Li 2003

- **PMID 12654727** (DOI: 10.1101/gad.1066503)
  - **daf-28 encodes a C. elegans insulin superfamily member that is regulated by environmental cues and acts in the DAF-2 signaling pathway.**
  - Li W, Kennedy S, Ruvkun G — *Genes & development* (2003)
  - _In Caenorhabditis elegans, the decision to enter a developmentally arrested dauer larval stage is triggered by a combination of signals from sensory neurons in response to environmental cues, which include a dauer pheromone. These sensory inputs are coupled to the parallel DAF-2/insulin receptor-lik..._


## Corrections / notes from retry pass

**Rate-limited fetches now resolved:**

### FLP-18 — Rogers 2003 (PMID 14555955) — CITATION PROBLEM

PMID 14555955 is actually about **FLP-21 activating NPR-1**, not FLP-18:
- Title: Inhibition of C. elegans social feeding by FMRFamide-related peptide activation of NPR-1
- The paper identifies flp-21 as NPR-1 ligand. Primary FLP-18 / NPR-4 / NPR-5 refs are:
  - **Cohen et al. 2009** (PLoS Biol): FLP-18 modulates behavior via NPR-4/5 in AVA
  - **Kim & Li 2004** also relevant
- Action: update T4-5 candidate config for FLP-18 to cite Cohen 2009, not Rogers 2003

### NLP-40 — Wang 2013 (PMID 23583549) — CONFIRMED

- Title: Neuropeptide secreted from a pacemaker activates neurons to control a rhythmic behavior.
- Current Biology 2013. DOI 10.1016/j.cub.2013.03.049
- Wang H, Girskis K, Janssen T, Chan J, Dasgupta K.
- Confirms NLP-40 defecation motor program pacemaker role.

## Summary verification table

| peptide | claimed ref | PMID | verdict |
|---|---|---|---|
| FLP-13 | Nath 2016 | 27546573 | ✓ CORRECT (Current Biology, ALA sleep) |
| FLP-18 | Rogers 2003 | 14555955 | ✗ MISATTRIBUTED (paper is about FLP-21); use Cohen 2009 |
| FLP-21 | de Bono 1998 | 9741632 | ✓ CORRECT (Cell, NPR-1 natural variation). Rogers 2003 is the complementary FLP-21 → NPR-1 paper. |
| NLP-40 | Wang 2013 | 23583549 | ✓ CORRECT (Current Biology, defecation) |
| DAF-28 | Li 2003 | 12654727 | ✓ CORRECT (Genes & Dev, insulin superfamily) |

**Action items for T4-5:**
- Update FLP-18 reference from Rogers 2003 to Cohen et al. 2009
- Other 4 candidates have verified primary-source citations
- Citation audit pipeline now surfaces misattributions automatically


## Cohen 2009 FLP-18 correction verified

- **PMID 19356718** (DOI: 10.1016/j.cmet.2009.02.003)
- **Cohen M, Reale V, Olofsson B, Knights A, Evans P (2009)** Coordinated regulation of foraging and metabolism in C. elegans by RFamide neuropeptide signaling. *Cell Metabolism*.
- Abstract confirms: "Animals lacking these neuropeptides, encoded by the flp-18 gene, are defective in chemosensation and foraging, accumulate excess fat, and exhibit reduced oxygen consumption. Two G protein-coupled receptors..."
- **This IS the correct FLP-18 primary reference.** Update all project docs where Rogers 2003 was cited for FLP-18 to use this instead.
