# Wave P — Empirical anchor matrix

**Status:** SCAFFOLDED. Citation hygiene's load-bearing artifact. Every quantitative biological claim in any Wave P document must appear in this file or be flagged as unsourced.

**Update protocol:** when a new claim enters a preregistration document or CSV, add a row to the relevant table here. When the corresponding phase finishes, fill in the simulator's value and pass/fail.

---

## Per-paper anchor table

| Anchor | Paper | PMID | Claim | Phase that validates | Predicted threshold | Match criterion | Sim value | Pass |
|---|---|---|---|---|---|---|---|---|
| 1 | Crowder 1996 *PNAS* | 8855256 | WT halothane EC50 in C. elegans is ~3% atm (~340 µM aqueous) | H | EC50 = 340 ± 100 µM aqueous | within 2× | TBD | TBD |
| 2 | Morgan 1995 (PMID lookup needed) | (lookup) | WT isoflurane EC50 ~5% atm | H | EC50 = 290 ± 100 µM aqueous | within 2× | TBD | TBD |
| 3 | Morgan & Sedensky 1995 *Genetics* | 7549290 | gas-1(fc21) iso EC50 leftward 2-3× | H | gas1/WT ratio 0.33-0.5 | within 50% (so 0.25-0.67) | TBD | TBD |
| 4 | Sedensky 1992 *Genetics* | 1346264 | unc-79(e1068) halothane EC50 rightward 2-3× | H | unc79/WT ratio 1.5-3 | within 50% | TBD | TBD |
| 5 | Sedensky 1992 *Genetics* | 1346264 | unc-80(e1069) similar to unc-79 | H | unc80/WT ratio 1.5-3 | within 50% | TBD | TBD |
| 6 | Sedensky 2001 *Am J Physiol Cell Physiol* | 11756669 | twk-18(cn110) halothane resistance 2-3× | H | twk18/WT ratio 1.5-3 | within 50% | TBD | TBD |
| 7 | van Swinderen 1999 (PMID lookup needed) | (lookup) | unc-13(s69) halothane hypersensitivity 2-3× | H | unc13/WT ratio 0.33-0.67 | within 50% | TBD | TBD |
| 8 | Boddington 2017 (PMID lookup needed) | (lookup) | propofol immobilization in µM range | H | EC50 = 1 µM ± order of magnitude | within 10× | TBD | TBD |

---

## Mechanistic anchor table

| # | Paper | PMID | Claim | Phase that uses it | Notes |
|---|---|---|---|---|---|
| M1 | Hibbs & Gouaux 2011 *Nature* | 21572436 | C. elegans GluCl crystal structure (PDB 3RHW) | A, B | Direct structural anchor for GLC-1/-2/AVR-14/-15 |
| M2 | Mihic 1997 *Nature* | 9311784 | GABA-A / glycine receptor anesthetic-sensitive M2 site | B, D | Mammalian; extrapolated to UNC-49 |
| M3 | Sutton 1998 *Nature* | 9759724 | Neuronal SNARE complex crystal structure (PDB 1SFC) | A | Anchor for UNC-64/RIC-4/SNB-1 SNARE bundle |
| M4 | Richmond 1999 *Nat Neurosci* | 10570485 | unc-13(s69) hypomorph reduces release 80-90% | E | Markov synapse module validation |
| M5 | Kayser 2001 (PMID lookup needed) | (lookup) | gas-1 reduces Complex I activity 30-50% | F | Metabolic layer baseline |
| M6 | van Swinderen 2004 (PMID lookup needed) | (lookup) | Halothane reduces Ca cooperativity n in C. elegans NMJ from 3.5 to 2.0 | E | SNARE-machinery anesthetic shift |
| M7 | Lolicato 2017 *Cell* | 28729657 | TREK-1 K2P fenestration anesthetic site (PDB 6CQ6) | A, B | Mammalian K2P anchor for TWK-18 family |
| M8 | Xie 2020 *Nature* | 33020732 | Mammalian NALCN cryoEM structure (PDB 7SX3) | A, B | Mammalian anchor for NCA-1/NCA-2 |
| M9 | Zhu 2016 *Nature* | 27548872 | Bovine Complex I cryoEM (PDB 5LDX) | A | Mammalian anchor for GAS-1 / NUO-* |
| M10 | Nicoletti 2024 *PLOS ONE* | 38551921 | C. elegans cellular biophysics; 22 channel set | (Wave 2; Wave P consumes) | Wave P uses Wave 2's translations of these channels |
| M11 | Bentley 2016 *PLOS Comp Bio* | (PMID lookup needed) | Peptide-receptor mapping in C. elegans | (notebook pipeline; Wave P consumes via A_peptide) | |
| M12 | Cook 2019 *Nature* | 31270481 | C. elegans connectome (Witvliet/Cook) | (notebook pipeline) | |

---

## Methodological-tool anchor table

| # | Tool | Paper | PMID/DOI | Phase using |
|---|---|---|---|---|
| T1 | ColabFold | Mirdita 2022 *Nat Methods* | 35637307 | A |
| T2 | RoseTTAFold-AllAtom | Krishna 2024 *Science* | 38386700 | A (backup) |
| T3 | AlphaFold-Multimer | Evans 2022 bioRxiv | 10.1101/2021.10.04.463034 | A |
| T4 | FoldSeek | van Kempen 2023 *Nat Biotech* | 37156916 | A |
| T5 | AutoDock Vina 1.2 | Eberhardt 2021 *J Chem Inf Model* | 34003684 | B |
| T6 | DiffDock | Corso 2023 ICLR | 10.48550/arXiv.2210.01776 | B |
| T7 | GNINA | McNutt 2021 *J Cheminformatics* | 34108026 | B |
| T8 | fpocket | Le Guilloux 2009 *BMC Bioinformatics* | 19486540 | B |
| T9 | OpenMM | Eastman 2017 *PLOS Comp Bio* | 28746567 | D |
| T10 | AMBER ff14SB | Maier 2015 *J Chem Theory Comput* | 26574453 | D |
| T11 | GAFF | Wang 2004 *J Comput Chem* | 10.1002/jcc.20035 | D |
| T12 | Brian2 | Stimberg 2019 *eLife* | 10.7554/eLife.47314 | E, G |
| T13 | Gillespie SSA | Gillespie 1977 *J Phys Chem* | 10.1021/j100540a008 | E |

---

## Pre-flight blocking items (PMIDs to verify before phase entry)

| Phase | Blocking citations | Status |
|---|---|---|
| A | Rahman 2022 (Torpedo nAChR PDB 7QL5); Laverty 2019 (GABA-A 6X3X PMID re-verify) | Blocking |
| B | Yip 2013 propofol GABA-A photolabel; Jayakar 2014 propofol GABA-A; Trudell isoflurane GABA-A | Blocking |
| C | Morgan 1995 isoflurane EC50; Boddington 2017 propofol; halothane K_p primary source | Blocking |
| D | van Swinderen 2004; Kayser 2001; Hales & Lambert (GABA-A); Mihic 1997 re-verify | Blocking |
| E | van Swinderen 2004; Dodge & Rahamimoff 1967; Krasowski & Harrison 1999; Liu/Hu/Wang NMJ paper | Blocking |
| F | Kayser 2001/2004/2008; Falk 2006; Nichols 2006 K-ATP; Munro 1990 ATP rates | Blocking |
| G | Morgan 1995; van Swinderen 1999; van Swinderen 2004 | Blocking |
| H | Morgan 1995; van Swinderen 1999; Boddington 2017 | Blocking |

The above list is the **pre-flight verification queue** for the program. Each PMID lookup is a half-day's work; the entire queue resolves in 2-3 days of focused citation-verification work. This is the first work block before Phase A executes.

---

## Citation-misattribution lessons from Wave 2 (do not repeat)

Wave 2 caught four primary-source misattributions; Wave P enforces from day 1:

1. **"20 mV / 600 ms in AVA"** as a Mellem 2008 target — Mellem 2008 explicitly reports AVA does NOT show plateau. The plateau characterization is in RMD, with no specific 20 mV / 600 ms quantification. Cost Wave 2 ~3-4 weeks.
2. **Wang 2001 → SHK-1** — Wang 2001 is about SLO-1 at the NMJ, not SHK-1. The SHK-1 anchor papers are Wei 2005, Gu 2012, Dobosiewicz 2019, Liu 2018.
3. **Liu 2018 *Cell*** — actually 2020 Nat Commun in the relevant Nicoletti reference; year drift through agent fabrication.
4. **`10.1371/journal.pcbi.1007611`** "Nicoletti 2019 PLOS Comp Bio" — wrong DOI; resolves to a glioma paper unrelated to *C. elegans*. Real Nicoletti 2019 is `10.1371/journal.pone.0218738`.

Wave P guards against all four patterns by:

- Requiring PMID/DOI inline at point of citation.
- Marking unverified citations explicitly.
- Pre-flight verification before phase entry.
- Cross-checking quantitative biological claims (e.g., "EC50 is 3% atm") against the cited figure.

---

## Mutant panel (consumed by `validation/mutant_panel.csv`)

The simulator should predict phenotypes for every mutant the validation runs reference. See `mutant_panel.csv`.
