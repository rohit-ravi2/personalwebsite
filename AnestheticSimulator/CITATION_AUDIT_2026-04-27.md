# Citation and identifier audit — 2026-04-27

**Scope:** Initial pass on the 4 desk-work blocking items from `SETUP_COMPLETE.md`. Items 3 (Wave 2 ship status) and 4 (`/mnt/ssd4tb/` storage) resolved fully. Items 1 (PMID pre-flight) and 2 (UniProt ID re-verification) revealed systematic citation-hygiene failures requiring a follow-up systematic audit.

**Status:** PARTIAL — items 3 and 4 closed; items 1 and 2 escalated.

---

## Block 4: Storage allocation — RESOLVED

`/mnt/ssd4tb/` reports:

```
/dev/nvme1n1    3.6T  1.3T  2.2T  36% /mnt/ssd4tb
```

2.2 TB free. Wave P peak need is ~120 GB. **PASS, no action needed.**

`~/` (root SSD, `/dev/nvme0n1p6`): 712 GB free. Available as overflow if ever needed.

---

## Block 3: Wave 2 IRK + UNC-103 ship status — RESOLVED

Both channels SHIPPED. Found at:

- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels/irk.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels/unc103.py`

Plus all 13 other Nicoletti-2024 essential channels in the same directory: `cca1.py`, `egl19.py`, `egl2.py`, `kqt1.py`, `kqt3.py`, `nca.py`, `shk1.py`, `shl1.py`, `slo1_egl19_coupled.py`, `slo1_iso_dynamic_ca.py`, `slo1_iso.py`, `unc2.py`. AVA cell wrapper at `wave2/option_alpha_ava_cell.py`. Validation harness at `wave2/validate_phase_f_gate2.py`.

**Phase G of Wave P is unblocked when it gets there in month 4.** No action needed now.

---

## Block 1: PMID pre-flight verification — CRITICAL FINDINGS

Verified 8 of 9 unique PMIDs that were either marked `(PMID lookup needed)` or already cited in the kickoff docs. **3 of the already-cited PMIDs are wrong, and 4 of the previously-marked-"lookup needed" citations turned out to be fabricated entirely (the cited paper does not exist).**

### Verified correct (no action)

| Cited as | Actual | Status |
|---|---|---|
| Hibbs 2011 *Nature* PMID 21572436 | Hibbs RE, Gouaux E. "Principles of activation and permeation in an anion-selective Cys-loop receptor." *Nature* 474:54-60 (2011). DOI 10.1038/nature10139 | CORRECT — Tier-1 GluCl structural anchor (PDB 3RHW, 3RIF, 3RI5) |
| Nicoletti 2024 *PLOS ONE* PMID 38551921 | Nicoletti M, Chiodo L, Loppini A, Liu Q, Folli V, Ruocco G, Filippi S. *PLOS ONE* 19(3):e0298105 (2024). DOI 10.1371/journal.pone.0298105 | CORRECT — 22-channel library + 7-cell wrappers |

### Newly verified (was `(PMID lookup needed)`)

| Cited as | Actual | PMID | Notes |
|---|---|---|---|
| van Swinderen 1999 | van Swinderen B, Saifee O, Shebester L, Roberson R, Nonet ML, Crowder CM. *PNAS* 96(5):2479-2484 (1999) | **10051668** | ⚠ Paper is about **unc-64 (syntaxin)** + secondary effects on synaptobrevin (snb-1) and SNAP-25 (ric-4). The Wave P docs claim it as "unc-13 hypersensitivity" anchor — **this is a misattribution**. The paper does not characterize unc-13 phenotypes. Need to find the actual unc-13 anesthesia paper if that claim is to remain in the validation set. |
| Kayser 2001 (gas-1 Complex I) | Kayser EB, Morgan PG, Hoppel CL, Sedensky MM. *J Biol Chem* 276(23):20551-20558 (2001). DOI 10.1074/jbc.M011066200 | **11278828** | Confirms gas-1 is the 49-kDa Complex I subunit homolog (NDUFS2). Load-bearing for Phase F metabolic layer. |
| Bentley 2016 *PLoS Comp Biol* | Bentley B, Branicky R, Barnes CL, Chew YL, Yemini E, Bullmore ET, Vértes PE, Schafer WR. *PLoS Comput Biol* 12(12):e1005283 (2016). DOI 10.1371/journal.pcbi.1005283 | **27984591** | Multilayer connectome (synaptic + monoamine + neuropeptide). Source of A_peptide adjacency in the notebook pipeline. |
| Atanas 2023 *Cell* | Atanas AA et al. (Flavell lab, MIT). *Cell* 186(19):4134-4151.e31 (2023). DOI 10.1016/j.cell.2023.07.035 | **37607537** | Whole-brain calcium imaging in freely moving worms (~60 animals). Public data at wormwideweb.org. Phase I inverse-design ground truth. |

### Confirmed wrong PMID — already cited in docs

| Cited as | Actual citation | Real PMID | Cited PMID was wrong because |
|---|---|---|---|
| Crowder 1996 *PNAS* PMID 8855256 | Crowder CM, Shebester LD, Schedl T. "Behavioral effects of volatile anesthetics in *Caenorhabditis elegans*." ***Anesthesiology*** 85(4):901-912 (1996) | **8873562** | Wrong journal (Anesthesiology, not PNAS) AND wrong PMID. The cited PMID 8855256 does not correspond to any Crowder C. elegans paper. |
| Morgan & Sedensky 1995 PMID 7549290 | Morgan PG, Sedensky MM. "Mutations conferring new patterns of sensitivity to volatile anesthetics in *Caenorhabditis elegans*." *Anesthesiology* 81(4):888-898 (**1994**) | **7943840** | Wrong year (1994, not 1995) AND wrong PMID. This is the foundational gas-1 paper; it predates the Kayser 2001 molecular characterization. |
| Sedensky 1992 PMID 1346264 (unc-79 anchor) | Sedensky MM, Meneely PM. "Genetic Analysis of Halothane Sensitivity in *Caenorhabditis elegans*." ***Science*** 236:952-954 (**1987**). DOI 10.1126/science.3576211 | **3576211** | Wrong year (1987, not 1992), wrong PMID, wrong journal. The actual paper that ties unc-79/unc-80 to halothane resistance. |

### Confirmed fabricated — citation does not exist as written

| Cited as | What appears to be the actual reference | Notes |
|---|---|---|
| Sedensky 2001 PMID 11756669 (twk-18 cn110 halothane resistance) | NOT FOUND. Closest: Kunkel MT, Johnstone TB, Thomas JH, Salkoff L. "Mutants of a Temperature-Sensitive Two-P Domain Potassium Channel." *J Neurosci* 20(20):7517-7524 (**2000**). PMID **11027209** | Kunkel 2000 is the canonical *twk-18* gain-of-function paper, but it does NOT focus on halothane. The "twk-18 confers halothane resistance" claim needs a different anchor — needs targeted lit search to find the actual paper or drop the claim. |
| van Swinderen 2004 (halothane reduces Ca cooperativity n at C. elegans NMJ from 3.5 to 2.0) | NOT FOUND. Closest in topic: Stewart BA, Mohtashami M, Trimble WS, Boulianne GL. "SNARE proteins contribute to calcium cooperativity of synaptic transmission." *PNAS* 97(25):13955-13960 (**2000**). PMID **11095753** | Stewart 2000 is mammalian/Drosophila NMJ, not C. elegans, and doesn't address halothane. The specific "n: 3.5 → 2.0 in C. elegans NMJ under halothane" claim has no surface in the literature search. **Likely fabricated quantitative anchor.** Phase E Markov synapse calibration must be re-grounded against a real measurement. |
| Boddington 2017 (propofol immobilization in C. elegans, µM range) | NOT FOUND under that name+year. Closest: Awal MR, Austin D, Florman J, Alkema M, Gabel CV, Connor CW. "Breakdown of Neural Function under Isoflurane Anesthesia: In Vivo, Multineuronal Imaging in *Caenorhabditis elegans*." *Anesthesiology* 129(4):733-743 (**2018**). PMID **30004907** | Awal 2018 is isoflurane (not propofol) but is the standard contemporary C. elegans anesthesia + multineuronal imaging anchor. The Boddington 2017 propofol claim needs a real source or to be dropped from Phase H validation matrix. |

---

## Block 2: UniProt ID re-verification — CRITICAL FINDINGS

**Spot-check on the load-bearing UNC-49 entry revealed the UniProt ID is wrong.**

### Specific failure

`tier1_targets.csv` row for UNC-49:

```
UNC-49,WBGene00006765,Q17791,GABA-A homolog;...
```

**Both identifiers are wrong:**

- **Q17791** is the UniProt entry for `C07E3.8`, a 233-aa protein with predicted α-(1→3)-fucosyltransferase activity (DUF223 domain, PHA-1 regulator family). Cross-references to **WBGene00007418**. **It is not UNC-49.**
- **WBGene00006765** is also wrong. Per WormBase + the Bamber et al. 1999 *J Neurosci* paper that originally characterized UNC-49 (PMID 10377345), the correct gene ID is **WBGene00006784** (sequence T21C12.1).
- The correct UniProt entry for UNC-49 (TrEMBL): **Q0PDK2** (annotated GABA receptor, integral plasma membrane, neuromuscular junction). Note: TrEMBL not Swiss-Prot; the C. elegans GABA receptor isoform structure (UNC-49A/B/C) means a single Swiss-Prot reviewed entry may not exist.

### Implication

The UniProt IDs across the 25-row Tier-1 CSV (and 25-row Tier-2 CSV) cannot be trusted without per-row verification. The same likely applies to the WormBase IDs. The kickoff agent appears to have plausibly-formatted but unverified identifiers at the per-target level.

This is exactly the pattern Wave 2's citation audits caught (Mellem misattribution, Wang 2001 → SHK-1, Liu 2018 → 2020 year drift). The discipline did not propagate into the Wave P kickoff. **Flagging this as a methodology lesson** — the kickoff agent should have been required to verify each PMID/UniProt ID against a primary source before populating the CSV, and instead generated identifiers from pattern-matching.

### Conservative path forward

Two options:

**(A) Systematic re-verification pass.** Spawn a follow-up agent whose entire job is to verify each of the 50 (Tier 1 + Tier 2) target rows against UniProt + WormBase, and each PMID against PubMed. Rough cost: 1-2 hours of agent time. Output: a single corrected CSV per tier with verified columns + a `CITATION_CORRECTIONS_LOG.md` enumerating each change.

**(B) Defer until Phase A starts.** Phase A (structural priors) needs the structures, not the cited PMIDs, and the UniProt → AlphaFold DB lookup will surface ID mismatches naturally (a wrong UniProt ID → wrong predicted structure → fpocket / blastp sanity check fails). This pushes the audit cost into Phase A's prep step rather than treating it as a separate audit.

**Recommendation: (A).** The cost is bounded (~2 hours) and clearing the audit before Phase A means Phase A can be executed cleanly without surfacing identifier issues mid-flight. Wave 2's discipline says verify before committing.

---

## Pre-flight verification queue snapshot

For convenience, the corrected citations to use across `validation/empirical_anchors.md`, `targets/target_panel_rationale.md`, `preregistration/phase_*.md`, and any other doc:

| Topic | Use this | Drop this |
|---|---|---|
| WT halothane EC50 | Crowder 1996 *Anesthesiology* PMID **8873562** | Crowder 1996 *PNAS* PMID 8855256 |
| Mutations conferring anesthetic sensitivity / gas-1 origin | Morgan & Sedensky **1994** PMID **7943840** | Morgan & Sedensky 1995 PMID 7549290 |
| unc-79 / unc-80 halothane resistance | Sedensky & Meneely **1987** *Science* PMID **3576211** | Sedensky 1992 PMID 1346264 |
| gas-1 Complex I molecular characterization | Kayser et al. 2001 *JBC* PMID **11278828** | Kayser 2001 (PMID lookup needed) |
| Multilayer connectome / Bentley A_peptide | Bentley et al. 2016 *PLoS Comput Biol* PMID **27984591** | Bentley 2016 (PMID lookup needed) |
| Whole-brain calcium imaging (Phase I anchor) | Atanas et al. 2023 *Cell* PMID **37607537** | Atanas 2023 (PMID lookup needed) |
| Syntaxin (unc-64) halothane resistance | van Swinderen et al. 1999 *PNAS* PMID **10051668** | van Swinderen 1999 (mis-cited as "unc-13 hypersensitivity") |
| C. elegans isoflurane multineuronal anchor | Awal et al. 2018 *Anesthesiology* PMID **30004907** | Boddington 2017 (does not exist) |
| GluCl structure | Hibbs & Gouaux 2011 *Nature* PMID **21572436** | (already correct) |
| Nicoletti channel library | Nicoletti et al. 2024 *PLoS ONE* PMID **38551921** | (already correct) |

### Citations to re-find or drop

| Claim | Status |
|---|---|
| twk-18(cn110) confers halothane resistance | Specific paper not located. Kunkel 2000 PMID 11027209 characterizes twk-18 GoF but doesn't address halothane. Need targeted search or drop the claim from Phase D / Phase H validation. |
| van Swinderen 2004: halothane reduces Ca cooperativity n in C. elegans NMJ from ~3.5 to ~2.0 | Specific paper not located. Stewart 2000 PMID 11095753 is the SNARE-Ca-cooperativity paper but is mammalian/Drosophila. The specific quantitative C. elegans claim is likely fabricated. Phase E Markov synapse calibration must use a verified measurement. |
| unc-13(s69) halothane hypersensitivity 2-3× | Originally pinned to van Swinderen 1999, which is actually about unc-64. Need real paper or drop. Possible candidate: Saifee et al. or Hosono et al. unc-13 characterizations — needs further search. |

---

## Action items remaining

1. **Spawn citation-audit agent** to do option (A): systematic UniProt/WormBase/PMID re-verification across all 50 target rows, all PMIDs in `validation/empirical_anchors.md`, all PMIDs cited in any preregistration document, and all PMIDs in skeleton Python files. ~2 hours expected. Apply corrections inline (use Edit tool, not Write). Produce a `CITATION_CORRECTIONS_LOG.md`.

2. **Update `STATUS.md`** to reflect that desk-work blockers 3 and 4 are CLOSED, and blockers 1 and 2 are now expanded into a single "systematic citation audit" task that subsumes both.

3. **Update `SETUP_COMPLETE.md`** with the same.

4. **Methodology note for paper 4:** the kickoff agent's failure to verify identifiers before populating CSVs is exactly the pattern Wave 2 caught (Mellem misattribution → 3-4 weeks of wrong-target work avoided). Add this to the methodology-paper case-study catalog.

---

## Summary

| Desk-work blocker | Status |
|---|---|
| 1. PMID pre-flight verification | PARTIAL — 9 PMIDs verified (8 actionable corrections), but the audit revealed broader citation hygiene failures requiring systematic re-verification |
| 2. UniProt ID re-verification | ESCALATED — spot-check of UNC-49 row revealed wrong UniProt ID + wrong WormBase ID; 25-row CSV cannot be trusted without per-row verification |
| 3. Wave 2 IRK + UNC-103 ship status | CLOSED — both channels SHIPPED at `wave2/channels/irk.py` and `wave2/channels/unc103.py` plus all 13 other essential channels |
| 4. /mnt/ssd4tb/ storage check | CLOSED — 2.2 TB free, 36% used; Wave P peak ~120 GB easily fits |

**Net:** 2 of 4 blockers closed. The remaining 2 collapse into a single systematic-audit task that should run before Phase A executes. Estimated cost: ~2 agent-hours, $0 external spend.
