# Task 8 — Ripoll-Sánchez 2023 cross-reference

Generated: 2026-04-21

## Paper identification

**Ripoll-Sánchez L, Watteyne J, Sun H, Fernandez R, Taylor SR, Weinreb A,
Bentley BL, Hammarlund M, Miller DM 3rd, Hobert O, Beets I, Vértes PE,
Schafer WR (2023)** The neuropeptidergic connectome of *C. elegans*.
*Neuron* 111(22):3570-3589.e5.
DOI: [10.1016/j.neuron.2023.09.043](https://doi.org/10.1016/j.neuron.2023.09.043).
PMID: 37935195. PMC: PMC7615469 (open access).

## Methods of the paper (relevant to our cross-reference)

- Integrates 3 data sources:
  - Taylor et al. 2021 (CeNGEN scRNAseq) — same source we use
  - Beets et al. 2023 (biochemical receptor-ligand pairing)
  - White 1986 + Albertson/Thomson 1976 + Witvliet et al. (anatomy)
- Emphasizes **volume transmission** — extrasynaptic peptide release
  reaches neurons not directly connected synaptically. Our simulator's
  T1c volume-transmission layer follows this principle.
- Three datasets generated: (1) comprehensive NPP-GPCR network,
  (2) range-constrained diffusion networks, (3) 92 per-peptide
  adjacency matrices.

## Key results relevant to our project

- **Peptidergic rich club = 52% of all neurons** (vs 11 neurons for the
  synaptic rich club from Towlson et al. 2013). Implication: peptidergic
  signaling is structurally dominant in C. elegans.
- **Three core communities** share input-connectivity patterns,
  suggesting functional specialization within peptidergic signaling.
- Highly interconnected core; autocrine foci (neurons signaling to
  themselves peptidergically).
- Several "key network hubs are little-studied neurons that appear
  specialized for peptidergic neuromodulation." — this matches the
  broadcaster-neuron pattern we use in Task 5 (I1-I5, M5, NSM).

## Cross-reference for our T4-5 candidates

| peptide | confirmed in Ripoll-Sánchez 2023? | their receptor(s) | our receptor assignment | discrepancy |
|---|---|---|---|---|
| FLP-13 | **✓ Yes** (versatile-neuropeptide group with FLP-4, FLP-9, FLP-10) | not surfaced in fetch | DMSR-1, DMSR-2 (per Nath 2016) | receptor detail not fetchable; likely consistent — DMSR family is known FLP-13 receptor |
| FLP-18 | **✓ Yes** (Fig 4A, pervasive network) | **NPR-5** specifically cited | NPR-1, NPR-4, NPR-5 | NPR-5 confirmed; NPR-1/NPR-4 are from other refs (Cohen 2009, Kim & Li 2004) |
| FLP-21 | ✓ Yes (implied by NPR-1 scaffold) | not surfaced in fetch | NPR-1 (per de Bono 1998 + Rogers 2003) | consistent |
| NLP-40 | **Not explicit in fetched content** | — | AEX-2 (per Wang 2013) | can't verify from fetch; need supplementary access |
| DAF-28 | **Not explicit in fetched content** | — | DAF-2 (per Li 2003) | can't verify from fetch; need supplementary access |

## Accessibility limitations encountered

- **GitHub repository** (github.com/LidiaRipollSanchez/Neuropeptide-Connectome)
  found. Contains CSV adjacency matrices + 92 per-peptide networks +
  scRNAseq expression data. WebFetch couldn't drill into individual
  files.
- **bioRxiv preprint** (doi 10.1101/2022.10.30.514396): blocked
  (HTTP 403).
- **NemaMod interactive website** (nemamod.org): not attempted via
  WebFetch; designed for interactive browsing not API access.
- **Figshare dataset** (doi 10.6084/m9.figshare.c.6895870.v1): not
  attempted; would require specific file download.

## What's verified, what's pending manual check

- **Verified from abstract + main text fetch:**
  - FLP-13 in the paper's connectome as a versatile-group peptide
  - FLP-18 → NPR-5 interaction
  - FLP-21 implied via NPR-1 scaffold
  - General methodology + rich-club statistics

- **Pending user access for verification:**
  - NLP-40 presence in RS23 connectome and its receptor assignment
  - DAF-28 presence (it may not be classified as "neuropeptide" per
    the paper's definition — insulin-family peptides are sometimes
    excluded from "peptide" networks as they fall into a separate
    family)
  - Full receptor sets per peptide (need supplementary tables)
  - Rich-club membership of our chosen releaser neurons
    (RIS, ALA, AVK, DVA, AVB, NSM, PDE, RIM, RIC, ASI/ASJ)

## Implication for project / paper framing

The project's T1c volume-transmission layer is **consistent with RS23's
central methodological finding** that peptide signaling is
extrasynaptic. This validates our architectural choice.

The **52% peptidergic rich club** statistic is worth citing in the
paper's methods section as background — it motivates why peptidergic
ablations should be expected to produce widespread network effects
(Mode 1 readout-blind phenomenon) rather than focal synaptic-cascade
effects (Mode 2/3 readout-trivial/cascade).

The **"key network hubs are little-studied neurons"** observation
directly supports the paper's argument about readout-architecture
sensitivity: the 18-neuron classifier intersection excludes many of
these hub neurons (PVW, RIG, I1-I5, AVJ, PVT, URB, etc.) which is
exactly the structural reason modulator effects stay invisible in
Mode 1.

## Action items for T4-5

1. **Update FLP-18 primary citation** from Rogers 2003 to
   Cohen et al. 2009 (Rogers 2003 is actually FLP-21 / NPR-1 per
   Task 7 verification). RS23 confirms NPR-5 as an FLP-18 receptor.
2. **Verify NLP-40 and DAF-28 presence** in RS23 manually when paper
   access available. If absent, note that these are separately-
   annotated peptide families and our citations (Wang 2013, Li 2003)
   remain primary.
3. **Paper citation to add:** Ripoll-Sánchez 2023 as the
   comprehensive neuropeptidergic connectome reference. Our target-set
   work should cite their supplementary tables once the full data
   is accessed.

## Sources cited

- [Ripoll-Sánchez 2023 Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(23)00756-0)
- [PubMed 37935195](https://pubmed.ncbi.nlm.nih.gov/37935195/)
- [PMC7615469](https://pmc.ncbi.nlm.nih.gov/articles/PMC7615469/)
- [OpenWorm Connectome Toolbox entry](http://openworm.org/ConnectomeToolbox/RipollSanchez_2023/)
- [GitHub repository](https://github.com/LidiaRipollSanchez/Neuropeptide-Connectome)
- [Figshare dataset DOI](https://doi.org/10.6084/m9.figshare.c.6895870.v1)
