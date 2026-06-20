# References — verification log

Paper 1: *Local connectomic topology in C. elegans is largely self-determining.*
Each entry verified via web search on 2026-06-20 (title + DOI/PMID confirmed against
publisher / PubMed / PMC). "Verified" = the work exists with the stated venue and
identifier. Substrate / simulator citations are intentionally excluded.

| key | citation | DOI / PMID | verified | used where |
|---|---|---|---|---|
| white1986 | White JG, Southgate E, Thomson JN, Brenner S. (1986) The structure of the nervous system of the nematode *C. elegans*. *Phil. Trans. R. Soc. Lond. B* 314(1165):1–340. | doi:10.1098/rstb.1986.0056 | yes | §1.2, §2.1 — foundational connectome |
| varshney2011 | Varshney LR, Chen BL, Paniagua E, Hall DH, Chklovskii DB. (2011) Structural properties of the *C. elegans* neuronal network. *PLoS Comput. Biol.* 7(2):e1001066. | doi:10.1371/journal.pcbi.1001066; PMID 21304930 | yes | §1.2, §2.1 — network-quantified edge list |
| cook2019 | Cook SJ, Jarrell TA, Brittin CA, et al. (2019) Whole-animal connectomes of both *C. elegans* sexes. *Nature* 571(7763):63–71. | doi:10.1038/s41586-019-1352-7; PMID 31270481 | yes | §1.2, §2.1 — primary connectome w/ weights |
| witvliet2021 | Witvliet D, Mulcahy B, Mitchell JK, et al. (2021) Connectomes across development reveal principles of brain maturation. *Nature* 596(7871):257–261. | doi:10.1038/s41586-021-03778-8 | yes | §1.2, §5.3 — developmental connectome |
| hammarlund2018 | Hammarlund M, Hobert O, Miller DM 3rd, Sestan N. (2018) The CeNGEN project. *Neuron* 99(3):430–433. | doi:10.1016/j.neuron.2018.07.042; PMID 30092212 | yes | §1.2, §2.2 — CeNGEN project descriptor |
| taylor2021 | Taylor SR, Santpere G, Weinreb A, et al. (2021) Molecular topography of an entire nervous system. *Cell* 184(16):4329–4347. | doi:10.1016/j.cell.2021.06.023 | yes | §1.2, §2.2 — single-cell expression atlas |
| pereira2015 | Pereira L, Kratsios P, Serrano-Saiz E, et al. (2015) A cellular and regulatory map of the cholinergic nervous system of *C. elegans*. *eLife* 4:e12432. | doi:10.7554/eLife.12432; PMID 26705699 | yes | §2.1 — NT identity backbone |
| loer2022 | Loer CM, Rand JB. (2022) The evidence for classical neurotransmitters in *C. elegans* (WormAtlas Neurotransmitter Table; updated 27 Sep 2022). WormAtlas. | resource (no journal DOI) | yes (resource exists; not a peer-reviewed article) | §1.2, §2.1 — NT identities |
| hobert2016 | Hobert O. (2016) A map of terminal regulators of neuronal identity in *C. elegans*. *WIREs Dev. Biol.* 5(4):474–498. | doi:10.1002/wdev.233; PMID 27136279 | yes | §1.2, §4.3, §5.1 — terminal selectors |
| arnatkeviciute2018 | Arnatkevičiūtė A, Fulcher BD, Pocock R, Fornito A. (2018) Hub connectivity, neuronal diversity, and gene expression in the *C. elegans* connectome. *PLoS Comput. Biol.* 14(2):e1005989. | doi:10.1371/journal.pcbi.1005989; PMID 29432412 | yes | §1.3, §4.2, §5.2 — prior topology↔expression |
| fenyves2020 | Fenyves BG, Szilágyi GS, Vassy Z, Sőti C, Csermely P. (2020) Synaptic polarity and sign-balance prediction using gene expression data in the *C. elegans* chemical synapse connectome. *PLoS Comput. Biol.* 16(12):e1007974. | doi:10.1371/journal.pcbi.1007974 (PMC7785220) | yes | §1.3, §4.2, §5.2 — expression predicts synaptic sign |
| barabasi2020neuron | Barabási DL, Barabási A-L. (2020) A genetic model of the connectome. *Neuron* 105(3):435–445.e5. | doi:10.1016/j.neuron.2019.10.031; PMID 31806491 | yes | §1.1, §1.3, §5.1, §5.2 — genetic blueprint pole |
| kovacs2020 | Kovács IA, Barabási DL, Barabási A-L. (2020) Uncovering the genetic blueprint of the *C. elegans* nervous system. *PNAS* 117(52):33570–33577. | doi:10.1073/pnas.2009093117; PMID 33318182 | yes | §1.1, §1.3, §5.1, §5.2 — genetic blueprint pole |
| hotelling1936 | Hotelling H. (1936) Relations between two sets of variates. *Biometrika* 28(3–4):321–377. | doi:10.1093/biomet/28.3-4.321 | yes | §3.1 — CCA method origin |
| gilmer2017 | Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE. (2017) Neural message passing for quantum chemistry. *Proc. ICML* (PMLR 70):1263–1272. | proceedings.mlr.press/v70/gilmer17a.html | yes | §3.2 — message-passing GNN framework |
| kipf2017 | Kipf TN, Welling M. (2017) Semi-supervised classification with graph convolutional networks. *ICLR 2017*. | openreview SJU4ayYgl | yes | §3.2 — GCN |
| mordvintsev2020 | Mordvintsev A, Randazzo E, Niklasson E, Levin M. (2020) Growing neural cellular automata. *Distill* 5(2):e23. | doi:10.23915/distill.00023 | yes | §3.2 — neural cellular automata |

## Notes / caveats

- **loer2022** is a curated WormAtlas data table (last updated September 2022), not a
  peer-reviewed journal article. The project docs cited it as "Loer & Rand 2022"; it is
  real and citeable as a resource, but should be referenced as the WormAtlas
  Neurotransmitter Table rather than as a paper. A peer-reviewed alternative now exists
  (Wang et al. 2024, *eLife*, "A neurotransmitter atlas of C. elegans males and
  hermaphrodites"; PMC11185579) and could be cited alongside or instead if a
  journal-article reference is preferred — NOT yet verified in detail here, so it is not
  used in the manuscript.

## Considered but NOT used (to keep substrate/simulator out)

- Atanas et al. 2023 (whole-brain calcium + behavior; DANDI 000776) — used by the
  downstream simulator, not by the static topology-vs-expression analysis. Excluded.
- Any Brian2 / NEURON / channel-biophysics / anesthesia references — substrate effort,
  out of scope by directive.

## Could NOT verify / flagged

- No citation in the manuscript is unverified. All 17 used references were confirmed
  with venue + identifier on 2026-06-20.
- The original project docs referenced "Andrew Gao 2023" deep-learning neuron
  classification as a possible classifier prior. The closest real match found is an
  arXiv 2023 paper "Neuronal Cell Type Classification using Deep Learning" (mammalian /
  Allen Cell Types Database), whose first-author identity and exact relevance I could
  NOT confirm; it was therefore NOT cited. Flagged here rather than invented.
