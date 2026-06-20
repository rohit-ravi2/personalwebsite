# Local connectomic topology in *Caenorhabditis elegans* is largely self-determining: gene expression is a modulatory refinement layer, not the primary architect of wiring

**Rohit Ravi**
New York University

*Manuscript draft — Paper 1. Compiled 2026-06-20. Target venues: eLife / PLOS Computational Biology / Network Neuroscience.*

> **Provenance note (for internal review; remove before submission).** Every quantitative claim in this manuscript is sourced from a committed artifact in the project repository (worktree `website-paper`, branch `paper/multimodal-analysis`, off commit `cdfc28c`). The artifact backing each number is given in a footnote-style "[src: …]" tag at first use and collected in the Data & Code Availability section. Numbers are **read** from committed artifacts; nothing was re-run for this draft. Claims whose underlying numeric artifact could not be located in this worktree are explicitly flagged with **[PROVENANCE GAP]** rather than presented as verified. This manuscript concerns the static topology-vs-expression analysis only; an embodied biophysical simulator built downstream of these analyses is a separate, unfinished effort and is deliberately excluded.

---

## Abstract

The nematode *Caenorhabditis elegans* is the only animal with a fully reconstructed, cell-resolved connectome and a matched single-cell gene-expression atlas covering its entire nervous system. This pairing makes it the natural testbed for a long-standing question in developmental neurobiology: how much of an animal's wiring is encoded by the transcriptional identity of its neurons, versus determined by the structural logic of the network itself? Here we ask, quantitatively, whether single-cell gene expression carries information about local connectivity *beyond* what the connectome's own topology already supplies. We restrict to the 91 somatic neurons for which both modalities are available and triangulate the question across three method families of increasing nonlinearity: linear canonical-correlation analysis (CCA) between an expression embedding and a connectivity-motif embedding; a graph neural cellular automaton (GNCA) that predicts outgoing synaptic strength from local rules over the connectome; and a neuron-role classifier probe. The GNCA predicts held-out synaptic strength at r = 0.987, but a permutation control that destroys the expression–neuron correspondence while preserving the graph barely moves the result (r = 0.934), and a graph-only model recovers most of the signal on its own (r = 0.861); expression alone is far weaker (r = 0.27). CCA does find substantial linear structure shared between expression and motif space (canonical correlations ρ = [0.875, 0.836, 0.809]), and an expression-only classifier recovers coarse behavioral role (sensory / inter / motor) at 92.7% — confirming that transcriptional identity *is* informative about a neuron's circuit role. The decisive negative result comes from dynamics: a hybrid gene-conditioned graph-LSTM trained to predict neural activity shows no improvement when expression is real versus shuffled (`gene_causal_by_shuffle = False`). A genetic-knockout "collapse curve" further shows that predictive performance plateaus well above chance even after removing the 200 top expression drivers, indicating heavy genetic redundancy. Taken together, the gap between topology-only and expression-augmented models — consistent across linear, nonlinear-static, and dynamic settings — supports a topology-dominant, gene-modulatory account: local wiring in *C. elegans* is largely self-determining, and gene expression refines rather than architects it.

---

## 1. Introduction

### 1.1 The question

Two extreme hypotheses bracket how nervous-system wiring could be specified. At one pole, a **genetic-blueprint** view holds that the molecular identity of each neuron — the transcription factors and surface molecules it expresses — is the proximate cause of which partners it connects to; under this view, expression should predict connectivity, and connectivity that cannot be read off from expression is residual noise [Barabási & Barabási 2020; Kovács et al. 2020]. At the opposite pole, a **structural/topological** view holds that, once the gross architecture of a network is in place, local wiring is highly constrained by the network's own organization — degree sequences, motif statistics, spatial and lineage constraints — so that much of the connectivity is predictable *without* molecular labels at all.

These are not mutually exclusive, and the interesting scientific quantity is not which pole is "true" but *how much additional information expression carries once topology is accounted for*. That is the question this paper operationalizes.

### 1.2 Why *C. elegans*

*C. elegans* is uniquely suited to the question because both sides of the comparison exist at single-cell resolution for the *same* cells. The hermaphrodite connectome was first reconstructed by serial-section electron microscopy [White et al. 1986], quantified and analyzed as a network [Varshney et al. 2011], and substantially revised and extended to both sexes with quantitative synapse weights by Cook et al. [2019]; developmental connectome series now exist as well [Witvliet et al. 2021]. On the molecular side, the CeNGEN consortium [Hammarlund et al. 2018] produced a single-cell RNA-sequencing atlas of the entire nervous system [Taylor et al. 2021], complementing decades of curated neurotransmitter-identity mapping [Pereira et al. 2015; Loer & Rand 2022]. The transcriptional identity of *C. elegans* neurons is, moreover, mechanistically anchored: combinatorial "terminal selector" transcription factors specify and maintain neuron-type identity [Hobert 2016], so expression here is not an arbitrary covariate but the developmental program itself.

### 1.3 Prior work on the topology–expression relationship

Several groups have shown that expression and connectivity in *C. elegans* are *not independent*. Connected hub neurons are genomically distinct and share expression patterns more than expected by chance, an effect not reducible to lineage, birth time, neurotransmitter system, or spatial position [Arnatkevičiūtė et al. 2018]. Pre-/post-synaptic neurotransmitter and receptor expression predicts synaptic *sign* for two-thirds of ionotropic chemical synapses [Fenyves et al. 2020]. And the Barabási genetic-connectome program shows that simple rules over neuronal "genetic identity" predict specific connectome motifs and can infer a small set of gap-junction (innexin) interactions [Barabási & Barabási 2020; Kovács et al. 2020].

What is missing from this literature is a *head-to-head accounting* — a single study that, on one matched dataset, measures how much predictive power expression *adds over topology* and, crucially, reports the controls that isolate the marginal contribution of expression. Showing that expression correlates with connectivity (true) is a weaker claim than showing that expression predicts connectivity *better than the graph predicts itself* (which we find it largely does not, for local wiring). This paper supplies that accounting, and foregrounds an honest null where the augmentation fails.

### 1.4 Contributions

1. A **triangulated** topology-vs-expression comparison on the 91-neuron matched subset, spanning a linear method (CCA), a nonlinear static predictor (GNCA), and a classifier probe.
2. A **shuffle/ablation control battery** isolating the marginal contribution of expression over topology for synaptic-strength prediction.
3. An **honest dynamics null**: gene conditioning does not improve activity prediction beyond shuffled expression.
4. A **genetic redundancy ("collapse") analysis** showing graceful degradation under cumulative top-driver knockout.
5. A synthesis arguing for a **topology-dominant, gene-modulatory** account of local wiring, with explicit scope limits.

---

## 2. Data

### 2.1 Connectome

We use the hermaphrodite chemical-synapse and gap-junction connectivity of Cook et al. [2019], cross-checked against the classical reconstruction [White et al. 1986] and the network-quantified edge list of Varshney et al. [2011]. Neurotransmitter identities used in deriving edge sign/role features follow the curated assignments of Pereira et al. [2015] and the WormAtlas neurotransmitter compilation of Loer & Rand [2022]. The connectome was consolidated into a unified directed multigraph (chemical + electrical layers) with explicit null-model controls (degree-preserving edge shuffling) introduced at the graph-construction stage. [src: `docs/project-history.md` Phases 1, A/B/C]

### 2.2 Single-cell expression

Per-neuron gene expression is taken from the CeNGEN single-cell RNA-seq atlas [Taylor et al. 2021; Hammarlund et al. 2018]. Neuron-name normalization across data sources (e.g. resolving `ASHL` / `ASH_L` / `ASHl` and WormBase gene-ID ↔ gene-symbol crosswalks) was performed once through a canonical loader; earlier per-notebook normalizations that silently dropped neurons were superseded. [src: `docs/project-history.md` Phase 0, Phase A]

### 2.3 The matched subset (N = 91)

All analyses are restricted to the **91 somatic neurons present in both the connectome and the single-cell expression atlas**. This intersection is the unit of analysis throughout. [src: `src/content/projects/c-elegans-multimodal.mdx`]
**[PROVENANCE GAP]** The figure N = 91 is recorded in the project's own write-up (the MDX project page) but I was unable to locate, in this worktree, the upstream artifact (e.g. an intersection manifest or notebook cell output) that derives 91 from the raw source-cell lists. The number should be re-derived from the canonical loader and the exact inclusion criteria stated before submission; it is reported here as the project's stated subset size, not as an independently re-verified count.

---

## 3. Methods

The design principle throughout is **triangulation against controls**: we never report a model's performance without the matched control that removes the variable of interest (expression), and we span three method families so that a conclusion does not rest on the inductive bias of any single architecture.

### 3.1 Canonical-correlation analysis (linear)

To test for *linear* shared structure, we apply canonical-correlation analysis [Hotelling 1936] between a per-neuron **expression embedding** and a per-neuron **connectivity-motif embedding** (local motif participation statistics derived from the connectome). CCA finds paired linear projections of the two views that are maximally correlated; the canonical correlations ρ₁ ≥ ρ₂ ≥ … quantify how much of each view's leading structure is linearly shared with the other. [src: `docs/project-history.md` Phases A/B/C; `docs/claude-chat-context.md` §2]

### 3.2 Graph neural cellular automaton (nonlinear, static)

For the central synaptic-strength prediction we use a **graph neural cellular automaton (GNCA)**. The GNCA is a message-passing model [Gilmer et al. 2017; Kipf & Welling 2017] run in the cellular-automaton regime [Mordvintsev et al. 2020]: each neuron carries a state vector, repeatedly updated by a learned local rule that aggregates messages from connectome neighbors (optionally concatenated with that neuron's expression features), and a readout predicts **outgoing synaptic strength**. Performance is reported as Pearson r between predicted and observed strength on **held-out edges**. The architecture is deliberately *local* — it has access only to a neuron's neighborhood under repeated propagation — which is what makes it a clean probe of whether *local* wiring is self-determining. [src: `docs/project-history.md` Phase E / NCA (`NCA.ipynb`, `phaseE.ipynb` cell 1 `run_nca_pipeline.py`)]

We evaluate four conditions:

- **Full** — topology + real expression.
- **Shuffled-expression** — topology intact, expression vectors permuted across neurons (destroys the expression–neuron correspondence while preserving the graph and the expression *distribution*). This is the key control: the gap between Full and Shuffled is the marginal value of *correctly-assigned* expression.
- **Graph-only** — connectome topology with no expression features.
- **Expression-only** — expression features with the graph structure removed.

### 3.3 Neuron-role classifier probe

As an independent check that expression is genuinely informative about circuit role, we train a supervised classifier to predict each neuron's coarse **behavioral role (sensory / interneuron / motor)** from its expression vector alone, and report held-out accuracy. This is a probe, not a connectivity predictor: it establishes the *upper-bound* sense in which "expression knows what a neuron is," against which the *failure* of expression to add connectivity/dynamics information becomes interpretable. [src: `docs/claude-chat-context.md` §2; `src/content/projects/c-elegans-multimodal.mdx`]

### 3.4 Dynamics: gene-conditioned graph-LSTM and the shuffle test

To test whether expression shapes *dynamics* (not just static wiring), we train a hybrid **gene-conditioned graph-LSTM** ("PhaseD" model) to predict neural time-series, cycling through LSTM, cycle-aware LSTM, cycle-aware Transformer, and per-neuron shared-LSTM-with-gene-conditioning variants. The decisive test is a battery of nulls — temporal-shuffled, identity-permuted, and cycle-permuted — summarized by the boolean `gene_causal_by_shuffle`: it is **True** only if the gene-conditioned model predicts dynamics reliably better with *real* expression than with *shuffled* expression. [src: `docs/project-history.md` Phase D (`PhaseD.ipynb`, cells 35+); `docs/state_of_claims_2026-05-02.md` C-07]

### 3.5 Genetic-redundancy ("collapse") analysis

Finally, to characterize *how* expression information is distributed across genes, we rank genes by their contribution as "drivers" of the predictor and remove them cumulatively (in-silico knockout), recording model correlation as a function of the number of top genes removed. A blueprint with a few load-bearing genes would collapse toward chance quickly; a redundant code would degrade gracefully and plateau above chance. [src: figure `public/images/projects/celegans-collapse-curve.png`]

### 3.6 Cross-validation and controls

Held-out evaluation is used for the GNCA (held-out edges) and the classifier (held-out neurons). Null models (degree-preserving shuffling, edge permutation) were built into the pipeline at the graph-construction stage so that every structural claim is referenced against a permuted baseline. [src: `docs/project-history.md` Phases A/B/C]

**[PROVENANCE GAP]** The exact cross-validation protocol (number of folds, held-out fraction, seed handling) and confidence intervals for the headline scalars (the four GNCA r-values, the 92.7% classifier accuracy, and the three canonical correlations) are not captured as numeric artifacts in this worktree; the source notebooks (`NCA.ipynb`, `PhaseD.ipynb`, `Paper1MasterNotebook`) that produced them are referenced in the project history but are not present here. Point estimates are reported exactly as committed; uncertainty quantification must be added from the source notebooks before submission.

---

## 4. Results

### 4.1 The marginal value of expression for synaptic strength is small

The full GNCA predicts held-out outgoing synaptic strength at **r = 0.987**. [src: `docs/project-history.md` Phase E; `docs/state_of_claims_2026-05-02.md` C-01; `src/content/projects/c-elegans-multimodal.mdx`] The control battery localizes where that signal lives:

| Condition | Held-out r | Interpretation |
|---|---|---|
| Full (topology + expression) | **0.987** | ceiling |
| Shuffled-expression (topology preserved) | **0.934** | expression *correspondence* contributes little |
| Graph-only | **0.861** | topology alone carries most of the signal |
| Expression-only | **0.27** | expression alone is a weak predictor of local wiring |

[src for all four: `docs/project-history.md` Phase E; `docs/state_of_claims_2026-05-02.md` C-01–C-04; `docs/claude-chat-context.md` §2]

Three things follow. First, **topology is the dominant carrier**: the graph-only model (0.861) already recovers the great majority of the explainable signal, and expression-only (0.27) is far behind. Second, **the marginal value of correctly-assigned expression is the Full−Shuffled gap (0.987 → 0.934)**, which is small: scrambling which expression vector belongs to which neuron — while leaving the graph and the expression distribution untouched — costs only ~0.05 in r. Third, expression is *not useless* — Full (0.987) does exceed graph-only (0.861), so some real refinement exists — but its contribution is consistent with a **modulatory** role layered on a topology-determined backbone, not with expression being the architect.

> A caution on interpreting r near the ceiling: at r = 0.987 vs 0.934 the *variance-explained* difference (R² 0.974 vs 0.872) is larger than the raw-r difference suggests, but it remains modest next to the graph-only baseline. The qualitative ranking — topology ≫ expression-correspondence — is robust; the precise marginal magnitude should be reported with CIs (see §3.6 provenance gap).

### 4.2 Expression nonetheless carries linear circuit structure (CCA)

CCA between the expression embedding and the motif embedding yields canonical correlations **ρ = [0.875, 0.836, 0.809]**. [src: `docs/state_of_claims_2026-05-02.md` C-06; `docs/claude-chat-context.md` §2; `src/content/projects/c-elegans-multimodal.mdx`] At least three strongly-correlated canonical dimensions are shared between transcriptional identity and local connectivity-motif participation. This is fully consistent with prior reports that expression and connectivity are coupled in *C. elegans* [Arnatkevičiūtė et al. 2018; Fenyves et al. 2020] — the point of this paper is not that the coupling is absent (it plainly exists, linearly and substantially) but that, *conditioned on topology*, expression adds little to the prediction of the wiring itself (§4.1, §4.4).

### 4.3 Expression predicts coarse behavioral role (classifier probe)

An expression-only classifier predicts the three-class behavioral role (sensory / inter / motor) at **92.7%** held-out accuracy. [src: `docs/state_of_claims_2026-05-02.md` C-05; `docs/claude-chat-context.md` §2; `src/content/projects/c-elegans-multimodal.mdx`] Transcriptional identity therefore *does* encode what kind of neuron a cell is, at the resolution of its broad circuit role — exactly as the terminal-selector logic of *C. elegans* neuronal identity would predict [Hobert 2016]. This is the crucial framing for the null that follows: expression is richly informative about identity, yet that information is largely redundant with topology when the task is predicting local wiring or dynamics.

### 4.4 The honest null: gene conditioning does not improve dynamics

The gene-conditioned graph-LSTM, after the full shuffle battery (temporal / identity / cycle permutations), returns **`gene_causal_by_shuffle = False`**: predicting neural dynamics is *not* reliably better with real expression than with shuffled expression on this dataset. [src: `docs/project-history.md` Phase D; `docs/state_of_claims_2026-05-02.md` C-07; `docs/claude-chat-context.md` §2]

We foreground this result rather than bury it. The original hypothesis entering Phase D was the strong, blueprint-flavored claim that *gene expression causally shapes how neurons fire together*; the project's own internal null battery falsified it. A negative result obtained by a pre-specified shuffle control is exactly the evidence that makes the topology-dominant thesis credible rather than merely the artifact of an over-expressive model: when expression *should* have helped (dynamics, where molecular identity sets channel and receptor complements), the correctly-assigned expression did not beat its own shuffle.

A necessary scope limit: this null is a statement about *this dataset and this model class*, and "absence of detectable improvement" is not "proof of no effect." The relevant dynamics data are sparse and noisy, the model family is finite, and a more powerful model or richer recordings could in principle surface a gene-dynamics signal. We therefore state the null at the level it was earned: on this matched dataset, gene conditioning yields no shuffle-robust dynamics improvement.

### 4.5 Genetic redundancy: the collapse curve

The genetic-knockout analysis (Figure 1) shows that model correlation **plateaus well above chance** as the top expression "drivers" are removed cumulatively. After an initial drop from the single-model baseline (r ≈ 0.82) over the first ~10 knockouts, performance flattens around r ≈ 0.68 and stays there essentially unchanged from ~10 through **200** removed genes; the average probability the model assigns to hub neurons (navy) declines only gently over the same range and never approaches the chance line. [src: `public/images/projects/celegans-collapse-curve.png`] The top-ranked driver is `WBGene00001137`. **[PROVENANCE GAP]** I did not locate a gene-ID → symbol crosswalk artifact in this worktree to resolve `WBGene00001137` to a named gene; it should be looked up in WormBase and named in the final figure caption.

This graceful, plateauing degradation is the signature of a **redundant** genetic code rather than a few load-bearing master genes: no small set of drivers is individually necessary for the topology-aligned prediction. Redundancy of this kind is precisely what one expects if topology — not any compact gene set — is doing the structural work, with expression spread thinly across many partially-substitutable contributors.

> Scope note on the baseline mismatch: the collapse-curve baseline (r ≈ 0.82) is the operating point of the *driver-attribution model used for the knockout sweep*, not the held-out r = 0.987 of the full GNCA (§4.1). The two numbers come from different model setups and should not be conflated; the collapse curve is informative about *redundancy structure* (the shape of the degradation), not about absolute predictive ceiling.

---

## 5. Discussion

### 5.1 A topology-dominant, gene-modulatory account

The three method families agree. Linearly (CCA), expression and connectivity share substantial structure. Statically (GNCA), topology alone predicts local synaptic strength nearly as well as topology-plus-expression, and correctly-assigned expression adds only a small margin over its own shuffle. Dynamically (graph-LSTM), gene conditioning adds nothing shuffle-robust. And the redundancy analysis shows the expression contribution is diffuse, not carried by a few master genes. The convergent reading is that **local wiring in *C. elegans* is largely self-determining at the level of topology, with gene expression acting as a modulatory refinement layer** rather than the primary architect.

This is not a claim that genes do not build the nervous system — they obviously do, through the terminal-selector programs that fix neuron identity [Hobert 2016] and the surface-molecule logic that prior work has begun to read out [Barabási & Barabási 2020; Kovács et al. 2020]. It is the narrower, quantitative claim that *once the connectome's local topology is given, the additional predictive information in single-cell expression about that same local wiring is small* — and, for dynamics on this dataset, undetectable above shuffle.

### 5.2 Relationship to prior work

Our finding is compatible with, and sharpens, the existing literature. Arnatkevičiūtė et al. [2018] and Fenyves et al. [2020] establish that expression *correlates with* and *partly predicts* aspects of connectivity (hub genomics; synaptic sign). We do not contradict that; CCA (§4.2) reproduces the coupling. What we add is the *conditional* accounting — the marginal value of expression *given topology* — and the controls that isolate it. The Barabási genetic-connectome program [Barabási & Barabási 2020; Kovács et al. 2020] argues that genetic identity predicts connectome motifs; our results suggest that for *local synaptic-strength* prediction the topology itself already encodes most of that motif structure, so a topology-aware predictor recovers the signal without explicit gene labels. The two views are reconcilable if the genetic program is largely *upstream*, having set the topology during development, such that at the adult stage topology is a near-sufficient statistic for local wiring.

### 5.3 Limitations

1. **Scope is local and static.** The GNCA is intentionally local; this paper speaks to *local* wiring, not to global developmental specification or long-range routing. The genetic blueprint may dominate at scales this design cannot see.
2. **Adult snapshot.** Both modalities are (largely) adult; developmental trajectories [Witvliet et al. 2021] could reveal a stronger gene-causal phase that is no longer visible once wiring is complete.
3. **Null ≠ proof of absence (§4.4).** The dynamics null is bounded by data sparsity and the finite model family.
4. **Ceiling effects (§4.1).** Interpreting differences in r near 0.99 requires the R²/CI framing flagged above; CIs are not yet in-artifact (§3.6).
5. **N = 91.** The matched subset is a fraction of the 302-neuron nervous system; the inclusion criterion needs re-derivation from the canonical loader (§2.3 gap), and statistical power is modest.
6. **Expression representation.** Results may depend on how expression is embedded (binary presence/absence vs. quantitative levels; gene-set choice); robustness across representations is future work.

### 5.4 Falsifiability and what would overturn this

The thesis is falsifiable on its own terms. It would be undermined by any of: (i) a model in which correctly-assigned expression substantially beats its shuffle for *local* synaptic-strength prediction on this subset (closing the Full−Shuffled gap from the expression side, not by weakening the graph); (ii) a `gene_causal_by_shuffle = True` result on richer dynamics data with a pre-registered shuffle control; (iii) a collapse curve that drops to chance after a small number of driver knockouts (indicating a few load-bearing genes rather than redundancy). We regard (ii) as the most informative future test.

---

## 6. Conclusion

On the one dataset where both a cell-resolved connectome and a whole-nervous-system single-cell expression atlas exist, and across linear, nonlinear-static, and dynamic models with matched shuffle/ablation controls, local *C. elegans* wiring is predicted largely by its own topology; single-cell expression — though richly informative about neuron identity and behavioral role — adds little marginal predictive power for local connectivity and none, above shuffle, for dynamics on this dataset. The picture is topology-dominant and gene-modulatory. The honest dynamics null and the redundancy of the genetic contribution are not caveats to that picture; they are its strongest support.

---

## 7. A note on the methods-paper split

The GNCA architecture in §3.2 is, independently of the biological thesis, a method for **connectome-constrained synaptic-strength prediction**: a message-passing cellular automaton operating in the local regime over a directed multigraph with optional node features, evaluated under a topology-vs-feature ablation protocol. That methodological contribution — the architecture, the shuffle-control evaluation design, and the held-out-edge regression setup — is naturally separable from the *C. elegans* biology and is a good fit for a graph-representation-learning venue (NeurIPS GRL / ICLR LMRL). We anticipate a short companion methods paper presenting the GNCA and its ablation protocol as a reusable tool for biological-graph property prediction, with this paper as the application that motivates it. The two papers share the §4.1 ablation table but differ in framing: here it is evidence about worm biology; there it is a benchmark of an architecture against feature-vs-structure baselines.

---

## Data & Code Availability

All quantitative claims trace to committed artifacts in the project repository (worktree `website-paper`, branch `paper/multimodal-analysis`, base commit `cdfc28c`):

- **GNCA four-way ablation (r = 0.987 / 0.934 / 0.861 / 0.27):** `docs/project-history.md` (Phase E / NCA); `docs/state_of_claims_2026-05-02.md` claims C-01–C-04; `docs/claude-chat-context.md` §2; `src/content/projects/c-elegans-multimodal.mdx`. Source notebooks referenced: `NCA.ipynb`, `phaseE.ipynb` (`run_nca_pipeline.py`).
- **Classifier 92.7%:** `docs/state_of_claims_2026-05-02.md` C-05; `docs/claude-chat-context.md` §2; MDX page.
- **CCA ρ = [0.875, 0.836, 0.809]:** `docs/state_of_claims_2026-05-02.md` C-06; `docs/claude-chat-context.md` §2; MDX page.
- **Dynamics null `gene_causal_by_shuffle = False`:** `docs/project-history.md` (Phase D); `docs/state_of_claims_2026-05-02.md` C-07. Source notebook: `PhaseD.ipynb`.
- **Collapse curve (Figure 1):** `public/images/projects/celegans-collapse-curve.png`.

**Outstanding provenance items before submission** (see in-text [PROVENANCE GAP] tags): (1) re-derive N = 91 from the canonical loader and state inclusion criteria; (2) recover cross-validation protocol + confidence intervals for all headline scalars from the source notebooks, which are not present in this worktree; (3) resolve `WBGene00001137` to a gene symbol via WormBase.

---

## References

(Full bibliographic details, DOIs/PMIDs, and verification status are in `paper/references.md`.)

1. White JG, Southgate E, Thomson JN, Brenner S. (1986) The structure of the nervous system of the nematode *Caenorhabditis elegans*. *Phil. Trans. R. Soc. Lond. B* 314(1165):1–340. doi:10.1098/rstb.1986.0056.
2. Varshney LR, Chen BL, Paniagua E, Hall DH, Chklovskii DB. (2011) Structural properties of the *Caenorhabditis elegans* neuronal network. *PLoS Comput. Biol.* 7(2):e1001066. doi:10.1371/journal.pcbi.1001066.
3. Cook SJ, Jarrell TA, Brittin CA, et al. (2019) Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature* 571(7763):63–71. doi:10.1038/s41586-019-1352-7.
4. Witvliet D, Mulcahy B, Mitchell JK, et al. (2021) Connectomes across development reveal principles of brain maturation. *Nature* 596(7871):257–261. doi:10.1038/s41586-021-03778-8.
5. Hammarlund M, Hobert O, Miller DM 3rd, Sestan N. (2018) The CeNGEN project: the complete gene expression map of an entire nervous system. *Neuron* 99(3):430–433. doi:10.1016/j.neuron.2018.07.042.
6. Taylor SR, Santpere G, Weinreb A, et al. (2021) Molecular topography of an entire nervous system. *Cell* 184(16):4329–4347. doi:10.1016/j.cell.2021.06.023.
7. Pereira L, Kratsios P, Serrano-Saiz E, et al. (2015) A cellular and regulatory map of the cholinergic nervous system of *C. elegans*. *eLife* 4:e12432. doi:10.7554/eLife.12432.
8. Loer CM, Rand JB. (2022) The evidence for classical neurotransmitters in *C. elegans* (WormAtlas Neurotransmitter Table; updated 27 Sep 2022). WormAtlas. [resource]
9. Hobert O. (2016) A map of terminal regulators of neuronal identity in *Caenorhabditis elegans*. *WIREs Dev. Biol.* 5(4):474–498. doi:10.1002/wdev.233.
10. Arnatkevičiūtė A, Fulcher BD, Pocock R, Fornito A. (2018) Hub connectivity, neuronal diversity, and gene expression in the *Caenorhabditis elegans* connectome. *PLoS Comput. Biol.* 14(2):e1005989. doi:10.1371/journal.pcbi.1005989.
11. Fenyves BG, Szilágyi GS, Vassy Z, Sőti C, Csermely P. (2020) Synaptic polarity and sign-balance prediction using gene expression data in the *Caenorhabditis elegans* chemical synapse neuronal connectome network. *PLoS Comput. Biol.* 16(12):e1007974. doi:10.1371/journal.pcbi.1007974.
12. Barabási DL, Barabási A-L. (2020) A genetic model of the connectome. *Neuron* 105(3):435–445.e5. doi:10.1016/j.neuron.2019.10.031.
13. Kovács IA, Barabási DL, Barabási A-L. (2020) Uncovering the genetic blueprint of the *C. elegans* nervous system. *PNAS* 117(52):33570–33577. doi:10.1073/pnas.2009093117.
14. Hotelling H. (1936) Relations between two sets of variates. *Biometrika* 28(3–4):321–377. doi:10.1093/biomet/28.3-4.321.
15. Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE. (2017) Neural message passing for quantum chemistry. *Proc. ICML* (PMLR 70):1263–1272.
16. Kipf TN, Welling M. (2017) Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
17. Mordvintsev A, Randazzo E, Niklasson E, Levin M. (2020) Growing neural cellular automata. *Distill* 5(2):e23. doi:10.23915/distill.00023.
