# Local connectomic topology in *Caenorhabditis elegans* is largely self-determining: gene expression is a modulatory refinement layer, not the primary architect of wiring

**Rohit Ravi**
New York University

*Paper 1 — assembled manuscript draft. Compiled 2026-06-20. Target venues: eLife / PLOS Computational Biology / Network Neuroscience.*

> **Provenance note (for internal review; remove before submission).** Every quantitative claim in this manuscript is *read* from a committed artifact in the original analysis repository (`/home/rohit/Desktop/C-Elegans/`); nothing was re-run for this draft (zero-compute). Each number carries a `[src: …]` tag at or near first use, and the full claim→artifact mapping is given in the **Provenance Table** (Appendix A), which lists known provenance *gaps* at the top so they cannot be missed. Where a number's backing artifact contradicts an earlier informal write-up — most importantly the words "held-out" and "N = 91" — the corrected, artifact-grounded statement is used and the discrepancy is flagged in place with **[PROVENANCE NOTE]**. This manuscript concerns the static topology-vs-expression analysis only; a downstream embodied biophysical simulator is a separate, unfinished effort and is deliberately excluded — none of its claims, struggles, or numbers are load-bearing here.

---

## Abstract

The nematode *Caenorhabditis elegans* is the only animal with a fully reconstructed, cell-resolved connectome and a matched single-cell gene-expression atlas spanning its entire nervous system. This pairing makes it the natural testbed for a long-standing question in developmental neurobiology: how much of an animal's wiring is encoded by the transcriptional identity of its neurons, and how much is determined by the structural logic of the network itself? We ask, quantitatively, whether single-cell gene expression carries information about local connectivity *beyond* what the connectome's own topology already supplies, and we triangulate the question across three method families of increasing nonlinearity, each paired with a matched control that removes the variable of interest. (i) A graph neural cellular automaton (GNCA) — a local message-passing model over the directed synapse-count graph with per-neuron expression features — predicts mean outgoing synaptic strength at Pearson r = 0.987 (MAE = 0.097). This is an **in-sample full-fit**, not a held-out score [src: NCA.ipynb cell 4]; its evidentiary weight comes not from the absolute value but from a battery of controls measured the same way: permuting which expression vector belongs to which neuron, while leaving the graph and the expression distribution untouched, costs only ~0.05 in r (shuffled r = 0.934), a graph-only model recovers most of the signal with no expression at all (r ≈ 0.857–0.861), and an expression-only model is far weaker (r ≈ 0.27). (ii) Canonical-correlation analysis (CCA) finds substantial *linear* structure shared between an expression embedding and a connectivity-motif embedding (canonical correlations ρ = [0.875, 0.836, 0.809]), confirming that expression and connectivity are genuinely coupled. (iii) An expression-only classifier recovers coarse behavioral role (sensory / interneuron / motor) at 92.7% cross-validated accuracy — transcriptional identity *does* encode what kind of neuron a cell is. The decisive negative result comes from dynamics: a hybrid gene-conditioned graph-LSTM trained to predict neural time series shows no improvement when expression is real versus shuffled (`gene_causal_by_shuffle = False`), a null obtained under a pre-specified shuffle gate and reproduced in an independent clean re-implementation — though, importantly, on *simulated* dynamics, since the only real *C. elegans* activity recordings of comparable scale lack the neuron identities needed to attach expression. Finally, a cumulative gene-knockout "collapse curve" for an expression→hub-status predictor degrades gracefully and plateaus well above chance after removing the 200 top driver genes, the signature of a redundant rather than a load-bearing genetic code. Across linear, nonlinear-static, and dynamic settings, the gap between topology-only and expression-augmented models is consistent and small. Together with the honest dynamics null and the redundancy result — the two findings that make the thesis credible rather than the artifact of an over-expressive model — these results support a topology-dominant, gene-modulatory account: local wiring in *C. elegans* is largely self-determining, and gene expression refines rather than architects it.

---

## 1. Introduction

### 1.1 The question

Two extreme hypotheses bracket how nervous-system wiring could be specified. At one pole, a **genetic-blueprint** view holds that the molecular identity of each neuron — the transcription factors and surface molecules it expresses — is the proximate cause of which partners it connects to; under this view, expression should predict connectivity, and connectivity that cannot be read off from expression is residual noise [Barabási & Barabási 2020; Kovács et al. 2020]. At the opposite pole, a **structural/topological** view holds that, once the gross architecture of a network is in place, local wiring is heavily constrained by the network's own organization — degree sequences, motif statistics, spatial and lineage constraints — so that much of the connectivity is predictable *without* molecular labels at all.

These poles are not mutually exclusive, and the scientifically interesting quantity is not which one is "true" but **how much additional information expression carries once topology is accounted for**. A study that merely shows expression *correlates with* connectivity establishes the weaker claim; the stronger, decision-relevant claim is whether expression predicts connectivity *better than the graph predicts itself*. That conditional accounting — and the controls that isolate it — is what this paper operationalizes.

### 1.2 Why *C. elegans*

*C. elegans* is uniquely suited to the question because both sides of the comparison exist at single-cell resolution for the *same* cells. The hermaphrodite connectome was first reconstructed by serial-section electron microscopy [White et al. 1986], quantified and analyzed as a network [Varshney et al. 2011], and substantially revised and extended to both sexes with quantitative synapse weights by Cook et al. [2019]; developmental connectome series across maturation now exist as well [Witvliet et al. 2021]. On the molecular side, the CeNGEN consortium [Hammarlund et al. 2018] produced a single-cell RNA-sequencing atlas of the entire nervous system [Taylor et al. 2021], complementing decades of curated neurotransmitter-identity mapping [Pereira et al. 2015; Loer & Rand 2022]. Crucially, the transcriptional identity of *C. elegans* neurons is mechanistically anchored: combinatorial "terminal selector" transcription factors specify and maintain neuron-type identity throughout life [Hobert 2016], so expression here is not an arbitrary covariate but a readout of the developmental program itself. If gene expression is going to predict wiring anywhere, it should do so in this animal.

### 1.3 Prior work on the topology–expression relationship

Several groups have established that expression and connectivity in *C. elegans* are *not independent*. Connected hub neurons are genomically distinct and share expression patterns more than expected by chance, an effect not reducible to lineage, birth time, neurotransmitter system, or spatial position [Arnatkevičiūtė et al. 2018]. Pre- and post-synaptic neurotransmitter and receptor expression jointly predict synaptic *sign* for roughly two-thirds of ionotropic chemical synapses [Fenyves et al. 2020]. And the Barabási genetic-connectome program shows that simple rules over neuronal "genetic identity" reproduce specific connectome motifs and can infer a small set of gap-junction (innexin) interactions [Barabási & Barabási 2020; Kovács et al. 2020].

What is missing from this literature is a **head-to-head accounting on one matched dataset** — a single study that measures how much predictive power expression *adds over topology* and reports the shuffle/ablation controls that isolate that marginal contribution. Demonstrating that expression correlates with connectivity (true, and reproduced here) is a weaker statement than demonstrating that expression predicts connectivity *better than the graph predicts itself* (which, for local wiring, we find it largely does not). This paper supplies that conditional accounting and, equally importantly, foregrounds an honest null where the expression augmentation fails.

### 1.4 Approach: triangulation against controls

The design principle is **triangulation against matched controls**: we never report a model's performance without the control that removes expression, and we span three method families so that no conclusion rests on the inductive bias of a single architecture.

- **Linear — CCA.** Canonical-correlation analysis [Hotelling 1936] between a per-neuron expression embedding and a per-neuron connectivity-motif embedding quantifies, dimension by dimension, how much of each view's leading structure is *linearly* shared with the other.
- **Nonlinear, static — GNCA.** A graph neural cellular automaton — a message-passing model [Gilmer et al. 2017; Kipf & Welling 2017] run in the cellular-automaton regime [Mordvintsev et al. 2020] — predicts outgoing synaptic strength from a deliberately *local* rule over the connectome, with optional per-neuron expression features. Its locality is exactly what makes it a clean probe of whether *local* wiring is self-determining. The four-way ablation (full / shuffled-expression / graph-only / expression-only) is the spine of the paper.
- **Probe — expression-only classifier.** A supervised classifier predicting coarse behavioral role from expression alone establishes the upper-bound sense in which "expression knows what a neuron is," against which the *failure* of expression to add connectivity or dynamics information becomes interpretable.

To these we add two findings that supply the credibility of a negative result rather than merely the success of a positive one: (iv) a **dynamics null**, in which a gene-conditioned graph-LSTM does no better with real than with shuffled expression under a pre-specified shuffle gate (`gene_causal_by_shuffle = False`); and (v) a **genetic-redundancy "collapse curve,"** in which cumulative in-silico knockout of the top driver genes degrades an expression→hub-status predictor only gently, plateauing well above chance through 200 knockouts.

### 1.5 Preview of results and the credibility spine

The three method families agree. Linearly, expression and connectivity share substantial structure (CCA ρ up to 0.875). Statically, topology alone predicts local synaptic strength nearly as well as topology-plus-expression: graph-only r ≈ 0.857–0.861 versus full r = 0.987, while correctly-*assigned* expression beats its own shuffle by only ~0.05 (0.987 vs 0.934) and expression-only collapses to r ≈ 0.27. As a probe, expression nonetheless classifies behavioral role at 92.7%, so the topology-dominance is *not* because expression is uninformative about neurons — it is informative, yet largely redundant with topology when the task is predicting local wiring. The dynamics null and the redundant collapse curve close the argument: when expression *should* have helped (dynamics, where molecular identity sets channel and receptor complements), the correctly-assigned expression did not beat its own shuffle, and no small set of genes is individually load-bearing. The convergent reading is **topology-dominant and gene-modulatory**: once the connectome's local topology is given, the additional predictive information in single-cell expression about that same local wiring is small.

We make four contributions: (1) a triangulated topology-vs-expression comparison spanning a linear method, a nonlinear static predictor, and a classifier probe; (2) a shuffle/ablation control battery that isolates the marginal contribution of correctly-assigned expression for synaptic-strength prediction; (3) an honest dynamics null obtained under a pre-specified shuffle gate; and (4) a genetic-redundancy analysis showing graceful, plateauing degradation under cumulative top-driver knockout. We state scope limits explicitly throughout (§6), including the points enumerated below as integrity flags.

### 1.6 Statements of scope and provenance integrity

We surface, at the outset rather than in a footnote, the limits that govern how the headline numbers should be read:

1. **The GNCA r = 0.987 is an in-sample full-fit, not a held-out score.** [PROVENANCE NOTE] The synaptic-strength model and its three controls (full / shuffled / graph-only / expression-only) are trained and evaluated on the full node set with no train/test split; Pearson r is computed in-sample [src: NCA.ipynb cells 3–8; project scorecard labels it "GNCA (full)"]. The relative comparison across the four conditions — all measured the same in-sample way — is what carries the thesis and is unaffected; the absolute r is *not* a generalization estimate and is not reported as one. A 5-fold cross-validated motif-participation regression on the same data gives a more conservative r ≈ 0.64 (folds [0.25, 0.62, 0.82, 0.86, 0.64]) [src: NCA.ipynb cell 24], which we report alongside the in-sample fit.

2. **Two cohorts with distinct provenance.** [PROVENANCE NOTE] The GNCA four-way ablation is computed on the N = 838 expression-aligned node graph of NCA.ipynb (CeNGEN-mapped features; the saved prediction CSV `analysis/gnca/gnca_pred_outgoing_strength.csv` contains 838 rows, with some duplicate and whitespace-padded neuron names — a cleanliness issue noted in Methods). The "N = 91 somatic neurons with both modalities" cohort used for the degree/CCA analyses is the connectome ∩ averaged single-cell expression intersection built in the master pipeline. The classifier and collapse-curve analyses run on still other cohorts (179 and 222 neurons respectively). We attach each N to its own analysis and do **not** attach N = 91 to the GNCA number.

3. **The dynamics null is on simulated, not real, activity.** [PROVENANCE NOTE] The gene-conditioned graph-LSTM was trained against simulated trajectories, because the only real *C. elegans* activity recordings of comparable scale [Kato et al. 2015] lack the per-neuron identities required to attach expression. The null therefore bounds what the clean pipeline can demonstrate on the available identity-resolved data; it does not refute a gene→dynamics effect in biology, and we frame it accordingly.

4. **The collapse curve concerns a different task than the GNCA edge model.** Its baseline (r ≈ 0.82) is the operating point of an expression→hub-status (out-degree > 5) predictor, not the GNCA synaptic-strength regression; the two should not be conflated. The curve is informative about *redundancy structure*, not absolute predictive ceiling.

5. **No confidence intervals are reported on the headline scalars** under the present zero-recompute constraint; bootstrap CIs on the modest cohorts are an explicit pre-submission gap.

These are not hedges against the thesis — items (3) and the redundancy result are, if anything, its strongest support — but the conditions under which it is earned, stated plainly so that the conditional claim ("given topology, expression adds little to local wiring; nothing shuffle-robust to dynamics on these data") is read at exactly the strength the artifacts support.

---

## 2. Methods

The design principle throughout is **triangulation against controls**: no model's performance is reported without the matched control that removes the variable of interest (gene expression), and the question is posed across three method families of increasing nonlinearity — a linear method (canonical-correlation analysis), a nonlinear static predictor (a graph neural cellular automaton), and a classifier probe — so that the conclusion does not rest on the inductive bias of any single architecture. A fourth family (gene-conditioned sequence models) supplies the dynamics test, and a fifth (cumulative in-silico knockout) characterizes how the expression contribution is distributed across genes.

> **Provenance and honesty note (for internal review).** Every quantitative and architectural statement below is read from a *committed, executed* artifact — primarily the notebook `NCA.ipynb` (with cell outputs), the dynamics notebook `PhaseD.ipynb`, the knockout notebook `phaseE.ipynb`, the master pipeline `Paper1MasterNotebook.ipynb`, the saved prediction CSVs under `analysis/gnca/`, and the figure `public/images/projects/celegans-collapse-curve.png`. The canonical analysis repository is `/home/rohit/Desktop/C-Elegans/`; the manuscript worktree (`website-paper`) contains only the narrative MDX and the collapse-curve figure, **not** the analysis artifacts, so each method below cites the artifact in the analysis repo by path. Nothing was re-run for this draft (zero-compute). Numbers verified from artifacts are tagged **[verified-in-artifact]**; numbers that are correct but whose framing in the project's narrative summaries was wrong are tagged **[corrected]**; items I could not back from an artifact are tagged **[PROVENANCE GAP]** rather than invented.

### 2.1 Data

#### 2.1.1 Connectome

Connectivity is loaded from the project's consolidated edge table `data/processed/csv/merged_edges_with_relatedness.csv` (fallback `master_connectome_dataset.csv`), with columns `Source`, `Target`, `Type`, `SynapseCount`, `SourceFile`, `Relatedness`. The full table is 27,727 edges over 1,533 nodes. The corpus is assembled from the classical serial-section EM reconstruction of White et al. [1986], the network-quantified edge list of Varshney et al. [2011], the quantitative weighted hermaphrodite connectome of Cook et al. [2019], and the developmental connectome series of Witvliet et al. [2021]. [src: `NCA.ipynb` cells 0/9; `CLAUDE_CONTEXT.md` connectome-source block]

For the GNCA, the directed source→target chemical+gap multigraph is filtered to edges with `SynapseCount ≥ 0` and the synapse count is used as the edge weight. The adjacency is **out-degree row-normalized** — $A_{\text{norm}} = D_{\text{out}}^{-1}A$ — so that message passing aggregates a convex combination of a neuron's outgoing neighbors. On the GNCA's expression-aligned node set this yields a working graph of **N = 838 nodes / 16,673 edges**. [src: `NCA.ipynb` cell 1, executed output `N=838, edges kept=16673`]

Neurotransmitter identities used in deriving role/edge features follow the curated cholinergic map of Pereira et al. [2015] and the WormAtlas neurotransmitter compilation of Loer & Rand [2022] (a curated data table, not a peer-reviewed article; see References).

#### 2.1.2 Single-cell expression

The study uses **two distinct expression provenances depending on the analysis**, and the manuscript states each explicitly rather than collapsing them:

- **GNCA + role classifier + CCA (the `NCA.ipynb` track):** per-neuron gene expression from the **CeNGEN** single-cell L4 atlas [Taylor et al. 2021; Hammarlund et al. 2018], mapped to WormBase gene IDs in `data/processed/csv/gene_expr_CeNGEN_base_mapped.csv` — a matrix of **838 nodes × 22,469 genes**. [src: `NCA.ipynb` cell 0, `expr_df: (838, 22469)`]
- **Master-pipeline degree/CCA cohort (the `Paper1MasterNotebook.ipynb` track):** the **GSE136049** (Packer/Zhu 2019) gene-by-barcode matrix (22,468 genes × 100,954 cells) averaged to per-neuron profiles and intersected with the connectome. [src: `Paper1MasterNotebook.ipynb` cells 7, 14, 27, 30–32, output `Neuron expression matrix: (91, 22468)`; saved `expression_neuron_mean.csv` shape (91, 22468)]

#### 2.1.3 The matched cohorts (cohort-N disclosure)

**[corrected]** The headline figure of **N = 91 somatic neurons with both connectome and single-cell expression** is the matched cohort of the *master pipeline* (`Paper1MasterNotebook.ipynb` cell 31, `(91, 22468)`), used for the per-neuron degree-prediction and the master-notebook CCA analyses. It is **not** the cohort on which the headline GNCA r was computed. The GNCA synaptic-strength model runs on the **N = 838** CeNGEN-aligned node set described in §2.1.1–2.1.2; the role classifier and the `NCA.ipynb` CCA run on a 179-neuron alignment (`NCA.ipynb` cell 23, `179 neurons × 22469 genes; 27727 edges`); the collapse curve runs on 222 common neurons (§2.6). These distinct Ns are reported with their own provenance throughout, and no single N is attached uniformly to all results.

**[PROVENANCE GAP]** The saved GNCA prediction CSV (`analysis/gnca/gnca_pred_outgoing_strength.csv`, 838 data rows) contains duplicate neuron-name rows and trailing-whitespace names (e.g. `RMGR `, ` ASGL`), an artifact of the 838-row CeNGEN join; consequently 838 rows ≠ 838 unique somatic neurons. This is a reviewer-facing cleanliness issue: the clean unique-neuron count for the GNCA cohort should be re-derived (under a future, non-zero-compute pass) before submission. The N = 91 master-cohort intersection manifest is recorded in the cohort notebook but not as a standalone artifact in the manuscript worktree.

### 2.2 Canonical-correlation analysis (linear)

To test for *linear* shared structure between transcriptional identity and local connectivity, we apply canonical-correlation analysis (CCA) [Hotelling 1936] between a per-neuron **expression embedding** and a per-neuron **connectivity-motif embedding**.

- **Motif embedding.** A 5-dimensional per-neuron motif matrix is built from the connectome (`NCA.ipynb` cells 21–22, `motif_counts_safe`): counts of reciprocal, feedforward, feedback-triangle, shared-output, and lateral-inhibition participation per neuron.
- **Embeddings.** Expression is reduced by `PCA(n_components=50, random_state=0)` on the standardized expression matrix `X_expr`; the motif matrix is reduced by `PCA(n_components=5, random_state=0)` to `Y_motif`.
- **CCA.** `sklearn.cross_decomposition.CCA(n_components=3, max_iter=1000)` is fit on (PCA-50 expression, PCA-5 motif); the canonical correlation for component $i$ is $\rho_i = \mathrm{corrcoef}(U_{:,i}, V_{:,i})$.

[src: `NCA.ipynb` cell 35, executed `Canonical correlations (expr <-> motif): [0.875, 0.836, 0.809]`] **[verified-in-artifact]**

**Honesty caveat (carried to Results/Limitations).** These canonical correlations are **in-sample**: there is no train/test split and no permutation null on the $\rho_i$ in this cell, and they are computed in a high-PC space (50 expression PCs vs 5 motif PCs) over a small N, a regime in which CCA correlations are upward-biased. The values are reported as **descriptive linear-alignment magnitudes**, not as held-out generalization estimates; a permutation/shuffle null on the canonical correlations is the recommended pre-submission addition.

### 2.3 Graph neural cellular automaton (nonlinear, static)

The central synaptic-strength predictor is a **graph neural cellular automaton (GNCA)** — a message-passing model [Gilmer et al. 2017; Kipf & Welling 2017] run in the cellular-automaton regime [Mordvintsev et al. 2020]. The full architecture and training configuration below are read directly from `NCA.ipynb` cells 1–3 (no re-run).

#### 2.3.1 Inputs and target

- **Node features.** $X_{\text{feat}}$ = the top-**32** principal components (`USE_PCA=True, N_PCA=32`, `NCA.ipynb` cell 1) of the per-gene `StandardScaler`-normalized CeNGEN expression matrix (§2.1.2).
- **Graph.** The out-degree row-normalized directed synapse-count adjacency $A_{\text{norm}}$ of §2.1.1.
- **Target.** The **z-scored mean outgoing synaptic strength** per node: outgoing edge weights grouped by source and averaged, then standardized $(x-\mu)/\sigma$.

#### 2.3.2 Model (`NCA.ipynb` cell 2, class `GNCA`)

- Input encoder: $\text{Linear}(32 \to 64)$ with tanh activation (tanh-initialized 64-dim hidden state per node).
- Message: $\text{Linear}(64 \to 64)$ without bias, aggregated as $A_{\text{norm}}\, m(x)$.
- Update: a residual MLP on the concatenation of self-state and aggregated message — $\text{Linear}(128 \to 128)\!\to\!\text{ReLU}\!\to\!\text{Linear}(128 \to 64)$ applied to $[\,x,\; A_{\text{norm}} m(x)\,]$, with residual update $x \leftarrow x + \Delta x$.
- Stochastic asynchronous **fire mask**: a per-node $\text{Bernoulli}(0.5)$ mask gates the update during training (asynchronous-CA flavor; disabled at eval).
- Readout: $\text{LayerNorm}(64) \to \text{Linear}(64 \to 1)$.
- Unroll: **8** message-passing/update steps (`STEPS=8`).
- Parameter count (printed): **0.031169 M** (~31k parameters). [src: `NCA.ipynb` cell 2/3] **[verified-in-artifact]**

#### 2.3.3 Training (`NCA.ipynb` cell 3)

AdamW ($\text{lr}=3\times10^{-3}$, weight decay $1\times10^{-4}$); `CosineAnnealingLR` ($T_{\max}=200$, $\eta_{\min}=3\times10^{-4}$); MSE loss on the z-scored target; gradient clipping at 1.0; `torch.manual_seed(42)`; **200 epochs**; best-training-loss checkpoint saved to `gnca_best.pt`; device = CUDA.

#### 2.3.4 Reported performance and a critical framing correction

The full GNCA reaches **Pearson r = 0.987, MAE = 0.097**. [src: `NCA.ipynb` cell 4, executed `Final Pearson r=0.987 | MAE=0.097`; saved predictions `analysis/gnca/gnca_pred_outgoing_strength.csv` (838 rows; recomputed Pearson r = 0.984 from the saved snapshot, vs 0.987 reported from the best checkpoint); `CLAUDE_CONTEXT.md` line 38 labels this `GNCA (full)`]

**[corrected — do NOT call this "held-out".]** `NCA.ipynb` cells 3–8 (the synaptic-strength model and its three controls) contain **no train/test split**: Pearson r is computed **in-sample on the full node set**. `train_test_split` / `KFold` appear only in later cells (20, 24–26, 31, 36–39), which drive the motif-prediction and role-classifier analyses — not the synaptic-strength GNCA. The project's own scorecard correctly names this model `GNCA (full)`. The manuscript therefore reports **r = 0.987 as an in-sample fit**, not as held-out performance. The thesis does not depend on the absolute r: it rests on the *relative* comparison among the four conditions below, all measured the same in-sample way, so the comparison is valid. A more honest cross-validated number also exists for a related task — a 5-fold CV of per-neuron motif regression in `NCA.ipynb` cell 24 gives mean r ≈ 0.639 (folds 0.62 / 0.25 / 0.82 / 0.86 / 0.64) — and is reported alongside the in-sample GNCA fit so readers can see both.

#### 2.3.5 The four-condition control battery

All four conditions reuse the same `GNCA` class; they differ only in what information is supplied. r-values are the load-bearing contrast for the topology-dominant / gene-modulatory thesis.

| Condition | What it removes | r | Artifact (source) |
|---|---|---|---|
| **Full** (topology + real expression) | nothing | **0.987** | `NCA.ipynb` cell 4 |
| **Shuffled-expression** (topology preserved) | the expression↔neuron correspondence | **0.934** | `NCA.ipynb` cell 5 |
| **Graph-only** | all expression features | **0.861** (peak) / 0.857 (final-epoch) | `NCA.ipynb` cell 8 |
| **Expression-only** | the informative graph | **0.27** (final-epoch 0.269) | `NCA.ipynb` cell 8 |

- **Shuffled-expression** (`NCA.ipynb` cell 5, "Cell A — Shuffle control", `seed 123`, 80 epochs): `torch.randperm` permutes the expression-feature **rows** (breaks neuron↔expression alignment) while $A_{\text{norm}}$ is unchanged. r drops only 0.987→0.934 — i.e. topology carries the signal. **Provenance precision:** 0.934 is the value at epoch 80, the **last** logged epoch of this 80-epoch run (the main model trains 200), so it is the final reported value, not a cherry-picked mid-run peak. [src: `NCA.ipynb` cell 5, executed `[SHUF] Epoch 080 | loss=0.1297 | r=0.934`] **[verified-in-artifact]**

- **Graph-only** and **Expression-only** (`NCA.ipynb` cell 8, "Cell D — Compare graph-only vs expression-only", via `train_gnca_baseline`, 100 epochs, `state_dim=64`). **[corrected — minor discrepancy to disclose]** The headline graph-only **0.861** is the per-epoch *peak* (epoch 80); the notebook's final `=== Baseline Comparison ===` summary line (last epoch, 100) prints **`Graph-only r = 0.857`**. Both come from the same run. The manuscript reports "graph-only r ≈ 0.86 (peak 0.861 / final-epoch 0.857)"; the qualitative claim (graph-only < full, ≫ expression-only) is unaffected. Expression-only plateaus near **r ≈ 0.27** (`[EXPR-ONLY] Epoch 080 | r=0.270`; final-epoch 0.269; summary `Expression-only r = 0.269`). [src: `NCA.ipynb` cell 8] **[verified-in-artifact]** The expression-only weakness is independently corroborated in `Paper1MasterNotebook.ipynb` cells 37–38, where raw-gene → degree regression gives strongly **negative** out-of-sample $R^2$ (e.g. `total_degree R² = -15.3`), confirming expression alone has little predictive power for connectivity targets.

- **Robustness run.** A separate edge-dropout (50%) + `state_dim=8` GNCA (120 epochs) was run as a robustness check on the architecture's reliance on the full graph. [src: `NCA.ipynb`]

#### 2.3.6 Committed prediction artifact

Per-node predictions are saved to `analysis/gnca/gnca_pred_outgoing_strength.csv` (columns `neuron`, `y_true_z`, `y_pred_z`; 838 data rows), permitting recomputation of the correlation **without re-running the model** (recomputed Pearson r = 0.984 from the saved snapshot; 0.987 is the best-checkpoint value). Companion artifacts in the same directory — `gnca_motif_performance.csv`, `gnca_saliency_features.csv`, `gnca_integrated_gradients_{features,genes}.csv`, and the per-gene knockout sweep `gnca_gene_perturb_fast.csv` — support the downstream attribution and collapse-curve analyses. [src: `analysis/gnca/`] **[verified-in-artifact]**

### 2.4 Neuron-role classifier probe (expression alone)

As an independent check that expression is genuinely informative about circuit role, a supervised classifier predicts each neuron's coarse **behavioral role (sensory / interneuron / motor)** from its expression vector alone.

- **Features.** PCA-reduced expression features $X$.
- **Labels.** A 3-class role assigned in-notebook by a **neuron-name-prefix heuristic**: sensory (AF/AL/AS/AVM/AWA/AWC/ASE/ASH/URX/OL...), motor (DA/DB/DD/VA/VB/VD/AS...), else interneuron.
- **Model.** `nn.Sequential(Linear(d,128) → ReLU → Linear(128,64) → ReLU → Linear(64,3))`; Adam ($\text{lr}=2\times10^{-3}$); cross-entropy loss; 100 epochs/fold; `StratifiedKFold` (5-split `kf`, defined cell 25).
- **Metric.** Mean fold accuracy = **0.9274603174603175 (92.7%)**. [src: `NCA.ipynb` cell 28, "TASK D: Predict Behavioral Role (3-class)", executed `Mean Accuracy (role): 0.9274603174603175`] **[verified-in-artifact]**

**Honesty caveats (disclosed in Results).** (1) The role labels are a name-prefix heuristic, **not** curated WormAtlas annotations; the `AS` prefix appears in *both* the sensory and motor lists, so `AS*` neurons are labeled motor (the motor check runs second) — a known ambiguity in the labeling rule. (2) No class-balance / chance-level baseline is printed in this cell; 3-class chance is ~0.33 but the role distribution is imbalanced, so balanced accuracy would be the more honest headline metric. (3) This is the **role** task from `NCA.ipynb` only; a separate clean-pipeline notebook (`New Notebooks/04_synthetic_dynamics_and_classification.ipynb`) reports a *different* task — 3-class neurotransmitter (cholinergic/GABA/glutamate) at 0.727 ± 0.077 (N = 84) **with a proper permutation null** (null 95th percentile 0.629) — and the two must not be conflated. The 92.7% is the name-prefix role probe; it establishes the upper-bound sense in which expression "knows what a neuron is," against which the *failure* of expression to add connectivity/dynamics information becomes interpretable.

### 2.5 Dynamics: gene-conditioned graph-LSTM and the shuffle test (the honest null)

To test whether expression shapes *dynamics* rather than only static wiring, a hybrid **gene-conditioned graph-LSTM** (the "PhaseD" model) is trained to predict neural time-series, and subjected to a pre-specified ablation/shuffle battery.

- **Model (D3 `HybridGeneGraphLSTM`).** Maps sliding-window dynamics (`X_win` shape `(B, T, N, 1)`) plus a per-neuron CeNGEN **gene embedding** (PCA of expression) over a normalized sparse adjacency.
- **Tests.** A **zero-gene ablation** (`no_gene`) and a **gene-shuffle** (`gene_shuf`, permute the per-neuron gene embedding across neurons).
- **Gate.** `gene_causal_ok = (mse_shuf − mse_hybrid) > 0.005`.
- **Result.** `mse_hybrid = 0.9879`, `mse_no_gene = 0.9857`, `mse_gene_shuf = 0.9879`. The hybrid equals the shuffled-gene model exactly, and zeroing genes is *marginally better*; the gate evaluates to $0.9879 - 0.9879 = 0.0000 \not> 0.005$ → **`gene_causal_by_shuffle = False`**. [src: `PhaseD.ipynb` cell 66 `hybrid: 0.9879, no_gene: 0.9857, gene_shuf: 0.9879`; cell 68 `VERDICT: {'mse_gain_over_identity': True, 'gene_causal_by_shuffle': False}`] **[verified-in-artifact]**

**Corroborating clean re-implementation.** The cleaner re-derivation in `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` (Part A) reaches the **same null** via a preregistered criterion: synthetic linear dynamics from the Witvliet adjacency plus a small gene-conditioned term, four Ridge($\alpha=1$) predictors on features $[\,y[n],\,(A_{\text{norm}}y)[n],\,\text{gene}[n]\,]$ (identity / graph-only / hybrid / shuffled-gene), trained on the first half of the timesteps and tested on the second; the preregistered gate $(\text{mse}_{\text{shuf}} - \text{mse}_{\text{hybrid}}) > 0.01$ evaluates to **0.001283 → NOT GENE-CAUSAL (null)** (mse_identity 0.0034, mse_graph 0.0038, mse_hybrid 0.0025, mse_shuf 0.0038; summary saved to `data_derived/nb04_dynamics_summary.csv`).

**Critical scope statement (also in Results/Limitations).** This null is on **synthetic / simulated dynamics**, not real calcium imaging. As stated in the clean notebook (`nb04` cell 0): the PhaseD dynamics test ran on a simulated trajectory (`PhaseA_A9_trajectories_continuous.npy`); the only real dynamics set (Kato et al. 2015) lacks neuron identities, so gene→dynamics cannot be tested on real data with the available data. The null therefore **bounds what the clean pipeline can show** — gene conditioning adds no shuffle-robust improvement under this model class and this (synthetic) dynamics target — rather than refuting gene→dynamics causality in biology. It is foregrounded, not buried: a negative result obtained by a pre-specified shuffle control is the evidence that makes the topology-dominant thesis credible rather than an artifact of an over-expressive model.

### 2.6 Genetic-redundancy ("collapse") analysis

To characterize *how* the expression contribution is distributed across genes, we rank genes by their influence on the predictor and remove them cumulatively (in-silico knockout), recording model performance as a function of the number of top drivers removed.

- **Task and target.** This sweep is run on the **expression → hub-topology** map, **not** the GNCA edge-regression. The target is `Hub_Div = 1` if out-degree > 5 else 0. [src: `phaseE.ipynb` cell 6]
- **Cohort and features.** N = **222** common neurons; features = the top-2000-variance CeNGEN genes (`StandardScaler`-normalized).
- **Model.** `DirectMLP(2000 → 32 → 1)`, Adam ($\text{lr}=1\times10^{-3}$), `BCEWithLogitsLoss` with L1 (Lasso) regularization (`L1_LAMBDA=0.005`) on the first layer, 401 epochs; metric = `pearsonr(Hub_Div, sigmoid(logits))`.
- **Knockout method.** Cumulative zeroing of the top-K driver genes (ranked by influence on the MLP's hub prediction) for $K \in \{0,1,2,5,10,20,30,40,50,75,100,200\}$, re-evaluating both the model correlation R and the mean probability assigned to hub neurons at each K.
- **Result (the collapse curve, Figure 1).** Trained R = 0.818; the cumulative-KO trajectory is R(K=0)=0.818, 0.819, 0.802, 0.799 (K=5), then 0.678 (K=10), 0.679, 0.686, 0.685, 0.665 (K=50), 0.683, 0.682, **0.687 (K=200)** — i.e. an initial drop over the first ~10 knockouts followed by a **plateau ≈ 0.665–0.687 well above the chance line through 200 removed genes**; the average hub-probability declines only gently (0.952 → 0.886). [src: figure `public/images/projects/celegans-collapse-curve.png`; source+output `phaseE.ipynb` cell 6, "Combinatorial Gene Knockout Analysis (The Collapse Curve)"; original saved figure `/mnt/ssd4tb/Desktop/C-Elegans/figures/crispr_results/genetic_collapse_curve.png`] **[verified-in-artifact]**

**Interpretation and scope.** Graceful, plateauing degradation is the signature of a **redundant** genetic code rather than a few load-bearing master genes — no small driver set is individually necessary for the topology-aligned prediction. The collapse-curve baseline (R ≈ 0.82) is the operating point of *this hub-prediction MLP*, a different model setup from the GNCA edge regression (§2.3); the curve is informative about **redundancy structure** (the shape of the degradation), not about the GNCA's absolute predictive ceiling, and the two numbers are not conflated.

**[PROVENANCE GAP — figure label discrepancy to resolve.]** The committed PNG title reads `Top Driver: WBGene00001137`, but the `phaseE.ipynb` cell-6 run read for this draft identifies the top driver as `WBGene00019961` (the cell-4 single-gene version). The two runs share the same plateau pattern and R/hub-prob trajectory; only the reported top-driver gene-ID label differs (the PNG was produced by a sibling/re-run). The discrepancy must be resolved and the top driver resolved to a gene symbol via WormBase before the final figure caption is fixed.

### 2.7 Corroborating variance-partition (topology-dominant spine)

Beyond the four headline conditions, a variance partition for motif prediction (`NCA.ipynb` cell 31, mean $R^2$) independently supports the topology-dominant reading: expression-only 0.388, degree-only 0.838, both 0.882, expression-UNIQUE 0.044, degree-UNIQUE 0.494 — expression adds little *unique* variance over topology/degree. [src: `NCA.ipynb` cell 31] **[verified-in-artifact]**

---

## 3. Results

The question this section answers is not whether gene expression and connectivity are coupled in *C. elegans* — they demonstrably are — but whether single-cell expression carries information about local wiring **beyond what the connectome's own topology already supplies**. We answer it by triangulation: the same conditional accounting is run across three method families of increasing nonlinearity — a linear method (canonical-correlation analysis), a nonlinear static predictor (a graph neural cellular automaton with a matched shuffle/ablation battery), and a supervised classifier probe — and is then stress-tested against a dynamics model whose pre-specified null is the decisive result, and against a genetic-knockout sweep that characterizes how the (small) expression contribution is distributed across genes.

Every quantitative claim below is tagged with the committed artifact it is read from. Where the backing artifact contradicts a previously-circulated framing of a number (e.g. the word "held-out," or the cohort size attached to a result), the discrepancy is **FLAGGED** in place rather than smoothed over. Nothing was re-run for this section.

> **Cohort note — two distinct Ns, stated up front.** The analyses below do **not** all run on the same neuron set, and the manuscript must not conflate them. The GNCA synaptic-strength model and its three controls (§3.1) run on the **N = 838 expression-aligned node set** of `NCA.ipynb` (input `gene_expr_CeNGEN_base_mapped.csv`, 838 rows; the saved prediction CSV contains duplicate neuron-name rows and trailing-whitespace names, so 838 is *not* 838 unique somatic neurons). The CCA (§3.2) and the role classifier (§3.3) run on a ~**179-neuron** alignment in the same notebook; the collapse curve (§3.5) runs on **N = 222** common neurons. The widely-quoted figure **N = 91** is the somatic-neurons-with-both cohort built in `Paper1MasterNotebook.ipynb` (CeNGEN/GSE136049 averaged per-neuron, intersected with the connectome) and is the unit for the master-notebook degree/CCA work — **it is not the cohort of the headline GNCA r = 0.987.** [src for N = 838: `notebooks/archive/nca/NCA.ipynb` cell 0 (`expr_df: (838, 22469)`) and `analysis/gnca/gnca_pred_outgoing_strength.csv` (838 data rows, verified by direct read). src for N = 91: `notebooks/Paper1MasterNotebook.ipynb` cell 31 (`Neuron expression matrix: (91, 22468)`). All artifact paths are under `/home/rohit/Desktop/C-Elegans/`.]
>
> **[PROVENANCE GAP — N = 91]** I could not locate, in this worktree or the analysis repo, an intersection-manifest artifact that re-derives 91 from the raw source-cell lists; it appears only in the project's own MDX write-up and the context scorecard. Report N = 91 as the master-notebook cohort size, and re-derive it from the canonical loader with explicit inclusion criteria before submission.

### 3.1 The marginal value of expression for synaptic strength is small (GNCA + controls)

The central static test predicts each neuron's mean **outgoing synaptic strength** (z-scored) from a graph neural cellular automaton run over the connectome, with optional per-neuron expression features (top-32 PCA components of CeNGEN expression). The full model reaches **Pearson r = 0.987 (MAE = 0.097)**. [src: `notebooks/archive/nca/NCA.ipynb` cell 4 executed output `Final Pearson r=0.987 | MAE=0.097`; saved predictions `analysis/gnca/gnca_pred_outgoing_strength.csv` (838 rows); tabulated in `CLAUDE_CONTEXT.md` line 38 as **"GNCA (full)"**.]

> **FLAG — this is an in-sample fit, NOT "held-out."** The synaptic-strength cell and its three controls (`NCA.ipynb` cells 3–8) contain **no train/test split**: Pearson r is computed on the full node set. `train_test_split`/`KFold` appear only in later cells (20, 24–26, 31, 36–39), which drive the motif/classifier analyses, not this regression. `CLAUDE_CONTEXT.md` correctly labels the model "GNCA (full)." The manuscript must report r = 0.987 as an **in-sample full-fit**, not a held-out result. Any prior text (including the project MDX) that says "held-out r = 0.987" is **incorrect on this point** and should be corrected.
>
> **Secondary check.** Recomputing Pearson r directly from the committed prediction CSV (838 rows) yields **r = 0.984** — consistent with, but not identical to, the notebook's 0.987. The notebook value is from the best-loss checkpoint; the saved CSV recovers r ≈ 0.984. Report 0.987 as the notebook figure and note the CSV recovers ≈ 0.984. [verified by direct read of `analysis/gnca/gnca_pred_outgoing_strength.csv`.]

The control battery localizes where that signal lives. All four conditions are measured the same in-sample way, so the **relative** comparison is valid even though the absolute r is not held-out:

| Condition | r | Provenance | What it rules out |
|---|---|---|---|
| **Full** (topology + real expression) | **0.987** | `NCA.ipynb` cell 4; `CLAUDE_CONTEXT.md` l.38 | ceiling (in-sample) |
| **Shuffled-expression** (topology intact, expression rows permuted) | **0.934** | `NCA.ipynb` cell 5 (`[SHUF] Epoch 080 \| r=0.934`); `CLAUDE_CONTEXT.md` l.39 | that *correctly-assigned* expression is load-bearing — it is not |
| **Graph-only** (no expression features) | **≈ 0.86** | `NCA.ipynb` cell 8 (`[GRAPH-ONLY] Epoch 080 r=0.861`; final-epoch summary `r = 0.857`); `CLAUDE_CONTEXT.md` l.40 | that topology alone is insufficient — it is nearly sufficient |
| **Expression-only** (graph removed) | **0.27** | `NCA.ipynb` cell 8 (`[EXPR-ONLY] r=0.270`, final-epoch `0.269`); `CLAUDE_CONTEXT.md` l.41 | that expression alone predicts local wiring — it largely does not |

> **FLAG — graph-only value.** The headline 0.861 is the **per-epoch peak (epoch 80)**; the notebook's final `=== Baseline Comparison ===` summary line (last epoch, 100) prints **`Graph-only r = 0.857`**. Both come from the same run. Report "**≈ 0.86 (peak 0.861 / final 0.857)**." Expression-only is likewise peak 0.270 / final 0.269; report **≈ 0.27**.
>
> **Provenance precision — shuffled-expression.** 0.934 is the value at epoch 80, the *last* logged epoch of an 80-epoch run (the shuffle control runs fewer epochs than the 200-epoch main model), so it is this control's final reported value, not a cherry-picked mid-run peak. Mechanism: `torch.randperm` permutes the expression-feature rows (breaking the neuron↔expression alignment) while the out-degree-normalized adjacency `A_norm` is left untouched.

**Three conclusions follow, and each is anchored to a specific control:**

1. **Topology is the dominant carrier of the signal.** Graph-only (≈ 0.86) already recovers the great majority of the explainable variance, while expression-only (0.27) is far behind. *Control that establishes this:* the graph-only vs. expression-only gap.

2. **The marginal value of correctly-assigned expression is the Full − Shuffled gap (0.987 → 0.934), which is small.** Scrambling *which* expression vector belongs to *which* neuron — while preserving the graph and the full expression distribution — costs only ≈ 0.05 in r. *Control that establishes this:* the shuffle. This is the single most important number in the section, because it isolates the contribution of expression *identity* from the contribution of merely having feature-shaped inputs.

3. **Expression is not useless, but it is modulatory.** Full (0.987) does exceed graph-only (≈ 0.86), so a real refinement exists; its size is consistent with a modulatory layer on a topology-determined backbone, not with expression being the architect.

A caution on reading r near the ceiling: at r = 0.987 vs 0.934 the variance-explained difference (R² 0.974 vs 0.872) is larger than the raw-r gap suggests, but it remains modest beside the graph-only baseline, and the qualitative ranking (topology ≫ expression-identity) is robust. **[PROVENANCE GAP — CIs]** No bootstrap/permutation confidence intervals exist for any of these four r-values; `CLAUDE.md` line 41 and `CLAUDE_CONTEXT.md` line 62 both flag "No bootstrap CIs on N=91 neurons" as a known pre-submission gap. Under the zero-compute constraint we report point estimates exactly as committed and acknowledge the absence of CIs.

**Corroborating evidence (same thesis, independent artifact).** In `Paper1MasterNotebook.ipynb` (cells 37–38) a raw-gene → degree regression gives strongly **negative R²** (e.g. total-degree R² = −15.3), independently confirming that expression alone has little predictive power for connectivity targets. And `NCA.ipynb` cell 31's variance partition for motif prediction gives expression-only R² = 0.388 vs degree-only R² = 0.838 (both = 0.882; expression-**unique** = 0.044 vs degree-unique = 0.494) — i.e. expression adds little *unique* variance over topology/degree. [src: `notebooks/Paper1MasterNotebook.ipynb` cells 37–38; `notebooks/archive/nca/NCA.ipynb` cell 31.]

### 3.2 Expression nonetheless carries linear circuit structure (CCA)

To establish that the topology-dominant result is not an artifact of expression being uninformative, we measure the *linear* shared structure between an expression embedding and a connectivity-**motif** embedding by canonical-correlation analysis. The top three canonical correlations are **ρ = [0.875, 0.836, 0.809]**. [src: `notebooks/archive/nca/NCA.ipynb` cell 35 executed output `Canonical correlations (expr <-> motif): [0.875, 0.836, 0.809]`; tabulated in `CLAUDE_CONTEXT.md` line 44. Exact match.] Method: `sklearn.cross_decomposition.CCA(n_components=3, max_iter=1000)` on PCA(50) of expression vs PCA(5) of a 5-column motif matrix (reciprocal / feedforward / feedback-triangle / shared-output / lateral-inhibition counts per neuron, built in `NCA.ipynb` cells 21–22); ρᵢ = corrcoef(U[:,i], V[:,i]).

At least three strongly-correlated dimensions are shared between transcriptional identity and local motif participation — fully consistent with prior reports that expression and connectivity are coupled in *C. elegans* [Arnatkevičiūtė et al. 2018; Fenyves et al. 2020]. The point of this paper is *not* that the coupling is absent (it plainly exists, linearly and substantially) but that, **conditioned on topology**, expression adds little to predicting the wiring itself (§3.1) or the dynamics (§3.4).

> **FLAG — descriptive, upward-biased.** These ρ are **in-sample** (no held-out split, no permutation null on the canonical correlations) and are computed in a high-PC space (50 expression PCs, 5 motif PCs) over a small N — a regime in which CCA correlations are upward-biased. Frame them as **descriptive magnitudes of linear alignment**, or add a shuffle null before submission. They are not, on their own, evidence of a large *conditional* expression contribution; §3.1 is the conditional accounting.

### 3.3 Expression predicts coarse behavioral role (classifier probe)

As an independent upper-bound check that expression *is* genuinely informative about circuit role, an expression-only classifier predicts the three-class behavioral role (sensory / interneuron / motor) at **mean CV accuracy = 92.7%** (0.9274603…). [src: `notebooks/archive/nca/NCA.ipynb` cell 28 (`# === TASK D: Predict Behavioral Role (3-class) ===`), output `Mean Accuracy (role): 0.9274603174603175`; tabulated in `CLAUDE_CONTEXT.md` line 45. Exact match.] Method: MLP 128-64-3, CrossEntropyLoss, Adam lr = 2e-3, 100 epochs/fold, StratifiedKFold (5-split), features = PCA-reduced expression.

Transcriptional identity therefore *does* encode what kind of neuron a cell is, at the resolution of its broad circuit role — as the terminal-selector logic of *C. elegans* neuronal identity predicts [Hobert 2016]. This is the load-bearing framing for the null in §3.4: expression is richly informative about *identity*, yet that information is largely **redundant with topology** when the task is predicting local wiring or dynamics.

> **FLAGS — three caveats the manuscript must disclose.**
> (1) **Labels are a name-prefix heuristic** assigned in-notebook, not curated WormAtlas annotations; the prefix `AS` appears in *both* the sensory and the motor lists, and because the motor check runs second, `AS*` neurons are labeled motor — an ambiguity/bug to fix or at least disclose.
> (2) **No chance/permutation baseline is printed in this cell.** Three-class chance is ≈ 0.33, but the role distribution is imbalanced, so **balanced accuracy** would be the honest metric; report it before submission.
> (3) **Do not conflate with the NT task.** The cleaner sibling pipeline `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` reports a *different* 3-class task (neurotransmitter chol/GABA/glu at 0.727 ± 0.077, N = 84) *with* a proper permutation null (null 95th pct 0.629). The 92.7% is the role task from the archive `NCA.ipynb` only.

### 3.4 The honest null: gene conditioning does not improve dynamics

The decisive negative result comes from dynamics. A hybrid **gene-conditioned graph-LSTM** (`HybridGeneGraphLSTM`, "PhaseD") was trained to predict neural time-series from sliding-window dynamics plus a per-neuron CeNGEN gene embedding over a normalized sparse adjacency. Its pre-specified shuffle test returns:

**VERDICT: `{'mse_gain_over_identity': True, 'gene_causal_by_shuffle': False}`.**

The supporting MSEs: **hybrid = 0.9879, no_gene (zeroed) = 0.9857, gene_shuf (permuted) = 0.9879.** The hybrid model exactly equals its shuffled-gene control, and *zeroing* the genes is marginally **better** than keeping them. The gate `gene_causal_ok = (mse_shuf − mse_hybrid) > 0.005` evaluates to 0.0000 → **False**. [src: `notebooks/archive/phases/PhaseD.ipynb` cell 66 (`hybrid: 0.9879, no_gene: 0.9857, gene_shuf: 0.9879`) and cell 68 (`VERDICT: {...gene_causal_by_shuffle: False}`); tabulated in `CLAUDE_CONTEXT.md` line 47. Exact match.]

**We foreground this result rather than bury it.** The hypothesis entering Phase D was the strong, blueprint-flavored claim that *gene expression causally shapes how neurons fire together*; the project's own pre-specified null battery falsified it. A negative result obtained from a *shuffle control that was specified before the test* is precisely the evidence that makes the topology-dominant thesis credible rather than an artifact of an over-expressive model: in the one setting where expression *should* most plausibly help (dynamics, where molecular identity sets channel and receptor complements), correctly-assigned expression did not beat its own shuffle.

**Corroborating null (independent, preregistered).** The cleaner re-implementation `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` (Part A) reaches the *same* null through a preregistered criterion: `gene_causal_delta = (mse_shuf − mse_hybrid) = 0.001283 < 0.01 threshold` → "NOT GENE-CAUSAL (null)" (mse_identity 0.0034, mse_graph 0.0038, mse_hybrid 0.0025, mse_shuf 0.0038), summary saved to `data_derived/nb04_dynamics_summary.csv`. [src: `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` Part A.]

> **FLAG — scope of the null (critical, do not overstate).** Per the source (nb04 cell 0), the PhaseD dynamics test ran on a **simulation** (`PhaseA_A9_trajectories_continuous.npy`), **not real calcium imaging**. The only real dynamics set with the right structure (Kato et al. 2015) lacks neuron identities, so a gene→dynamics test cannot be run on real, identity-resolved data with the available data. The manuscript must therefore state the null at the level it was earned: it bounds **what this clean pipeline can show on synthetic, simulation-derived dynamics**, and is *not* a refutation of gene→dynamics causality in biology. "Absence of detectable improvement" on this dataset and model class ≠ "proof of no effect." This is the most important future test (richer, identity-resolved recordings with a preregistered shuffle).

### 3.5 Genetic redundancy: the collapse curve

Finally, to characterize *how* the (small) expression contribution is distributed across genes, top "driver" genes were ranked and removed cumulatively (in-silico knockout) from a predictor of **hub status** (out-degree > 5), recording model correlation as a function of the number of genes removed. A blueprint with a few load-bearing genes would collapse toward chance quickly; a redundant code would degrade gracefully and plateau above chance.

The curve plateaus **well above chance** (Figure 1). After an initial drop from the trained baseline (**r ≈ 0.818**) over the first ~10 knockouts, performance flattens around **r ≈ 0.665–0.687** and stays essentially unchanged from ~K = 10 through **K = 200** removed genes; the average probability assigned to hub neurons declines only gently (**0.952 → 0.886**) and never approaches the chance line at 0. Verified knockout trajectory (R at K = 0,1,2,5,10,20,30,40,50,75,100,200): 0.818, 0.819, 0.802, 0.799, 0.678, 0.679, 0.686, 0.685, 0.665, 0.683, 0.682, 0.687. [src — FIGURE: `/home/rohit/Desktop/website-paper/public/images/projects/celegans-collapse-curve.png` (read directly; red r-curve and navy hub-prob trajectory confirmed visually); SOURCE CODE + OUTPUT: `notebooks/archive/phases/phaseE.ipynb` cell 6 (`Combinatorial Gene Knockout Analysis (The Collapse Curve)`); original saved fig `/mnt/ssd4tb/Desktop/C-Elegans/figures/crispr_results/genetic_collapse_curve.png`.] Method: target `Hub_Div` (out-degree > 5); features = top-2000-variance CeNGEN genes (StandardScaler'd); N = 222 common neurons; `DirectMLP(2000-32-1)`, Adam lr = 1e-3, BCEWithLogitsLoss + L1 (λ = 0.005) on the first layer, 401 epochs; metric = Pearson r between `Hub_Div` and the model's sigmoid output; cumulative zeroing of the top-K driver genes.

This graceful, plateauing degradation is the signature of a **redundant** genetic code, not of a few master genes: no small set of drivers is individually necessary for the topology-aligned prediction. Redundancy of this kind is exactly what one expects if topology — not a compact gene set — is doing the structural work, with the expression contribution spread thinly across many partially-substitutable genes. This supports the topology-dominant thesis from the gene side, complementing the shuffle (§3.1) and the dynamics null (§3.4).

> **FLAGS — two provenance items.**
> (1) **Top-driver gene-ID discrepancy.** The committed PNG title reads "Top Driver: **WBGene00001137**," but the `phaseE` cell-6 run on record identifies the top driver as **WBGene00019961** (the cell-4 single-gene version). The PNG was produced by a sibling/re-run with the same plateau pattern but a different reported top-driver ID; the numeric trajectory (R and hub-prob) is consistent across them, only the gene-ID label differs. **[PROVENANCE GAP]** Neither WBGene ID is resolved to a gene symbol in this worktree; look both up in WormBase, reconcile which run the published figure comes from, and name the gene in the final caption.
> (2) **Different task from the GNCA.** The knockout sweep predicts **hub status** (out-degree > 5), **not** the GNCA outgoing-synaptic-strength regression of §3.1. Describe the collapse curve as redundancy in the **expression → hub-topology** map, not in the GNCA edge model; the baseline r ≈ 0.82 here is a different model setup from the GNCA r = 0.987 and the two must not be conflated.

### 3.6 Synthesis of the results

| Method family | Result | Sourced artifact | What it contributes to the thesis |
|---|---|---|---|
| Linear (CCA) | ρ = [0.875, 0.836, 0.809] | `NCA.ipynb` c.35 | expression *is* coupled to circuit structure (in-sample) |
| Classifier probe | role acc = 92.7% | `NCA.ipynb` c.28 | expression *is* richly informative about neuron identity |
| Nonlinear static (GNCA) | full 0.987 / shuf 0.934 / graph ≈0.86 / expr 0.27 | `NCA.ipynb` c.4–8 | **conditioned on topology, expression adds little** |
| Dynamics (graph-LSTM) | `gene_causal_by_shuffle = False` | `PhaseD.ipynb` c.66/68 | gene conditioning adds **nothing shuffle-robust** (on synthetic dynamics) |
| Redundancy (collapse) | plateau r ≈ 0.68 through 200 KOs | `phaseE.ipynb` c.6 + figure | the small expression contribution is **diffuse/redundant** |

The two informative-about-identity results (CCA, classifier) and the three conditional/marginal results (GNCA controls, dynamics null, collapse curve) are not in tension: expression knows *what a neuron is*, but once the connectome's local topology is given, that knowledge is largely redundant for predicting the *local wiring* and — on this synthetic-dynamics dataset — undetectable above shuffle for predicting *activity*. The convergence of a linear method, a nonlinear static predictor, a classifier probe, a dynamics null, and a redundancy analysis is what licenses the topology-dominant, gene-modulatory reading, and the honest null and the redundancy plateau are its strongest support rather than caveats to it.

---

## 4. Discussion

### 4.1 A topology-dominant, gene-modulatory account of local wiring

Three method families of increasing nonlinearity converge on a single reading. **Linearly**, canonical-correlation analysis finds that transcriptional identity and local connectivity-motif participation share substantial structure: the three leading canonical correlations are ρ = [0.875, 0.836, 0.809] [src: `NCA.ipynb` cell 35]. **Statically**, a graph neural cellular automaton predicting per-neuron outgoing synaptic strength reaches r = 0.987 with topology + expression, but a graph-only model already reaches r ≈ 0.86, and permuting which expression vector belongs to which neuron — while leaving the graph and the expression *distribution* untouched — costs only ≈ 0.05 in r (0.987 → 0.934) [src: `NCA.ipynb` cells 4, 5, 8; saved predictions `analysis/gnca/gnca_pred_outgoing_strength.csv`]. **Dynamically**, a gene-conditioned hybrid graph-LSTM shows no shuffle-robust improvement from real versus permuted expression (`gene_causal_by_shuffle = False`) [src: `PhaseD.ipynb` cell 68; cell 66]. And the redundancy analysis (§4.5, Fig. 1) shows the expression contribution is diffuse, surviving removal of the 200 top driver genes.

The convergent interpretation is that **local wiring in *C. elegans* is largely self-determining at the level of topology, with gene expression acting as a modulatory refinement layer rather than the primary architect.** This is *not* the claim that genes do not build the nervous system — they obviously do, through the combinatorial terminal-selector programs that fix and maintain neuron-type identity [Hobert 2016] and the surface-molecule logic that prior work has begun to read out [Barabási & Barabási 2020; Kovács et al. 2020]. It is the narrower, quantitative claim: *once the connectome's local topology is given, the additional predictive information that single-cell expression carries about that same local wiring is small, and for dynamics on this dataset it is undetectable above shuffle.*

The most natural reconciliation with the genetic-blueprint program is **temporal**: the genetic code acts largely *upstream*, sculpting topology during development, so that in the adult the topology is a near-sufficient statistic for local wiring. Under that reading, expression and connectivity are not independent (CCA confirms they are not), but expression's causal work has already been "compiled into" the graph by the time we read both modalities out — leaving only a modulatory residue measurable at the adult snapshot.

### 4.2 Why the classifier result and the wiring result are not in tension

A reader might object that an expression-only classifier recovers coarse behavioral role (sensory / interneuron / motor) at 92.7% mean cross-validated accuracy [src: `NCA.ipynb` cell 28] — surely that means expression "knows" connectivity? The resolution is the crux of the paper. Expression is *richly* informative about **what kind of neuron a cell is**; it is *not* correspondingly informative about **which specific partners it wires to, beyond what topology already encodes**. Knowing a neuron is a sensory cell (which expression tells you well) is a coarse-grained fact that is largely redundant with the neuron's position in the graph; it does not pin down the fine local wiring that the GNCA is scored on. The classifier is therefore an *upper-bound* probe — it establishes the strongest sense in which "expression encodes identity" — and it is precisely against that strong identity signal that the *failure* of expression to add wiring/dynamics information becomes interpretable rather than merely a weak-feature artifact.

Two honesty caveats attach to the 92.7% and must travel with it (§6, Limitations): the role labels are a neuron-name-prefix heuristic assigned inside the notebook (not curated WormAtlas annotations), with an `AS`-prefix collision that places `AS*` neurons in the motor class, and the cell prints no permutation/balanced-accuracy baseline [src: `NCA.ipynb` cell 28]. A sibling clean-pipeline notebook runs a *different* classification task — 3-class neurotransmitter identity (cholinergic / GABAergic / glutamatergic) at 0.727 ± 0.077 with a proper permutation null (null 95th-pct 0.629), N = 84 [src: `New Notebooks/04_synthetic_dynamics_and_classification.ipynb`, Part B]. We report the 92.7% role result as the headline because it is the expression→role probe, but we do **not** conflate it with the NT task, and we flag that it lacks an in-cell null.

### 4.3 Why the honest null *strengthens* the thesis rather than weakening it

The decisive result of this study is a negative one, and we foreground it deliberately. The hypothesis entering the dynamics phase was the strong, blueprint-flavored claim that *gene expression causally shapes how neurons fire together*. The project's own pre-specified shuffle battery falsified it: the hybrid model is no better with real expression than with permuted expression (mse_hybrid = mse_gene_shuf = 0.9879 to four decimals; zeroing genes entirely is *marginally better* at 0.9857), so the causal gate (mse_shuf − mse_hybrid > 0.005) evaluates to 0.0000 → False [src: `PhaseD.ipynb` cells 66, 68]. The cleaner re-implementation reaches the *same* null by an independent preregistered criterion: gene_causal_delta = mse_shuf − mse_hybrid = 0.001283 < 0.01 → "NOT GENE-CAUSAL (null)" [src: `New Notebooks/04_synthetic_dynamics_and_classification.ipynb`, Part A; summary `data_derived/nb04_dynamics_summary.csv`].

This is exactly the evidence that converts the topology-dominant claim from "an over-expressive graph model can fit anything" into a credible scientific conclusion. The danger with a high-capacity model that reaches r = 0.987 is that it could be ascribing the result to whatever feature it is fed. The shuffle control is the antidote: it asks whether the *correspondence* between a neuron and its real expression vector matters, holding the graph and the expression distribution fixed. When that correspondence is destroyed and the model barely notices (static: 0.987 → 0.934; dynamics: no change at all), the parsimonious reading is that the model is reading topology, not molecular identity. A study that reported only the r = 0.987 and the CCA coupling would be *weaker*, not stronger, than one that also reports where correctly-assigned expression fails to help. The triangulation — linear, nonlinear-static, dynamic — plus the matched nulls is the entire epistemic basis of the thesis.

We state the null at exactly the level it was earned (and no further; see §6): the PhaseD dynamics test was run on a **simulated** trajectory (`PhaseA_A9_trajectories_continuous.npy`), not on real calcium imaging, because the only real whole-brain dynamics set of the relevant kind (Kato et al. 2015) lacks the neuron identities needed to align gene vectors [src: `New Notebooks/04_synthetic_dynamics_and_classification.ipynb`, cell 0 notes]. The null therefore bounds *what this clean pipeline can demonstrate on the available data*; it does not refute a gene→dynamics relationship in the animal. That distinction is part of the honesty, not a hedge against it.

### 4.4 The marginal-value framing, read against the ceiling

A caution on interpreting correlations near the ceiling. At r = 0.987 vs 0.934 the *variance-explained* gap (R² ≈ 0.974 vs 0.872) is larger than the raw-r difference looks; but it remains modest next to the graph-only baseline (r ≈ 0.86), and the ranking topology ≫ expression-correspondence is what carries the argument, not the exact marginal magnitude. The headline r = 0.987 is, moreover, an **in-sample full-fit, not a held-out estimate** — the GNCA synaptic-strength cell trains and evaluates Pearson r on the full node set with no train/test split (KFold/`train_test_split` appear only in the later motif/classifier cells), and the project's own scorecard labels the model "GNCA (full)" [src: `NCA.ipynb` cells 3–8 contain no split; `CLAUDE_CONTEXT.md` line 38]. We therefore report it as an in-sample fit (correcting the project page, which calls it "held-out"). Crucially, the comparison that carries the thesis is unaffected: the shuffle control (0.934), graph-only (≈ 0.86) and expression-only (≈ 0.27) baselines are all measured the *same* in-sample way, so the *relative* gap structure is valid even though the absolute r is not held-out. A genuinely cross-validated number does exist for a related motif-regression task — 5-fold CV average r = 0.639 (folds 0.62 / 0.25 / 0.82 / 0.86 / 0.64) [src: `NCA.ipynb` cell 24] — and a variance partition for motif prediction shows expression adds little *unique* variance over degree: degree-only R² = 0.838, expression-only R² = 0.388, both = 0.882, expression-unique = 0.044, degree-unique = 0.494 [src: `NCA.ipynb` cell 31]. The honest summary is therefore stronger, not weaker, than the single headline scalar: under proper cross-validation and variance partitioning, the topology-dominant pattern holds, and expression's *unique* marginal contribution is small (≈ 0.04 R²).

### 4.5 Genetic redundancy: the collapse curve

The in-silico knockout sweep (Fig. 1) characterizes *how* expression information is distributed across genes. Ranking genes by their contribution to the predictor and removing them cumulatively, model correlation drops from the trained baseline r ≈ 0.818 over the first ~10 knockouts and then **plateaus around r ≈ 0.665–0.687, holding essentially flat from K ≈ 10 through K = 200**, never approaching the chance line; the average probability the model assigns to hub neurons declines only gently (0.952 → 0.886) over the same range [src: `phaseE.ipynb` cell 6, executed K-sweep]. This graceful, plateauing degradation is the signature of a **redundant** genetic code, not a few load-bearing master genes: no small driver set is individually necessary for the topology-aligned prediction. Redundancy of this kind is precisely what one expects if topology — not a compact gene set — is doing the structural work, with expression spread thinly across many partially-substitutable contributors.

Two scope notes must travel with this figure. First, the knockout *task* here is **hub-status prediction** (Hub_Div = 1 if out-degree > 5, a DirectMLP 2000→32→1 with L1 regularization on the top-2000-variance genes, N = 222 common neurons), **not** the GNCA synaptic-strength regression [src: `phaseE.ipynb` cell 6]. The collapse curve therefore evidences redundancy in the **expression→hub-topology map**, and we describe it as such, not as redundancy in the GNCA edge model. Second, its baseline r ≈ 0.82 is the operating point of *that* driver-attribution model and must not be conflated with the GNCA's r = 0.987 — they are different setups; the curve is informative about the *shape* of degradation (redundancy structure), not about absolute predictive ceiling.

### 4.6 Relationship to prior work

Our finding is compatible with — and sharpens — the existing literature. Arnatkevičiūtė et al. [2018] showed that connected hub neurons are genomically distinct beyond lineage, birth time, neurotransmitter system, and spatial position; Fenyves et al. [2020] showed that pre-/post-synaptic neurotransmitter and receptor expression predicts synaptic *sign* for two-thirds of ionotropic chemical synapses. We do not contradict either: CCA (ρ up to 0.875) reproduces the coupling. What we add is the **conditional accounting** — the marginal value of expression *given topology* — and the matched shuffle/ablation controls that isolate it. The Barabási genetic-connectome program [Barabási & Barabási 2020; Kovács et al. 2020] argues that genetic identity predicts connectome motifs and can infer gap-junction (innexin) rules; our results suggest that for *local synaptic-strength* prediction the topology already encodes most of that motif structure, so a topology-aware predictor recovers the signal without explicit gene labels. The two views reconcile if the genetic program is largely upstream (§4.1): genes set the topology during development, and at the adult stage topology is a near-sufficient statistic for local wiring.

---

## 5. Conclusion

On the one animal where both a cell-resolved connectome and a whole-nervous-system single-cell expression atlas exist for the same neurons, the marginal predictive value of single-cell gene expression over the connectome's own local topology is small. Three method families of increasing nonlinearity — linear (CCA), nonlinear-static (GNCA with a matched four-way shuffle/ablation battery), and a classifier probe — converge with two credibility-supplying controls — a pre-specified dynamics null and a genetic-redundancy collapse curve — on a single reading: local wiring in *C. elegans* is largely self-determining at the level of topology, and gene expression acts as a modulatory refinement layer rather than the primary architect. Expression is genuinely coupled to circuit structure and is richly informative about neuron identity; that information is simply largely redundant with topology when the task is the local wiring itself. The honest null and the redundancy plateau are not caveats to this thesis but its strongest support.

---

## 6. Limitations

We separate **scope/design limits** (intrinsic to what the study can claim) from **provenance/rigor gaps** (fixable before submission, flagged honestly).

### 6.1 Scope and design limits

1. **Local and static by construction.** The GNCA is intentionally a *local* propagator over a neighborhood; this paper speaks to *local* wiring, not to global developmental specification or long-range routing. A genetic blueprint may dominate at scales this design cannot see.
2. **Adult snapshot.** Both modalities are essentially adult. Developmental connectome series [Witvliet et al. 2021] could reveal a stronger gene-causal phase that is no longer visible once wiring is complete — consistent with the "genes act upstream" reconciliation, which this study cannot itself confirm.
3. **Correlational, not causal.** Every result here is observational. The shuffle controls establish that expression *correspondence* adds little predictive value over topology; they do not establish a causal direction. The in-silico "knockout" is an ablation of model inputs, not a biological perturbation.
4. **Dataset-specific null on simulated dynamics (§4.3).** The `gene_causal_by_shuffle = False` result was obtained on a *simulated* trajectory, because the real dynamics set lacks neuron identities. "Absence of detectable improvement" is not "proof of no effect": the dynamics data are sparse, the model family finite, and a more powerful model or richer (identity-resolved) recordings could surface a gene→dynamics signal. The null bounds what this pipeline can show, not biology.
5. **Hermaphrodite somatic neurons only.** The analysis is restricted to the hermaphrodite; the male-specific nervous system and the pharyngeal nervous system are out of scope. Generalization across sex and across the full 302/391-neuron complement is untested.
6. **Expression representation dependence.** Results may depend on the embedding (top-32 PCA of StandardScaler-normalized CeNGEN expression for the GNCA; PCA(50) for CCA), on quantitative vs. binary presence/absence encoding, and on gene-set choice. Robustness across representations is future work.

### 6.2 Provenance and rigor gaps (to close before submission)

7. **Cohort-N must be stated per analysis — and N = 91 is not yet reproduced in any artifact.** The project page frames the study as "N = 91 somatic neurons with both modalities," but no artifact in the analysis repo reproduces 91. The artifacts actually run on different node sets: the GNCA edge regression on **N = 838** CeNGEN-aligned nodes (16,673 edges, 32 PCA features) [src: `NCA.ipynb` cells 0–1]; the role classifier and CCA at the **179**-neuron alignment [src: `NCA.ipynb` cell 23]; the collapse curve at **N = 222** common neurons [src: `phaseE.ipynb` cell 6]; the NT classifier at **N = 84** [src: `New Notebooks/04…ipynb`]. **The headline GNCA r = 0.987 must not be attached to N = 91.** Before submission, either (a) re-derive the matched subset from the canonical loader and re-run the GNCA on that clean cohort (not permitted under the current zero-compute constraint), or (b) report each result with its own N and provenance, explicitly distinguishing the GNCA's 838-node graph from the master-notebook degree/CCA cohorts. **[PROVENANCE GAP]** No upstream intersection manifest deriving 91 was found in this worktree.
8. **The GNCA prediction CSV is name-dirty.** The committed per-node predictions (`gnca_pred_outgoing_strength.csv`, 838 data rows, columns neuron / y_true_z / y_pred_z) recover r ≈ 0.984 without re-running the model, honoring zero-compute — **but** the file contains duplicate neuron-name rows and trailing/leading whitespace in names (e.g. `"RMGR "`, `" ASGL"`), an artifact of the 838-row CeNGEN join. So "838" is not 838 unique somatic neurons. This is a reviewer-facing cleanliness issue to disclose in Methods [src: `analysis/gnca/gnca_pred_outgoing_strength.csv`].
9. **"Held-out" is unsupported for the GNCA synaptic-strength result (§4.4).** Report r = 0.987 as in-sample; the project MDX page must be corrected. The relative comparison (shuffle/graph-only/expression-only) is unaffected because all four are measured the same way.
10. **No confidence intervals on any headline scalar.** There are no bootstrap CIs on the GNCA r-values, the 92.7% classifier, or the three canonical correlations; the project's own notes flag this as a known pre-submission gap ("No bootstrap CIs on N=91 neurons — reviewers will ask"). Under zero-compute they cannot be generated here; they must be added from the source notebooks before submission, on whichever cohort is finally adopted. **[PROVENANCE GAP]**
11. **CCA and the role classifier lack in-cell nulls.** The canonical correlations are in-sample, computed in a high-PC space (50 expression PCs × 5 motif PCs) over small N, with no held-out split or permutation null on ρ — conditions under which canonical correlations are upward-biased. They should be framed as descriptive linear-alignment magnitudes (or given a shuffle null) before submission [src: `NCA.ipynb` cell 35]. The role classifier likewise prints no permutation/balanced-accuracy baseline (§4.2).
12. **Graph-only value: peak vs final epoch.** The headline graph-only r = 0.861 is the per-epoch peak (epoch 80); the notebook's final-epoch summary line prints r = 0.857 (same run) [src: `NCA.ipynb` cell 8]. Report as "≈ 0.86 (final-epoch 0.857 / peak 0.861)" to be exact. The qualitative claim (graph-only < full, > expression-only) holds either way. The shuffle r = 0.934 is the final logged epoch (epoch 80 of an 80-epoch run), not a cherry-picked mid-run peak [src: `NCA.ipynb` cell 5].
13. **Collapse-curve figure top-driver label discrepancy.** The committed PNG titles the top driver `WBGene00001137`, but the phaseE cell-6 run identifies a different top driver (`WBGene00019961` in the cell-4 single-gene version); the PNG was produced by a sibling/re-run with the same plateau pattern but a different reported gene ID. The R-trajectory and hub-probability numbers are consistent across both; only the gene-ID label differs [src: `celegans-collapse-curve.png` vs `phaseE.ipynb` cells 4, 6]. **[PROVENANCE GAP]** No WBGene→symbol crosswalk artifact was found to resolve either ID to a named gene; the final caption should name it via WormBase.

---

## 7. Figure list (with provenance)

> Only **Figure 1** is currently committed as a rendered image in this worktree (`public/images/projects/celegans-collapse-curve.png`). The remaining figures are *to be generated from committed artifacts* under zero-compute (the underlying CSVs and notebook outputs already exist; they need only plotting, which changes no numbers). Each entry gives the source artifact, the proposed caption, and current status.

### Figure 1 — The genetic "collapse curve" (committed)
- **Image:** `/home/rohit/Desktop/website-paper/public/images/projects/celegans-collapse-curve.png` (original: `/mnt/ssd4tb/Desktop/C-Elegans/figures/crispr_results/genetic_collapse_curve.png`).
- **Source data:** `/home/rohit/Desktop/C-Elegans/notebooks/archive/phases/phaseE.ipynb` cell 6 (cumulative-knockout sweep); driver-influence CSV `/home/rohit/Desktop/C-Elegans/analysis/gnca/gnca_gene_perturb_fast.csv`.
- **Caption (proposed):** *Genetic redundancy in the expression→hub-topology map. Red: model correlation r between predicted and observed hub status (out-degree > 5) as the top-K driver genes are removed cumulatively (K = 0…200). Navy: mean probability the model assigns to true hub neurons. Performance drops from the trained baseline (r ≈ 0.82) over the first ~10 knockouts, then plateaus at r ≈ 0.68 and remains essentially flat through 200 removed genes, never approaching the chance line (dotted) — the signature of a redundant genetic code rather than a few load-bearing master genes. (Model: DirectMLP 2000→32→1 with L1 regularization on top-2000-variance CeNGEN genes; N = 222 common neurons. The top-driver gene ID printed in the figure title requires reconciliation with the source run and a WormBase symbol lookup before submission.)*
- **Status:** COMMITTED. Caption flagged for the gene-ID reconciliation (Limitation 13).

### Figure 2 — The four-way GNCA ablation (to generate)
- **Source data:** `/home/rohit/Desktop/C-Elegans/notebooks/archive/nca/NCA.ipynb` cells 4 (full r = 0.987), 5 (shuffle r = 0.934), 8 (graph-only ≈ 0.86, expression-only ≈ 0.27); tabulated in `/home/rohit/Desktop/C-Elegans/CLAUDE_CONTEXT.md` lines 38–41.
- **Caption (proposed):** *The marginal value of expression for local synaptic-strength prediction is small. In-sample Pearson r for four GNCA conditions: Full (topology + real expression) = 0.987; Shuffled-expression (graph and expression distribution preserved, neuron↔expression correspondence destroyed) = 0.934; Graph-only = 0.86; Expression-only = 0.27. Topology alone (graph-only) recovers most of the explainable signal; the Full−Shuffled gap (≈ 0.05) is the marginal value of correctly-assigned expression. All four bars are measured the same in-sample way, so the relative gap structure is valid; absolute r is a full-fit, not held-out (see Methods).*
- **Status:** TO GENERATE — a 4-bar chart from values already in committed artifacts; no model re-run required.

### Figure 3 — CCA expression↔motif alignment (to generate)
- **Source data:** `NCA.ipynb` cell 35 (canonical correlations ρ = [0.875, 0.836, 0.809]); motif matrix built in cells 21–22.
- **Caption (proposed):** *Linear shared structure between transcriptional identity and local connectivity-motif participation. Canonical correlations from CCA (3 components) between PCA(50) of single-cell expression and PCA(5) of a five-dimensional motif embedding (reciprocal / feed-forward / feedback-triangle / shared-output / lateral-inhibition counts per neuron): ρ = [0.875, 0.836, 0.809]. At least three strongly-correlated canonical dimensions are shared — expression and connectivity are coupled — but this coupling does not translate into marginal predictive power over topology (Figs. 2, 4). In-sample; ρ are upward-biased without a held-out or permutation null (Limitation 11).*
- **Status:** TO GENERATE — a 3-bar (or scatter-of-canonical-variates) plot from committed values.

### Figure 4 — The dynamics null (to generate)
- **Source data:** `PhaseD.ipynb` cells 66, 68 (mse_hybrid 0.9879 / mse_no_gene 0.9857 / mse_gene_shuf 0.9879; verdict `gene_causal_by_shuffle = False`); corroborating `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` (gene_causal_delta = 0.001283 < 0.01 → null; summary `data_derived/nb04_dynamics_summary.csv`).
- **Caption (proposed):** *Gene conditioning does not improve dynamics prediction above shuffle. Test-set MSE for a hybrid gene-conditioned graph-LSTM predicting neural time-series: real-gene (hybrid) and shuffled-gene conditions are equal to four decimals (0.9879), and zeroing genes is marginally better (0.9857); the preregistered causal gate (mse_shuf − mse_hybrid > 0.005) evaluates to 0.0000 → False. An independent clean re-implementation reaches the same null (Δ = 0.001283 < 0.01). The dynamics were simulated (real identity-resolved whole-brain recordings are unavailable); the null bounds what the pipeline can show, not biology (§4.3).*
- **Status:** TO GENERATE — a paired-bar MSE chart with the gate annotation, from committed values.

### Figure 5 (optional) — Variance partition for motif prediction (to generate)
- **Source data:** `NCA.ipynb` cell 31 (degree-only R² 0.838 / expression-only 0.388 / both 0.882 / expression-unique 0.044 / degree-unique 0.494); cross-validated motif r = 0.639 from cell 24.
- **Caption (proposed):** *Expression adds little unique variance over topology. Commonality analysis for per-neuron motif prediction: degree alone explains R² = 0.838, expression alone 0.388, both jointly 0.882; the variance uniquely attributable to expression is only 0.044, versus 0.494 unique to degree. Consistent with the topology-dominant, gene-modulatory reading under a properly cross-validated estimator (motif-regression 5-fold CV r = 0.639).*
- **Status:** TO GENERATE (optional, strengthens §4.4) — from committed values.

---

## 8. References

All entries verified via web search on 2026-06-20 (title + DOI/PMID confirmed against publisher / PubMed / PMC). "Verified" = the work exists with the stated venue and identifier. Substrate / simulator citations are intentionally excluded. No phantom references appear below; the single un-cited candidate considered (the project docs' "Andrew Gao 2023" classifier prior) could not be confirmed and is therefore **not** cited (see note after the list).

1. **Arnatkevičiūtė A, Fulcher BD, Pocock R, Fornito A.** (2018) Hub connectivity, neuronal diversity, and gene expression in the *Caenorhabditis elegans* connectome. *PLoS Computational Biology* 14(2):e1005989. doi:10.1371/journal.pcbi.1005989; PMID 29432412. *[verified]*
2. **Barabási DL, Barabási A-L.** (2020) A genetic model of the connectome. *Neuron* 105(3):435–445.e5. doi:10.1016/j.neuron.2019.10.031; PMID 31806491. *[verified]*
3. **Cook SJ, Jarrell TA, Brittin CA, et al.** (2019) Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature* 571(7763):63–71. doi:10.1038/s41586-019-1352-7; PMID 31270481. *[verified]*
4. **Fenyves BG, Szilágyi GS, Vassy Z, Sőti C, Csermely P.** (2020) Synaptic polarity and sign-balance prediction using gene expression data in the *Caenorhabditis elegans* chemical synapse neuronal connectome network. *PLoS Computational Biology* 16(12):e1007974. doi:10.1371/journal.pcbi.1007974; PMC7785220. *[verified]*
5. **Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE.** (2017) Neural message passing for quantum chemistry. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1263–1272. *[verified]*
6. **Hammarlund M, Hobert O, Miller DM 3rd, Sestan N.** (2018) The CeNGEN project: the complete gene expression map of an entire nervous system. *Neuron* 99(3):430–433. doi:10.1016/j.neuron.2018.07.042; PMID 30092212. *[verified]*
7. **Hobert O.** (2016) A map of terminal regulators of neuronal identity in *Caenorhabditis elegans*. *WIREs Developmental Biology* 5(4):474–498. doi:10.1002/wdev.233; PMID 27136279. *[verified]*
8. **Hotelling H.** (1936) Relations between two sets of variates. *Biometrika* 28(3–4):321–377. doi:10.1093/biomet/28.3-4.321. *[verified]*
9. **Kipf TN, Welling M.** (2017) Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*. OpenReview SJU4ayYgl. *[verified]*
10. **Kovács IA, Barabási DL, Barabási A-L.** (2020) Uncovering the genetic blueprint of the *Caenorhabditis elegans* nervous system. *Proceedings of the National Academy of Sciences (PNAS)* 117(52):33570–33577. doi:10.1073/pnas.2009093117; PMID 33318182. *[verified]*
11. **Loer CM, Rand JB.** (2022) The evidence for classical neurotransmitters in *C. elegans* (WormAtlas Neurotransmitter Table; updated 27 Sep 2022). WormAtlas. *Resource (curated data table, not a peer-reviewed article); confirmed to exist. [verified — as resource]*
12. **Mordvintsev A, Randazzo E, Niklasson E, Levin M.** (2020) Growing neural cellular automata. *Distill* 5(2):e23. doi:10.23915/distill.00023. *[verified]*
13. **Pereira L, Kratsios P, Serrano-Saiz E, et al.** (2015) A cellular and regulatory map of the cholinergic nervous system of *C. elegans*. *eLife* 4:e12432. doi:10.7554/eLife.12432; PMID 26705699. *[verified]*
14. **Taylor SR, Santpere G, Weinreb A, et al.** (2021) Molecular topography of an entire nervous system. *Cell* 184(16):4329–4347. doi:10.1016/j.cell.2021.06.023. *[verified]*
15. **Varshney LR, Chen BL, Paniagua E, Hall DH, Chklovskii DB.** (2011) Structural properties of the *Caenorhabditis elegans* neuronal network. *PLoS Computational Biology* 7(2):e1001066. doi:10.1371/journal.pcbi.1001066; PMID 21304930. *[verified]*
16. **White JG, Southgate E, Thomson JN, Brenner S.** (1986) The structure of the nervous system of the nematode *Caenorhabditis elegans*. *Philosophical Transactions of the Royal Society of London B* 314(1165):1–340. doi:10.1098/rstb.1986.0056. *[verified]*
17. **Witvliet D, Mulcahy B, Mitchell JK, et al.** (2021) Connectomes across development reveal principles of brain maturation. *Nature* 596(7871):257–261. doi:10.1038/s41586-021-03778-8. *[verified]*

**Data sources referenced in Methods (not in the citation list above but named for provenance):** CeNGEN single-cell L4 atlas [Taylor et al. 2021]; GSE136049 (Packer/Zhu 2019) single-cell dataset, used only for the master-pipeline degree/CCA cohort; the whole-brain calcium dataset of **Kato et al. 2015** (referenced in §2.5/§4.3 as the real-dynamics set that lacks neuron identities). The Kato 2015 reference is named in-text as the reason the dynamics null is on simulated rather than real activity; if the §4.3 framing is retained at submission, a full verified citation for Kato et al. 2015 (*Cell* 163:656–669) should be added to this list — it was **not** independently DOI-verified in this zero-compute pass and is therefore marked **[UNVERIFIED — verify before submission]**.

**Considered but NOT cited (integrity note).** The project docs referenced an "Andrew Gao 2023" deep-learning neuron-classification prior. The closest real match found is an arXiv 2023 paper on mammalian/Allen-Cell-Types neuron classification, whose first-author identity and relevance could not be confirmed; it is therefore **not** cited rather than invented. A peer-reviewed neurotransmitter atlas (Wang et al. 2024, *eLife*, PMC11185579) exists as an alternative to the Loer & Rand WormAtlas table but was not individually detail-verified and is not used. Substrate/simulator references (Atanas et al. 2023; Brian2/NEURON/channel-biophysics/anesthesia) are deliberately excluded as out of scope.

---

## Appendix A — Provenance Table (every quantitative claim → artifact + status)

All artifact paths are under `/home/rohit/Desktop/C-Elegans/` unless an absolute path is given. Verification reflects the zero-compute harvest of 2026-06-20.

### A.0 Outstanding gaps (read these first)

| # | Gap | Where it bites | Required pre-submission action |
|---|---|---|---|
| G1 | **N = 91 not reproduced in any artifact.** The N=91 "somatic-with-both" cohort appears only in the project MDX / scorecard; no intersection-manifest artifact re-derives it. | §1.6(2), §2.1.3, §3 cohort note, §6.2(7) | Re-derive matched subset from canonical loader with explicit inclusion criteria; or report each result with its own N. |
| G2 | **No confidence intervals** on any headline scalar (GNCA r-values, 92.7%, CCA ρ). | §1.6(5), §3.1, §6.2(10) | Add bootstrap/permutation CIs from source notebooks on the finally-adopted cohort. |
| G3 | **Collapse-curve top-driver gene-ID** discrepancy (PNG `WBGene00001137` vs phaseE run `WBGene00019961`); neither resolved to a gene symbol. | §2.6, §3.5(1), §6.2(13), Fig. 1 | Reconcile which run produced the PNG; resolve ID via WormBase; fix caption. |
| G4 | **GNCA cohort CSV is name-dirty** (duplicate rows, leading/trailing whitespace), so "838" ≠ 838 unique somatic neurons. | §2.1.3, §3 cohort note, §6.2(8) | Clean the join; re-derive unique-neuron count (non-zero-compute). |
| G5 | **Kato et al. 2015** in-text citation not independently DOI-verified in this pass. | §2.5, §4.3, §8 | Verify *Cell* 163:656–669 before submission if the framing is retained. |

### A.1 The five headline results — all verified-in-artifact

| Claim (as stated in paper) | Value | Status | Artifact (path under `/home/rohit/Desktop/C-Elegans/`) |
|---|---|---|---|
| **(1a)** GNCA predicts outgoing synaptic strength | r = 0.987, MAE = 0.097 (**in-sample full-fit, NOT held-out**) | **verified-in-artifact** (number); **[corrected]** ("held-out" was wrong) | `notebooks/archive/nca/NCA.ipynb` cell 4 (`Final Pearson r=0.987 \| MAE=0.097`); `analysis/gnca/gnca_pred_outgoing_strength.csv` (838 rows; CSV recovers r ≈ 0.984); `CLAUDE_CONTEXT.md` l.38 ("GNCA (full)") |
| **(1b)** Shuffled-expression control | r = 0.934 (epoch 80, final of 80-ep run) | **verified-in-artifact** | `NCA.ipynb` cell 5 (`[SHUF] Epoch 080 \| loss=0.1297 \| r=0.934`); `CLAUDE_CONTEXT.md` l.39 |
| **(1c)** Graph-only baseline | r ≈ 0.86 (peak 0.861 / final-epoch 0.857) | **verified-in-artifact**; **[corrected]** (peak vs final disclosed) | `NCA.ipynb` cell 8 (`[GRAPH-ONLY] Epoch 080 r=0.861`; summary `r = 0.857`); `CLAUDE_CONTEXT.md` l.40 |
| **(1d)** Expression-only baseline | r ≈ 0.27 (final 0.269 / peak 0.270) | **verified-in-artifact** | `NCA.ipynb` cell 8 (`[EXPR-ONLY] Epoch 080 r=0.270`; summary `r = 0.269`); corroborated `Paper1MasterNotebook.ipynb` c.37–38 (degree R² = −15.3); `CLAUDE_CONTEXT.md` l.41 |
| **(2)** Neuron-role classifier (sensory/inter/motor) from expression alone | 92.7% (0.9274603174603175 mean CV acc) | **verified-in-artifact** (exact match) | `NCA.ipynb` cell 28 (`Mean Accuracy (role): 0.9274603174603175`); `CLAUDE_CONTEXT.md` l.45 |
| **(3)** CCA expression↔motif canonical correlations | ρ = [0.875, 0.836, 0.809] | **verified-in-artifact** (exact match) | `NCA.ipynb` cell 35 (`Canonical correlations (expr <-> motif): [0.875, 0.836, 0.809]`); `CLAUDE_CONTEXT.md` l.44 |
| **(4)** Dynamics honest null | `gene_causal_by_shuffle = False`; hybrid 0.9879 / no_gene 0.9857 / gene_shuf 0.9879; gate (0.0000 ≯ 0.005) | **verified-in-artifact** (exact match) | `notebooks/archive/phases/PhaseD.ipynb` cell 66 + cell 68 (`VERDICT: {…gene_causal_by_shuffle: False}`); corroborated `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` Part A (Δ = 0.001283 < 0.01); `CLAUDE_CONTEXT.md` l.47 |
| **(5)** Genetic collapse curve (plateau above chance through 200 KOs) | trained R = 0.818 → plateau R ≈ 0.665–0.687 (K=10…200); hub-prob 0.952 → 0.886 | **verified-in-artifact** (numbers consistent with committed PNG) | FIGURE: `/home/rohit/Desktop/website-paper/public/images/projects/celegans-collapse-curve.png`; CODE+OUTPUT: `notebooks/archive/phases/phaseE.ipynb` cell 6; original fig `/mnt/ssd4tb/Desktop/C-Elegans/figures/crispr_results/genetic_collapse_curve.png`; driver CSV `analysis/gnca/gnca_gene_perturb_fast.csv` |

### A.2 Supporting / corroborating claims

| Claim | Value | Status | Artifact |
|---|---|---|---|
| GNCA architecture (64-dim state, 8 steps, Bernoulli(0.5) fire, ~0.031M params) | as in §2.3.2 | **verified-in-artifact** | `NCA.ipynb` cells 1–3 (param count `0.031169M`) |
| Cross-validated motif regression | 5-fold mean r = 0.639 (folds 0.62/0.25/0.82/0.86/0.64) | **verified-in-artifact** | `NCA.ipynb` cell 24 |
| Variance partition (motif prediction) | expr-only R² 0.388; degree-only 0.838; both 0.882; expr-unique 0.044; degree-unique 0.494 | **verified-in-artifact** | `NCA.ipynb` cell 31 |
| Expression-only → degree (corroborating expr weakness) | negative R² (e.g. total-degree R² = −15.3) | **verified-in-artifact** | `Paper1MasterNotebook.ipynb` cells 37–38 |
| NT classifier (chol/GABA/glu) — *not the headline role task* | 0.727 ± 0.077, N = 84, perm-null 95th-pct 0.629 | **verified-in-artifact** | `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` Part B |
| Connectome corpus | 27,727 edges / 1,533 nodes; GNCA working graph 838 nodes / 16,673 edges | **verified-in-artifact** | `NCA.ipynb` cells 0/9; cell 1 (`N=838, edges kept=16673`) |
| Expression sources | CeNGEN (`gene_expr_CeNGEN_base_mapped.csv`, 838×22,469) for NCA track; GSE136049 (91×22,468) for master track | **verified-in-artifact** | `NCA.ipynb` cell 0; `Paper1MasterNotebook.ipynb` cells 7/14/27/30–32 |

### A.3 Items flagged as gaps (cross-reference to A.0)

| Item | Gap | Tag |
|---|---|---|
| N = 91 cohort | no intersection manifest reproduces 91 | **[PROVENANCE GAP]** (G1) |
| CIs on headline scalars | none exist | **[PROVENANCE GAP]** (G2) |
| Collapse-curve top driver | PNG label ≠ phaseE run; no symbol crosswalk | **[PROVENANCE GAP]** (G3) |
| GNCA prediction CSV | duplicate/whitespace-dirty names | **[PROVENANCE GAP]** (G4) |
| Kato et al. 2015 citation | not DOI-verified this pass | **[UNVERIFIED — verify before submission]** (G5) |

**No headline quantitative result is MISSING:** all five (GNCA + controls, role classifier, CCA, dynamics null, collapse curve) are verified against a named committed artifact. The gaps above are provenance/rigor issues on *supporting details* (cohort manifest, CIs, a gene-ID label, name hygiene, one citation), not fabricated or unbacked numbers.

---

## Appendix B — Note on the methods-paper split

The GNCA of §2.3 is, independently of the worm biology, a method for **connectome-constrained synaptic-property prediction**: a message-passing model run in the cellular-automaton regime [Mordvintsev et al. 2020] over a directed multigraph with optional node features, evaluated under a structured **topology-vs-feature ablation protocol**. Concretely the architecture is a 64-dim per-node hidden state initialized via tanh(Linear(32→64)) over top-32 PCA expression features, with 8 message-passing/update steps (message = out-degree-normalized adjacency × Linear(64→64); residual MLP update Linear(128→128)→ReLU→Linear(128→64) over concat[state, message]; stochastic Bernoulli(0.5) per-node "fire" mask during training; LayerNorm+Linear(64→1) readout), ≈ 0.031M parameters, trained AdamW (lr 3e-3, wd 1e-4) + cosine schedule, MSE on z-scored mean outgoing synaptic strength, seed 42 [src: `NCA.ipynb` cells 1–3].

That methodological contribution — the architecture, the **shuffle-control evaluation design** (full / shuffled-feature / graph-only / feature-only), and the regression setup (moved to held-out edges after the cohort/CV fixes of §6.2) — is naturally separable from the *C. elegans* biology and is a clean fit for a graph-representation-learning venue (**NeurIPS GRL workshop** or **ICLR LMRL** — Learning Meaningful Representations of Life). We anticipate a short companion methods paper presenting the GNCA and its feature-vs-structure ablation protocol as a reusable tool for biological-graph property prediction, benchmarked against standard GNN baselines [Gilmer et al. 2017; Kipf & Welling 2017]. The two papers share the §3.1/Fig. 2 ablation but differ in framing: **here** it is evidence about worm biology (does molecular identity add information over topology?); **there** it is a benchmark of an architecture against feature-vs-structure baselines. Two preconditions gate the methods split, both already identified: (i) the cohort definition must be cleaned and fixed (§6.2 item 7) so the benchmark runs on a well-specified node set, and (ii) the evaluation must be moved to genuinely held-out edges with CIs (§6.2 items 9–10), since a methods venue will scrutinize the train/test protocol more sharply than an application venue.

---

*End of manuscript. Source fragments: `paper/01_abstract_intro.md`, `paper/02_methods.md`, `paper/03_results.md`, `paper/04_discussion_figures.md`. Reference verification log: `paper/references.md`. Substrate / Tier-4 / Tier-5 / biophysical-simulator material is deliberately and entirely excluded.*
