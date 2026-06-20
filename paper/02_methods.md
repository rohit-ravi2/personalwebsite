# 2. Methods

*Manuscript draft — Paper 1. Compiled 2026-06-20. This file is the standalone
Methods section; it supersedes the embedded §3 of
`multimodal_topology_vs_expression.md` where the two conflict, because it is
written directly against the executed source notebooks rather than the project's
narrative summaries.*

> **Provenance and honesty note (for internal review).** Every quantitative and
> architectural statement below is read from a *committed, executed* artifact —
> primarily the notebook `NCA.ipynb` (with cell outputs), the dynamics notebook
> `PhaseD.ipynb`, the knockout notebook `phaseE.ipynb`, the master pipeline
> `Paper1MasterNotebook.ipynb`, the saved prediction CSVs under
> `analysis/gnca/`, and the figure `public/images/projects/celegans-collapse-curve.png`.
> The canonical analysis repository is `/home/rohit/Desktop/C-Elegans/`; the
> manuscript worktree (`website-paper`) contains only the narrative MDX and the
> collapse-curve figure, **not** the analysis artifacts, so each method below
> cites the artifact in the analysis repo by path. Nothing was re-run for this
> draft (zero-compute). Numbers verified from artifacts are tagged
> **[verified-in-artifact]**; numbers that are correct but whose framing in the
> project's narrative summaries was wrong are tagged **[corrected]**; items I
> could not back from an artifact are tagged **[PROVENANCE GAP]** rather than
> invented.

The design principle throughout is **triangulation against controls**: no model's
performance is reported without the matched control that removes the variable of
interest (gene expression), and the question is posed across three method families
of increasing nonlinearity — a linear method (canonical-correlation analysis), a
nonlinear static predictor (a graph neural cellular automaton), and a classifier
probe — so that the conclusion does not rest on the inductive bias of any single
architecture. A fourth family (gene-conditioned sequence models) supplies the
dynamics test, and a fifth (cumulative in-silico knockout) characterizes how the
expression contribution is distributed across genes.

---

## 2.1 Data

### 2.1.1 Connectome

Connectivity is loaded from the project's consolidated edge table
`data/processed/csv/merged_edges_with_relatedness.csv` (fallback
`master_connectome_dataset.csv`), with columns `Source`, `Target`, `Type`,
`SynapseCount`, `SourceFile`, `Relatedness`. The full table is 27,727 edges over
1,533 nodes. The corpus is assembled from the classical serial-section EM
reconstruction of White et al. [1986], the network-quantified edge list of
Varshney et al. [2011], the quantitative weighted hermaphrodite connectome of
Cook et al. [2019], and the developmental connectome series of Witvliet et al.
[2021]. [src: `NCA.ipynb` cells 0/9; `CLAUDE_CONTEXT.md` connectome-source block]

For the GNCA, the directed source→target chemical+gap multigraph is filtered to
edges with `SynapseCount ≥ 0` and the synapse count is used as the edge weight.
The adjacency is **out-degree row-normalized** — $A_{\text{norm}} = D_{\text{out}}^{-1}A$ —
so that message passing aggregates a convex combination of a neuron's outgoing
neighbors. On the GNCA's expression-aligned node set this yields a working graph
of **N = 838 nodes / 16,673 edges**. [src: `NCA.ipynb` cell 1, executed output
`N=838, edges kept=16673`]

Neurotransmitter identities used in deriving role/edge features follow the curated
cholinergic map of Pereira et al. [2015] and the WormAtlas neurotransmitter
compilation of Loer & Rand [2022] (a curated data table, not a peer-reviewed
article; see `references.md`).

### 2.1.2 Single-cell expression

The study uses **two distinct expression provenances depending on the analysis**,
and the manuscript states each explicitly rather than collapsing them:

- **GNCA + role classifier + CCA (the `NCA.ipynb` track):** per-neuron gene
  expression from the **CeNGEN** single-cell L4 atlas [Taylor et al. 2021;
  Hammarlund et al. 2018], mapped to WormBase gene IDs in
  `data/processed/csv/gene_expr_CeNGEN_base_mapped.csv` — a matrix of
  **838 nodes × 22,469 genes**. [src: `NCA.ipynb` cell 0, `expr_df: (838, 22469)`]
- **Master-pipeline degree/CCA cohort (the `Paper1MasterNotebook.ipynb` track):**
  the **GSE136049** (Packer/Zhu 2019) gene-by-barcode matrix
  (22,468 genes × 100,954 cells) averaged to per-neuron profiles and intersected
  with the connectome. [src: `Paper1MasterNotebook.ipynb` cells 7, 14, 27, 30–32,
  output `Neuron expression matrix: (91, 22468)`; saved
  `expression_neuron_mean.csv` shape (91, 22468)]

### 2.1.3 The matched cohorts (cohort-N disclosure)

**[corrected]** The headline figure of **N = 91 somatic neurons with both
connectome and single-cell expression** is the matched cohort of the *master
pipeline* (`Paper1MasterNotebook.ipynb` cell 31, `(91, 22468)`), used for the
per-neuron degree-prediction and the master-notebook CCA analyses. It is **not**
the cohort on which the headline GNCA r was computed. The GNCA synaptic-strength
model runs on the **N = 838** CeNGEN-aligned node set described in §2.1.1–2.1.2;
the role classifier and the `NCA.ipynb` CCA run on a 179-neuron alignment
(`NCA.ipynb` cell 23, `179 neurons × 22469 genes; 27727 edges`); the collapse
curve runs on 222 common neurons (§2.5). These distinct Ns are reported with
their own provenance throughout, and no single N is attached uniformly to all
results.

**[PROVENANCE GAP]** The saved GNCA prediction CSV
(`analysis/gnca/gnca_pred_outgoing_strength.csv`, 838 data rows) contains
duplicate neuron-name rows and trailing-whitespace names (e.g. `RMGR `, ` ASGL`),
an artifact of the 838-row CeNGEN join; consequently 838 rows ≠ 838 unique somatic
neurons. This is a reviewer-facing cleanliness issue: the clean unique-neuron
count for the GNCA cohort should be re-derived (under a future, non-zero-compute
pass) before submission. The N = 91 master-cohort intersection manifest is
recorded in the cohort notebook but not as a standalone artifact in the manuscript
worktree.

---

## 2.2 Canonical-correlation analysis (linear)

To test for *linear* shared structure between transcriptional identity and local
connectivity, we apply canonical-correlation analysis (CCA) [Hotelling 1936]
between a per-neuron **expression embedding** and a per-neuron
**connectivity-motif embedding**.

- **Motif embedding.** A 5-dimensional per-neuron motif matrix is built from the
  connectome (`NCA.ipynb` cells 21–22, `motif_counts_safe`): counts of
  reciprocal, feedforward, feedback-triangle, shared-output, and
  lateral-inhibition participation per neuron.
- **Embeddings.** Expression is reduced by `PCA(n_components=50, random_state=0)`
  on the standardized expression matrix `X_expr`; the motif matrix is reduced by
  `PCA(n_components=5, random_state=0)` to `Y_motif`.
- **CCA.** `sklearn.cross_decomposition.CCA(n_components=3, max_iter=1000)` is
  fit on (PCA-50 expression, PCA-5 motif); the canonical correlation for
  component $i$ is $\rho_i = \mathrm{corrcoef}(U_{:,i}, V_{:,i})$.

[src: `NCA.ipynb` cell 35, executed
`Canonical correlations (expr <-> motif): [0.875, 0.836, 0.809]`]
**[verified-in-artifact]**

**Honesty caveat (to be stated in Results/Limitations).** These canonical
correlations are **in-sample**: there is no train/test split and no permutation
null on the $\rho_i$ in this cell, and they are computed in a high-PC space
(50 expression PCs vs 5 motif PCs) over a small N, a regime in which CCA
correlations are upward-biased. The values are reported as **descriptive
linear-alignment magnitudes**, not as held-out generalization estimates; a
permutation/shuffle null on the canonical correlations is the recommended
pre-submission addition.

---

## 2.3 Graph neural cellular automaton (nonlinear, static)

The central synaptic-strength predictor is a **graph neural cellular automaton
(GNCA)** — a message-passing model [Gilmer et al. 2017; Kipf & Welling 2017] run
in the cellular-automaton regime [Mordvintsev et al. 2020]. The full architecture
and training configuration below are read directly from `NCA.ipynb` cells 1–3
(no re-run).

### 2.3.1 Inputs and target

- **Node features.** $X_{\text{feat}}$ = the top-**32** principal components
  (`USE_PCA=True, N_PCA=32`, `NCA.ipynb` cell 1) of the per-gene
  `StandardScaler`-normalized CeNGEN expression matrix (§2.1.2).
- **Graph.** The out-degree row-normalized directed synapse-count adjacency
  $A_{\text{norm}}$ of §2.1.1.
- **Target.** The **z-scored mean outgoing synaptic strength** per node: outgoing
  edge weights grouped by source and averaged, then standardized
  $(x-\mu)/\sigma$.

### 2.3.2 Model (`NCA.ipynb` cell 2, class `GNCA`)

- Input encoder: $\text{Linear}(32 \to 64)$ with tanh activation
  (tanh-initialized 64-dim hidden state per node).
- Message: $\text{Linear}(64 \to 64)$ without bias, aggregated as
  $A_{\text{norm}}\, m(x)$.
- Update: a residual MLP on the concatenation of self-state and aggregated
  message — $\text{Linear}(128 \to 128)\!\to\!\text{ReLU}\!\to\!\text{Linear}(128 \to 64)$
  applied to $[\,x,\; A_{\text{norm}} m(x)\,]$, with residual update
  $x \leftarrow x + \Delta x$.
- Stochastic asynchronous **fire mask**: a per-node $\text{Bernoulli}(0.5)$ mask
  gates the update during training (asynchronous-CA flavor; disabled at eval).
- Readout: $\text{LayerNorm}(64) \to \text{Linear}(64 \to 1)$.
- Unroll: **8** message-passing/update steps (`STEPS=8`).
- Parameter count (printed): **0.031169 M** (~31k parameters). [src: `NCA.ipynb`
  cell 2/3] **[verified-in-artifact]**

### 2.3.3 Training (`NCA.ipynb` cell 3)

AdamW ($\text{lr}=3\times10^{-3}$, weight decay $1\times10^{-4}$);
`CosineAnnealingLR` ($T_{\max}=200$, $\eta_{\min}=3\times10^{-4}$); MSE loss on
the z-scored target; gradient clipping at 1.0; `torch.manual_seed(42)`;
**200 epochs**; best-training-loss checkpoint saved to `gnca_best.pt`;
device = CUDA.

### 2.3.4 Reported performance and a critical framing correction

The full GNCA reaches **Pearson r = 0.987, MAE = 0.097**. [src: `NCA.ipynb`
cell 4, executed `Final Pearson r=0.987 | MAE=0.097`; saved predictions
`analysis/gnca/gnca_pred_outgoing_strength.csv` (838 rows; recomputed
Pearson r = 0.984 from the saved snapshot, vs 0.987 reported from the best
checkpoint); `CLAUDE_CONTEXT.md` line 38 labels this `GNCA (full)`]

**[corrected — do NOT call this "held-out".]** `NCA.ipynb` cells 3–8 (the
synaptic-strength model and its three controls) contain **no train/test split**:
Pearson r is computed **in-sample on the full node set**. `train_test_split` /
`KFold` appear only in later cells (20, 24–26, 31, 36–39), which drive the
motif-prediction and role-classifier analyses — not the synaptic-strength GNCA.
The project's own scorecard correctly names this model `GNCA (full)`. The
manuscript therefore reports **r = 0.987 as an in-sample fit**, not as held-out
performance. The thesis does not depend on the absolute r: it rests on the
*relative* comparison among the four conditions below, all measured the same
in-sample way, so the comparison is valid. A more honest cross-validated number
also exists for a related task — a 5-fold CV of per-neuron motif regression in
`NCA.ipynb` cell 24 gives mean r ≈ 0.639 (folds 0.62 / 0.25 / 0.82 / 0.86 /
0.64) — and is reported alongside the in-sample GNCA fit so readers can see both.

### 2.3.5 The four-condition control battery

All four conditions reuse the same `GNCA` class; they differ only in what
information is supplied. r-values are the load-bearing contrast for the
topology-dominant / gene-modulatory thesis.

| Condition | What it removes | r | Artifact (source) |
|---|---|---|---|
| **Full** (topology + real expression) | nothing | **0.987** | `NCA.ipynb` cell 4 |
| **Shuffled-expression** (topology preserved) | the expression↔neuron correspondence | **0.934** | `NCA.ipynb` cell 5 |
| **Graph-only** | all expression features | **0.861** (peak) / 0.857 (final-epoch) | `NCA.ipynb` cell 8 |
| **Expression-only** | the informative graph | **0.27** (final-epoch 0.269) | `NCA.ipynb` cell 8 |

- **Shuffled-expression** (`NCA.ipynb` cell 5, "Cell A — Shuffle control",
  `seed 123`, 80 epochs): `torch.randperm` permutes the expression-feature **rows**
  (breaks neuron↔expression alignment) while $A_{\text{norm}}$ is unchanged. r
  drops only 0.987→0.934 — i.e. topology carries the signal. **Provenance
  precision:** 0.934 is the value at epoch 80, the **last** logged epoch of this
  80-epoch run (the main model trains 200), so it is the final reported value, not
  a cherry-picked mid-run peak. [src: `NCA.ipynb` cell 5, executed
  `[SHUF] Epoch 080 | loss=0.1297 | r=0.934`] **[verified-in-artifact]**

- **Graph-only** and **Expression-only** (`NCA.ipynb` cell 8, "Cell D —
  Compare graph-only vs expression-only", via `train_gnca_baseline`, 100 epochs,
  `state_dim=64`). **[corrected — minor discrepancy to disclose]** The headline
  graph-only **0.861** is the per-epoch *peak* (epoch 80); the notebook's final
  `=== Baseline Comparison ===` summary line (last epoch, 100) prints
  **`Graph-only r = 0.857`**. Both come from the same run. The manuscript reports
  "graph-only r ≈ 0.86 (peak 0.861 / final-epoch 0.857)"; the qualitative claim
  (graph-only < full, ≫ expression-only) is unaffected. Expression-only plateaus
  near **r ≈ 0.27** (`[EXPR-ONLY] Epoch 080 | r=0.270`; final-epoch 0.269;
  summary `Expression-only r = 0.269`). [src: `NCA.ipynb` cell 8]
  **[verified-in-artifact]** The expression-only weakness is independently
  corroborated in `Paper1MasterNotebook.ipynb` cells 37–38, where raw-gene →
  degree regression gives strongly **negative** out-of-sample $R^2$ (e.g.
  `total_degree R² = -15.3`), confirming expression alone has little predictive
  power for connectivity targets.

- **Robustness run.** A separate edge-dropout (50%) + `state_dim=8` GNCA (120
  epochs) was run as a robustness check on the architecture's reliance on the full
  graph. [src: `NCA.ipynb`]

### 2.3.6 Committed prediction artifact

Per-node predictions are saved to
`analysis/gnca/gnca_pred_outgoing_strength.csv` (columns `neuron`, `y_true_z`,
`y_pred_z`; 838 data rows), permitting recomputation of the correlation
**without re-running the model** (recomputed Pearson r = 0.984 from the saved
snapshot; 0.987 is the best-checkpoint value). Companion artifacts in the same
directory — `gnca_motif_performance.csv`, `gnca_saliency_features.csv`,
`gnca_integrated_gradients_{features,genes}.csv`, and the per-gene knockout sweep
`gnca_gene_perturb_fast.csv` — support the downstream attribution and
collapse-curve analyses. [src: `analysis/gnca/`] **[verified-in-artifact]**

---

## 2.4 Neuron-role classifier probe (expression alone)

As an independent check that expression is genuinely informative about circuit
role, a supervised classifier predicts each neuron's coarse **behavioral role
(sensory / interneuron / motor)** from its expression vector alone.

- **Features.** PCA-reduced expression features $X$.
- **Labels.** A 3-class role assigned in-notebook by a **neuron-name-prefix
  heuristic**: sensory (AF/AL/AS/AVM/AWA/AWC/ASE/ASH/URX/OL...), motor
  (DA/DB/DD/VA/VB/VD/AS...), else interneuron.
- **Model.** `nn.Sequential(Linear(d,128) → ReLU → Linear(128,64) → ReLU →
  Linear(64,3))`; Adam ($\text{lr}=2\times10^{-3}$); cross-entropy loss; 100
  epochs/fold; `StratifiedKFold` (5-split `kf`, defined cell 25).
- **Metric.** Mean fold accuracy = **0.9274603174603175 (92.7%)**. [src:
  `NCA.ipynb` cell 28, "TASK D: Predict Behavioral Role (3-class)", executed
  `Mean Accuracy (role): 0.9274603174603175`] **[verified-in-artifact]**

**Honesty caveats (to be disclosed in Results).** (1) The role labels are a
name-prefix heuristic, **not** curated WormAtlas annotations; the `AS` prefix
appears in *both* the sensory and motor lists, so `AS*` neurons are labeled motor
(the motor check runs second) — a known ambiguity in the labeling rule.
(2) No class-balance / chance-level baseline is printed in this cell; 3-class
chance is ~0.33 but the role distribution is imbalanced, so balanced accuracy
would be the more honest headline metric. (3) This is the **role** task from
`NCA.ipynb` only; a separate clean-pipeline notebook
(`New Notebooks/04_synthetic_dynamics_and_classification.ipynb`) reports a
*different* task — 3-class neurotransmitter (cholinergic/GABA/glutamate) at
0.727 ± 0.077 (N = 84) **with a proper permutation null** (null 95th percentile
0.629) — and the two must not be conflated. The 92.7% is the name-prefix role
probe; it establishes the upper-bound sense in which expression "knows what a
neuron is," against which the *failure* of expression to add connectivity/dynamics
information becomes interpretable.

---

## 2.5 Dynamics: gene-conditioned graph-LSTM and the shuffle test (the honest null)

To test whether expression shapes *dynamics* rather than only static wiring, a
hybrid **gene-conditioned graph-LSTM** (the "PhaseD" model) is trained to predict
neural time-series, and subjected to a pre-specified ablation/shuffle battery.

- **Model (D3 `HybridGeneGraphLSTM`).** Maps sliding-window dynamics
  (`X_win` shape `(B, T, N, 1)`) plus a per-neuron CeNGEN **gene embedding**
  (PCA of expression) over a normalized sparse adjacency.
- **Tests.** A **zero-gene ablation** (`no_gene`) and a **gene-shuffle**
  (`gene_shuf`, permute the per-neuron gene embedding across neurons).
- **Gate.** `gene_causal_ok = (mse_shuf − mse_hybrid) > 0.005`.
- **Result.** `mse_hybrid = 0.9879`, `mse_no_gene = 0.9857`,
  `mse_gene_shuf = 0.9879`. The hybrid equals the shuffled-gene model exactly, and
  zeroing genes is *marginally better*; the gate evaluates to
  $0.9879 - 0.9879 = 0.0000 \not> 0.005$ →
  **`gene_causal_by_shuffle = False`**. [src: `PhaseD.ipynb` cell 66
  `hybrid: 0.9879, no_gene: 0.9857, gene_shuf: 0.9879`; cell 68
  `VERDICT: {'mse_gain_over_identity': True, 'gene_causal_by_shuffle': False}`]
  **[verified-in-artifact]**

**Corroborating clean re-implementation.** The cleaner re-derivation in
`New Notebooks/04_synthetic_dynamics_and_classification.ipynb` (Part A) reaches the
**same null** via a preregistered criterion: synthetic linear dynamics from the
Witvliet adjacency plus a small gene-conditioned term, four Ridge($\alpha=1$)
predictors on features $[\,y[n],\,(A_{\text{norm}}y)[n],\,\text{gene}[n]\,]$
(identity / graph-only / hybrid / shuffled-gene), trained on the first half of
the timesteps and tested on the second; the preregistered gate
$(\text{mse}_{\text{shuf}} - \text{mse}_{\text{hybrid}}) > 0.01$ evaluates to
**0.001283 → NOT GENE-CAUSAL (null)** (mse_identity 0.0034, mse_graph 0.0038,
mse_hybrid 0.0025, mse_shuf 0.0038; summary saved to
`data_derived/nb04_dynamics_summary.csv`).

**Critical scope statement (must appear in Results/Limitations).** This null is on
**synthetic / simulated dynamics**, not real calcium imaging. As stated in the
clean notebook (`nb04` cell 0): the PhaseD dynamics test ran on a simulated
trajectory (`PhaseA_A9_trajectories_continuous.npy`); the only real dynamics set
(Kato 2015) lacks neuron identities, so gene→dynamics cannot be tested on real
data with the available data. The null therefore **bounds what the clean pipeline
can show** — gene conditioning adds no shuffle-robust improvement under this model
class and this (synthetic) dynamics target — rather than refuting gene→dynamics
causality in biology. It is foregrounded, not buried: a negative result obtained
by a pre-specified shuffle control is the evidence that makes the topology-dominant
thesis credible rather than an artifact of an over-expressive model.

---

## 2.6 Genetic-redundancy ("collapse") analysis

To characterize *how* the expression contribution is distributed across genes, we
rank genes by their influence on the predictor and remove them cumulatively
(in-silico knockout), recording model performance as a function of the number of
top drivers removed.

- **Task and target.** This sweep is run on the **expression → hub-topology** map,
  **not** the GNCA edge-regression. The target is `Hub_Div = 1` if out-degree > 5
  else 0. [src: `phaseE.ipynb` cell 6]
- **Cohort and features.** N = **222** common neurons; features = the
  top-2000-variance CeNGEN genes (`StandardScaler`-normalized).
- **Model.** `DirectMLP(2000 → 32 → 1)`, Adam ($\text{lr}=1\times10^{-3}$),
  `BCEWithLogitsLoss` with L1 (Lasso) regularization (`L1_LAMBDA=0.005`) on the
  first layer, 401 epochs; metric = `pearsonr(Hub_Div, sigmoid(logits))`.
- **Knockout method.** Cumulative zeroing of the top-K driver genes (ranked by
  influence on the MLP's hub prediction) for
  $K \in \{0,1,2,5,10,20,30,40,50,75,100,200\}$, re-evaluating both the model
  correlation R and the mean probability assigned to hub neurons at each K.
- **Result (the collapse curve, Figure 1).** Trained R = 0.818; the cumulative-KO
  trajectory is R(K=0)=0.818, 0.819, 0.802, 0.799 (K=5), then 0.678 (K=10),
  0.679, 0.686, 0.685, 0.665 (K=50), 0.683, 0.682, **0.687 (K=200)** — i.e. an
  initial drop over the first ~10 knockouts followed by a **plateau ≈ 0.665–0.687
  well above the chance line through 200 removed genes**; the average hub-probability
  declines only gently (0.952 → 0.886). [src: figure
  `public/images/projects/celegans-collapse-curve.png`; source+output
  `phaseE.ipynb` cell 6, "Combinatorial Gene Knockout Analysis (The Collapse
  Curve)"; original saved figure
  `/mnt/ssd4tb/Desktop/C-Elegans/figures/crispr_results/genetic_collapse_curve.png`]
  **[verified-in-artifact]**

**Interpretation and scope.** Graceful, plateauing degradation is the signature of
a **redundant** genetic code rather than a few load-bearing master genes — no
small driver set is individually necessary for the topology-aligned prediction.
The collapse-curve baseline (R ≈ 0.82) is the operating point of *this
hub-prediction MLP*, a different model setup from the GNCA edge regression
(§2.3); the curve is informative about **redundancy structure** (the shape of the
degradation), not about the GNCA's absolute predictive ceiling, and the two
numbers are not conflated.

**[PROVENANCE GAP — figure label discrepancy to resolve.]** The committed PNG
title reads `Top Driver: WBGene00001137`, but the `phaseE.ipynb` cell-6 run read
for this draft identifies the top driver as `WBGene00019961` (the cell-4
single-gene version). The two runs share the same plateau pattern and R/hub-prob
trajectory; only the reported top-driver gene-ID label differs (the PNG was
produced by a sibling/re-run). The discrepancy must be resolved and the top driver
resolved to a gene symbol via WormBase before the final figure caption is fixed.

---

## 2.7 Corroborating variance-partition (topology-dominant spine)

Beyond the four headline conditions, a variance partition for motif prediction
(`NCA.ipynb` cell 31, mean $R^2$) independently supports the topology-dominant
reading: expression-only 0.388, degree-only 0.838, both 0.882, expression-UNIQUE
0.044, degree-UNIQUE 0.494 — expression adds little *unique* variance over
topology/degree. [src: `NCA.ipynb` cell 31] **[verified-in-artifact]**

---

## 2.8 Summary of honesty flags carried into Results

1. **GNCA r = 0.987 is in-sample, not held-out** (§2.3.4). Report as a full-fit;
   the thesis rests on the relative four-condition comparison and is unaffected.
2. **Cohort-N must not be conflated** (§2.1.3): GNCA on N = 838 (with
   duplicate/whitespace-dirty names in the saved CSV), role/CCA on 179, collapse
   curve on 222, master degree/CCA cohort on 91.
3. **Graph-only** = peak 0.861 / final-epoch 0.857 (§2.3.5); report ≈ 0.86.
4. **No bootstrap CIs / permutation nulls** exist for the headline scalars (GNCA
   r-values, 92.7% accuracy, CCA ρ) under the zero-compute constraint; the CCA and
   role tasks in particular lack permutation nulls in their own cells (the dynamics
   null and the NT classifier do have them). The absence of uncertainty
   quantification is acknowledged, not papered over.
5. **The dynamics null is on synthetic dynamics** (§2.5); it bounds the clean
   pipeline, it does not refute gene→dynamics causality in biology.
6. **Collapse-curve top-driver label discrepancy** (§2.6) is flagged for
   resolution.

[Bibliographic details, DOIs/PMIDs, and verification status for all cited works
are in `paper/references.md`.]
