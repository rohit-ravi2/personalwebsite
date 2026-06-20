# 4. Results

The question this section answers is not whether gene expression and connectivity
are coupled in *C. elegans* — they demonstrably are — but whether single-cell
expression carries information about local wiring **beyond what the connectome's own
topology already supplies**. We answer it by triangulation: the same conditional
accounting is run across three method families of increasing nonlinearity — a linear
method (canonical-correlation analysis), a nonlinear static predictor (a graph neural
cellular automaton with a matched shuffle/ablation battery), and a supervised
classifier probe — and is then stress-tested against a dynamics model whose pre-specified
null is the decisive result, and against a genetic-knockout sweep that characterizes how
the (small) expression contribution is distributed across genes.

Every quantitative claim below is tagged with the committed artifact it is read from.
Where the backing artifact contradicts a previously-circulated framing of a number
(e.g. the word "held-out," or the cohort size attached to a result), the discrepancy is
**FLAGGED** in place rather than smoothed over. Nothing was re-run for this section.

> **Cohort note — two distinct Ns, stated up front.** The analyses below do **not** all
> run on the same neuron set, and the manuscript must not conflate them.
> The GNCA synaptic-strength model and its three controls (§4.1) run on the **N = 838
> expression-aligned node set** of `NCA.ipynb` (input `gene_expr_CeNGEN_base_mapped.csv`,
> 838 rows; the saved prediction CSV contains duplicate neuron-name rows and
> trailing-whitespace names, so 838 is *not* 838 unique somatic neurons). The CCA (§4.2)
> and the role classifier (§4.3) run on a ~**179-neuron** alignment in the same notebook;
> the collapse curve (§4.5) runs on **N = 222** common neurons. The widely-quoted figure
> **N = 91** is the somatic-neurons-with-both cohort built in `Paper1MasterNotebook.ipynb`
> (CeNGEN/GSE136049 averaged per-neuron, intersected with the connectome) and is the unit
> for the master-notebook degree/CCA work — **it is not the cohort of the headline GNCA
> r = 0.987.** [src for N = 838: `notebooks/archive/nca/NCA.ipynb` cell 0 (`expr_df: (838, 22469)`)
> and `analysis/gnca/gnca_pred_outgoing_strength.csv` (838 data rows, verified by direct
> read). src for N = 91: `notebooks/Paper1MasterNotebook.ipynb` cell 31
> (`Neuron expression matrix: (91, 22468)`). All artifact paths are under
> `/home/rohit/Desktop/C-Elegans/`.]
>
> **[PROVENANCE GAP — N = 91]** I could not locate, in this worktree or the analysis repo,
> an intersection-manifest artifact that re-derives 91 from the raw source-cell lists; it
> appears only in the project's own MDX write-up and the context scorecard. Report N = 91
> as the master-notebook cohort size, and re-derive it from the canonical loader with
> explicit inclusion criteria before submission.

---

## 4.1 The marginal value of expression for synaptic strength is small (GNCA + controls)

The central static test predicts each neuron's mean **outgoing synaptic strength**
(z-scored) from a graph neural cellular automaton run over the connectome, with optional
per-neuron expression features (top-32 PCA components of CeNGEN expression). The full
model reaches **Pearson r = 0.987 (MAE = 0.097)**.
[src: `notebooks/archive/nca/NCA.ipynb` cell 4 executed output `Final Pearson r=0.987 | MAE=0.097`;
saved predictions `analysis/gnca/gnca_pred_outgoing_strength.csv` (838 rows);
tabulated in `CLAUDE_CONTEXT.md` line 38 as **"GNCA (full)"**.]

> **FLAG — this is an in-sample fit, NOT "held-out."** The synaptic-strength cell and its
> three controls (`NCA.ipynb` cells 3–8) contain **no train/test split**: Pearson r is
> computed on the full node set. `train_test_split`/`KFold` appear only in later cells
> (20, 24–26, 31, 36–39), which drive the motif/classifier analyses, not this regression.
> `CLAUDE_CONTEXT.md` correctly labels the model "GNCA (full)." The manuscript must report
> r = 0.987 as an **in-sample full-fit**, not a held-out result. Any prior text (including
> the project MDX and the companion draft `multimodal_topology_vs_expression.md` §4.1)
> that says "held-out r = 0.987" is **incorrect on this point** and should be corrected.
>
> **Secondary check.** Recomputing Pearson r directly from the committed prediction CSV
> (838 rows) yields **r = 0.984** — consistent with, but not identical to, the notebook's
> 0.987. The notebook value is from the best-loss checkpoint; the saved CSV recovers
> r ≈ 0.984. Report 0.987 as the notebook figure and note the CSV recovers ≈ 0.984.
> [verified by direct read of `analysis/gnca/gnca_pred_outgoing_strength.csv`.]

The control battery localizes where that signal lives. All four conditions are measured
the same in-sample way, so the **relative** comparison is valid even though the absolute r
is not held-out:

| Condition | r | Provenance | What it rules out |
|---|---|---|---|
| **Full** (topology + real expression) | **0.987** | `NCA.ipynb` cell 4; `CLAUDE_CONTEXT.md` l.38 | ceiling (in-sample) |
| **Shuffled-expression** (topology intact, expression rows permuted) | **0.934** | `NCA.ipynb` cell 5 (`[SHUF] Epoch 080 \| r=0.934`); `CLAUDE_CONTEXT.md` l.39 | that *correctly-assigned* expression is load-bearing — it is not |
| **Graph-only** (no expression features) | **≈ 0.86** | `NCA.ipynb` cell 8 (`[GRAPH-ONLY] Epoch 080 r=0.861`; final-epoch summary `r = 0.857`); `CLAUDE_CONTEXT.md` l.40 | that topology alone is insufficient — it is nearly sufficient |
| **Expression-only** (graph removed) | **0.27** | `NCA.ipynb` cell 8 (`[EXPR-ONLY] r=0.270`, final-epoch `0.269`); `CLAUDE_CONTEXT.md` l.41 | that expression alone predicts local wiring — it largely does not |

> **FLAG — graph-only value.** The headline 0.861 is the **per-epoch peak (epoch 80)**;
> the notebook's final `=== Baseline Comparison ===` summary line (last epoch, 100) prints
> **`Graph-only r = 0.857`**. Both come from the same run. Report "**≈ 0.86 (peak 0.861 /
> final 0.857)**." Expression-only is likewise peak 0.270 / final 0.269; report **≈ 0.27**.
>
> **Provenance precision — shuffled-expression.** 0.934 is the value at epoch 80, the
> *last* logged epoch of an 80-epoch run (the shuffle control runs fewer epochs than the
> 200-epoch main model), so it is this control's final reported value, not a cherry-picked
> mid-run peak. Mechanism: `torch.randperm` permutes the expression-feature rows (breaking
> the neuron↔expression alignment) while the out-degree-normalized adjacency `A_norm` is
> left untouched.

**Three conclusions follow, and each is anchored to a specific control:**

1. **Topology is the dominant carrier of the signal.** Graph-only (≈ 0.86) already recovers
   the great majority of the explainable variance, while expression-only (0.27) is far
   behind. *Control that establishes this:* the graph-only vs. expression-only gap.

2. **The marginal value of correctly-assigned expression is the Full − Shuffled gap
   (0.987 → 0.934), which is small.** Scrambling *which* expression vector belongs to *which*
   neuron — while preserving the graph and the full expression distribution — costs only
   ≈ 0.05 in r. *Control that establishes this:* the shuffle. This is the single most
   important number in the section, because it isolates the contribution of expression
   *identity* from the contribution of merely having feature-shaped inputs.

3. **Expression is not useless, but it is modulatory.** Full (0.987) does exceed graph-only
   (≈ 0.86), so a real refinement exists; its size is consistent with a modulatory layer
   on a topology-determined backbone, not with expression being the architect.

A caution on reading r near the ceiling: at r = 0.987 vs 0.934 the variance-explained
difference (R² 0.974 vs 0.872) is larger than the raw-r gap suggests, but it remains modest
beside the graph-only baseline, and the qualitative ranking (topology ≫ expression-identity)
is robust. **[PROVENANCE GAP — CIs]** No bootstrap/permutation confidence intervals exist for
any of these four r-values; `CLAUDE.md` line 41 and `CLAUDE_CONTEXT.md` line 62 both flag
"No bootstrap CIs on N=91 neurons" as a known pre-submission gap. Under the zero-compute
constraint we report point estimates exactly as committed and acknowledge the absence of CIs.

**Corroborating evidence (same thesis, independent artifact).** In
`Paper1MasterNotebook.ipynb` (cells 37–38) a raw-gene → degree regression gives strongly
**negative R²** (e.g. total-degree R² = −15.3), independently confirming that expression
alone has little predictive power for connectivity targets. And `NCA.ipynb` cell 31's
variance partition for motif prediction gives expression-only R² = 0.388 vs degree-only
R² = 0.838 (both = 0.882; expression-**unique** = 0.044 vs degree-unique = 0.494) — i.e.
expression adds little *unique* variance over topology/degree.
[src: `notebooks/Paper1MasterNotebook.ipynb` cells 37–38; `notebooks/archive/nca/NCA.ipynb` cell 31.]

---

## 4.2 Expression nonetheless carries linear circuit structure (CCA)

To establish that the topology-dominant result is not an artifact of expression being
uninformative, we measure the *linear* shared structure between an expression embedding and
a connectivity-**motif** embedding by canonical-correlation analysis. The top three
canonical correlations are **ρ = [0.875, 0.836, 0.809]**.
[src: `notebooks/archive/nca/NCA.ipynb` cell 35 executed output
`Canonical correlations (expr <-> motif): [0.875, 0.836, 0.809]`;
tabulated in `CLAUDE_CONTEXT.md` line 44. Exact match.]
Method: `sklearn.cross_decomposition.CCA(n_components=3, max_iter=1000)` on PCA(50) of
expression vs PCA(5) of a 5-column motif matrix (reciprocal / feedforward /
feedback-triangle / shared-output / lateral-inhibition counts per neuron, built in
`NCA.ipynb` cells 21–22); ρᵢ = corrcoef(U[:,i], V[:,i]).

At least three strongly-correlated dimensions are shared between transcriptional identity
and local motif participation — fully consistent with prior reports that expression and
connectivity are coupled in *C. elegans* [Arnatkevičiūtė et al. 2018; Fenyves et al. 2020].
The point of this paper is *not* that the coupling is absent (it plainly exists, linearly
and substantially) but that, **conditioned on topology**, expression adds little to
predicting the wiring itself (§4.1) or the dynamics (§4.4).

> **FLAG — descriptive, upward-biased.** These ρ are **in-sample** (no held-out split, no
> permutation null on the canonical correlations) and are computed in a high-PC space
> (50 expression PCs, 5 motif PCs) over a small N — a regime in which CCA correlations are
> upward-biased. Frame them as **descriptive magnitudes of linear alignment**, or add a
> shuffle null before submission. They are not, on their own, evidence of a large
> *conditional* expression contribution; §4.1 is the conditional accounting.

---

## 4.3 Expression predicts coarse behavioral role (classifier probe)

As an independent upper-bound check that expression *is* genuinely informative about circuit
role, an expression-only classifier predicts the three-class behavioral role
(sensory / interneuron / motor) at **mean CV accuracy = 92.7%** (0.9274603…).
[src: `notebooks/archive/nca/NCA.ipynb` cell 28 (`# === TASK D: Predict Behavioral Role
(3-class) ===`), output `Mean Accuracy (role): 0.9274603174603175`;
tabulated in `CLAUDE_CONTEXT.md` line 45. Exact match.]
Method: MLP 128-64-3, CrossEntropyLoss, Adam lr = 2e-3, 100 epochs/fold, StratifiedKFold
(5-split), features = PCA-reduced expression.

Transcriptional identity therefore *does* encode what kind of neuron a cell is, at the
resolution of its broad circuit role — as the terminal-selector logic of *C. elegans*
neuronal identity predicts [Hobert 2016]. This is the load-bearing framing for the null in
§4.4: expression is richly informative about *identity*, yet that information is largely
**redundant with topology** when the task is predicting local wiring or dynamics.

> **FLAGS — three caveats the manuscript must disclose.**
> (1) **Labels are a name-prefix heuristic** assigned in-notebook, not curated WormAtlas
> annotations; the prefix `AS` appears in *both* the sensory and the motor lists, and because
> the motor check runs second, `AS*` neurons are labeled motor — an ambiguity/bug to fix or
> at least disclose.
> (2) **No chance/permutation baseline is printed in this cell.** Three-class chance is
> ≈ 0.33, but the role distribution is imbalanced, so **balanced accuracy** would be the
> honest metric; report it before submission.
> (3) **Do not conflate with the NT task.** The cleaner sibling pipeline
> `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` reports a *different*
> 3-class task (neurotransmitter chol/GABA/glu at 0.727 ± 0.077, N = 84) *with* a proper
> permutation null (null 95th pct 0.629). The 92.7% is the role task from the archive
> `NCA.ipynb` only.

---

## 4.4 The honest null: gene conditioning does not improve dynamics

The decisive negative result comes from dynamics. A hybrid **gene-conditioned graph-LSTM**
(`HybridGeneGraphLSTM`, "PhaseD") was trained to predict neural time-series from
sliding-window dynamics plus a per-neuron CeNGEN gene embedding over a normalized sparse
adjacency. Its pre-specified shuffle test returns:

**VERDICT: `{'mse_gain_over_identity': True, 'gene_causal_by_shuffle': False}`.**

The supporting MSEs: **hybrid = 0.9879, no_gene (zeroed) = 0.9857, gene_shuf (permuted) =
0.9879.** The hybrid model exactly equals its shuffled-gene control, and *zeroing* the genes
is marginally **better** than keeping them. The gate
`gene_causal_ok = (mse_shuf − mse_hybrid) > 0.005` evaluates to 0.0000 → **False**.
[src: `notebooks/archive/phases/PhaseD.ipynb` cell 66 (`hybrid: 0.9879, no_gene: 0.9857,
gene_shuf: 0.9879`) and cell 68 (`VERDICT: {...gene_causal_by_shuffle: False}`);
tabulated in `CLAUDE_CONTEXT.md` line 47. Exact match.]

**We foreground this result rather than bury it.** The hypothesis entering Phase D was the
strong, blueprint-flavored claim that *gene expression causally shapes how neurons fire
together*; the project's own pre-specified null battery falsified it. A negative result
obtained from a *shuffle control that was specified before the test* is precisely the
evidence that makes the topology-dominant thesis credible rather than an artifact of an
over-expressive model: in the one setting where expression *should* most plausibly help
(dynamics, where molecular identity sets channel and receptor complements), correctly-assigned
expression did not beat its own shuffle.

**Corroborating null (independent, preregistered).** The cleaner re-implementation
`New Notebooks/04_synthetic_dynamics_and_classification.ipynb` (Part A) reaches the *same*
null through a preregistered criterion: `gene_causal_delta = (mse_shuf − mse_hybrid) =
0.001283 < 0.01 threshold` → "NOT GENE-CAUSAL (null)" (mse_identity 0.0034, mse_graph 0.0038,
mse_hybrid 0.0025, mse_shuf 0.0038), summary saved to `data_derived/nb04_dynamics_summary.csv`.
[src: `New Notebooks/04_synthetic_dynamics_and_classification.ipynb` Part A.]

> **FLAG — scope of the null (critical, do not overstate).** Per the source (nb04 cell 0),
> the PhaseD dynamics test ran on a **simulation** (`PhaseA_A9_trajectories_continuous.npy`),
> **not real calcium imaging**. The only real dynamics set with the right structure
> (Kato 2015) lacks neuron identities, so a gene→dynamics test cannot be run on real,
> identity-resolved data with the available data. The manuscript must therefore state the
> null at the level it was earned: it bounds **what this clean pipeline can show on
> synthetic, simulation-derived dynamics**, and is *not* a refutation of gene→dynamics
> causality in biology. "Absence of detectable improvement" on this dataset and model class
> ≠ "proof of no effect." This is the most important future test (richer, identity-resolved
> recordings with a preregistered shuffle).

---

## 4.5 Genetic redundancy: the collapse curve

Finally, to characterize *how* the (small) expression contribution is distributed across
genes, top "driver" genes were ranked and removed cumulatively (in-silico knockout) from a
predictor of **hub status** (out-degree > 5), recording model correlation as a function of
the number of genes removed. A blueprint with a few load-bearing genes would collapse toward
chance quickly; a redundant code would degrade gracefully and plateau above chance.

The curve plateaus **well above chance** (Figure: `celegans-collapse-curve.png`). After an
initial drop from the trained baseline (**r ≈ 0.818**) over the first ~10 knockouts,
performance flattens around **r ≈ 0.665–0.687** and stays essentially unchanged from
~K = 10 through **K = 200** removed genes; the average probability assigned to hub neurons
declines only gently (**0.952 → 0.886**) and never approaches the chance line at 0.
Verified knockout trajectory (R at K = 0,1,2,5,10,20,30,40,50,75,100,200):
0.818, 0.819, 0.802, 0.799, 0.678, 0.679, 0.686, 0.685, 0.665, 0.683, 0.682, 0.687.
[src — FIGURE: `/home/rohit/Desktop/website-paper/public/images/projects/celegans-collapse-curve.png`
(read directly; red r-curve and navy hub-prob trajectory confirmed visually);
SOURCE CODE + OUTPUT: `notebooks/archive/phases/phaseE.ipynb` cell 6 (`Combinatorial Gene
Knockout Analysis (The Collapse Curve)`); original saved fig
`/mnt/ssd4tb/Desktop/C-Elegans/figures/crispr_results/genetic_collapse_curve.png`.]
Method: target `Hub_Div` (out-degree > 5); features = top-2000-variance CeNGEN genes
(StandardScaler'd); N = 222 common neurons; `DirectMLP(2000-32-1)`, Adam lr = 1e-3,
BCEWithLogitsLoss + L1 (λ = 0.005) on the first layer, 401 epochs; metric = Pearson r
between `Hub_Div` and the model's sigmoid output; cumulative zeroing of the top-K driver genes.

This graceful, plateauing degradation is the signature of a **redundant** genetic code, not
of a few master genes: no small set of drivers is individually necessary for the
topology-aligned prediction. Redundancy of this kind is exactly what one expects if topology —
not a compact gene set — is doing the structural work, with the expression contribution
spread thinly across many partially-substitutable genes. This supports the topology-dominant
thesis from the gene side, complementing the shuffle (§4.1) and the dynamics null (§4.4).

> **FLAGS — two provenance items.**
> (1) **Top-driver gene-ID discrepancy.** The committed PNG title reads
> "Top Driver: **WBGene00001137**," but the `phaseE` cell-6 run on record identifies the top
> driver as **WBGene00019961** (the cell-4 single-gene version). The PNG was produced by a
> sibling/re-run with the same plateau pattern but a different reported top-driver ID; the
> numeric trajectory (R and hub-prob) is consistent across them, only the gene-ID label
> differs. **[PROVENANCE GAP]** Neither WBGene ID is resolved to a gene symbol in this
> worktree; look both up in WormBase, reconcile which run the published figure comes from,
> and name the gene in the final caption.
> (2) **Different task from the GNCA.** The knockout sweep predicts **hub status** (out-degree
> > 5), **not** the GNCA outgoing-synaptic-strength regression of §4.1. Describe the collapse
> curve as redundancy in the **expression → hub-topology** map, not in the GNCA edge model;
> the baseline r ≈ 0.82 here is a different model setup from the GNCA r = 0.987 and the two
> must not be conflated.

---

## 4.6 Synthesis of the results

| Method family | Result | Sourced artifact | What it contributes to the thesis |
|---|---|---|---|
| Linear (CCA) | ρ = [0.875, 0.836, 0.809] | `NCA.ipynb` c.35 | expression *is* coupled to circuit structure (in-sample) |
| Classifier probe | role acc = 92.7% | `NCA.ipynb` c.28 | expression *is* richly informative about neuron identity |
| Nonlinear static (GNCA) | full 0.987 / shuf 0.934 / graph ≈0.86 / expr 0.27 | `NCA.ipynb` c.4–8 | **conditioned on topology, expression adds little** |
| Dynamics (graph-LSTM) | `gene_causal_by_shuffle = False` | `PhaseD.ipynb` c.66/68 | gene conditioning adds **nothing shuffle-robust** (on synthetic dynamics) |
| Redundancy (collapse) | plateau r ≈ 0.68 through 200 KOs | `phaseE.ipynb` c.6 + figure | the small expression contribution is **diffuse/redundant** |

The two informative-about-identity results (CCA, classifier) and the three
conditional/marginal results (GNCA controls, dynamics null, collapse curve) are not in
tension: expression knows *what a neuron is*, but once the connectome's local topology is
given, that knowledge is largely redundant for predicting the *local wiring* and — on this
synthetic-dynamics dataset — undetectable above shuffle for predicting *activity*. The
convergence of a linear method, a nonlinear static predictor, a classifier probe, a dynamics
null, and a redundancy analysis is what licenses the topology-dominant, gene-modulatory
reading, and the honest null and the redundancy plateau are its strongest support rather than
caveats to it.
