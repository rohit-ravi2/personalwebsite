# The arc of the C. elegans project — from exploratory notebooks to a real simulator

## Phase 0 — Confusion

The earliest work was what it should be for a one-person undergraduate
research project without a clear mentor: chaos. `phase0.ipynb` is
twenty code cells with no markdown — raw loaders, key merged datasets,
a union of neurons across source/target/lineage fields, all written
before the author really knew what a connectome should look like.
`data_prep/cleanup.ipynb` counts file extensions and builds a
stringified schema signature because someone had to actually read
what had been downloaded. `data_prep/DataOverview.ipynb` eventually
formalised this into a programmatic inventory, but the intellectual
centre of gravity at this stage was just *"I have thousands of files
from WormBase, CeNGEN, Cook 2019, Loer & Rand, DANDI, Atanas
recordings — what's in all this and what fits together?"* Nothing
here was science yet; it was prerequisite file hygiene. There were
real mistakes baked in at this point — inconsistent neuron-name
normalisations (ASHL vs ASH_L vs ASHl), WormBase gene ID vs gene
symbol crosswalks that silently dropped neurons, ad-hoc ID cleaners
that each notebook reinvented slightly differently — and every one
of these errors propagated downstream until it was hunted down months
later. Several of the `.bak` files from August 2025 are literally
the fossilised record of redundant pipelines that got deleted once a
single canonical loader existed.

## Phase 1 — Graph construction

`phase1.ipynb` is the first thing that looks like a deliberate
project. It's 117 code cells with ten markdown sections, titled
*"Phase 1 — Graph Construction and Validation"*, consolidating
chemical, electrical and lineage information into a unified NetworkX
graph. This is also where the first round of validation happened:
checking symmetry of gap junctions, reconciling the Cook 2019 counts
against Loer & Rand 2022's NT identities, tracking which neurons
appear in which sources, auditing orphan edges. The notebook is
huge because a lot of it was throwaway exploration left in — `# TODO
— check if symmetry holds`, `# why does this drop 4 neurons?` — but
by the end of it there was a graph object that was defensible enough
to feed downstream. `phase3.ipynb` from August 2025 is the unfortunate
cousin: *"Sprint to insight — Planetary Ultimate Dataset Analysis"*
— which is exactly the kind of title an undergrad writes when they
think enough EDA will produce meaning on its own. It didn't, and
most of its conclusions got quietly abandoned when the more rigorous
phases replaced them.

## Phases A / B / C — The first real structure

The Phase-A/B/C notebooks from late August 2025 are when the project
started to behave like research rather than like exploration.
`phaseA.ipynb` does the canonical ID-normalisation pass once and for
all. `phaseb.ipynb` introduces **null models** — degree-preserving
shuffling, edge permutation — because whoever was mentoring the
author (or the author's own reading) had pointed out that raw motif
statistics without a null are meaningless. This matters because the
thesis only works if you can show that real biology differs from
permuted biology; shipping a number without that comparison is the
undergraduate mistake everyone makes first. `phasec.ipynb` then
builds the "gold subgraph" (well-characterised connections with
independent literature support) and refines the enrichment analysis.
These three notebooks are where the project became falsifiable — every
claim now had at least one explicit control.

## Phase D — Time dynamics and the first honest null

`PhaseD.ipynb` is sixty-five cells long and it's where the project
first confronted neural dynamics instead of static graph structure.
The pitch was *"gene expression causally shapes how neurons fire
together"* — a strong claim — and the notebook cycles through LSTM
prototypes, cycle-aware LSTM training, cycle-aware Transformers, and
a per-neuron shared LSTM with gene conditioning. The crucial moment
is Cell 35 onwards: temporal shuffled nulls, identity-permuted nulls,
cycle-permuted nulls. After all of this, the honest finding was
`gene_causal_by_shuffle: False`. The PhaseD hybrid gene-graph LSTM
did **not** predict worm dynamics better when gene expression was
real versus when it was shuffled. That result is in the project's
MDX page now as the honest null, but it must have been difficult to
write down at the time — it meant a major part of the narrative had
been falsified by the project's own internal audit.

## Phase E / NCA — Graph Neural Cellular Automata

Having been burned once on a too-strong dynamics claim, the project
retreated to a more conservative question: can topology predict
synaptic strength from local rules? `NCA.ipynb` and
`phaseE.ipynb` cell 1 (`run_nca_pipeline.py`) implement a Graph
Neural Cellular Automaton that predicts outgoing synaptic strength
per neuron from connectome structure plus expression, and this is
where the numbers that anchor the paper come from: **r = 0.987** on
held-out edges with the full model; **r = 0.934** when expression is
shuffled (topology carries almost all the signal); **r = 0.861**
graph-only; **r = 0.27** expression-only. The story settled into its
final shape here — *topology-dominant, gene-modulatory* — because
that's what the four-way comparison actually said. Paper1MasterNotebook
is the write-up flow that turns these four numbers into an argument.

## The simulator era — Phase 1a through Phase 3d

The project then pivoted from pure analysis to simulation. Phase 1a
built a 20-segment MuJoCo body. Phase 2a added
resistive-force hydrodynamics and an imitation-trained PPO controller
against Boyle-Berri-Cohen 2012 biological parameters. Phase 3a
imported the Cook 2019 connectome into a 300-neuron Brian2 LIF
network. Phase 3b trained an 8-event classifier bank on the 18-neuron
strict cross-worm intersection of the Atanas 2023 recordings
(AIBL, ASEL, AUAL, AVEL, AVER, CEPDL, I3, IL2DL, M3L, M3R, NSML,
NSMR, OLQDL, OLQDR, OLQVL, RMER, SMDVL, URXL) — the only neurons
reliably identified in all ten worms. Phase 3c integrated it all:
sensory stim → LIF brain → synthetic Ca → classifier → FSM → CPG →
MuJoCo → proprioception → back to brain. Phase 3d layered nine
neuropeptidergic/monoaminergic modulators on top with
CeNGEN-derived releaser tables. This architecture deliberately
mirrored the pattern of **Shiu et al. 2024's Drosophila brain
simulator** but keyed to worm biology, and it filled a gap neither
**FlyWire** (a reference connectome, no dynamics) nor **Eon Systems'
2026 embodied fly** (closed-source, no exposed audit trail) had
filled for any organism except the fly.

## The audit era — v3.0 through v3.3

Everything looked good until the numbers got stress-tested. The
v3.0 perturbation suite reported two apparent phenotype reproductions:
RIS → quiescence collapse ΔQUI = −0.53, AVA → reversal abolished
ΔREV = −0.15. Then biological corrections were added
(5-HT pharyngeal exclusion, CeNGEN per-edge glutamate receptors) and
both numbers appeared to regress. Instead of iterating on the
calibration, the project ran a **proper reproducibility audit**: 3
seeds × 3 configs × 6 ablations × 2 conditions, 108 runs, Brian2 and
numpy seeds locked. The audit overturned the initial story in both
directions. AVA under touch reliably abolished reversal at
**ΔREV = −0.57 ± 0.37** across three seeds — a cleaner reproduction
than the original. But the RIS flagship result averaged to
**ΔQUI = −0.24 ± 0.33**, directionally consistent with Turek 2016
but not statistically robust at 20 s runs. Every other claim (NSM,
RIM, PDE, AVB ablation) was noise. The biological corrections had
**systematically degraded phenotype reproduction**, which was
uncomfortable but informative: adding more accurate biology shifted
the network out of the regime the v3.0 numbers came from.

## Tier 1 — The biology-forward pivot

Rather than keep tuning v3 to pass tests, the next move was to
correct the biology even at the cost of short-term phenotype numbers.
**T1a** replaced LIF spiking with graded σ(V) dynamics
(Kunert-Graf 2014) because most *C. elegans* neurons don't
fire vertebrate-style action potentials (Goodman, Hall & Avery 1998).
**T1b** added L-type Ca plateau currents on fourteen command
neurons so reversal-bout sustainment became emergent. **T1c**
replaced uniform modulator broadcasting with 3D distance-weighted
volume transmission — FLP-11 from RIS now preferentially reaches
head neurons rather than every cell equally. **T1d** fixed a v1
O(n²) Brian2 recompile bug by making proprioception use persistent
PoissonGroups with mutable rates, closing the brain-body loop
properly. **T1e** added a 2D chemotaxis environment. When the
Tier 1 stack was audited, all 36 runs produced identical
QUI=0.92 state distributions — the graded brain's σ(V) output
didn't cross the classifier bank's threshold because the classifier
had been trained on LIF-derived synthetic calcium. This was the
moment the readout layer became the constraint.

## P0 / P1 — Honest architectural upgrades

At this point the project had enough scar tissue to know what was
needed. **P0 #1** exposed the full 300-neuron raster alongside the
validated-18 set so the simulator stopped pretending it only
simulated 18 neurons. **P0 #2** wired CeNGEN expression directly
into the dashboard as a 133-gene × 16-category polar chart per
neuron. **P0 #3** added the O₂/CO₂ aerotaxis sensory system
(URX/AQR/PQR via GCY-35/36, BAG via GCY-9) so a whole new phenotype
class — Gray 2004 aerotaxis — became reachable. **P1 #4**
implemented an `ActivityFSM` that reads command-neuron firing rates
directly, bypassing the classifier bank entirely — the architectural
fix for the Tier 1 regression. **P1 #6** added a molecular
visualisation layer (synthetic Ca traces, spatial modulator
diffusion fields). **P1 #8** replaced direct-injection sensory
stubs with five cascade ODEs for ASE, AWC, ASH, AFD, and ALM/PLM.
**P1 #2** and **P1 #4-body** shipped as scaffolds for compartmental
neurons and 95-muscle bodies respectively, with explicit labels that
integration is pending. The upgraded dashboard went live on
rohitravi.com; all six scenarios regenerated in the new format; the
codebase now carries every layer twice (legacy vs upgraded) so the
comparison itself becomes an instrument.

## T0 — The foundation exposed

The Tier 0 validation run on 2026-04-21 did what honest audits
always do. Profiling v3 LIF under a touch stim showed that ALM/AVM
fire cleanly — 1.7 Hz → 78 Hz — but AVA/AVE **decrease** firing on
touch (AVER drops 36 → 28 Hz). The previously validated
*"AVA ablation abolishes reversal"* phenotype runs through the
classifier's trained multi-neuron correlation pattern, not through a
biologically correct ALM → AIB → AVA command cascade. This isn't a
bug in the new `ActivityFSM`; it's a bug in the brain the FSM is
reading. The simulator had been passing a phenotype test by pattern
recognition rather than circuit dynamics. This is the finding that
now sits at the top of the project's open-issues list, and it's the
thing that makes the methodological paper writeable — other people's
simulators almost certainly have the same gap but weren't structured
to expose it.

## T0 resolution — the sign convention behind the gap (2026-04-25)

Four days of diagnostic work resolved the T0 cascade question at the
architectural level rather than through synaptic weight calibration.
The actual mechanism turned out to be the simulator's default
glutamate sign convention: per-presynaptic-neuron NT-sign treats
glutamate edges to iGluR-dominant postsynaptic cells as inhibitory,
which silently breaks the touch reversal cascade. The connectome
already contained a precomputed alternative (CeNGEN-derived
postsynaptic-receptor signs in `W_chem_per_edge`); switching to it
via the constructor flag `use_per_edge_glu_signs=True` makes the
operative cascade fire at +60 Hz on touch — and the operative
cascade turns out to be ALM/AVM → PVC → AVD/AVE → AVA, not the
ALM → AIB → AVA pathway the project had been planning to calibrate
(AIB has zero chemical edges to AVD in this connectome). The April
21 phenotype reproduction was a sign-convention artifact on the
dREV channel; the AVA-ablation behavioral effect persists under
per-edge but in dPIR (mean −0.117, 9/10 negative seeds), suggesting
the FSM/classifier was calibrated to read circuit responses through
a channel whose meaning shifts with the sign convention. PVC over-
activation under per-edge mode is an open question — CeNGEN
expression may diverge from functional dominance at specific
synapses. Full record: `docs/t0_resolution_report.md`. Two
suspects (voltage regime, gap conductance) were cleanly falsified
along the way.

## Where the project sits now

It is no longer exploratory notebooks. It is a connectome-constrained,
modulation-aware, embodied simulator with literature-grounded
transduction ODEs, aerotaxis wiring, a CeNGEN gene-expression ring
per neuron, a compartmental neuron scaffold ready for calibration,
and a 95-muscle innervation table ready for a muscle-driver layer.
It has two genuine validated phenotype reproductions and a known,
documented brain-calibration gap with a three-step fix plan. The
live dashboard at **rohitravi.com/projects/c-elegans-multimodal**
exposes every one of these layers to readers who want to see where
pattern recognition ends and circuit dynamics would have to begin.

FlyWire gave the world a fly connectome. Eon gave the world an
embodied fly. Neither of them documents, on a public page, the
specific place where the classifier is faking circuit biology.
That's what this project now does. The next year's work is turning
that documented gap into closed-loop circuit dynamics that actually
compute the phenotypes they claim.
