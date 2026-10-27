import type { InvestigationEntry } from "./types";

export const ENTRIES: InvestigationEntry[] = [
  {
    n: 1,
    date: "2026-09-16",
    q: "Does whole-brain activity predict where the worm's speed is heading?",
    verdict: "ruled-out",
    tag: "Self-correction: look-ahead",
    tried:
      "An earlier pipeline of mine reported that neural activity improved prediction of forward speed beyond the speed's own history (gain +0.0531 R² over an AR(3) baseline on 38 Atanas animals). I re-audited how it handled time: the lag search was allowed to pick negative lags.",
    found:
      "Every positive branch had picked a lag at which the neural window contained the target frame and up to two frames after it, while the baseline saw only the past. With the neural window restricted to strictly earlier frames, the same representation scores −0.0048 R² [−0.0078, −0.0018].",
    ruled:
      "The claim that the worm's brain 'predicts future speed'. What survives is that current speed can be read out from current activity.",
    why:
      "A lag-sign convention was inverted in one module's comment, and the selection procedure exploited it. The frozen result was correct as arithmetic and wrong as science.",
    next:
      "Every later benchmark is strictly past-only by construction, and the lag grid is part of a frozen spec written before scoring.",
  },
  {
    n: 2,
    date: "2026-09-16",
    q: "Does the connectome predict which neurons respond when one is stimulated?",
    verdict: "kept",
    tag: "Randi functional atlas",
    tried:
      "Randi et al. (2023) stimulated single neurons optogenetically and recorded the whole brain. I scored how well the wiring diagram, and separately the physical distance between the two cells, rank the significant responses in their published map (23,902 pairs, 1,151 significant).",
    found:
      "Distance alone reaches AUC 0.727; connectome weight reaches 0.566. Adding the connectome to distance changes AUC by −0.0005 [−0.0068, +0.0057]. Within the same labelled pair, responses are 1.75× [1.48, 2.11] more frequent in animals where the two cells sit close together.",
    ruled:
      "Connectome weight as an independent predictor of the functional map, at this resolution. Ruled in: a proximity effect that does not depend on neuron identity.",
    why:
      "The same pair of neurons responds more when they happen to be near each other. Identity is fixed; only geometry changed. That points at either light bleeding between nearby nuclei or genuine local coupling.",
    next:
      "Separate the two: build a measurement model of optical contamination and see whether the near-field effect survives it.",
  },
  {
    n: 3,
    date: "2026-09-17",
    q: "Is the proximity effect just light from the stimulated cell leaking into its neighbours?",
    verdict: "partial",
    tag: "Mechanism campaign",
    tried:
      "Froze a preregistration, then split the 110 animals into discovery, confirmation and reserve sets. I fitted an optical kernel from the static image alone (independent of any response) and unmixed each evoked volume into per-nucleus light plus a free halo around the target.",
    found:
      "After unmixing, near receivers still respond more than mid-distance ones: 0.064 [0.037, 0.096] on confirmation (30 contributing animals). On reserve the estimate is similar but its interval reaches below the preregistered floor (0.060 [0.019, 0.105]): inconclusive.",
    ruled:
      "Pure optical contamination, as I modelled it, does not explain everything. No rival (off-target stimulation, electrical coupling, local field) could be separated either.",
    why:
      "Unmixed 'own responders' still showed evoked signal pointing back toward the target, so part of the residual is probably light my model does not capture. The data cannot tell the mechanisms apart.",
    next:
      "The terminal verdict was 'non-identifiable from existing data'. The next job was to check how much of the effect the contamination model really accounts for.",
  },
  {
    n: 4,
    date: "2026-10-04",
    q: "How much of the near-field effect does the optical kernel explain?",
    verdict: "ruled-out",
    tag: "Self-correction: discovery set",
    tried:
      "Re-read the kernel-subtracted test on all three partitions instead of quoting the discovery figure.",
    found:
      "I had reported that the static kernel explains about 30% of the raw near-versus-far difference. That was the discovery set only (29%). On held-out animals it is 7% [−0.3%, 17%] on confirmation and 13% [3%, 25%] on reserve. The discovery set turned out to be the animals with the most stimulations, not a random sample.",
    ruled:
      "The 'about a third' magnitude, and any calibration taken from the discovery set.",
    why:
      "The discovery partition had been sorted by stimulation count before it was frozen. It reads higher than held-out data on every contamination measure.",
    next:
      "Check the other contamination number I had leaned on: the copy ratio.",
  },
  {
    n: 5,
    date: "2026-10-04",
    q: "Does fast signal transfer between neighbours look like contamination or like coupling?",
    verdict: "partial",
    tag: "Estimator calibration",
    tried:
      "Calibrated my copy-ratio estimator on synthetic worlds with known ground truth. The one-way version read 1.6–2.7 under pure contamination (it should read 1), because light also leaks back from receiver to target. I built a two-way estimator and preregistered a replication rule before running it on held-out animals.",
    found:
      "The two-way estimator reads 0.86–1.17 on pure contamination and 4.3–5.2 on simulated electrical coupling. On real held-out animals it reads 0.45–0.52 (confirmation) and 0.38–0.48 (reserve), every interval below 1. The preregistered rule fails.",
    ruled:
      "A coupling-sized signature in the fast transfer. Also ruled out: using the kernel as a magnitude calibration.",
    why:
      "Held-out fast transfer is about half what the static kernel predicts and far below what coupling would produce. The kernel works as an upper bound and a flag, not as a measurement.",
    next:
      "One connectome lead was still standing on the Randi data. Test it with a matched control.",
  },
  {
    n: 6,
    date: "2026-10-04",
    q: "Once near pairs are excluded, does the connectome add anything?",
    verdict: "ruled-out",
    tag: "Matched control",
    tried:
      "On the part of the map where neurons are not near each other, connectome plus distance had beaten distance alone by about +0.03 AUC. I ran a pair-matched control: drop the same number of random pairs and refit, 50 draws.",
    found:
      "The control reproduces most of the gain: observed +0.030 against a control mean of +0.023 [0.017, 0.032]; under a second exclusion rule, +0.028 against +0.027.",
    ruled:
      "The last connectome-specific lead in the stimulation data.",
    why:
      "Removing pairs changes what the score is measured on. A matched random removal produces nearly the same increment, so the connectome was not the cause.",
    next:
      "The Randi map is hard to read without a better measurement model. Start again from spontaneous whole-brain activity, with clean frozen benchmarks and geometry as a required baseline.",
  },
  {
    n: 7,
    date: "2026-10-08",
    q: "Do a neuron's connectome partners help forecast its activity?",
    verdict: "partial",
    tag: "Linear forecasting (A2)",
    tried:
      "Rebuilt the work as a package: content-verified loaders for Atanas 2023 (DANDI) and Kato 2015 (OSF), a frozen 5-fold animal split, and ridge models that forecast each of 89 identified neurons from its own history plus a source set. Source sets: connectome partners, five degree-preserving shuffles, five random sets, and the k nearest physical neighbours (k matched to the connectome).",
    found:
      "The connectome beats its shuffles by a small, real margin (+0.0016, +0.0023, +0.0015 R² at 0.6, 2.4 and 6 s). The nearest physical neighbours match or beat it at every horizon (connectome − neighbours −0.0004, −0.0006 [−0.0015, +0.0002], −0.0016).",
    ruled:
      "Ruled out: a connectome advantage over geometry. Kept: a small advantage over random graphs.",
    why:
      "Connected neurons tend to be physically close, so 'connected' and 'nearby' overlap. Once nearby is offered directly, the wiring adds nothing.",
    next:
      "A linear model might miss what the wiring is for. Give a nonlinear model the same choice.",
  },
  {
    n: 8,
    date: "2026-10-08",
    q: "Does neural activity predict behaviour beyond behaviour's own history?",
    verdict: "kept",
    tag: "Behaviour, past-only (A1)",
    tried:
      "Strictly past-only models: behaviour history alone vs. behaviour history plus neural activity, at 0 to 6 s ahead, on 38 held-out animals.",
    found:
      "Reversal state is the robust case: +0.036, +0.019, +0.033 and +0.035 AUROC at 0, 0.6, 2.4 and 6 s, all intervals above zero. Velocity gains +0.016 R² concurrently and +0.003 at 0.6 s; head curvature only concurrently.",
    ruled:
      "Nothing new. This is a positive control: the pipeline finds the known reversal signal under the past-only rule that killed entry 1.",
    why:
      "Reversal is driven by well-characterised command interneurons, so this is expected. It shows the benchmark can detect real neural information.",
    next:
      "Test the connectome where it should matter most: predicting the effect of an intervention.",
  },
  {
    n: 9,
    date: "2026-10-08",
    q: "Does the connectome predict what happens when AVA is silenced?",
    verdict: "partial",
    tag: "Held-out perturbation (K1)",
    tried:
      "Trained on wild-type Kato recordings, simulated AVA clamped off in each model, and compared the predicted change in 20 other neurons with recordings of AVA::HisCl animals.",
    found:
      "The manipulation check passes (AVA log-variance ratio −1.44 and −2.26). But rank correlations are spread out: connectome 0.39, shuffles 0.29–0.52, random graphs 0.04–0.52, physical neighbours 0.52.",
    ruled:
      "Nothing either way. With 5 animals per arm the benchmark cannot separate the graphs, and I report it as uninformative.",
    why:
      "One intervention, five animals, twenty readout neurons: the spread across null graphs is wider than any plausible connectome effect.",
    next:
      "Return to the large dataset with a learned model.",
  },
  {
    n: 10,
    date: "2026-10-09",
    q: "With a learned graph network, does the real wiring beat a shuffled copy?",
    verdict: "ruled-out",
    tag: "Typed GNN (L2)",
    tried:
      "One fixed architecture (per-neuron recurrent state, separate message maps for chemical and gap-junction edges, a learned gate per edge), with only the edge set varied. Config frozen from an inner check on training animals; 38 held-out animals, 2.4 s ahead.",
    found:
      "Connectome − shuffled = +0.0001 [−0.0007, +0.0009]. Connectome − nearest neighbours = −0.0023 [−0.0036, −0.0007]. Connectome − no edges = +0.0034. GNN − linear on the same graph = +0.0147. The complete graph collapses to R² −0.17.",
    ruled:
      "A connectome-specific advantage in this learned model. Some sparse graph helps a little; which one does not matter.",
    why:
      "Nonlinearity buys about 1.5 points of R²; graph identity buys about nothing. Dense message passing is destructive at this sample size.",
    next:
      "The connectome's learned gates were fairly consistent across folds. Before treating them as candidate important synapses, check whether shuffled graphs give consistent gates too.",
  },
  {
    n: 11,
    date: "2026-10-10",
    q: "Do the learned edge gates point at real, important synapses?",
    verdict: "ruled-out",
    tag: "Gate-stability control",
    tried:
      "Measured how well each graph's learned gates agree across the five folds (mean pairwise correlation), for the connectome, all three shuffles, and the neighbour graph.",
    found:
      "Connectome gates agree at 0.61 (chemical) and 0.78 (gap junction). Shuffled-graph gates agree just as well: 0.38, 0.64, 0.76 (chemical) and 0.74, 0.82, 0.78 (gap). Neighbour-graph gates agree much less (0.20, 0.09).",
    ruled:
      "Reading gate stability as evidence about the wiring. The gates are not mechanistic candidates.",
    why:
      "Stability comes from the training procedure on a sparse graph of a given degree structure, not from the biology of the edges.",
    next:
      "I paused the project here. The one test that could change the picture is the Randi stimulation benchmark with a measurement-aware score (see below).",
  },
];
