import type { InvestigationEntry } from "./types";

export const ENTRIES: InvestigationEntry[] = [
  {
    n: 1,
    date: "Sep 21",
    q: "Can the test be run without my seeing the answer first?",
    verdict: "kept",
    tag: "Blinding",
    tried:
      "Before computing anything, I permuted the functional profiles of all 13,805 co-registered presynaptic neurons with a stored random key, wrote the analysis plan, and stored its SHA-256 hash. The code that loads the true labels refuses to run unless the plan's hash still matches.",
    found:
      "The plan was committed on Sep 21 and the labels were unblinded on Sep 22. Every later extension got its own predictions file, hashed and committed before its true-label run.",
    ruled:
      "Choosing the analysis after seeing which version gives a positive.",
    why:
      "Dendritic clustering has many reasonable definitions (window size, tier, synapse choice, similarity measure). Without a frozen plan, one of them would come out significant by chance.",
    next:
      "Calibrate the planned tests on blind data: do they find clustering when there is none, and do they find it when it is planted?",
  },
  {
    n: 2,
    date: "Sep 21–22",
    q: "Does the standard test report clustering when the labels carry only geometry?",
    verdict: "kept",
    tag: "Planted-geometry check",
    tried:
      "Replace each presynaptic neuron's functional label with its soma position (no function at all), then run the usual analysis: compare near pairs (≤ 10 µm apart along the dendrite) with far pairs (40–200 µm) and test against a shuffle of presynaptic identities within each postsynaptic cell.",
    found:
      "The shuffle reported 'clustering' at p = 0.001: d = 0.095 on pyramidal dendrites and d = 0.139 on interneuron dendrites. A geometry-stratified shuffle flagged it too. Only the regression that adjusts for presynaptic soma distance, lateral separation and axon co-travel (β_adj) passed, with |z| ≤ 0.74.",
    ruled:
      "The naive within-cell shuffle as a test of functional placement. It reports geometry as function. I demoted the stratified shuffle and the raw path-vs-Euclidean contrast to descriptive before unblinding.",
    why:
      "Neurons whose somata sit near each other send axons through the same neuropil and share retinotopy, so their synapses land near each other on a dendrite. Any label correlated with soma position inherits that.",
    next:
      "Check that the adjusted test is not simply blind: false-positive rate and power on planted effects.",
  },
  {
    n: 3,
    date: "Sep 22",
    q: "Can the geometry-adjusted test still find a real effect?",
    verdict: "kept",
    tag: "Calibration",
    tried:
      "Run β_adj on 100 fresh blind keys to get its false-positive rate, then plant placement effects of known size and count how often it recovers them.",
    found:
      "False-positive rates were 0.03–0.08 at α = 0.05, and β_adj had at least 0.88 power at d = 0.10 in all four confirmatory tests (1.00 on interneuron dendrites).",
    ruled:
      "The worry that the adjustment removes everything, signal included.",
    why:
      "A null result is only worth reporting if the test could have produced a positive.",
    next:
      "Unblind and run the four pre-registered tests.",
  },
  {
    n: 4,
    date: "Sep 22",
    q: "Do functionally similar inputs land near each other, beyond geometry?",
    verdict: "ruled-out",
    tag: "Primary test",
    tried:
      "The four confirmatory tests: pyramidal and interneuron dendrites, each with two similarity measures from the digital-twin model of the recorded activity (receptive-field location and a 512-d tuning feature vector).",
    found:
      "The raw clustering is real: d = 0.056 (p = 0.011) for RF location on pyramidal dendrites and d = 0.032 (p = 0.012) for the tuning feature on interneuron dendrites. It is smaller than what soma position alone produced in the planted check. After adjustment all four are null (Holm p = 1.0 for each).",
    ruled:
      "Functional placement beyond presynaptic geometry at the strict tier, down to upper bounds of d ≤ 0.034–0.103.",
    why:
      "Similarity also falls off smoothly out to about 200 µm rather than as a sharp ≤ 10 µm cluster, which is what a broad geometric gradient looks like.",
    next:
      "The weakest bound (pyramidal RF, 0.103) is too loose to say much. Throw the robustness battery at it and use the larger broad tier.",
  },
  {
    n: 5,
    date: "Oct 4",
    q: "Is the null an artifact of one analysis choice?",
    verdict: "ruled-out",
    tag: "Robustness battery",
    tried:
      "75 pre-registered runs varying the proofreading tier, synapse choice, label quality, depth matching, synapse size, 5 and 20 µm windows, five postsynaptic subclasses, three dendritic compartments, merge controls, dropping hub axons, orientation similarity and response correlation.",
    found:
      "None leaked geometry on the planted check and none was Holm-significant (smallest Holm p = 0.279). The broad tier, with 8,737 pyramidal near pairs, tightened the pyramidal RF bound from d ≤ 0.103 to d ≤ 0.019.",
    ruled:
      "A placement effect hiding in a particular tier, window, cell type or compartment, except where power is too low to say (the apical tuft).",
    why:
      "If an effect existed at d ≈ 0.1, some of these slices should have shown it; their bounds sit well below that.",
    next:
      "All of this uses the same hand-checked functional labels. Test on presynaptic neurons that were never used.",
  },
  {
    n: 6,
    date: "Oct 4",
    q: "Does the result hold on presynaptic neurons that were never analysed?",
    verdict: "ruled-out",
    tag: "Held-out tier (A11)",
    tried:
      "Build pairs only from 12,522 automatically co-registered presynaptic neurons, disjoint from every label used before, on the same postsynaptic cells.",
    found:
      "Raw clustering appeared in 6 of 6 tests (p ≤ 0.022), and the pyramidal RF raw effect was larger than in the main set (d = 0.119), not smaller as I had predicted. Adjusted, all six were null (smallest Holm p = 0.50). I had also predicted the pyramidal RF bound would come in under 0.05; it was 0.074, so that prediction failed.",
    ruled:
      "The null as a property of one label set.",
    why:
      "The pyramidal RF point estimate was again small and positive (+0.035, z = 1.73), as in the strict tier (+0.027). Not significant, but it is the place an effect could still be.",
    next:
      "Ask the question in a way that does not depend on my geometry model.",
  },
  {
    n: 7,
    date: "Oct 4",
    q: "At matched 3-D distance, does similarity fall with distance along the dendrite?",
    verdict: "ruled-out",
    tag: "Path vs Euclidean",
    tried:
      "Compare synapse pairs that are equally far apart in space but differ in how far apart they are along the dendrite. Input-specific placement predicts the along-dendrite distance matters; geometry predicts only 3-D distance does.",
    found:
      "No fall-off along the dendrite in 6 of 6 tests (|z| ≤ 1.11).",
    ruled:
      "Dendrite-specific placement that my soma-geometry model might have absorbed.",
    why:
      "This test does not use the geometry regression at all, so it checks the result against geometry I did not model.",
    next:
      "Test the one recent claim of structured input groups on dendrites in this same volume.",
  },
  {
    n: 8,
    date: "Oct 4",
    q: "Are the Sarup & Boahen axon groups functional ensembles?",
    verdict: "ruled-out",
    tag: "Published axon groups",
    tried:
      "Sarup & Boahen (2026) reported axons that repeatedly co-cluster on the same dendritic branches, with no functional data. I rebuilt their groups from co-registered axons (1,421 pair-groups against 92,197 comparison pairs) and asked whether group members are functionally more similar.",
    found:
      "No function beyond geometry: adjusted |z| ≤ 1.08 for RF, feature and response similarity; raw contrasts d ≤ 0.04. Their within-group ordering was above chance (733 consistent groups vs about 685 expected, p = 0.006), but the excess sits in pairs whose order also matches cortical depth (55% consistent, vs 48% in the rest), which geometry already explains.",
    ruled:
      "The reading of these anatomical groups as co-tuned ensembles. Groups of three or more axons could not be tested: too few carried functional labels.",
    why:
      "A grouping found from anatomy alone inherits the same soma-geometry structure the naive shuffle picks up.",
    next:
      "Check the two cases closest to the imaging literature: spine-only synapses and distal displacement of inputs.",
  },
  {
    n: 9,
    date: "Oct 4",
    q: "Is there an effect on spines, or in how far inputs sit from the soma?",
    verdict: "ruled-out",
    tag: "Spine-only and distal displacement",
    tried:
      "Restrict to synapses where both inputs land on spines, as in spine imaging. Separately, test the prediction (Landau, Sabatini & Clopath) that distal inputs are more displaced in receptive field from the cell's own.",
    found:
      "Spine-only: raw pyramidal RF d = 0.041 (p = 0.004), adjusted null in all three measures, RF upper bound 0.035. Distal displacement: a raw basal slope at z = 3.56 that soma geometry explains entirely (adjusted z = 0.04); none on apical dendrites. My first run of the planted-geometry check here divided by the wrong null and inflated its z; I fixed the scale before reading the corrected numbers, and the true-label values did not change.",
    ruled:
      "Spine-restricted placement and distal displacement beyond geometry.",
    why:
      "Both raw effects exist and match what the literature describes. Both are what presynaptic geometry predicts.",
    next:
      "Everything so far is one animal with model-derived labels. Go to a second animal with measured activity.",
  },
  {
    n: 10,
    date: "Oct 4",
    q: "Does it replicate in a second animal with measured activity?",
    verdict: "partial",
    tag: "Second animal (V1DD)",
    tried:
      "The Allen V1 Deep Dive volume, using measured ΔF/F correlations of 193 co-registered axons across three stimulus sets. Path distance needs authenticated skeleton access, so I used Euclidean windows, and checked first that the same Euclidean version is null on MICrONS (6 of 6, |z| ≤ 0.98).",
    found:
      "Raw clustering for natural-movie correlations: d = 0.087 (p = 0.022). The naive shuffle again flagged planted soma geometry (d = 0.201, p = 0.002). Adjusted: null for all three stimuli (smallest Holm p = 0.38).",
    ruled:
      "Nothing new on its own: with 958 near pairs the smallest detectable effect is d ≈ 0.1–0.2, so this is agreement, not a tight bound.",
    why:
      "It does replicate the methods finding in a second animal: the naive test calls geometry function there too.",
    next:
      "Open: a path-distance version in V1DD, tuning-based V1DD labels, and an apical-tuft test with enough pairs.",
  },
];
