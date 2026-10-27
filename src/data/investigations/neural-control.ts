import type { InvestigationEntry } from "./types";

// Every number here is traced to a result file in docs/stage1/neural-control_FACTS.md.
export const ENTRIES: InvestigationEntry[] = [
  {
    n: 1,
    date: "2026-09-21",
    q: "Does the published connectome model hold up when I rerun it?",
    verdict: "kept",
    tag: "Reproduction",
    tried:
      "Score Creamer, Leifer and Pillow's (2025) published connectome-constrained model, their fully connected model and their shuffled-connectome control on their own shipped test maps, with their own metric code and no refitting.",
    found:
      "The direction reproduces: connectome-constrained 0.140 against 0.040 for a shuffled connectome (p < 0.001). The headline fraction does not: I get 81% of train-to-test reproducibility, not the reported 92%. And the connectome-constrained model does not significantly beat the model with no wiring constraint at all (0.140 vs 0.120, p = 0.29).",
    ruled:
      "That the wiring diagram is irrelevant. A real connectome beats a shuffled one.",
    why:
      "The shuffled control is the key comparison, and it holds. But a fully connected model doing about as well means the constraint is not what carries most of the prediction.",
    next:
      "If the wiring is not doing most of the work, what is? Try a model that never sees the connectome and only knows each neuron's name.",
  },
  {
    n: 2,
    date: "2026-10-04",
    q: "Can neuron identity alone beat the connectome model?",
    verdict: "kept",
    tag: "Identity-indexed model",
    tried:
      "One shared recurrent core, with a learned table row per named neuron shared across animals. No connectome input. Trained on 80 animals, scored zero-shot on 30 animals it never saw, with the authors' estimator. Control: the same model with names shuffled independently in each animal.",
    found:
      "0.223 (neuron level) and 0.248 (class level, left/right merged) against 0.105 for the connectome-constrained model and 0.153 for the fully connected one on the same animals. Shuffling names drops it to about zero (within ±0.02 across 10 runs). Hiding every pre-stimulus observation leaves it unchanged (0.225 / 0.250).",
    ruled:
      "That the gain comes from the animal's ongoing state. It comes from a learned, state-independent map keyed by identity.",
    why:
      "Knowing which neuron is which, and learning its typical response from other animals, captures more of the measured causal map than the wiring diagram does.",
    next:
      "Identity tables cannot say anything about a neuron the model never trained on. Can expression or wiring stand in for the missing row?",
  },
  {
    n: 3,
    date: "2026-10-04",
    q: "For never-trained neurons, does gene expression beat wiring as a stand-in?",
    verdict: "partial",
    tag: "Self-correction",
    tried:
      "Hold out each of 80 neuron classes once (5 folds) and predict its causal map from CeNGEN expression or connectome features. First with a small network mapping features to the table row, then with a model-free version: average the measured maps of the most similar training neurons.",
    found:
      "The first version said expression wins: 0.106 vs 0.076 (fallback 0.013), in 5 of 5 folds. The model-free version, run the same day, gave a tie: 0.137 (expression) vs 0.134 (connectome), against 0.031 for random features. Combining both gives 0.147.",
    ruled:
      "My own 'expression beats connectome' claim. The gap came from how the network used connectome features, not from a difference in information.",
    why:
      "A result that flips under a second, simpler operationalization was a property of the instrument. What survives is that expression and wiring predict an unseen neuron's map about as well as each other, both far above random.",
    next:
      "The connectome carries real information, but less than identity. Where in the map does it carry it?",
  },
  {
    n: 4,
    date: "2026-10-06",
    q: "Where in the causal map does the connectome help?",
    verdict: "kept",
    tag: "Locality",
    tried:
      "Split map entries by the physical distance between the responding neuron and the stimulated one (per-animal NeuroPAL positions), and rescore Creamer's models on their own test maps. Control: a shuffle of the anatomical connectome that preserves distance.",
    found:
      "Within 10 µm the connectome-constrained model scores 0.214 against −0.005 for the shuffled connectome. Beyond 15 µm it is 0.008 against 0.009. The near-field effect beats the distance-preserving shuffle (p = 0.002).",
    ruled:
      "The connectome as a predictor of long-range causal influence. Its advantage over a shuffled connectome lives entirely in close neighbours.",
    why:
      "This reconciles two published readings: anatomy predicts propagation poorly overall (Randi et al. 2023) and a connectome model captures much of the reproducible signal (Creamer et al. 2025). They hold in different spatial regimes.",
    next:
      "My first draft also said the far field was barely reproducible at all. That depended on how strong a model I had. Push the identity model to its ceiling and look again.",
  },
  {
    n: 5,
    date: "2026-10-06",
    q: "How much of a neuron's response is shared across worms?",
    verdict: "partial",
    tag: "Individuality",
    tried:
      "Model-free: compare single stimulation events for the same target within one animal and across animals, subtracting different-target pairs as the baseline. Controls for gain, recording date, ID confidence, whether the target was actually activated, and distance to the target.",
    found:
      "The shared fraction (stereotypy index) is 0.37 [0.29, 0.48] at the neuron level, so roughly 60% of the reliable, target-specific response is individual. Merging left/right pairs raises the shared fraction to 0.59 [0.42, 0.87].",
    ruled:
      "A fully stereotyped causal map. Part of what any shared-identity model misses is individual, not noise.",
    why:
      "It puts a bound on what a species-level model can reach, but the estimate moves with the resolution (left/right merged or not), so I treat it as a range rather than a number.",
    next:
      "If much of the response is individual, how close to the shared ceiling can a shared model get?",
  },
  {
    n: 6,
    date: "2026-10-07",
    q: "Did the model really reach the noise ceiling?",
    verdict: "ruled-out",
    tag: "Self-correction",
    tried:
      "Train the identity model with an event-level response loss, then estimate the noiseless ceiling of the test map from split-half reliability. Control I added afterwards: the same model with shuffled identities, scored with and without the pre-stimulus activity it was given.",
    found:
      "Before: 0.413 on the fixed split (94% of a 0.44 ceiling) and 0.597 in cross-validation over 110 animals (about 100% of 0.586). After: the shuffled-identity model scored 0.215 with real pre-stimulus activity and 0.004 without it. The model was predicting each event's ongoing dynamics, not the stimulus effect.",
    ruled:
      "Both ceiling headlines. They were state leakage, not causal prediction.",
    why:
      "The stimulus-triggered map has no sham control, so a model that sees the pre-stimulus trace can score by forecasting the trace. Every number after this one is scored state-free: an impulse response from rest.",
    next:
      "Rescore everything state-free and see what is left.",
  },
  {
    n: 7,
    date: "2026-10-07",
    q: "State-free, how much of the reproducible map can identity predict?",
    verdict: "kept",
    tag: "Ceiling, corrected",
    tried:
      "Score the identity model with every pre-stimulus observation hidden, on the fixed 80/30 split and in 5-fold cross-validation over all 110 animals. Controls: shuffled identities, and Creamer's connectome-constrained model on the same targets.",
    found:
      "Cross-validation: 0.451, 77% of the 0.586 ceiling, against 0.001 with shuffled names and 0.127 for the connectome-constrained model (whose weights were fit on many of these animals, which favours it). Fixed split: 0.278, 63% of the 0.44 ceiling, against 0.105 for the connectome-constrained model. On the fixed split, for responders beyond 15 µm, the identity model still predicts 0.125, where the connectome-constrained model gets 0.000.",
    ruled:
      "That the far field holds no reproducible, identity-specific structure. It does; the connectome just does not see it.",
    why:
      "With more training animals per stimulated pair the fraction of ceiling rises from 63% to 77%, so the remaining headroom looks limited by data rather than by model capacity.",
    next:
      "The model predicts evoked responses. Does that knowledge also help forecast what a worm does on its own, with nothing stimulated?",
  },
  {
    n: 8,
    date: "2026-10-10",
    q: "Does knowing the causal map help forecast spontaneous activity in new worms?",
    verdict: "ruled-out",
    tag: "Forecasting",
    tried:
      "Forecast held-out, freely moving worms (Atanas et al. 2023) with class-indexed linear models, adding a drive term weighted by the measured causal map, by the connectome, or by a label-shuffled map. Then recurrent forecasters pretrained on the causal model, on unstimulated data, or with shuffled names.",
    found:
      "No drive term improved forecast skill by more than 0.004 at any horizon or training-set size, and the causal map, the connectome and the shuffled map were indistinguishable. Pretraining helped only generically: causal, unstimulated and shuffled-name pretraining did about as well as each other. One positive survived: the causal map predicts which neurons co-fluctuate spontaneously beyond anatomy and distance (partial ρ 0.085, p = 0.01), with no directional information.",
    ruled:
      "Perturbation knowledge as a forecasting aid for natural dynamics, in this linear, class-level, 2 Hz form.",
    why:
      "What is forecastable in a new worm is mostly each neuron's own mean reversion plus a few global modes. The causal map shares a symmetric co-activity pattern with spontaneous activity, but nothing that carries forward in time at this sampling rate.",
    next:
      "Back to anatomy. If the synaptic connectome helps only up close, is it really synapses that matter there, or just physical neighbourhood?",
  },
  {
    n: 9,
    date: "2026-10-14",
    q: "Is it synapses, membrane contact, or cell-body distance that predicts the map?",
    verdict: "partial",
    tag: "Geometry",
    tried:
      "Put three anatomical layers in competition: cell-body distance, nerve-ring membrane contact from published EM contact data, and synapses. Partial correlations of each with the causal map given the others, against a distance-preserving null and a degree-preserving rewiring of contact.",
    found:
      "Cell-body proximity keeps a large effect after contact and synapses are accounted for (partial 0.29). Contact adds a small but real signal beyond distance and synapses (0.040, p = 0.003). Synapses add nothing once contact is known (−0.015).",
    ruled:
      "Synapses as the predictive anatomical layer, at this resolution. Contact subsumes them.",
    why:
      "Cell-body proximity looks like a measurement layer (optical crosstalk and neighbourhood effects) rather than wiring. Where neurites actually touch is the more informative anatomy, though the effect is small and rests on two EM contact maps.",
    next:
      "The decisive test is a dynamical model whose coupling is restricted to contacting pairs instead of synaptic pairs. It needs a full refit of the Creamer model, which is open.",
  },
];
