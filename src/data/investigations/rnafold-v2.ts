import type { InvestigationEntry } from "./types";

// Every number here is traced to a repo file or notebook output cell in docs/stage1/rnafold-v2_FACTS.md.
// Dates: 2026-03-07 is the timestamp in the pipeline2 / evaluation notebook outputs; the later notebooks
// were committed on 2026-04-17 and carry no earlier timestamps.
export const ENTRIES: InvestigationEntry[] = [
  {
    n: 1,
    date: "2026-03-07",
    q: "Can a 488M-parameter RNA diffusion model run on an 8 GB GPU?",
    verdict: "kept",
    tag: "RNAPro inference",
    tried:
      "RNAPro with MSAs and RibonanzaNet2 features, no templates, 2 recycles, 8 diffusion steps, a 500-nt cap and expandable CUDA memory segments, on the 28 competition test sequences (pipeline1, pipeline2).",
    found:
      "24 of 28 targets produced 5 samples each. Two sequences (4,640 and 1,460 nt) exceeded the cap and two ran out of GPU memory. Model ranking scores averaged 0.030.",
    ruled: "That the model could not be run locally at all.",
    why: "The length cap and memory limit decide which targets can be scored, so they define the local evaluation set.",
    next: "Score the samples against the deposited PDB structures.",
  },
  {
    n: 2,
    date: "2026-03-07",
    q: "Can TM-align score the predictions against the PDB structures?",
    verdict: "ruled-out",
    tag: "TM-align harness",
    tried:
      "Download the references from RCSB, convert mmCIF to PDB, compile TM-align from source, and score all 130 sample files (evaluation.ipynb).",
    found:
      "Zero scores. TM-align could not parse any of the 125 converted predictions, and one reference (9I9W) failed conversion because its chain ID is too long for PDB format. The result CSVs record 125 empty rows and 5 failures.",
    ruled: "This conversion route as a scorer.",
    why: "Without a working scorer there is no local number at all.",
    next: "Compute TM-score directly in Python from the coordinates.",
  },
  {
    n: 3,
    date: "Mar–Apr 2026",
    q: "Does a Python TM-score on backbone atoms give a usable number?",
    verdict: "ruled-out",
    tag: "Atom-level scoring",
    tried:
      "Extract P, C4′ and C1′ atoms, superimpose by Kabsch, compute an approximate TM-score, keep the best of five samples per target (pipeline3).",
    found:
      "A best-of-five mean of 0.093, but atoms were paired by position, so a 30-nt prediction was compared with the first atoms of a much larger reference.",
    ruled: "Position-based pairing. The 0.093 is an artefact and is not reported as a result.",
    why: "A score is only meaningful if each predicted residue is compared with the same residue in the reference.",
    next: "Move to one point per nucleotide and pair residues by sequence.",
  },
  {
    n: 4,
    date: "Mar–Apr 2026",
    q: "What does RNAPro score once residues are paired correctly?",
    verdict: "kept",
    tag: "Sequence-matched scoring",
    tried:
      "One pseudo-atom per nucleotide (pipeline4), then a local sequence alignment of each prediction to its reference before superposition (pipeline5). One sample per target.",
    found:
      "Mean TM-score 0.072, median 0.043, best 0.238 (9G4J) over 24 targets, up from 0.059 with position-based pairing. Longer targets score higher.",
    ruled: "Nothing about RNAPro's inputs: MSAs and RibonanzaNet2 were never ablated.",
    why: "This is the only held-out local number in the repo.",
    next: "Try a template-based approach that does not depend on the length cap.",
  },
  {
    n: 5,
    date: "Mar–Apr 2026",
    q: "Can a non-neural template folder over the PDB do better?",
    verdict: "partial",
    tag: "Pipeline5 template folder",
    tried:
      "45 stages over 9,564 PDB RNA structures: FAISS retrieval, alignment, coordinate transfer, weighted fusion, gap repair, and six refinement stages each benchmarked on its own.",
    found:
      "The folder runs end to end with inspectable artifacts. But top-1 retrieval returned a sequence-identical entry for every target, usually the target's own structure, and the query was the reference sequence itself.",
    ruled: "Every Pipeline5 score as an estimate of prediction quality. They measure coordinate transfer from a copy of the answer.",
    why: "A template bank that contains the test structures cannot tell you how the method does on new RNAs.",
    next: "Rerun with each target and its close homologs removed from the bank. Not yet done.",
  },
];
