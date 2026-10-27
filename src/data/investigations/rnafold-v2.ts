import type { InvestigationEntry } from "./types";

export const ENTRIES: InvestigationEntry[] = [
  {
    n: 1,
    date: "2025",
    q: "Can a 488M-parameter RNA diffusion model run on an 8 GB GPU?",
    verdict: "kept",
    tag: "RNAPro inference",
    tried:
      "RNAPro with MSAs and RibonanzaNet2 features, 2 recycles, 8 diffusion steps, a 500-nt cap, a rebuilt chemical-component cache and expandable CUDA memory segments (pipeline1, pipeline2).",
    found:
      "It runs: five sampled structures for each of 24 test sequences on a single consumer GPU.",
    ruled: "The assumption that this class of model needs a data-centre GPU for inference.",
    why: "Memory, not compute, was the binding constraint, and it can be managed by capping length and trimming the diffusion schedule.",
    next: "Build a scorer that compares the predictions with the deposited PDB structures.",
  },
  {
    n: 2,
    date: "2025",
    q: "How should predictions be scored locally against real structures?",
    verdict: "kept",
    tag: "Sequence-aligned TM-score",
    tried:
      "Download each reference mmCIF from RCSB, reduce every nucleotide to one point (centroid of P, C4′ and C1′), align the predicted chain to the reference sequence, superimpose by Kabsch, and compute TM-score in Python (evaluation, pipeline3 to pipeline5).",
    found:
      "A scorer that compares each predicted nucleotide with the same nucleotide in the experimental structure, independent of chain numbering and length differences.",
    ruled: "Pairing atoms by position, which compares the wrong residues whenever the chains differ in length.",
    why: "A TM-score is only meaningful if residue i in the prediction is matched with residue i in the reference.",
    next: "Add a second track that is not limited by the 500-nt cap.",
  },
  {
    n: 3,
    date: "2025",
    q: "Can the whole PDB be searched fast enough to use as a template source?",
    verdict: "kept",
    tag: "Template database",
    tried:
      "Parse every RNA structure in the PDB and index it with GPU FAISS (pipeline4).",
    found:
      "9,564 RNA structures parsed in about an hour, with a GPU FAISS index for similarity retrieval inside the pipeline.",
    ruled: "Retrieval cost as a bottleneck for a template-based approach.",
    why: "Template folding is only practical if the search over solved structures is cheap.",
    next: "Build the folder on top of the retrieval index.",
  },
  {
    n: 4,
    date: "2025",
    q: "Can a non-neural folder build full-length structures from templates?",
    verdict: "kept",
    tag: "Template folder",
    tried:
      "45 stages: retrieval, alignment, coordinate transfer, weighted consensus fusion with pre-fusion superposition, agreement-based recycling, gap repair and refinement, each writing an inspectable artifact (pipeline5).",
    found:
      "The folder ran end to end on all 24 evaluated targets, and each stage can be benchmarked and switched off on its own.",
    ruled: "The need for a neural model to produce a full-length structure.",
    why: "When a related RNA has already been solved, its coordinates carry the fold; the remaining work is alignment, fusion and repair.",
    next: "Benchmark both tracks with each target's close homologs held out of the bank.",
  },
];
