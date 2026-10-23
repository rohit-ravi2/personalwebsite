export const tools = [
  {
    slug: "connectome",
    title: "C. elegans connectome",
    kind: "Interactive visualization",
    project: "c-elegans-multimodal",
    description:
      "Inspect neuron classes and connections in a small nervous system.",
    scope:
      "Displays a published connectivity graph. Node categories and connectivity do not establish physiological function.",
  },
  {
    slug: "anatomy",
    title: "Inside a neural model",
    kind: "3D model explorer",
    project: "substrate-anatomy",
    description:
      "Inspect modeled neurons, channels, receptors, and integration status.",
    scope:
      "Requires WebGL. This is an inspection interface for a computational model, not a validated reconstruction of all cellular biology.",
  },
  {
    slug: "neural-states",
    title: "Neural-state perturbations",
    kind: "Precomputed simulation results",
    project: "anesthesia-pipeline",
    description:
      "Compare modeled dose responses and published pharmacology across species.",
    scope:
      "Loads saved simulation ensembles. Reduced model activity is not evidence of unconsciousness, and this is not a dosing tool.",
  },
  {
    slug: "specification",
    title: "Development and wiring",
    kind: "Interactive evidence ledger",
    project: "specified-information",
    description:
      "Explore how contact opportunities and developmental constraints shape connectivity.",
    scope:
      "An interactive presentation of the investigation and its controls; it does not run new experiments.",
  },
  {
    slug: "resistance",
    title: "Mutation trajectories",
    kind: "Precomputed model predictions",
    project: "amr-mutation-trajectory",
    description:
      "Compare ranked next-mutation predictions with observed resistance trajectories.",
    scope:
      "Loads saved transition predictions rather than generating novel mutations. Research exploration, not antibiotic-selection advice.",
  },
  {
    slug: "methylation",
    title: "DNA mixture playground",
    kind: "Browser inference",
    project: "cfdna-cancer-detection",
    description:
      "Change a simulated mixture and inspect the model's methylation-based predictions.",
    scope:
      "A small research demonstration with synthetic inputs. It is not a diagnostic test or validated clinical service.",
  },
  {
    slug: "trials",
    title: "Trial-outcome model",
    kind: "Browser inference",
    project: "clinical-trial-failure-prediction",
    description:
      "Explore how an illustrative model responds to trial features.",
    scope:
      "Uses a demonstration model and example inputs. Outputs do not establish the probability of a real trial succeeding.",
  },
  {
    slug: "art",
    title: "Artwork classifier",
    kind: "Browser inference",
    project: "ai-vs-human-art",
    description:
      "Upload an image and inspect a small model's source and style predictions.",
    scope:
      "The image is processed in your browser. This smaller demo model is separate from the full research classifier; a prediction does not establish an artwork's provenance.",
  },
  {
    slug: "movement",
    title: "Humanoid motion studies",
    kind: "Recorded simulation",
    project: "robotic-movement",
    description:
      "Watch jumping and throwing trajectories derived through physics and optimization.",
    scope:
      "Video replays of MuJoCo experiments. No live optimizer or physical robot is controlled from this page.",
  },
] as const;
