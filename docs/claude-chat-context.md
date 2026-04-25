# C. elegans multi-modal simulator — Claude.ai Projects context pack

**Purpose of this file:** Upload to a Claude.ai "C. elegans simulator"
Project as knowledge. Every chat in that project will inherit this
context, so Rohit can discuss ideas without re-explaining the setup.

**Maintenance:** Live doc. When the simulator's state changes
materially (new phenotype validated, brain recalibrated, new paper
angle), update this file and re-upload.

Current as of **2026-04-21**, post-T0 validation run.

---

## 1 · One-paragraph identity

A connectome-constrained, modulation-aware, embodied *C. elegans*
digital twin. 300-neuron Brian2 LIF brain derived from the Cook 2019
hermaphrodite wiring + Loer & Rand 2022 neurotransmitter table,
driving a 20-segment MuJoCo body through a 5-state behavioural FSM.
Nine peptidergic/monoaminergic modulators (FLP-11, FLP-1, FLP-2,
NLP-12, PDF-1, 5-HT, DA, TA, OA) with CeNGEN-derived releaser +
receptor assignments and literature-grounded diffusion lengths.
Published at https://www.rohitravi.com/projects/c-elegans-multimodal.
Code at https://github.com/rohit-ravi2/personalwebsite (private).

## 2 · Current thesis

*Local connectomic topology in C. elegans is largely self-
determining, while gene expression acts as a modulatory refinement
layer rather than a primary architect of wiring.*

Evidence supporting this from the analyses side:
- **GNCA** (Graph Neural Cellular Automata) predicting synaptic
  strength: r = 0.987 on held-out edges. Shuffled-expression
  control: r = 0.934 (topology carries the signal). Graph-only:
  0.861. Expression-only: 0.27.
- **Neuron-role classifier** 92.7% accuracy (sensory/inter/motor)
  from expression alone.
- **CCA** between expression space and motif space: ρ = [0.875,
  0.836, 0.809].
- **Honest null**: PhaseD hybrid gene-graph LSTM showed
  `gene_causal_by_shuffle: False` — dynamics prediction doesn't
  improve with gene expression on this dataset.

## 3 · Architecture — current live state

Four layers, each with a 'legacy' and an 'upgraded' path:

### Brain
- **Default**: `LIFBrain` (Brian2, 300 neurons, chemical + gap
  synapses, W_syn=0.8 mV tuned for balanced regime).
- **Alternative**: `GradedBrain` (T1a Kunert-Graf 2014, σ(V)
  continuous release + L-type Ca plateau on 14 command neurons).
- **Scaffold for v4**: 15-neuron compartmental pool (soma+dendrite
  + L-type Ca + slow h-inactivation) in `compartmental_neurons.py`.
  Compiles in Brian2; plateau parameters not yet calibrated against
  Gao & Hobert 2020 voltage-clamp.

### Sensory
- **Default**: `sensory_injection.py` — direct Poisson spike trains
  into target neurons per preset ("touch_anterior" → ALM/AVM @ 180 Hz).
- **Upgraded (P1 #8)**: `sensory_transduction.py` — five ODE
  cascades (ASE GCY-22/cGMP/TAX-4, AWC ODR-10/Gα_i/cGMP-drop
  OFF-cell, ASH OSM-9/OCR-2+TRPA-1, AFD GCY-8/18/23, ALM MEC-4/10).
  Each cascade has τ_rise, τ_decay, and adaptation constants with
  literature citations. Selected via `sensory_mode="transduction"`.
- **P0 #3**: `AerotaxisSensory` adds URX/AQR/PQR (high-O2) + BAG
  (O2-off + CO2-on) via Gray 2004 / Zimmer 2009. Drives an
  `aerotaxis` scenario with a 7% → 21% O2 linear gradient.

### FSM (behavioural state)
- **Default**: `BehavioralFSM` driven by 8-event Atanas-trained
  classifier bank (logistic regression on 18-neuron strict
  cross-worm intersection readout, AUC 0.75-0.90 per event).
- **Upgraded (P1 #4)**: `ActivityFSM` reads command-neuron firing
  rates directly (AVA/AVE for reversal, AVB/PVC for forward,
  SMDV/RIV for omega, RIS for quiescence, NSM for feeding dwell)
  and triggers on z-scored deviation from a 20 s EMA baseline,
  with 2 s warmup window. Selected via `fsm_mode="activity"`.

### Body
- **Default**: `wormbody.xml` — 20 segments, 19 hinge actuators,
  CPG-driven per-state.
- **Scaffold for v4**: `wormbody_v2.xml` — same skeleton + 80
  quadrant (DL/DR/VL/VR) actuators + sites, plus
  `motor_innervation.json` (540 sparse neuron→muscle weights from
  White 1986 + Cook 2019 + Pereira 2015 rules). Quadrant actuators
  are position-typed on hinge joints (MuJoCo semantics bug — they
  should be `<muscle>` on `<tendon><spatial>`); muscle-driver
  code that reads motor rates and writes to the 80 actuators
  doesn't exist yet.

### Dashboard / UI
Built in React + Astro + Tailwind. Lives at
`src/components/react/CelegansDashboard.tsx` (~3.7k LOC). Renders:
- 20-seg body with state-coloured glow + trail + D/V compass
- 3D 300-neuron brain (rotatable, NT-filter chips, search)
- Raster view (clickable, NT-colored)
- 2D arena (chemotaxis food patch OR aerotaxis O2 gradient)
- 9-modulator concentration strip (heatmap + line overlay)
- FSM timeline with event-fire carets + stim labels
- Event-probability line plot with time ticks
- CeNGEN gene-expression polar chart on locked-neuron popover
- Synthetic Ca trace + ego-network + firing-rate history on
  locked neuron
- Live stats + circuit badges + activity-FSM z-role badges
- URL-hash sharable links, CSV/PNG export, ?-help overlay

## 4 · What's validated (honest)

**AVA ablation abolishes reversal under touch.** ΔREV = −0.57 ± 0.37
across 3 seeds, all three correctly negative. Clears 2·SEM bar.
Genuine Chalfie 1985 reproduction. **BUT** — see §5 below.

**RIS / Turek 2016 quiescence pathway.** ΔQUI = −0.24 ± 0.33
across 3 seeds (2/3 negative). Directionally consistent but not
statistically robust at 20 s runs.

Pipeline works end-to-end: stim → brain → classifier → FSM → body →
MuJoCo → state distribution → dashboard JSON.

## 5 · What the Tier-0 validation run revealed (2026-04-21)

**Critical finding:** profiling v3 LIF with touch stim showed
AVA/AVE *decrease* firing on touch (AVER drops 36→28 Hz) rather
than producing the expected reversal burst. ALM/AVM sensory cells
fire cleanly (1.7 → 78 Hz). Top ΔHz responders are head motor
neurons (SIB/RIV/SMD), not command interneurons.

**Interpretation:** the AVA/Chalfie phenotype reproduction runs
through the CLASSIFIER's trained multi-neuron correlation pattern,
not through biologically-correct AVA command cascade. My
ActivityFSM reading AVA directly therefore fails to trigger
reversals (touch-activity scenario: QUIESCENT=91%).

**This is a publishable methodological finding.** For the paper
methods section: *"Connectome-constrained LIF simulators that pass
classifier-based phenotype reproductions may do so via distributed
pattern recognition rather than command-neuron cascades. Directly
reading command-neuron activity can serve as a falsification test
for whether the simulator has captured circuit-level dynamics vs.
only readout-level statistics."*

Full T0 report: `scripts/brain/artifacts/t0_run_report.md`.

## 6 · Roadmap (prioritised)

### Tier 2 — unblock ActivityFSM + complete scaffolds (~2 mo)
1. **Synaptic weight calibration** so ALM→AIB→AVA cascade actually
   depolarises AVA (target: AVA baseline 2-5 Hz, during-touch ≥20 Hz).
2. **Compartmental integration**: `LIFBrain.replace_neurons_with_
   compartmental([...])` that substitutes the 15 compartmental
   models, re-wires synapses to soma compartment. Calibrate against
   Gao & Hobert 2020.
3. **Muscle driver**: new `muscle_driver.py` reads motor-neuron
   rates, applies innervation matrix, writes to v2 actuators.
   Replace position actuators with real `<muscle>` on
   `<tendon><spatial>`.
4. **Sensory transduction calibration** against Chalasani 2007 AWC,
   Suzuki 2008 ASE, Hilliard 2005 ASH traces.

### Tier 3 — validation + publication-grade claims (~3 mo)
5. **Ensemble audit with corrected brain** (n≥5 seeds × 60 s × 6
   ablations × 3 configs). Does ActivityFSM now reproduce
   AVA/Chalfie? Does RIS/Turek clear 2·SEM?
6. **Aerotaxis phenotype** validation. Does the sim navigate toward
   preferred O2 (12%)?
7. **Parameter uncertainty quantification**: 200-point Latin-
   hypercube sample over ~50 dominant parameters, propagate to
   phenotype statements.

### Tier 4 — the unique-in-field stuff (~6 mo)
8. **CeNGEN-conductance coupling**: scale per-neuron ion-channel
   conductance by CeNGEN TPM. Closes the connectomics-
   transcriptomics loop architecturally, not just statistically.
9. **WebGPU-compiled brain** for live in-browser sim (10 kHz on a
   4060 Ti is plausible).
10. **Pheromone / multi-worm** environment.

## 7 · Tech stack + constraints

**Sim backend:** Brian2 2.9 + MuJoCo (Python) + numpy. Conda env at
`~/miniconda3/envs/ml/bin/python`.

**Frontend:** Astro 4.16 + React 18 + Tailwind. Bundle pre-renders
JSON scenarios (no live Brian2 in browser yet).

**Deploy:** Vercel auto-deploy on push to main.

**Local compute:** RTX 4060 Ti, 8 GB VRAM. Real-time ratio 2.6× for
LIF brain + MuJoCo body (30 s sim = 80 s wall).

**Storage:** `/home/rohit/Desktop/website/personalwebsite/` is the
Astro repo. Brain code under `scripts/brain/`, generated JSONs
under `public/data/`, artifacts under `scripts/brain/artifacts/`.

**Hard constraints:** No PhD (industry route, health-driven).
No wet-lab work. NJ geographic anchor. Theoretical +
computational only.

## 8 · Current publication plan

**Paper 1** — multi-modal analysis: *eLife* / *PLOS Computational
Biology* / *Network Neuroscience*. Topology-dominant /
gene-modulatory framing. Anchors: GNCA r=0.987, NT-classifier
92.7%, CCA ρ≈0.87, honest null on gene-causal LSTM. Draft in
progress.

**Paper 2** — methods paper: NeurIPS GRL / ICLR LMRL workshop
track. GNCA architecture for connectome-constrained synaptic
prediction. Could include the T0 falsification-test methodology
(§5) as a secondary contribution.

**Potential paper 3 (if Tier 2 lands):** the first *C. elegans*
simulator with validated ActivityFSM + transduction cascades +
compartmental dynamics + CeNGEN-conductance coupling. Single-
author accessible.

## 9 · Who I am (for context in chats)

NYU undergrad, Data Science major with Philosophy minor. Industry-
track, not PhD. Working toward AI roles that bridge technical and
philosophical domains. Strong linear algebra / calculus /
probability foundations. Intellectual interests span
neuroscience, consciousness studies, quantum computing, and
Vedantic non-dualism.

**Working style preferences** (important):
- Plan first for non-trivial work, execute second.
- Rigor over brevity. Full-credit-quality reasoning when explaining.
- Push back on speculative proposals before elaborating; ask for
  falsifiability.
- No wet-lab bio work ever. Only theoretical + computational.
- Vedanta / non-dualist framings welcome when they sharpen
  technical work; avoid ideological overlays.
- Direct, no-sugarcoat assessments. Honest scope labels (shipped
  vs. scaffolded vs. calibration-pending).

## 10 · How to use this project in chats

**Questions that should cite this doc:**
- "Where are we on the simulator?"
- "Is X phenotype reproduced?"
- "What would a v3.5 brain look like?"
- "Where does CeNGEN data plug in?"
- "What's in the publication plan?"

**Questions that should defer to rohitravi.com/projects/c-elegans-multimodal:**
- "What does the dashboard show?"
- "Can I see a live example of X?"

**Questions that should ask for current state:**
- "What changed since §5 was written?" (this file drifts)
- Anything about commit hashes, specific file contents, or
  exact code paths.

## 11 · Suggested Claude.ai Project setup

**Project name:** `C. elegans multi-modal simulator`

**Custom instructions (paste into the Project's instructions field):**

> You are helping Rohit Ravi think through an in-silico *C. elegans*
> simulator project combining connectomics, transcriptomics, and
> embodied simulation. Knowledge files describe the current state,
> architecture, validated phenotypes, open issues, and roadmap.
>
> Working style:
> - Be direct and non-sugarcoating. Label shipped vs. scaffolded
>   vs. calibration-pending work honestly.
> - Push back on speculative claims — ask for falsifiability before
>   elaborating.
> - Prefer rigor over brevity when explaining technical concepts.
> - No wet-lab suggestions; theoretical + computational only.
> - Vedanta / non-dualist framings are welcome when they sharpen
>   technical work, but avoid ideological overlays in straight
>   scientific discussion.
>
> When asked about the simulator state, cite §5 (Tier 0 findings)
> as the load-bearing current limitation: the v3 LIF brain's
> AVA/Chalfie phenotype reproduction runs through classifier
> correlation patterns, not biologically-correct command-cascade
> dynamics. This is a known gap with a documented Tier 2 fix path.
>
> Defer to rohitravi.com/projects/c-elegans-multimodal for live
> visual reference. Ask Rohit for current code state when that
> matters — this document drifts from the repo.

**Files to upload as project knowledge:**

1. **This file** (`claude-chat-context.md`) — the primary
   reference.
2. **`scripts/brain/artifacts/t0_run_report.md`** — most recent
   audit findings, referenced from §5.
3. **`src/content/projects/c-elegans-multimodal.mdx`** — the
   public-facing summary, shows what's externally claimed.

Optional (only if discussing specific aspects):
- `scripts/brain/activity_fsm.py` if debating FSM design
- `scripts/brain/compartmental_neurons.py` if debating v4 brain
- Any specific figure from `/home/rohit/Desktop/website/
  personalwebsite/public/images/projects/` if discussing visuals.

**What NOT to upload:**
- Full Python backend (too large; chats burn budget re-reading)
- Raw scenario JSONs (huge, no human-relevant context)
- Dashboard TSX (3.7k LOC, specific-to-UI)

## 12 · Maintenance

Update this file when:
- A Tier-2/3/4 item ships — update §6 state.
- A phenotype is newly validated or newly invalidated — §4 / §5.
- Architecture changes (new brain class, FSM mode, etc.) — §3.
- Publication plan shifts — §8.
- Personal/career context shifts meaningfully — §9.

Re-upload to the Claude.ai Project after substantial updates. Old
version will be replaced; conversations after re-upload see new
context automatically.
