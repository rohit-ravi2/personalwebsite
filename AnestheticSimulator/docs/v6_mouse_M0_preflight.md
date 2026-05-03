# V6 mouse cross-phylum pre-flight (M0) — *Mus musculus*

**Status: M0 PASS — go for M1 with one substantive caveat (see §P0).**

Date: 2026-05-03
Substrate: generic LIF random graph (no mammalian connectome required, per V5 M2 finding that connectome topology beyond cell-type aggregates is not load-bearing).
Behavioral anchor source: Sonner / Eger / Lambert / Belelli labs — gold-standard mammalian MAC and LRR data.

---

## Why mouse, not zebrafish

The V5 M2 result removed the connectome bottleneck that previously made mammal generalization expensive (no need for MICrONS-scale wiring). That makes mouse the strongest next step:

| candidate | data quality | feasibility post-M2 | clinical relevance |
|---|---|---|---|
| **Mouse** | gold-standard MAC + LRR + extensive mutant panel | random graph works | DIRECT — anesthesia studied for human use |
| Rat | comparable MAC; smaller mutant panel | random graph works | direct |
| Zebrafish | thinner anesthesia behavioral lit; smaller mutant panel | random graph works | indirect |
| Aplysia / Hydra | minimal anesthesia work | random graph works | low |

Mouse adds a **deuterostome vertebrate** to the existing protostome invertebrate set (worm + fly), making the cross-species claim cross-PHYLA, not just cross-organism.

---

## P0 — Substrate (generic LIF random graph)

### What's needed

- ~2,000–4,000 neurons (similar scale to fly Winding 2023; tractable in Brian2 cython)
- E:I ratio ~80:20 (mammalian cortical norm)
- Mean degree ~30–60 (matches mammalian cortex local connectivity stats)
- Random connectivity (per V5 M2: connectome topology beyond cell-type aggregates is not load-bearing in the architecture)

### Substantive caveat — what this approach DOES NOT model

Mammalian anesthesia involves **brain-region-specific phenomena** that a flat random graph cannot capture:

1. **Cortical-thalamic dynamics**: NREM-like slow oscillations, EEG burst suppression, gamma suppression — all signatures of "deep" anesthesia in mammals
2. **Sleep-gating circuits**: ventrolateral preoptic nucleus, parabrachial nucleus, locus coeruleus — anesthesia hijacks endogenous sleep machinery in mammals
3. **Cortex-thalamic relay**: anesthesia disrupts thalamic firing modes (tonic → burst), which the current architecture has no analog for
4. **State transitions**: the "loss of consciousness" component of mammalian anesthesia is structurally different from invertebrate immobilization

**The mouse V6 test is therefore restricted to the LRR / immobilization phenotype** — the most "invertebrate-like" component of mammalian anesthesia. Higher-order mammalian features (consciousness suppression, EEG signatures, NREM-like state transitions) are out of scope and must NOT be claimed by V6.

This is the single most important caveat to surface in the page when V6 results are reported. Honest framing: "the same mechanism-class architecture that recovers worm + fly behavioral immobilization also recovers mouse loss-of-righting-reflex EC50, but the higher-order cortical-thalamic features of mammalian anesthesia are not captured."

### Brian2 scale (carried over from fly P3)

Fly was 2,952 neurons × 110K edges in 42s wall for 60s sim. Mouse generic graph at ~3,000 neurons × 50K edges should run faster (sparser graph). V3-protocol ensemble (~700 sims) projects to **~40-60 min on 8 cores**, comparable to fly V4.

**Verdict P0: usable, with the caveat that mouse V6 tests only the LRR/immobilization phenotype, not higher-order mammalian anesthesia features.**

---

## P1 — Receptor ortholog mapping

The mouse case is structurally easier than worm or fly: **the perturbation-table EC50 anchors are mammalian electrophys data to begin with** (Mihic 1997 GABA-A α1β2γ2 was rat receptor; Patel & Honoré 1999 TREK-1 was mouse; Forman 1996 nAChR was rat; Hanley 2002 Complex I was rat liver). So receptor ortholog mapping is mostly **already native** for mouse.

| mechanism class | worm orth | fly orth | mouse — direct EC50 source |
|---|---|---|---|
| GABA-A potentiation | UNC-49 | Rdl | **rat α1β2γ2** (Mihic 1997 PMID 9311785) |
| K2P potentiation | TWK-18 | Sandman | **mouse TREK-1** (Patel 1999 PMID 10321245) |
| nAChR antagonism | UNC-29/38/63 | Dα/Dβ | **rat α4β2** (Forman 1996 PMID 8633440) |
| Complex I block | GAS-1, NUO-1 | ND-49, ND-75 | **rat liver mitochondria** (Hanley 2002 PMID 12411414) |
| SNARE cooperativity | UNC-64 | Syx1A | **rat NMJ** (Stewart 2000 PMID 11095753) |
| NCA leak block | UNC-79/80 | dunc-79/80 | **mouse NALCN** (Lu 2007 PMID 17972040) |
| GluCl potentiation | GLC-* | GluClα | (no mammalian ortholog — DEFERRED for mouse) |
| Complex II block | MEV-1 | SdhC | mouse SDHC (analogy) |

**Deferred class for mouse**: GluCl potentiation. Mammals don't have GluCl receptors (glutamate-gated chloride is invertebrate-only). Drop this row from the mouse perturbation table.

**New mammalian-specific class to consider**: NMDA antagonism (ketamine target). Worm/fly version was deferred. For mouse, NMDA antagonism is a significant ketamine mechanism. Optionally add a `nmda_antagonism` class.

**Verdict P1: ortholog mapping is essentially trivial for mouse. Drop GluCl row; optionally add NMDA antagonism.**

---

## P2 — Behavioral EC50 anchors (mouse MAC + LRR)

Sonner / Eger / Lambert / Belelli labs have published extensive mouse MAC and LRR EC50 data. The mouse anesthesia field is the gold standard; behavioral anchors here are higher quality than worm or fly.

### Wild-type MAC values (aqueous equivalents)

| anesthetic | mouse MAC (vol%) | aqueous eq. (µM) | source |
|---|---|---|---|
| **halothane** | 0.95–1.05 vol% | **~350 µM** | Sonner 1999 PMID 9856681; Eger 2002 |
| isoflurane | 1.30–1.40 vol% | ~290 µM | Sonner 1999; Eger 2002 |
| sevoflurane | 2.5–2.7 vol% | ~270 µM | Sonner 1999; Eger 2002 |
| desflurane | 6.5–7.0 vol% | ~290 µM | Eger 2002 |
| diethyl ether | 3.2–3.5 vol% | ~3 mM | Eger 2002 |

### Wild-type IV anesthetic LRR EC50

| anesthetic | mouse LRR EC50 | source |
|---|---|---|
| propofol | 10–15 µM | Belelli 1997 PMID 9298537; Antognini lab |
| etomidate | 1.5–2.5 µM | Belelli 1997 PMID 9298537; Lambert lab |
| ketamine | 25–40 µM | (Mott / Drummond — secondary lit) |

**Striking finding**: mouse halothane MAC at ~350 µM aqueous is essentially the SAME as worm Crowder 1996 (340 µM) and fly van Swinderen 1999 (340 µM). The aqueous concentrations of volatile anesthetics at MAC are conserved across phyla — same molecules, same conserved targets, same effective concentrations. The cross-species calibration anchor is essentially universal.

**Verdict P2: mouse anchors are gold-standard. Calibration target = halothane WT MAC ≈ 350 µM aqueous.**

---

## P3 — Mouse mutant directional panel

### Established directional anchors with primary literature

| gene / allele | direction | anesthetic | source | mechanism class |
|---|---|---|---|---|
| **GABA-A β3(N265M) knock-in** | RESISTANT | etomidate | **Jurd 2003 FASEB J PMID 12521989** | gaba_potentiation |
| GABA-A β3(N265M) | RESISTANT | propofol | Jurd 2003; Reynolds 2003 | gaba_potentiation |
| **GABA-A β3(N265M)** | RESISTANT | halothane (LRR loss) | Jurd 2003 (smaller effect) | gaba_potentiation |
| GABA-A α1(H101R) | partial RESIST | propofol (in vivo) | Reynolds 2003 PMID 14563696 | gaba_potentiation |
| **TREK-1 KO** | RESISTANT | halothane | **Heurteaux 2004 EMBO J PMID 15212942** | k2p_potentiation |
| TREK-1 KO | RESISTANT | isoflurane | Heurteaux 2004 | k2p_potentiation |
| **TASK-1 KO** | RESISTANT | halothane | **Linden 2007 PMID 17215344** | k2p_potentiation |
| TASK-3 KO | partial RESIST | halothane | Linden 2008 | k2p_potentiation |
| TASK-1/TASK-3 double KO | RESISTANT | halothane (strong) | Linden 2008 | k2p_potentiation |
| **GIRK2 KO** | RESISTANT | etomidate, halothane | Bayliss lab; Lazarenko 2010 | (not in current panel — would add) |
| Stx1A hypomorph | HYPER | halothane | Jung 2011 (partial data) | snare_cooperativity |
| **NDUFS4 conditional KO** | HYPER | halothane | **Quintana 2010 PMID 20847272** | complex_i_block |
| Synaptobrevin (VAMP2) KO conditional | HYPER (partial) | halothane | Wölfel 2008 | snare_cooperativity |

**13 well-anchored directional mutants** with primary literature PMIDs.

### Direction balance

- HYPER (predict ratio < 1): NDUFS4, Stx1A, Vamp2 — 3 anchors
- RESISTANT (predict ratio > 1): β3(N265M) etomidate/propofol/halothane, α1(H101R), TREK-1 (halothane + iso), TASK-1, TASK-3, TASK-1/3, GIRK2 — 10 anchors

Direction skew is OPPOSITE to worm V3 (mostly HYPER) and fly V4 (mixed). Mouse panel is RESISTANT-rich because mammalian anesthesia genetics has focused on knock-out / knock-in mutants of receptor-binding sites where loss-of-function reduces drug effect → resistant.

**Implication**: V6 Gate 3 directional accuracy will test the model's ability to predict RESISTANT directions cleanly — which is exactly the direction the V3/V4 architecture struggled with most (initial Gαo bug, etc.). **This is genuinely diagnostic** for whether the architecture handles RESISTANT mutants properly.

**Verdict P2 (mutants): mouse mutant panel is large enough (n=13), well-anchored, and structurally complementary to worm/fly (more RESISTANT-direction anchors).**

---

## M0 GO / NO-GO summary

| check | result | implication |
|---|---|---|
| P0 — substrate (generic random graph) | **PASS** with caveat | mouse V6 tests LRR phenotype only; higher-order mammalian features out of scope |
| P1 — ortholog mapping | **PASS** | EC50 anchors are mammalian electrophys to begin with; transfer is direct |
| P2 — behavioral anchors | **PASS** | gold-standard MAC + LRR; same ~350 µM halothane anchor as worm/fly |
| P3 — mutant panel | **PASS** | 13 directional anchors, RESISTANT-rich (complementary to V3/V4) |

**Verdict: GO for M1.**

---

## What's structurally different vs worm V3 / fly V4

1. **Substrate is a random graph**, not a published connectome. Per V5 M2, this is honest given the connectome topology isn't load-bearing. The page must explicitly state this (the V6 substrate is NOT "the mouse connectome").

2. **Drop GluCl class** from perturbation table (no mammalian ortholog).

3. **Mutant panel is RESISTANT-rich** — most mammal anesthesia genetics works on receptor-binding-site knock-ins where LoF reduces drug effect. Tests the architecture's ability to predict resistance directions.

4. **Optional add NMDA antagonism class** — ketamine in mammals engages NMDA prominently. If included, ketamine becomes a more meaningful test in V6 than it was in worm/fly (where ketamine had only weak nAChR + K2P engagement).

5. **The behavioral readout is LRR** (loss of righting reflex), which maps cleanly to "command-interneuron firing rate suppression" in the network model. Same metric as worm/fly. **The higher-order features of mammalian anesthesia (cortical EEG, NREM-like states, consciousness disruption) are NOT testable here and must not be claimed.**

---

## Honest pre-registration of what mouse V6 will and won't claim

**Will claim if it passes**:
- The same conserved-target perturbation table that recovers worm + fly behavioral phenotypes also recovers mouse loss-of-righting-reflex EC50 and mutant phenotypes.
- The architecture transfers across two phyla (protostome invertebrates → deuterostome vertebrates).
- One-parameter calibration (α) on a single behavioral anchor (halothane MAC) generalizes to held-out anesthetics and mutants.

**Will NOT claim**:
- Mouse cortical EEG features
- NREM-like slow oscillations
- Consciousness suppression
- Burst suppression
- Anything requiring brain-area-specific dynamics
- Anything claiming the substrate IS the mouse connectome (it's a generic graph)

The narrowed honest claim under either outcome: **the mechanism-map (conserved targets + Hill perturbation + literature EC50s) is what does the work — confirmed across nematode, dipteran, and mammalian substrates at the immobilization phenotype level.**

---

## Next: M1 — mouse perturbation tables

Three CSVs paralleling worm V3 + fly V4 structure:

- `data/state_validation_mouse/mouse_anesthetic_perturbation_table.csv`
  - Mostly identical to worm/fly tables (mammalian electrophys was already the source)
  - Drop GluCl class
  - Optionally add NMDA class for ketamine
- `data/state_validation_mouse/mouse_immobilization_anchors.csv`
  - 5 PRIMARY anchors: halothane, isoflurane, sevoflurane, desflurane, diethyl ether (Sonner 1999, Eger 2002)
  - 2 PRIMARY for IV: propofol, etomidate (Belelli 1997)
- `data/state_validation_mouse/mouse_directional_mutants.csv`
  - 13 anchors: β3(N265M), α1(H101R), TREK-1, TASK-1, TASK-3, GIRK2, NDUFS4, Stx1A, Vamp2 + halothane and isoflurane
- `data/state_validation_mouse/mouse_mutant_baseline_perturbations.csv`
  - LIF entry-point factors per mutant (analogous to worm/fly tables)

Estimated effort: 1.5–2 days. M2 (generic LIF brain) follows: ~1 day.

**Total V6 timeline: 7–8 working days + 1–2 overnight ensemble runs. Same compute budget as fly V4.**
