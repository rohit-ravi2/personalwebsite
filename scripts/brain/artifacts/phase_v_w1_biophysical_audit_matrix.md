# Phase V Wave 1 — Biophysical Feasibility Audit Matrix

**Survey-level assessment, single-session.** For each mechanism: current state, worm data tier, adjacent priors, implementation complexity (order-of-magnitude), validation pathway, research utility, OpenWorm/c302/Nicoletti positioning, and feasibility tier (production / research-artifact / aspirational).

**Feasibility tiers:**
- **Production-grade:** kinetic data exists in worm-specific voltage-clamp literature with quantitative parameters; existing implementations (c302, Nicoletti) provide validated models that can be imported/calibrated; deployable with research-tool fidelity.
- **Research-artifact-grade:** mechanism is implementable from worm/adjacent priors but parameters require fitting/calibration; useful for mechanistic exploration but not production claims.
- **Aspirational:** worm-specific data sparse; substantial wet-lab generation required; implementation possible only with mammalian priors, low confidence on quantitative claims.

**Data tier within each:** rich (multiple worm papers), moderate (1-3 worm papers + adjacent), sparse (mostly adjacent), absent (must extrapolate).

---

## Category A — Ion channel biophysics

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl (c302/Nicoletti) | Tier |
|---|---|---|---|---|---|---|---|---|
| **EGL-19 (L-type Ca)** | implemented in `compartmental_neurons.py` (m_inf only, h via custom equation; v_rest correction needed); also in `graded_brain.py` h_kca variants | rich (Lainé 2014, Liu 2011, Shtonda 2005) | mammalian L-type extensive | low — already done; needs h-equation fix (existing 0.3 ratio insufficient) | Mellem 2008 voltage-clamp targets (existing in `phase0_plateau_diagnostic.py`) | central to plateau dynamics, anesthesia (volatile reduces L-type), drug discovery (Ca channel blockers) | Nicoletti 2019, c302 muscle models | **production** (with h fix) |
| **UNC-2 (P/Q-type Ca)** | absent | moderate (Schafer 1995, Saheki 2009 synaptic role) | mammalian P/Q extensive | low — HH-style, parameters from Nicoletti 2019 | knockout phenotype: reduced synaptic transmission | synaptic Ca for vesicle release, drug discovery (gabapentin-class) | Nicoletti 2019 implements | **research-artifact** (importable) |
| **CCA-1 (T-type Ca)** | absent | moderate (Shtonda 2005, Steger 2005) | mammalian T-type rich | low — HH-style | RMD bistability validation per Nicoletti 2019 | low-threshold activation, rebound burst, sleep-wake oscillations | Nicoletti 2019 implements | **research-artifact** (importable) |
| **SHL-1 (Kv4 A-type)** | absent | rich (Fawcett 2006: V0.5a=2.4mV, ka=7.4, V0.5i=-19.5, ki=3.7, biexponential inactivation) | Drosophila Shal extensive | low — full HH params known | shl-1(ok1168) null mutant phenotype | repolarization, AHP shaping, fast neuronal dynamics | Nicoletti 2019, c302 implement | **production** (importable) |
| **SHK-1 (Kv1 delayed rectifier)** | absent | rich (Fawcett 2006: V0.5a=25mV, ka=14, τ_m≈5ms at 0mV, τ_h=1400ms; Liu 2011 muscle) | Shaker extensive | low — full HH params known | shk-1 LOF mutant, primary K+ in muscle | AP repolarization, membrane potential setting | Nicoletti 2019, c302 implement | **production** (importable) |
| **EGL-36 (Kv3 Shaw)** | absent | rich (Johnstone 1997: V1/2=+63 mV, slope 28 mV/e, complex multi-state activation) | Drosophila Shaw moderate | low — known params | egl-36 GOF mutants (egg-laying defects) | high-V activation, behavioral state regulation | Nicoletti 2019, c302 implement | **production** (importable) |
| **EGL-2 (ether-a-go-go)** | absent | moderate (Weinshenker 1999) | mammalian eag extensive | low — HH-style | egl-2 mutant phenotype | slow modulation, behavioral state | Nicoletti 2019, c302 implement | **production** (importable) |
| **KQT-1/2/3 (KCNQ-class)** | absent | moderate (Wei 2005) | mammalian KCNQ rich (M-current) | low — HH-style | KCNQ openers (retigabine analog) phenotype | M-current, anesthesia (volatiles modulate) | Nicoletti 2019 implements KQT-3 | **production** (importable) |
| **EXP-2 (delayed rectifier, pharynx)** | absent | rich (Davis 1999, Shtonda 2005) | none — unique to nematodes | low | pharyngeal AP shape | feeding behavior, pharynx-specific | c302 muscle models implement | **production** (importable; pharynx-specific) |
| **IRK-1/2/3 (inward rectifier)** | absent | moderate | mammalian IRK extensive | low — HH-style | knockout phenotypes | resting potential setting | Nicoletti 2019 implements IRK | **production** (importable) |
| **KVS-1 (KCNB-class)** | absent | moderate | mammalian Kv2 moderate | low — HH-style | functional role unclear | unclear | Nicoletti 2019 implements | **research-artifact** (importable) |
| **SLO-1 (BK Ca-activated K)** | partially in h_kca patch (generic K_Ca) | rich (Wang 2001, Davis 2008, Salkoff 2006) | mammalian BK extensive | low-medium — needs Ca-binding kinetics + voltage gating | slo-1 LOF mutant (locomotion defects, alcohol resistance) | plateau termination, anesthesia (volatile inhibits SLO-1), alcohol | c302 muscle, Nicoletti uses BK | **production** (with Ca-K + voltage coupling) |
| **SLO-2 (BK Cl-activated K)** | absent | rich (Liu 2011, Yuan 2003) | mammalian SLACK | low | slo-2 LOF mutant (reduces muscle current) | unique nematode K+, distinct from SLO-1 | c302 muscle implements | **production** (importable) |
| **SK channels (KCNL-1/2/3)** | absent | sparse | mammalian SK rich | medium — Ca-activation kinetics differ from BK | knockout phenotypes incomplete | AHP shaping, after-burst dynamics, anesthesia | Nicoletti 2019 implements KCNL | **research-artifact** (data thin but importable) |
| **NCA-1/NCA-2 (NALCN Na+ leak)** | absent | rich (Yeh 2008, Humphrey 2007, Xie 2013) | NALCN family in mammals | low — passive linear | nca-1; nca-2 double mutant (severe locomotion defects) | resting membrane potential, baseline excitability, central to ON-state | Nicoletti 2019 implements NCA leak | **production** (importable) |
| **HCN (h-current)** | absent | sparse — C. elegans has TPH-2-related but role unclear | mammalian HCN extensive | medium — needs cyclic-nucleotide gating + voltage | unclear in worm; pacemaking role hypothetical | pacemaking, sag response, anesthesia (volatile inhibits HCN) | not in c302 standard, Nicoletti doesn't implement | **research-artifact** (worm role unclear, would be exploratory) |
| **TWK family (two-pore K, K2P)** | absent | sparse — TWK-7 (Gottschalk lab) and TWK-40 (sleep) characterized | mammalian K2P extensive (TREK, TASK) | low-medium — passive K+ leak with allosteric gating | twk-40 mutant (sleep defects per Nichols 2017) | resting potential, **central to volatile anesthetic action**, sleep | not in c302 standard | **research-artifact** for general; **production-priority for anesthesia work** (data exists for TWK-40) |
| **GluCl (glutamate-gated chloride)** | implicit at synaptic level (sign convention via CeNGEN ratio); not as kinetic channel | rich (Cully 1994, Vassilatis 1997, Etter 1996; AVR-14/15, GLC-1/2/3/4) | unique to invertebrates; mammalian glycine receptor structurally similar | medium — need ligand-gated kinetics (binding/unbinding/channel open) at receptor level | ivermectin sensitivity (canonical), avr-14 mutants | inhibitory glutamate (AVA-precedent finding), drug discovery (avermectin class) | c302 has at synaptic level | **production** (with kinetic upgrade) |
| **GABA_A-like (UNC-49)** | implicit at synaptic level via per-neuron sign | rich (Bamber 1999, Schuske 2004) | mammalian GABA_A extensive | medium — Cl- channel kinetic scheme | unc-49 LOF (motor defects) | inhibitory transmission, **central to anesthetic action (propofol/etomidate)**, drug discovery | c302 at synaptic level | **production-priority for anesthesia** |
| **CLH-1/CLH-3 (CLC chloride)** | absent | moderate (Schriever 1999) | mammalian CLC family | medium | mutant phenotypes | Cl- homeostasis, intracellular pH | not in c302 | **aspirational** (lower utility) |

**Category A summary:** ~15 of 18 listed channels have rich-to-moderate worm-specific kinetic data, AND existing implementations in Nicoletti 2019 / c302 / OpenWorm muscle models. **The single highest-leverage near-term move is integrating Nicoletti 2019's channel library into the simulator's compartmental-cell roster rather than re-implementing from primary sources.** Estimated time: 1-2 weeks of integration work.

The exceptions: HCN (worm role unclear, exploratory), TWK family (data exists but limited; central for anesthesia work — high priority for that application specifically), CLC chloride (lower utility, sparse data).


---

## Category B — Synaptic transmission machinery

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **SNARE complex** (UNC-64/syntaxin, RIC-4/SNAP-25, SNB-1/synaptobrevin) | absent — abstracted as instantaneous current | rich genetic dissection (Liu 2020, Saifee 1998); structural data via mammalian | mammalian extensive | medium-high — kinetic schemes need rate constants | open-syntaxin KI rescues unc-13 (Liu 2020); aldicarb sensitivity assays | drug discovery (botulinum toxin targets SNARE), anesthesia (some volatile effects on SNARE) | not in c302; addressed as "release rate" abstraction | **research-artifact** (genetic data rich, quantitative kinetics from mammalian priors) |
| **UNC-13/Munc13 priming** | absent | rich (Richmond 1999, Zhou 2013, Zhou 2019) — quantitative phenotypes | mammalian Munc13 extensive | medium — priming/unpriming rate constants needed | unc-13(s69) shows 97% reduction in primed vesicle response | central to release probability and short-term plasticity | not in c302 detail | **research-artifact** (importable from mammalian models with worm validation) |
| **UNC-18/Munc18 docking** | absent | rich (Weimer 2003, Liu 2020) | mammalian Munc18 extensive | medium | docking quantification via EM | basic vesicle traffic, not anesthesia-relevant | not in c302 | **research-artifact** |
| **UNC-10/RIM (active zone organizer)** | absent | rich (Weimer 2006, Zhou 2019) | mammalian RIM extensive | medium — couples Ca channels to vesicles | unc-10 mutant reduces evoked release | release probability modulation, plasticity | not in c302 | **research-artifact** |
| **Vesicle pool dynamics** (docked / primed / recycling / reserve) | absent — single "release rate" abstraction | moderate worm-specific (FM4-64 imaging + EM) | mammalian extensive | medium-high — multi-pool ODEs with transition kinetics | tomosyn (TOM-1) mutant shows enlarged primed pool | short-term plasticity (depression, facilitation), drug screening | not in c302 | **research-artifact** (importable from mammalian models like Pan-Vázquez 2024) |
| **Synaptotagmin (SNT-1, SNT-3) Ca-sensing** | absent — graded sigmoid release proxy in GradedBrain | rich (Mahoney 2008, Nonet 1993) | mammalian Syt-1 extensive (Hill function with n≈4-5) | medium — Ca-binding kinetics with cooperative gating | snt-1 LOF (locomotion + release defects) | release probability dependence on presynaptic Ca, anesthesia (volatiles affect Syt-1) | not in c302 detail | **research-artifact** |
| **Active-zone Ca channels (UNC-2 + EGL-19 nanodomains)** | absent — synaptic Ca abstracted as instantaneous | moderate (Liu 2018, Saheki 2009) | mammalian active-zone Ca rich | high — spatially-resolved Ca microdomain dynamics | unc-2 mutants reduce release | release probability calibration, drug discovery | not in c302 | **research-artifact** (with simplification, full microdomain is **aspirational**) |
| **Endocytosis (clathrin-mediated, ultrafast)** | absent | moderate (UNC-26 / synaptojanin, AP-2 complex genetics) | mammalian extensive | medium — but lower priority for steady-state dynamics | endocytosis-defective mutants | high-frequency stimulation effects, replenishment kinetics | not in c302 | **research-artifact** (lower priority) |
| **Receptor binding kinetics (closed/open/desensitized)** | absent — synaptic current is instantaneous step | moderate (heterologous expression for some receptors) | mammalian extensive (well-characterized for nAChR, NMDA, AMPA, GABA_A) | medium-high — Markov state schemes per receptor | dose-response curves, desensitization timecourse | drug discovery, anesthesia central, plasticity (NMDA in spike-timing) | not in c302 | **research-artifact** for major receptors; **production-priority for anesthesia** |
| **DCV-specific machinery (UNC-31/CAPS, IDA-1, UNC-104 transport)** | partially — modulator layer abstracts release rate from firing | rich (Speese 2007, Hammarlund 2008, Zhou 2007) | mammalian CAPS/IA-2 moderate | medium — separate DCV pool with distinct release kinetics from SVs | unc-31 LOF (no peptide release, locomotion + egg-laying defects) | peptidergic signaling, neuromodulation, IDPs in CNS drugs | not in c302 (modulator layer abstracts) | **research-artifact** (already partially in modulator layer; refinement) |

**Category B summary:** Worm genetic data is rich for the molecular machinery; **quantitative release kinetics (rate constants, dwell times, pool sizes) are inferred mostly from mammalian priors**. The key implementation question: do you need full kinetic resolution (multi-pool ODEs with rate constants) or graded-release abstraction sufficient for behavioral claims? Answer depends on application: anesthesia mechanism work needs receptor binding kinetics (because volatile/intravenous anesthetics act at the receptor level); drug discovery for synaptic-transmission-modifying compounds needs SNARE/vesicle pool dynamics; basic research-artifact level is satisfied by graded release with proper Ca-dependence.

**Highest-leverage near-term:** receptor binding kinetics for the major ligand-gated channels (UNC-49 GABA_A, GluCl, NMR-1/2 NMDA, GLR family AMPA, nAChR family). This is also where Category C overlaps.


---

## Category C — Receptor pharmacodynamics

| Receptor | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **GLR-1/2/3/4/5/6/7/8 (AMPA/kainate-like iGluR)** | implicit synaptic current | rich (Brockie 2001, Mellem 2002, Maricq 1995); CeNGEN expression | mammalian AMPA/kainate extensive (Markov schemes, well-characterized desensitization) | medium — kinetic scheme: closed → open → desensitized | glr-1 LOF (locomotion, AVA function) | central to fast excitation, drug discovery (AMPA modulators), cognition research | not in c302 | **research-artifact** (importable from mammalian) |
| **NMR-1/2 (NMDA-like)** | implicit synaptic current | rich (Brockie 2001, Korswagen 2002) | mammalian NMDA extensive (Mg2+ block, glycine co-agonism, slow kinetics) | medium-high — voltage-dependent Mg2+ block + Ca permeability | nmr-1 LOF (foraging defects) | plasticity-relevant (in mammalian sense), ketamine target | not in c302 | **research-artifact** (importable) |
| **UNC-49 (GABA_A-like)** | implicit per-neuron sign | rich (Bamber 1999, Schuske 2004) | mammalian GABA_A extensive (allosteric sites, anesthetic binding) | medium-high — multiple binding sites including allosteric for anesthetics | unc-49 LOF (motor defects) | **central to volatile + IV anesthetic action**, drug discovery | not in c302 detail | **production-priority for anesthesia** |
| **AVR-14/15, GLC-1/2/3/4 (GluCl, glutamate-gated Cl)** | implicit at synaptic level | rich (Cully 1994, Etter 1996, Frazier 2003) | unique to invertebrates; mammalian glycine receptor structurally similar | medium — kinetic scheme + ivermectin allosteric binding | avr-14 LOF, ivermectin response | drug discovery (avermectin class anti-parasitics, multi-billion market) | not in c302 | **production** (worm-specific data, drug-discovery direct) |
| **UNC-29/38/63 + others (nAChR family)** | implicit | rich (Lewis 1980, Touroutine 2005, Boulin 2008); ~30+ subunit genes | mammalian nAChR rich | medium-high — multiple subunit combinations | levamisole / nicotine response, locomotion phenotypes | drug discovery (anthelmintics levamisole/pyrantel; nicotine-related), addiction research | not in c302 detail | **production** for nematode-specific nAChR (anthelmintic relevant) |
| **PDFR-1, NPR-1/22, DMSR-1, SER-1/4/7, OCTR-1, TYRA-2/3 (modulator GPCRs)** | direct current modulation in modulator layer | moderate (specific receptor characterization variable; some EC50 data heterologous) | mammalian GPCR extensive | high — full GPCR with G-protein cycle + downstream cascade | knockout phenotypes for many | central to neuromodulation, drug discovery (GPCR is largest drug-target class) | not in c302 detail | **research-artifact** for full kinetics; current abstraction sufficient for some applications |
| **Allosteric modulation framework (PAM/NAM)** | absent | sparse worm-specific; mostly mammalian for GABA_A anesthetic-binding | mammalian rich for GABA_A, NMDA, AMPA | high — modeling allosteric requires multi-state Markov scheme | benzodiazepine response in worm (limited), volatile anesthetic phenotypes | **central for anesthesia mechanism**, drug discovery | absent everywhere worm-specific | **research-artifact for development; aspirational for quantitative anesthesia claims** without wet-lab validation |
| **Receptor desensitization kinetics** | absent | moderate (some receptor-level worm data) | mammalian rich (multi-timescale) | medium — Markov state schemes per receptor | dose-response curves, paired-pulse responses | short-term plasticity, drug screening | absent | **research-artifact** (importable) |
| **Receptor trafficking (activity-dependent)** | absent | moderate (Burbea 2002, Grunwald 2004 for GLR-1) | mammalian extensive | medium-high — endocytosis + recycling kinetics | activity-dependent GLR-1 trafficking | plasticity, learning research | absent | **research-artifact** (lower priority near-term) |

**Category C summary:** Receptor genetics is rich; quantitative pharmacology (EC50, kinetic schemes, allosteric binding) is largely from heterologous expression or mammalian priors. Highest research utility for the project's stated applications: **GABA_A (UNC-49) + GluCl (AVR-14/15) + nAChR family** because they're directly drug-targetable. Anesthesia-mechanism work specifically requires GABA_A allosteric framework.

---

## Category D — Gene expression coupled to neural dynamics

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **Static channel/receptor density per CeNGEN TPM** | partially — expression loaded but not used to scale per-neuron channel densities | rich (CeNGEN bulk + L4 single-cell, Taylor 2021) | mammalian "scale by Allen Atlas TPM" approaches | low-medium — multiplier on per-cell g_max from TPM | Mellem-style cellular validation per cell type | foundational for "biology-grounded" claims; T4-#8 in original roadmap | not in c302 explicitly | **production-priority** (highest leverage in this category — nearly free with CeNGEN data already loaded) |
| **Activity-dependent immediate-early gene transcription** | absent | moderate (Carmie 2009: CRH-1/CREB; Greer 2009 stress response) | mammalian rich (fos/jun, egr-1) | medium-high — slow timescale ODEs coupled to activity history | activity-dependent CREB phosphorylation, stress response | learning, plasticity, **mechanistic insight into long-term effects of perturbations** | absent everywhere | **research-artifact** for IEG; **aspirational for quantitative dynamics in worm** |
| **mRNA localization, local translation** | absent | sparse (some axonal transport studies) | mammalian moderate | high — spatial RNA dynamics | hard to measure in worm | plasticity research at fine resolution | absent | **aspirational** |
| **TF cascade → channel insertion → altered excitability** | absent | sparse (UNC-42, UNC-86, etc. for cell fate; activity-dependent less characterized) | mammalian rich | high — multi-day timescale | knockout phenotypes; behavioral plasticity | long-term adaptation, learning research | absent | **aspirational** for full coupling |
| **Long-term excitability changes (hours-days)** | absent — simulator runs at minutes | sparse for adult-neuron-specific | mammalian moderate | high — different timescale than current sim | behavioral plasticity assays | long-term drug effects, chronic perturbations | absent | **aspirational** (timescale mismatch with current simulator) |

**Category D summary:** Single high-leverage entry: **per-cell channel densities scaled by CeNGEN TPM**. Foundational data exists, integration is a multiplier on g_max values. Everything else in this category is research-artifact or aspirational due to worm-specific dynamic data sparsity at the timescales required.

---

## Category E — Epigenetics

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **5-methylcytosine (5mC) DNA methylation** | absent | sparse — C. elegans has minimal 5mC; MET-2 methyltransferase weak | mammalian rich | high | hard to validate; minimal modification in worm | low for worm-specific work | absent | **aspirational** (not biologically prominent in worm) |
| **N6-methyladenine (6mA) DNA methylation** | absent | moderate (Greer 2015: DAMT-1 demethylase NMAD-1) | none — vertebrates lack 6mA at significant levels | high | damt-1 mutant phenotypes; transgenerational inheritance | limited mechanistic dissection at neural dynamics level | absent | **aspirational** for neural-dynamics application |
| **Histone modifications (H3K4me3, H3K9me3, H3K27me3)** | absent | rich genetic dissection (Greer lab, Strome lab) | mammalian rich | high — chromatin state changes are hours-days | LSD-1 / SET-2 mutant phenotypes | gene regulation, longevity research, transgenerational effects | absent | **research-artifact** (data rich for genetics, but coupling to neural dynamics quantitatively sparse) |
| **microRNA regulation (let-7 family, lin-4)** | absent | rich for development (lin-4 founding miRNA) | extensive | high — multi-target gene regulation | LOF phenotypes | development, longevity, neural plasticity | absent | **research-artifact** for development; sparse for adult neural plasticity |
| **piRNAs, lncRNAs** | absent | moderate (PRG-1, Argonautes) | extensive in mammals | high | germline mostly | low for adult neural work | absent | **aspirational** |
| **Activity-dependent epigenetic changes in adult neurons** | absent | sparse | mammalian moderate | very high | hard to measure in worm | learning/memory research | absent | **aspirational** |

**Category E summary:** Largely **aspirational for the project's research-tool goals**. C. elegans has unusual epigenetic landscape (minimal 5mC, primary modifications are 6mA + histone marks). Genetic data is rich but quantitative coupling to neural-dynamics applications is sparse to absent. Realistic ceiling for this category: not a near-term implementation target. May surface as a long-term research direction after other categories are mature.

---

## Category F — Peptide processing and release

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **Pre-pro-peptide processing (BLI-4, EGL-3, KPC-1, AEX-5)** | absent — modulator layer assumes mature peptide | rich (Husson 2007, Husson 2009 mass spec for FLPs/NLPs) | mammalian PC-1/PC-2 extensive | medium — sequential proteolytic cleavage with rate constants | egl-3 mutants (peptidergic defects) | drug discovery (proprotein processing inhibitors), peptide therapeutics | absent | **research-artifact** (data rich, kinetics from mammalian) |
| **DCV biogenesis + transport (UNC-104/KIF1A, IDA-1)** | absent — modulator layer assumes packaged/positioned | rich (Zhou 2007, UNC-104 transport at 1.6-2.7 µm/s) | mammalian extensive | medium | unc-104 LOF (severely reduced peptide signaling) | basic peptidergic biology | absent | **research-artifact** (lower priority near-term) |
| **Co-release of multiple peptides (FLP-1 has 8 peptides, NLP-12 has 2)** | absent — single peptide per modulator class | moderate (mass spec confirms co-release) | limited mammalian (some prohormones) | medium — separate peptides released with different stoichiometry | peptide-specific receptor activation | drug discovery — peptide-receptor specificity | absent | **research-artifact** |
| **Peptide degradation (extracellular proteases, NEP-class)** | partially — generic decay τ in modulator layer | sparse | mammalian extensive | medium | NEP inhibitor effects | drug duration-of-action, receptor desensitization downstream | absent | **research-artifact** (data sparse but importable from mammalian) |
| **Volume transmission (peptides diffuse beyond synapse)** | partially — diffusion length in modulator layer | moderate (Bentley 2016 biological prior on 4 mm diffusion-equivalent) | mammalian extensive | medium-high — spatial reaction-diffusion | peptide gradient measurements (limited in worm) | drug pharmacokinetics in CNS | absent | **research-artifact** |
| **DCV release Ca-dependence (UNC-31/CAPS, distinct from SV)** | partially — modulator release linked to firing rate | rich (Speese 2007) | mammalian CAPS moderate | medium — separate Ca-binding kinetics from synaptotagmin | unc-31 mutant phenotypes | peptidergic release mechanism, drug discovery | absent | **research-artifact** |

**Category F summary:** Worm-specific data for peptide processing + release is unusually rich (mass spec inventories, genetic dissection, partial functional). The current modulator layer in the simulator is a coarse abstraction; refining to per-peptide kinetic dynamics is **research-artifact-grade work that's importable from existing data**. Sub-priority for paper 2 (behavioral closure) but core for paper 3+ (mechanistic claims about modulation).


---

## Category G — Second messenger cascades

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **GPCR → G-protein cycle (Gαs/Gαi/Gαq)** | absent — modulators bypass to direct current modulation | rich genetic (GSA-1 Gαs, GOA-1 Gαo, EGL-30 Gαq); sparse quantitative kinetics | mammalian extensive (cycle rates, GTPase rates, GEF/GAP regulation) | medium-high — coupled ODEs with state-cycling | knockout phenotypes; pertussis toxin-style perturbations | drug discovery (GPCR is largest target class), neuromodulation mechanism | absent | **research-artifact** (importable from mammalian) |
| **cAMP / PKA pathway** | absent — would be downstream of Gαs GPCRs | moderate (KIN-1 PKA, ACY-1/2/3/4 adenylyl cyclase, PDE family) | mammalian extensive | medium — classic mass-action ODEs | KIN-1 conditional rescue experiments | central modulator effector, drug discovery | absent | **research-artifact** |
| **IP3 / DAG / Ca / PKC pathway** | absent — would be downstream of Gαq GPCRs | moderate (ITR-1 IP3R, EGL-8 PLCβ, PKC TPA-1/PKC-1) | mammalian extensive | medium-high — overlaps with Category I (Ca dynamics) | ITR-1 mutant phenotypes; PKC inhibitor responses | plasticity, modulation, anesthesia (some volatile effects on PKC) | absent | **research-artifact** |
| **cGMP / PKG pathway** | partially — present in sensory transduction cascades (5 cascades for ASE/AWC/ASH/AFD/ALM in `sensory_transduction.py`) | rich for sensory neurons (EGL-4 PKG, GCY family guanylyl cyclases, TAX-2/4 CNG channels) | mammalian moderate | already-implemented level: medium for extending beyond sensory | sensory neuron transduction validated against published traces | sensory cascade fidelity, drug discovery (PDE5 inhibitors as analogs) | partially in this project (sensory cascades) | **production for sensory cascades; research-artifact for broader use** |
| **Cross-talk between cascades** | absent | sparse worm-specific | mammalian moderate (specific cross-talks documented) | high — multi-pathway coupled ODEs | indirect via combined perturbations | realistic signaling fidelity | absent | **aspirational** for full network |

**Category G summary:** Genetic data is comprehensive (every cascade component has a worm gene); quantitative kinetic data is mostly mammalian priors. Sensory transduction cascades are **already implemented in the simulator** (`sensory_transduction.py`) at production level. Extending the same approach to GPCR-modulated cascades is research-artifact-grade work. Cross-talk modeling is aspirational.

---

## Category H — Plasticity machinery

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **Hebbian-style LTP/LTD** | absent | sparse — worm uses graded transmission, no classical LTP | mammalian extensive | high — mechanistically requires NMDA-like Ca-dependent gating | hard to measure in worm | learning research, **but largely doesn't translate to worm biology** | absent | **aspirational / not biologically apt** for worm |
| **Spike-timing-dependent plasticity (STDP)** | absent | not applicable — graded transmission | mammalian extensive | high — requires spikes that worm largely lacks | n/a | learning research | absent | **not applicable to worm** (architectural mismatch) |
| **Habituation/dishabituation (tap-withdrawal circuit)** | absent | rich (Rankin lab decades; Bozorgmehr 2013, Engel 2002) | invertebrate plasticity | medium — modulator-induced changes in synaptic gain | habituation timecourse, dishabituation by novel stimuli | learning research, behavioral plasticity | not in c302 | **research-artifact** (worm-grounded data exists) |
| **Modulator-induced plasticity** | absent — modulators have direct excitability effects only | moderate (5HT/DA effects on synaptic gain documented) | mammalian moderate | medium-high — modulator → 2nd messenger → receptor trafficking → gain change | behavioral assays, pharmacological perturbations | learning, drug discovery | absent | **research-artifact** |
| **Activity-dependent receptor trafficking (GLR-1)** | absent | moderate (Burbea 2002, Grunwald 2004) | mammalian extensive | medium — endocytosis/recycling kinetics | GLR-1::GFP imaging | plasticity research | absent | **research-artifact** |
| **Structural plasticity (synapse formation/elimination)** | absent — connectome is fixed | moderate (Witvliet 2020 developmental connectome shows changes) | mammalian extensive | very high — requires altering connectome over time | developmental data exists | development research, less relevant for adult worm | absent | **aspirational** (probably scope-limited to development) |

**Category H summary:** "Plasticity" framing largely imports mammalian assumptions that don't translate to worm. **Habituation/dishabituation is the worm-native plasticity** with strong literature (Rankin lab); modulator-induced gain changes are the second-most-grounded option. Hebbian LTP/STDP don't apply mechanistically. Realistic implementation: habituation circuit + modulator-induced gain at the network level. Lower priority for anesthesia/drug discovery; central for learning/memory research.

---

## Category I — Calcium signaling beyond pool

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **Bulk cytosolic Ca pool** | partially in h_kca patch (single ODE: d[Ca]/dt = α·I_Ca - [Ca]/τ) | rich GCaMP imaging | mammalian extensive | low — already abstracted | GCaMP traces | foundational Ca dynamics | partially in this project | **production** with α calibration (yesterday's work) |
| **IP3R Ca release (ITR-1)** | absent | moderate (Walker 2002, Yin 2004) | mammalian extensive | medium-high — Ca-IP3 dependent gating | itr-1 mutants (defecation cycle defects) | second messenger amplification, oscillations | absent | **research-artifact** (importable) |
| **Ryanodine receptor (UNC-68) Ca-induced-Ca-release** | absent | moderate (Maryon 1998, Sakube 1997) | mammalian extensive | medium-high — CICR dynamics | unc-68 mutants (locomotion defects) | CICR amplification, muscle dynamics | absent | **research-artifact** |
| **ER Ca pumps (SCA-1 SERCA)** | absent | moderate (Cho 2000) | mammalian extensive | medium — mass-action ODE | sca-1 mutants | ER Ca homeostasis | absent | **research-artifact** |
| **Mitochondrial Ca uptake/efflux** | absent | sparse worm-specific | mammalian extensive | high — coupled to mitochondrial dynamics | hard to validate in worm | bioenergetics, neurodegeneration research | absent | **aspirational** |
| **Calmodulin (CMD-1) buffering** | absent — α parameter approximates buffering effectively | rich (CMD-1 binding partners well-characterized) | mammalian extensive | medium — CaM binding partners | cmd-1 mutants (lethal) | Ca signaling specificity | absent | **research-artifact** |
| **Other Ca-binding proteins (calbindin homologs)** | absent | moderate | mammalian extensive | medium — fast/slow buffering ODEs | mutant phenotypes | Ca microdomain dynamics | absent | **research-artifact** |
| **Plasma membrane Ca extrusion (MCA-1/2/3 PMCA, NCX-2/3/9)** | absent — abstracted as decay τ | moderate (Gönczy 1997, Zhou 2018) | mammalian extensive | medium — explicit pump/exchanger ODEs | mutant phenotypes | Ca homeostasis, recovery kinetics | absent | **research-artifact** (current decay τ approximation may be sufficient) |
| **Ca microdomains/nanodomains** | absent | sparse worm-specific imaging | mammalian extensive | very high — spatial sub-cellular resolution | hard to measure in worm | release-site specificity, drug discovery | absent | **aspirational** |

**Category I summary:** Bulk Ca pool sufficient for current applications. ER Ca dynamics (ITR-1 IP3R, UNC-68 RyR, SCA-1 SERCA) is **research-artifact-grade work, importable from mammalian priors with worm validation**. Microdomains and mitochondrial Ca are aspirational. The α_Ca calibration question from yesterday's Wave 1 work specifically falls in this category — corrected α produces physiologically realistic [Ca] but more sophisticated buffering would improve fidelity.

---

## Category J — Other mechanisms identified

| Mechanism | Current state | Worm data | Adjacent priors | Impl complexity | Validation | Research utility | Existing impl | Tier |
|---|---|---|---|---|---|---|---|---|
| **Gap junction modulation (innexin gating)** | partially — UNC-9, UNC-7, INX-* gap junctions in connectome with fixed strength | rich (Liu 2017 antidromic-rectifying, Starich 2009) | invertebrate-specific (innexins ≠ vertebrate connexins) | medium — voltage-dependent gap junction conductance | unc-9 mutants | electrical coupling dynamics, drug discovery (Cx43 modulators) | partial in c302 | **research-artifact** |
| **Volume transmission of monoamines** | partially — modulator layer with diffusion length | moderate (5HT/DA spatial dynamics characterized in some neurons) | mammalian extensive | medium | imaging studies | drug pharmacokinetics | partial here | **research-artifact** |
| **Neurosteroid synthesis (DAF-9 pathway)** | absent | moderate (steroid hormones in dauer regulation) | mammalian extensive (allopregnanolone effects on GABA_A) | high | daf-9 mutants | anesthesia (neurosteroids modulate GABA_A), longevity | absent | **research-artifact** for endocrine; **research-artifact** for anesthesia coupling |
| **NaCl ion homeostasis (kidney-like)** | absent | sparse | mammalian moderate | medium | gcy-7 etc. | osmotic regulation, salt chemotaxis | absent | **aspirational** for now |
| **Neuron-glia interaction** | absent — no glia in current sim | moderate (CEPsh glia in nerve ring; AMsh, PHsh) | mammalian extensive | very high | glia ablation phenotypes | tripartite synapse, neurodegeneration research | absent | **aspirational** |
| **Activity-dependent metabolic coupling** | absent | sparse | mammalian extensive | very high | neuro-metabolic coupling | ATP demand, energy efficiency, anesthesia (volatile reduces metabolism) | absent | **aspirational** |

