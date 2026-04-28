# Tier-1 target panel rationale

**Status:** SCAFFOLDED. UniProt IDs are best-effort — re-verify against WormBase before Phase A entry.

This document provides the paper-by-paper justification for each Tier-1 target's inclusion. Every Tier-1 entry maps to either (a) a primary-source link to *C. elegans* anesthetic mechanism, or (b) a high-confidence mammalian-homolog anesthetic-binding established in the literature.

---

## Cys-loop receptor superfamily (12 targets)

### GABA receptor pentamers

**UNC-49** (homopentameric or heteropentameric inhibitory GABA-A homolog) — included because:

- *C. elegans*'s major inhibitory GABA receptor at the NMJ. Mutants are uncoordinated.
- Mammalian GABA-A potentiation is the most established anesthetic mechanism (volatile + intravenous). Mihic 1997 *Nature* defined the M2-domain anesthetic-sensitive site.
- Bamber 2003 (*Eur J Neurosci*) characterized UNC-49B receptor; established as the primary inhibitory channel.
- Predicted shift: τ_decay potentiation (slowed IPSC decay) at clinical anesthetic. Direction confirmed by mammalian.

**EXP-1** (cation-conducting GABA receptor) — included because:

- Unusual: GABA-gated cation channel (depolarizing, not hyperpolarizing).
- Beg & Jorgensen 2003 (*Nat Neurosci* PMID 14595442) characterized.
- May potentiate similarly to UNC-49 (M2 helix is conserved); mechanism diverges due to ion selectivity.

### GluCl pentamers (4 targets)

**AVR-14, AVR-15, GLC-1, GLC-2** — included because:

- *C. elegans*-specific GluCl is the **structurally best-anchored** Cys-loop in the panel: Hibbs & Gouaux 2011 (*Nature* PMID 21572436, PDB 3RHW) crystallized GluCl from *C. elegans*. Direct structural anchor.
- GluCl mediates ivermectin sensitivity; anesthetic potentiation has been characterized in mammalian glycine receptor (homologous α subunit).
- Dent 2000 (*EMBO J* — PMID lookup needed) characterized AVR-14/AVR-15 in pharyngeal pumping.
- AVR-14 is the strongest single anchor — direct *C. elegans* crystal structure is rare for ion channel literature.

### nAChR (6 targets)

**ACR-16** (homopentameric nAChR) — included because:

- Major NMJ acetylcholine receptor.
- Touroutine 2005 / Francis 2005 characterized ACR-16 as the dominant homomeric nAChR.
- Mammalian nAChR M2 anesthetic-sensitive site is well characterized.

**UNC-29 / UNC-38 / UNC-63 / LEV-1 / ACR-2** (heteropentameric levamisole receptor) — included because:

- The "levamisole-sensitive" receptor at the NMJ; together they form a heteropentameric nAChR.
- Anesthetic effects documented at mammalian heteromeric nAChRs.
- Boulin 2008 (*PNAS*) identified the subunit composition.

---

## K2P channels (3 targets)

**TWK-18** (gain-of-function halothane resistance) — **load-bearing anchor**:

- Sedensky 2001 *Am J Physiol Cell Physiol* PMID 11756669: TWK-18(cn110) is a gain-of-function allele of a K2P channel that confers halothane resistance.
- Mechanistically: GOF K2P → resting hyperpolarization → cell less excitable → resists anesthetic-induced quiescence (or, alternatively, reduces effect of anesthetic-driven K2P potentiation).
- Direction must reproduce: TWK-18 GOF → halothane EC50 right-shifted (resistance).

**TWK-7, TWK-29** — included because:

- Same family as TWK-18; expression patterns include locomotor and pharyngeal neurons.
- Anesthetic effect direction expected to match TWK-18.

---

## NCA channel complex (5 targets)

**NCA-1, NCA-2, UNC-79, UNC-80, NLF-1** — **load-bearing anchor**:

- Sedensky 1992 *Genetics* PMID 1346264: unc-79(e1068) and unc-80(e1069) are halothane-resistant.
- NCA-1, NCA-2 are NALCN homologs; UNC-79 and UNC-80 are auxiliary subunits forming the channel complex.
- Mammalian NALCN: Xie 2020 (PMID 33020732, PDB 7SX3) cryoEM. Anesthetic effects characterized.
- Mechanism: anesthetic blocks NCA → reduces Na+ leak → reduced excitability. unc-79 / unc-80 mutants without functional NCA are baseline less excitable → anesthetic does less, → resistance.
- UNC-79 and UNC-80 are large proteins (>1000 aa) with significant intrinsically disordered regions — AlphaFold confidence may be limited at full length. Phase A focuses on structured domains.

---

## SNARE machinery (6 targets)

**UNC-64, RIC-4, SNB-1** (the four-helix bundle SNARE) — included because:

- The SNARE complex is the membrane-fusion machine; all volatile anesthetics demonstrably reduce vesicle release in mammalian and worm preparations.
- van Swinderen 2004 (PMID lookup needed) — Ca cooperativity reduction by halothane in *C. elegans* NMJ.
- Sutton 1998 *Nature* PDB 1SFC — neuronal SNARE bundle structure.
- Predicted shift: reduced bundle assembly rate, reduced fusion efficiency.

**UNC-13** (priming factor) — **load-bearing anchor**:

- van Swinderen 1999 (PMID lookup needed): unc-13 mutants are halothane hypersensitive.
- Richmond 1999 (PMID 10570485): unc-13(s69) hypomorph has 80-90% reduced release.
- Predicted shift: anesthetic reduces UNC-13 priming activity → reduced release.

**UNC-18** (SM protein) — included because:

- Required for SNARE-complex formation; binds syntaxin-1A (UNC-64).
- Mammalian Munc18 anesthetic interaction documented.

**SNT-1** (synaptotagmin Ca sensor) — included because:

- Ca cooperativity in release is mediated by Synaptotagmin's C2A/C2B domains.
- The van Swinderen 2004 cooperativity reduction (n_Ca: 3.5 → 2.0) is most parsimoniously explained by anesthetic effect on the Ca-sensor.

---

## Mitochondrial Complex I (7 targets, plus mev-1 control)

**GAS-1** (NDUFS2 homolog, 49 kDa core subunit) — **load-bearing anchor**:

- Morgan & Sedensky 1995 *Genetics* PMID 7549290: gas-1(fc21) is hypersensitive to volatile anesthetics by 2-3×.
- Kayser 2001: gas-1 mutant has 30-50% reduced Complex I activity.
- The mechanistic link to immobilization: Complex I block → ATP drift → K-ATP partial open → resting hyperpolarization → cell more sensitive to anesthetic-induced shifts.

**NUO-1, NUO-2, NUO-3, NUO-4, NUO-5, NUO-6** — included because:

- Other Complex I subunits; Falk 2006 / Kayser follow-ups suggest milder phenotypes.
- nuo-1 mutants are also hypersensitive (smaller effect than gas-1).
- Predicted shift: same as GAS-1, smaller magnitude.

**MEV-1** (Complex II SDHC homolog) — control:

- mev-1(kn1) has Complex II defect; reduced anesthetic hypersensitivity vs Complex I mutants.
- Used to show Complex I is the dominant anesthetic-sensitive node — Complex II is not.

---

## What's NOT in Tier 1 (and why)

Items that might naively appear in Tier 1 but are deferred to Tier 2:

- **EGL-19** — already in Wave 2; Wave P consumes its translation. No need to re-examine for Tier 1.
- **NMR-1 / NMR-2 (NMDA)** — primary ketamine target. Ketamine is a control anesthetic in Wave P; NMR-1/-2 are most relevant if ketamine pivots to primary panel. Tier 2.
- **GLR-1 through GLR-8 (AMPA-like)** — anesthetic effect on AMPA is documented but secondary to GABA. Tier 2.
- **EGL-19, UNC-2, CCA-1 (Ca channels)** — anesthetic effects exist but are not the dominant mechanism for immobilization in mammals. Tier 2.
- **Voltage-gated K (SHK-1, SHL-1, KVS-1, SLO-1/2)** — many are Wave 2-translated; Tier 2 for anesthetic effect.
- **Inwardly rectifying K (IRK-1, IRK-2, IRK-3)** — IRK-1 is in AVA's channel set; Wave 2 in flight. Tier 2 for anesthetic effect.
- **Peptide processing (EGL-3, EGL-21, KPC-1, NEP-1)** — modulator-layer effects; Tier 2.
- **Monoamine receptors** — secondary to direct neurotransmitter receptors; Tier 2.

---

## Predicted oligomer state column rationale

The CSV's `predicted_oligomer_state` column drives Phase A's choice of monomer-only vs multimer prediction:

- **homopentamer** (Cys-loop alpha subunit) → AlphaFold-Multimer with 5 copies of the same sequence.
- **heteropentamer** → AlphaFold-Multimer with documented stoichiometry; if stoichiometry uncertain, predict the most-common composition first.
- **homodimer** (K2P) → AlphaFold-Multimer with 2 copies.
- **heterotetramer** (NCA complex) → predict NCA-1 + UNC-79 + UNC-80 assembly via AlphaFold-Multimer; computational expense.
- **monomer** (SNARE individual subunits, kinase-like) → AlphaFold DB pull.

---

## Pocket compartment column rationale

The CSV's `pocket_compartment` column drives Phase C's membrane-partition adjustment:

- **membrane_embedded** → use K_p × [aqueous] for anesthetic concentration at site.
- **membrane_interfacial** → use K_p × [aqueous] (close enough; the partition still applies near bilayer interface).
- **aqueous_extracellular** → use bulk aqueous concentration.
- **aqueous_intracellular** → use bulk cytosolic concentration (~equal to aqueous bulk for volatile anesthetics due to fast permeation).

For SNARE machinery, RIC-4 (SNAP-25) is annotated as `membrane_interfacial` because its lipid-anchored cysteines tether it to the membrane interface — anesthetic concentration at its surface is membrane-side. UNC-13 / UNC-18 / SNT-1 are predominantly cytosolic (or peripheral); SNT-1's C2 domains insert into the membrane upon Ca binding (transient `membrane_interfacial`).

---

## Pre-Phase-A blocking items

Before Phase A executes, the following must resolve:

1. **UniProt ID verification.** The UniProt IDs in `tier1_targets.csv` were assigned best-effort at kickoff. Re-verify each against current WormBase + UniProt before pulling AlphaFold DB structures.
2. **Pocket residue specification.** `targets/pocket_residues_homolog.csv` is empty; Phase A populates it from sequence alignments to mammalian homologs. Some entries (UNC-79, UNC-80, NLF-1) may have no homolog-derived pocket and require de novo identification via fpocket.
3. **License terms.** AlphaFold-Multimer + RoseTTAFold-AllAtom license verification.
4. **Citation verification.** The (PMID lookup needed) markers in this document and in phase preregistration documents must resolve before the first phase enters its execution work block.
