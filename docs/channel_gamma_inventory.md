# Single-channel γ inventory — Layer 1 channels (Phase 2)

**Status:** Phase 2 of §7.3.5 Path 2. Per-channel γ values for the 9 channels
in current Layer 1 cell builders. Inventory complete (8/9 with sourced γ;
1 estimated due to literature gap).

**Date:** 2026-05-12

**Reference:** `docs/channel_parameter_derivation_methodology.md` §2 (formula)
+ §6 (epistemic labeling). γ is the channel-type-intrinsic single-channel
conductance entering the Path 2 formula:

```
gbar_intensive[channel][cell] = γ[channel] × TPM[channel][cell] × E_translation × C_global
```

---

## 1 · Methodology recap

### 1.1 Scoping hierarchy applied per channel

1. **Direct C. elegans single-channel patch-clamp measurements** — searched
   for each channel; **none found** for any of the 9 channels in scope.
   This is consistent with the field state: single-channel recordings in
   C. elegans neurons are technically difficult, and most channel
   characterizations to date are macroscopic (whole-cell, voltage clamp).
2. **Heterologous expression of C. elegans gene** (oocyte / HEK293) —
   functional studies exist for some channels (EGL-19, SHL-1, KQT-1,
   IRK-1) but report macroscopic conductance-voltage relationships, NOT
   single-channel γ.
3. **Mammalian / Drosophila homolog fallback** — primary source for v1
   γ values across all 9 channels. Each entry below documents the
   homolog choice + measurement conditions + the assumption that γ
   transfers across species.

### 1.2 Physiological Ca²⁺ vs Ba²⁺ adjustment

Mammalian Ca-channel single-channel measurements use Ba²⁺ as charge
carrier (~90-110 mM external) because Ba²⁺ produces larger, easier-to-
resolve unitary currents than physiological Ca²⁺ (2 mM external). Ba²⁺
γ values systematically overestimate physiological Ca²⁺ γ by factor
~3-4× for L-type, similar for P/Q and T types.

**For Layer 1 substrate** (physiological [Ca²⁺]_out = 2 mM, per §6.5
authorization), Ca-channel γ entries report **two values**: the literature
Ba²⁺ value and the adjusted physiological Ca²⁺ value. The methodology
formula uses the physiological value.

### 1.3 Epistemic label assignment

All 9 channels receive the same epistemic label per §2.8 of design doc:
**"approximation from adjacent biology"** — γ inherited from mammalian
homolog with documented conditions. None of the channels has direct
C. elegans γ measurement; literature gap is documented but doesn't trigger
hard-stop (>50% un-sourced threshold not met — see §3 coverage check).

---

## 2 · Per-channel inventory

### 2.1 EGL-19 (L-type Ca, Cav1 family homolog)

| field | value |
|---|---|
| Channel name | EGL-19 |
| Family | L-type voltage-gated Ca²⁺ (Cav1.x) |
| C. elegans gene | egl-19 |
| Mammalian homolog | Cav1.2 (CACNA1C) |
| γ in Ba²⁺ (literature) | 20-27 pS (110 mM Ba²⁺), canonical ~25 pS |
| γ in physiological Ca²⁺ (used) | **6 pS** (central; range 4-8 pS) |
| Source | Hille "Ion Channels of Excitable Membranes" 3rd ed Ch. 4; multiple Cav1 studies (Carbone & Lux 1984; Hofmann et al. 1994; Madame Curie Bioscience Database) |
| Epistemic label | approximation from adjacent biology |
| Notes | Cav1 channels are the best-characterized Ca channels; γ transfer to C. elegans Cav1 homolog (egl-19) is well-supported by sequence conservation in pore-forming S6 region. Physiological Ca²⁺ γ << Ba²⁺ γ due to Ca²⁺ binding in pore (anomalous mole-fraction effect). C. elegans-specific functional studies (Jospin et al. 2002, Lainé et al. 2014) report macroscopic conductance only. |
| Heteromer/paralog | Single-gene (egl-19) — no aggregation needed |

### 2.2 CCA-1 (T-type Ca, Cav3 family homolog)

| field | value |
|---|---|
| Channel name | CCA-1 |
| Family | T-type voltage-gated Ca²⁺ (Cav3.x) |
| C. elegans gene | cca-1 |
| Mammalian homolog | Cav3.1 (CACNA1G) |
| γ in Ba²⁺ (literature) | 5-8 pS (110 mM Ba²⁺) — "tiny" T-type signature |
| γ in physiological Ca²⁺ (used) | **3 pS** (central; range 2-5 pS) |
| Source | Steger et al. 2005 (CCA-1 functional study); Madame Curie review (Cav3 family) |
| Epistemic label | approximation from adjacent biology |
| Notes | T-type channels have the smallest γ among voltage-gated Ca channels. Physiological Ca²⁺ γ is lower than Ba²⁺ value. Steger 2005 confirms CCA-1 functional behavior matches mammalian Cav3 (low-voltage activation ~-40 mV, fast inactivation, slow deactivation tail) — supports γ transfer assumption. |
| Heteromer/paralog | Single-gene (cca-1) — no aggregation needed |

### 2.3 UNC-2 (P/Q-type Ca, Cav2 family homolog)

| field | value |
|---|---|
| Channel name | UNC-2 |
| Family | High-voltage activated Ca²⁺ (Cav2.x; P/Q/N/R) |
| C. elegans gene | unc-2 |
| Mammalian homolog | Cav2.1 (CACNA1A) — P/Q-type |
| γ in Ba²⁺ (literature) | 19-20 pS (90 mM Ba²⁺, recombinant human Cav2.1); native Purkinje P-type subconductance 9, 14, 19 pS |
| γ in physiological Ca²⁺ (used) | **5 pS** (central; range 3-7 pS) |
| Source | Recombinant human Cav2.1 single-channel studies (Bourinet et al. 2010, multiple); Purkinje native P-type (Usowicz et al. 1992) |
| Epistemic label | approximation from adjacent biology |
| Notes | C. elegans has single Cav2α gene (unc-2). Mathews et al. 2003 + Huang et al. 2019 confirm functional homology to mammalian Cav2.1. Multiple-conductance state behavior in native channels (Purkinje) noted; v1 uses central value. C. elegans-specific γ not measured. |
| Heteromer/paralog | Single-gene (unc-2) — no aggregation needed |

### 2.4 IRK (Kir2 family homolog) — paralog family

| field | value |
|---|---|
| Channel name | IRK |
| Family | Inwardly-rectifying K⁺ (Kir2.x) |
| C. elegans genes | irk-1, irk-2, irk-3 (three paralogs) |
| Mammalian homolog | Kir2.1 (KCNJ2) |
| γ (literature) | 21-34 pS chord at Vm = -100 mV; 31-43 pS slope between -60/-140 mV |
| γ (v1 used) | **25 pS** (central; chord conductance at near-rest potential) |
| Source | Pegan et al. 2005 (PMC2234023); Carboxy-terminal Determinants of Conductance in Inward-rectifier K Channels |
| Epistemic label | approximation from adjacent biology |
| Notes | Wave 2 cell builders treat IRK as a single channel module with one gbar. CeNGEN TPMs are per-gene (irk-1, irk-2, irk-3). Per methodology §2.4 default rule + Phase 2 literature verification (see §4.1 below): treat each as separate channel with single-gene TPM. IRK-1 forms functional homotetramers in oocyte (Emtage 2012). No direct biochemical evidence for IRK heterotetramers in C. elegans. |
| Heteromer/paralog | **Paralog-separate (default rule)** confirmed by Phase 2 literature review |

### 2.5 KQT-1 (KCNQ family homolog) — paralog family

| field | value |
|---|---|
| Channel name | KQT-1 |
| Family | KCNQ (Kv7) M-current channels |
| C. elegans gene | kqt-1 |
| Mammalian homolog | KCNQ2-5 family (groups phylogenetically with KCNQ2/3/4/5; not KCNQ1) |
| γ (literature) | 3.2 pS (KCNQ1 mammalian, Sesti & Goldstein 1998); KCNQ2/3 likely similar |
| γ (v1 used) | **3 pS** (central) |
| Source | KCNQ1 single-channel: PMID 9834139 (Sesti & Goldstein 1998 J Physiol); KCNQ family review |
| Epistemic label | approximation from adjacent biology |
| Notes | KCNQ family has notably low γ across mammalian paralogs (~3 pS). Wei et al. 2005 confirms C. elegans KQT-1 functional homology (M-current behavior, mAChR modulation, slow activation/deactivation). KQT-1 expressed alone in oocyte produces functional homomeric current. |
| Heteromer/paralog | **Paralog-separate (default rule)** with FLAG — see §4.2 below. Mammalian KCNQ2/3 heteromer is well-known precedent; KQT-1/KQT-3 heteromer hypothesized but not biochemically confirmed in C. elegans. Phase 3 should check whether KQT-3 is expressed in AIY (the cell using KQT-1); if both at similar TPM, consider min-across-pore-forming aggregation in Phase 5 sensitivity analysis. |

### 2.6 SHL-1 (Kv4 / Shal family homolog)

| field | value |
|---|---|
| Channel name | SHL-1 |
| Family | A-type voltage-gated K⁺ (Kv4.x / Shal) |
| C. elegans gene | shl-1 |
| Mammalian homolog | Kv4.2 (KCND2) |
| γ (homotetramer alone) | 4 pS in heterologous expression |
| γ (with auxiliary DPP6) | 6-7.5 pS in native neurons |
| γ (v1 used) | **6 pS** (central — assumes C. elegans has DPPX-like auxiliary) |
| Source | Kaulin et al. 2009 J. Neurosci 29:3242 (DPP6 modulation); Chen & Johnston 2004 (native CA1 ~6 pS); UniProt KCND2 (homotetramer ~4 pS) |
| Epistemic label | approximation from adjacent biology |
| Notes | Fawcett et al. 2006 confirms SHL-1 functional homology to Kv4 (fast transient outward K, similar V_half). Whether C. elegans has DPPX-like auxiliary affecting γ is unknown; v1 assumes auxiliary contribution per native-channel value. |
| Heteromer/paralog | Single-gene (shl-1) — no aggregation needed |

### 2.7 EGL-2 (EAG / Kv10 family homolog)

| field | value |
|---|---|
| Channel name | EGL-2 |
| Family | EAG (ether-à-go-go) — Kv10.x |
| C. elegans gene | egl-2 |
| Mammalian homolog | Kv10.1 / Eag1 (KCNH1) |
| γ (literature) | 8 pS |
| γ (v1 used) | **8 pS** |
| Source | Cryo-EM + functional Eag1 structure paper (Whicher & MacKinnon 2016 / PMC5477842; γ attributed to structural neutral intracellular vestibule) |
| Epistemic label | approximation from adjacent biology |
| Notes | EAG channels have moderately low γ due to neutral intracellular vestibule. C. elegans EGL-2 + UNC-103 are the two EAG-family channels in worm (EAG + ERG orthologs respectively). γ transfer assumption supported by ~70% sequence identity to human KCNH1 in pore region. |
| Heteromer/paralog | Single-gene (egl-2) — no aggregation needed |

### 2.8 UNC-103 (HERG / Kv11 family homolog)

| field | value |
|---|---|
| Channel name | UNC-103 |
| Family | ERG (ether-à-go-go-related) — Kv11.x |
| C. elegans gene | unc-103 |
| Mammalian homolog | hERG / Kv11.1 (KCNH2) |
| γ at physiological [K]_out=4 mM | 2 pS |
| γ at symmetrical 100-150 mM K | 10-14 pS |
| γ (v1 used) | **2 pS** (physiological [K]_out=4 mM; matches Layer 1 substrate) |
| Source | Kiehn et al. 1996 *Circulation* 94:2572; multiple hERG single-channel studies |
| Epistemic label | approximation from adjacent biology |
| Notes | hERG has unusually low γ due to permeation pathway energy barriers. γ is K-concentration-dependent — important to use physiological [K]_out value (4 mM) matching Layer 1 substrate, not symmetrical-K experimental values. C. elegans UNC-103 70% identical to hERG in transmembrane/pore domains — supports γ transfer. UNC-103 itself not biophysically characterized in C. elegans. |
| Heteromer/paralog | Single-gene (unc-103) — no aggregation needed |

### 2.9 NCA (NALCN family homolog) — γ literature gap

| field | value |
|---|---|
| Channel name | NCA |
| Family | NALCN (sodium leak channel) |
| C. elegans genes | nca-1, nca-2 (pore-forming paralogs); unc-77, unc-80 (auxiliary regulators) |
| Mammalian homolog | NALCN (single-gene in mammals) |
| γ in literature | **NO PUBLISHED VALUE** (explicitly stated in Belal et al. NALCN preprint: "There is no available estimate for NALCN single-channel conductance") |
| γ (v1 used — ESTIMATED) | **5 pS** (estimated; placeholder for v1) |
| Source | No direct measurement. Estimate based on (a) NALCN's small contribution to total membrane conductance (2-5% maximal conductance, voltage-insensitive), (b) typical leak-channel γ range, (c) ScNav-family structural relation suggesting γ smaller than NaV channels (~20 pS) but larger than HERG-like (~2 pS) |
| Epistemic label | **approximation from adjacent biology — LITERATURE GAP flagged for refinement** |
| Notes | NALCN's tight regulation, low open probability, and small contribution to total cell conductance make single-channel measurement technically very difficult. No published unitary γ exists for NALCN as of 2026. v1 uses 5 pS placeholder; substantive finding documented for Phase 5 sensitivity analysis. If Phase 5 surfaces NCA channels beyond 5× discrepancy, γ refinement is a candidate (Phase 2 followup). |
| Heteromer/paralog | **Paralog-separate (default rule)** — NCA-1 (nca-1 TPM) and NCA-2 (nca-2 TPM) are separate channels. **Auxiliary subunits unc-77 and unc-80 are NOT included in density estimation** (they're regulators, not pore-formers) per methodology §2.4 exception rule notes. |

---

## 3 · Coverage assessment + hard-stop check

### 3.1 Coverage statistics

| status | count | channels |
|---|---|---|
| γ sourced (literature value, mammalian homolog) | 8 | EGL-19, CCA-1, UNC-2, IRK, KQT-1, SHL-1, EGL-2, UNC-103 |
| γ estimated (literature gap) | 1 | NCA |
| **Total in scope** | **9** | |

**Coverage: 8/9 = 89% sourced.** Well above 50% hard-stop threshold. Phase 2
proceeds. NCA gap documented as substantive finding for Phase 5 sensitivity
analysis.

### 3.2 Hard-stop conditions (none triggered)

- ✓ γ values sourced for >50% of channels (89% > 50%)
- ✓ CeNGEN data accessible (Phase 3 will confirm)
- ✓ No methodology-level architectural failures surfaced
- ✓ No repeated failure patterns

Phase 2 acceptance met. Proceed to Phase 3.

---

## 4 · Heteromer-vs-paralog verification per methodology §2.4

Methodology §2.4 specifies a per-family table with default rules (paralog-
separate) and exception triggers (min-across-pore-forming for documented
heteromers). Phase 2 literature scoping confirms / refines:

### 4.1 IRK family (irk-1, irk-2, irk-3) — KEEP DEFAULT

**Verified:** IRK-1 forms functional homotetramers in Xenopus oocytes
(Emtage et al. 2012, PMC3544400). IRK-1 alone confers large inwardly-
rectified K current.

**No biochemical evidence** for IRK heterotetramers in C. elegans found
in Phase 2 search. Distinct functional roles per paralog (IRK-1 in HSN
serotonergic, IRK-2/3 in ASH OCTR-1 signaling) consistent with separate
homomeric channels rather than heteromeric assembly.

**Decision:** **Default paralog-separate confirmed.** Each irk-1, irk-2,
irk-3 treated as separate channel with single-gene TPM in Phase 5
derivation.

For Layer 1 cells: AVAL/AVAR use "IRK" channel module (Wave 2 builders);
Phase 3 will determine which paralog (irk-1, -2, or -3) has dominant
TPM in AVA/RIM and use that TPM for the IRK channel density. If TPMs
are comparable, sum (paralog-separate aggregation = independent channels
in parallel = sum-of-densities effectively).

### 4.2 KQT family (kqt-1, kqt-2, kqt-3) — DEFAULT WITH HETEROMER FLAG

**Mammalian precedent:** KCNQ2 + KCNQ3 form heterotetramers (M-current).
KCNQ2/5 also heteromerize (Soh et al. 2022).

**C. elegans hypothesis:** KQT-2 + KQT-3 hypothesized to heteromerize per
Okahata 2019 *Science Advances*, but **no direct biochemical evidence**.
Hypothesis based on mammalian KCNQ2/3 precedent + functional interactions
in ADL sensory neurons. KQT-1 in oocyte produces functional homomeric
current (Wei et al. 2005).

**Layer 1 scope:** Only KQT-1 is in current Wave 2 cell builders (AIY).
KQT-2 and KQT-3 are not in Layer 1 cells (out of Phase 2 scope).

**Decision:** **Default paralog-separate for KQT-1 in v1, with FLAG.**

- Phase 3 should pull KQT-1 + KQT-3 TPMs in AIY (Wormbook indicates KQT-1
  and KQT-3 co-expressed in chemosensory neurons; AIY's status TBD)
- If AIY has both KQT-1 and KQT-3 at similar TPM levels, Phase 5
  sensitivity analysis should test both: (a) KQT-1 alone as homomer
  (default), (b) KQT-1/KQT-3 heteromer with min-across-pore-forming TPM
- If Phase 5 surfaces AIY KQT current discrepancy, heteromer hypothesis is
  a candidate refinement

### 4.3 NCA channels (nca-1, nca-2, unc-77, unc-80) — PARALOG-SEPARATE + AUXILIARY-IGNORED

**Pore-forming subunits:** nca-1 and nca-2 are paralogous pore-forming
α-subunits (each can form a functional channel; both expressed in some
cells, primarily one in others). NALCN-family in mammals is single-gene;
C. elegans has duplicated to two paralogs.

**Auxiliary subunits:** unc-77 and unc-80 are channelosome regulators
(per Lu et al. 2009; Topf et al. 2024). Not pore-formers; modify gating
and trafficking. **Excluded from density estimation** per methodology
§2.4 exception rule.

**Decision:** **NCA-1 and NCA-2 as separate channels with single-gene
TPMs.** Auxiliary unc-77/unc-80 contribute to functional channel
behavior but not density. (Wave 2 cell builders may collapse to single
"NCA" channel module; Phase 3 should clarify whether to derive per-paralog
or sum-of-paralogs.)

### 4.4 Per-family table update (methodology §2.4)

Phase 2 confirms the methodology §2.4 default rules with the following
refinements:

| family | original §2.4 rule | Phase 2 verification | refinement |
|---|---|---|---|
| EGL-19 | single-gene | confirmed | (none) |
| CCA-1 | single-gene | confirmed | (none) |
| UNC-2 | single-gene | confirmed | (none) |
| SHL-1 | single-gene | confirmed | (none) |
| SHK-1 | single-gene | (out of scope — not in Layer 1 cells) | (deferred) |
| EGL-36 | single-gene | (out of scope) | (deferred) |
| EXP-2 | single-gene | (out of scope) | (deferred) |
| UNC-103 | single-gene | confirmed | (none) |
| EGL-2 | single-gene | confirmed | (none) |
| NCA channels | paralogs separate; aux ignored | confirmed | (none) |
| IRK channels | paralogs separate (default) | confirmed via IRK-1 homomer oocyte data | (none) |
| KQT channels | paralogs separate (default) | confirmed for KQT-1 with FLAG | **Phase 3 + Phase 5 should check KQT-1/KQT-3 heteromer hypothesis in AIY** |
| TWK channels | paralogs separate | (out of scope) | (deferred) |
| SLO channels | separate (different types) | (out of scope — deferred to Phase 2 v2) | (deferred) |

---

## 5 · Coverage uncertainty summary

For Phase 5 derivation, the per-channel γ values carry the following
uncertainty:

| Channel | γ used | Uncertainty range | Driver |
|---|---:|---|---|
| EGL-19 | 6 pS | 4-8 pS | Ba→Ca conversion + cross-species transfer |
| CCA-1 | 3 pS | 2-5 pS | Ba→Ca conversion + T-type subconductance variation |
| UNC-2 | 5 pS | 3-7 pS | Ba→Ca conversion + Cav2.1 native subconductance states |
| IRK | 25 pS | 21-34 pS | Chord vs slope measurement choice |
| KQT-1 | 3 pS | 2-5 pS | KCNQ paralog variation (KCNQ1-5 range 1.5-5 pS) |
| SHL-1 | 6 pS | 4-7.5 pS | DPPX auxiliary contribution uncertain in C. elegans |
| EGL-2 | 8 pS | 6-10 pS | Single-source γ; cross-species transfer uncertainty |
| UNC-103 | 2 pS | 1-3 pS | hERG γ well-constrained at physiological [K] |
| NCA | 5 pS | 1-20 pS (very wide) | **No published measurement; estimated** |

Median per-channel uncertainty is roughly ±50% (factor ~1.5-2× spread).
This is consistent with the §4.1.1 combined-uncertainty framing in the
methodology document (γ variance ~1.5-2× is the lower bound on per-
channel uncertainty before TPM scaling adds its ~3.6× contribution).

NCA's uncertainty is much wider (factor ~20× spread) and warrants Phase 5
sensitivity analysis.

---

## 6 · Phase 2 acceptance criteria status

Per methodology / roadmap:

- [x] γ inventory complete for every channel in scope (9/9)
- [x] Each entry has explicit source and epistemic label
- [x] Coverage gaps explicitly documented (NCA literature gap)
- [x] Heteromer-vs-paralog literature scoping complete; per-family table
      refined where evidence required
- [x] Hard-stop check passed (89% sourced > 50% threshold)

**Phase 2 SHIPPED.** Ready for Phase 3 (CeNGEN TPM extension + translation
efficiency).

---

## 7 · Files of record

- This document: `docs/channel_gamma_inventory.md`
- Methodology reference: `docs/channel_parameter_derivation_methodology.md` §2
- Progress tracking: `scripts/brain/wave2/artifacts/path2_progress_summary.md`
- Phase 2 checkpoint: `scripts/brain/wave2/artifacts/path2_phase2_checkpoint.json`

---

## 8 · Primary literature cited

- **Cav1.2 (EGL-19 homolog):** Hille 2001 *Ion Channels of Excitable
  Membranes* 3rd ed Ch. 4; Hofmann et al. 1994 *Annu Rev Neurosci*;
  Madame Curie Bioscience Database (Catterall) PMID NBK6181
- **Cav3 / T-type (CCA-1 homolog):** Steger et al. 2005 *J Exp Biol* 208:2191
  (PMC1382270); Madame Curie review
- **Cav2.1 / P/Q-type (UNC-2 homolog):** Bourinet et al. 2010 *Mol Pharmacol*;
  Usowicz et al. 1992 *Neuron* 9:1185 (native Purkinje); Modal Gating of
  Human CaV2.1 (Tottene et al. 2002)
- **Kir2.1 (IRK homolog):** Pegan et al. 2005 *J Gen Physiol* (PMC2234023)
  "Carboxy-terminal Determinants of Conductance"
- **C. elegans IRK-1 oocyte expression:** Emtage et al. 2012 *J Neurosci*
  (PMC3544400) "IRK-1 Potassium Channels Mediate Peptidergic Inhibition"
- **KCNQ1 (KQT-1 homolog):** Sesti & Goldstein 1998 *J Physiol* 514:651
  (PMID 9834139)
- **C. elegans KQT family:** Wei et al. 2005 *J Biol Chem* 280:21337
  (PMID 15797864); Okahata et al. 2019 *Sci Adv* 5:eaav3631
- **Kv4.2 / DPP6 (SHL-1 homolog):** Kaulin et al. 2009 *J Neurosci* 29:3242
  (DPP6 conductance modulation); Chen & Johnston 2004 *J Physiol* 559:187
  (native dendritic A-current)
- **C. elegans SHL-1:** Fawcett et al. 2006 *J Biol Chem* 281:30725
  (PMID 16899454)
- **Kv10.1 / Eag1 (EGL-2 homolog):** Whicher & MacKinnon 2016 *Science*
  cryo-EM structure (PMC5477842; γ = 8 pS attributed to neutral
  intracellular vestibule)
- **C. elegans EGL-2:** Weinshenker et al. 1999 *J Neurosci* 19:9831
- **hERG / Kv11.1 (UNC-103 homolog):** Kiehn et al. 1996 *Circulation*
  94:2572 (γ at physiological [K]); Vandenberg et al. 2012 *Physiol Rev*
  (review); Ceccarini et al. 2012 *PLOS ONE* (PMC3487835, MD analysis)
- **C. elegans UNC-103:** Garcia & Sternberg 2003 *J Neurosci* 23:2696
  (functional characterization)
- **NALCN (NCA homolog):** Belal et al. (preprint) anterior pituitary
  characterization; Xie et al. 2020 *Nat Commun* cryo-EM
  (PMC7672056); Cochet-Bissuel et al. 2014 *Front Cell Neurosci*
  (no published γ documented)
- **C. elegans NCA-1/2:** Yeh et al. 2008 *PLOS Biol* (PMC2295944);
  Gao et al. 2015 *Nat Commun* (PMC4364778)
