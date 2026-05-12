# Layer 1 design decisions — ion concentrations + dynamic Nernst + pumps

**Status:** §6 ambiguities resolved 2026-05-12. Layer 1 §7.1 foundation work
block in progress.

**Date:** 2026-05-12 (v1: pre-flight; v2: post-authorization)

**Scope:** Per-cell `[K]_in`, `[Na]_in`, `[Cl]_in`, `[Ca]_in` as Brian2 state
variables for the four production-grade Wave 2 cells (AVAL, AVAR, AIY, RIM);
fixed extracellular reservoir; single whole-cell compartment;
Na/K-ATPase + Ca-extrusion + KCC-2 (+ ABTS-1, contingent); dynamic Nernst per
Brian2 dt; pump current routed into Phase F's metabolic balance; full validation
panel.

**Out of scope (Layer 1):** Extending to other 296 cells (Layer 1 v2, gated on
CeNGEN-conductance coupling refinement); intracellular Ca spatial buffering
(Layer 4); dynamic extracellular reservoir (Layer 1 v2 if validation surfaces
need); Mg/Pi/HCO3 (Layer 1 v2); compartmentalization (Layer 4); Layers 2-7
work blocks.

---

## 1 · Trajectory context

Phase G v1's LIFBrain integration hard-stopped on 2026-05-12 at CP2
calibration: 0% behavioral suppression across 5 orders of dose. Three
implementation bugs explained the null, but the deeper issue is substrate
adequacy — LIFBrain represents synapses as signed scalar weights, with no
GluCl conductance, K-ATP channel, or NCA leak channel for halothane /
sevoflurane / halogen mechanism classes to engage with. Even a bug-fixed Phase
G v2 would ship hooks that proxy biological mechanisms not present in the
substrate.

The full-stack bottom-up substrate redesign is committed (~30-44 work blocks
across Layers 1-7). Layer 1 lands ion concentration state + dynamic Nernst +
pumps on the four existing production-grade Wave 2 cells as the foundation
for everything downstream.

---

## 2 · Resolved design decisions (per Rohit authorization 2026-05-12)

### 2.1 V-vs-concentration coupling — Architecture (a)

Concentration-determined dynamics. `dV/dt = -(Σ I_chan + I_pump) / Cm` and
`d[X]/dt = -I_X / (z_X · F · vol)` enforced simultaneously per Brian2 step,
with Nernst potentials recomputed each step from concentrations. Membrane
charge conservation and ion mass conservation are coupled exactly. This is
what makes Phase G perturbations bite — anesthetic-driven [Cl]_in shifts
feed back into V via dynamic E_Cl.

### 2.2 Production-grade redefined

Nicoletti-5dp byte-match is the wrong validation target under a substrate
where Nernst emerges from ion gradient dynamics. New definition:

- Resting state: dynamically maintained gradients producing V within
  Nicoletti's published phenotype envelope (not byte-match).
- Voltage-clamp response: published I-V relationships within Nicoletti's
  reported SEM tolerance.
- Current-clamp response: phenotype category preserved (plateau / graded /
  spiking); rise/decay timescales within published ranges.
- Ion homeostasis: gradients return to rest after perturbation on biological
  timescales (~seconds for [K], [Na], [Cl]; ~seconds for [Ca]).

Rohit's framing: "moving from 'matches data' to 'reproduces data from
physics'." Documented as upgrade in epistemic rigor, not regression in
metric.

### 2.3 Cl cotransporter inclusion

**KCC-2 required.** Without K-Cl symport, GABA-A/GluCl-mediated Cl influx
accumulates intracellular Cl without clearance and the cell loses its
hyperpolarizing reversal within seconds — the exact mechanism Phase G is
trying to model would self-destruct.

**ABTS-1 added** (Na-Cl/HCO3 exchanger). CeNGEN T2 finding (§3.2) shows
abts-1 strongly co-expressed with kcc-2 in all three cells (AVA 569, AIY
206, RIM 224 TPM). Tanis 2015 (PMC4391577) + Bellemer 2011 (PMC3101993)
identify ABTS-1 as the second adult Cl extruder; some neurons (ventral cord
motor) express only KCC-2 but the three target interneurons co-express
both.

**NKCC-1 deferred.** CeNGEN T2 + T4 both show nkcc-1 = 0 in AVA, AIY, RIM —
consistent with adult Cl-loader absence in mature head interneurons (Tanis
2015 reports NKCC-1 in body wall muscles + "subset of head and tail
neurons" not enumerated; the target interneurons aren't in that subset at
either CeNGEN threshold).

### 2.4 Ca buffering — phenomenological factor (κ_B ≈ 100)

Phenomenological `d[Ca]_free/dt = (1/(1+κ_B)) · (-I_Ca / (2 · F · vol) -
I_clear)`. κ_B = 100 as central value; treated as parameter for tuning if
validation surfaces residuals. Explicit one-state buffer kinetics deferred
to Layer 1 v2 contingent on validation.

### 2.5 Cell volume — capacitance-derived with documented uncertainty

Nicoletti 2024 publishes total capacitance (AVAL 9.66 pF, AVAR 8.43 pF,
AIY 1.05 pF, RIM 1.55 pF). Surface area from `C_m / specific_Cm` with
specific_Cm = 0.86 μF/cm² (per AVAL Wave 2 spec). Volume requires
geometric assumption — see §6.1 ambiguity.

### 2.6 Initial conditions — biology-grounded with mammalian fallback

Initialize from C. elegans values where they exist (none directly
measured for these neurons; see §3.1); fall back on mammalian defaults
with explicit "mammalian-default, no C. elegans-specific empirical
refinement available" labels per ion per cell.

### 2.7 Pump electrogenicity — wired as both membrane current and concentration sink

Na/K-ATPase: 3 Na out / 2 K in per ATP → net +1 charge out → hyperpolarizing
membrane current, AND −3 Na flux per cycle to d[Na]/dt, +2 K flux per cycle
to d[K]/dt. Ca-ATPase similarly: hyperpolarizing membrane current + Ca
extrusion. Both wired simultaneously — not separated.

### 2.8 Cross-cutting: biophysical parameter inference under structural constraints

Direct empirical C. elegans-specific data is unavailable for many parameters
Layer 1 needs (cell volumes, ion concentration baselines, pump current
densities). Rather than block on absent measurements, Layer 1 ships with
explicit per-parameter epistemic labels and biophysical constraints:

- **Empirically grounded** — measured for these cells (e.g., Nicoletti
  capacitances, CeNGEN transporter TPMs).
- **Biophysically derived** — calibrated against equilibrium / conservation
  constraints (e.g., I_NaK_max tuned to balance K leak at rest).
- **Approximation from adjacent biology** — mammalian or different-cell C.
  elegans values applied with documented limitation (e.g., Payne 1997
  [Cl]_in for mammalian KCC-2-dominant neurons).
- **Free parameter with sensitivity sweep** — bounded by validation
  invariance (e.g., r_eff sweep §6.1).
- **Biophysically derived under assumptions inconsistent with substrate**
  *(new category, added 2026-05-12 from §7.3 finding)* — parameters fit
  against a model whose implicit ion-state or other state assumptions
  contradict the current substrate. The fit is internally consistent (it
  matches its target data) but not biophysically transferable. Channels
  inherited from Nicoletti 2024 currently fall into this category until
  §7.3.5 audit/refit; see §8.

The substrate is biophysically grounded in *structure* (mass conservation,
Nernst, pump stoichiometry, mechanism-class wiring) with explicit
uncertainty quantification in *parameters*. This is the right rigor mode:
the alternative — waiting for C. elegans-specific measurements on every
parameter — means the substrate never ships, and Phase G stays stuck on a
LIFBrain that can't represent its mechanism classes. Each parameter
defaults to its best-available value labeled honestly; refinement happens
when empirical data lands.

This framing is load-bearing across all subsequent §6 resolutions: §6.1
(volume as free parameter), §6.2 (Ca clearance lumping with refactor
trigger), §6.4 (ABTS-1 pH-coupling approximation), §6.5 (mammalian-Cl
default). Each is a deliberate epistemic-honesty choice, not a corner cut.

**Recurring methodology step — parameter audit before integration.** §7.3
surfaced that inherited fits (Nicoletti channels) encode implicit ion-state
assumptions invisible until the substrate is composed. The transferable
lesson: **before integrating an inherited parameter set into the substrate,
audit what state variables and reversal potentials its fit assumed; verify
those assumptions are consistent with the current substrate state; if not,
flag for refit before composition.** This applies recurringly across layers:

- Layer 1: Nicoletti channel fits assume E_Ca = 60 mV (= [Ca]_in ≈ 17 μM).
  Inconsistent with physiological [Ca]_in = 50 nM. Layer 1.5 / §7.3.5
  audits + refits. See §8.
- Layer 3 (anticipated): Wicks 1996 graded-release Boltzmann parameters
  were derived under Ascaris recording conditions with implicit
  intracellular Cl, Mg, ATP assumptions. Will need the same audit before
  any WB3-equivalent reuse.
- Layer 5+ (anticipated): peptide release rate-coupling constants, gap
  junction conductances inherited from electrophysiology under saline
  conditions — same audit pattern.

"Parameter audit before integration" is now a standing methodology step
in the substrate redesign roadmap. See `docs/substrate_redesign_roadmap.md`
cross-cutting tracks.

---

## 3 · Verification query findings (Q1-Q5)

### 3.1 Q1/Q4 — Baseline ion concentrations

**No directly-measured nematode neuron intracellular ion concentrations
were located for the four target cells.** Available references:

- **Avery 1995** *C. elegans* electrophysiology pipette saline:
  136.5 mM K-gluconate + 17.5 mM KCl + 9 mM NaCl + 1 mM MgCl₂ +
  10 mM HEPES (pH 7.2). This is the dialyzed-cell condition during
  whole-cell patch-clamp — not native intracellular composition.
- Calcium: GCaMP imaging widely used; resting [Ca]_in in C. elegans
  neurons consistent with mammalian baseline ~50-100 nM. Direct
  Cl⁻ imaging exists via SuperClomeleon (PMC9026654) but applied to
  glia, not the four target neurons.
- C. elegans does not encode voltage-gated Na channels — Na dynamics
  are not load-bearing for spike generation (Ca-dominant inward
  current).

**Layer 1 baseline (mammalian defaults, explicitly labeled):**

| ion | [X]_in (mM) | [X]_out (mM) | E_X @ 25°C (mV) | grounding |
|---|---:|---:|---:|---|
| K  | 140  | 4    | −91  | mammalian default |
| Na | 10   | 140  | +68  | mammalian default |
| Cl | 5    | 110  | −78  | mammalian default (low [Cl]_in per adult-neuron precedent) |
| Ca | 5×10⁻⁵ (= 50 nM) | 2 | +135 | mammalian default; GCaMP imaging consistent |

Note: AVAL's current Wave 2 spec assumes E_Ca = 60 mV (Nicoletti). The
mammalian default initial conditions give E_Ca ≈ +135 mV — substantially
different. This is one of the expected ways the Nicoletti 5dp match
breaks under the new substrate (per §2.2 accepted trade).

### 3.2 Q1/Q4 secondary — CeNGEN transporter expression (NEW FINDING)

The project's existing `public/data/cengen-panel.json` covers only the 41
channel/receptor genes used by Wave 2; **the four target transporters
(kcc-2, nkcc-1, abts-1, eat-6) are absent from the local CeNGEN panel.**
Resolved by direct query of CeNGEN bulk-integrated TMM-counts CSVs
(`021821_medium_threshold2.csv`, `021821_stringent_threshold4.csv`):

| gene | AVA T2 | AIY T2 | RIM T2 | AVA T4 | AIY T4 | RIM T4 | role |
|---|---:|---:|---:|---:|---:|---:|---|
| **kcc-2**   | 598.6 | 26.9   | 234.2 | 598.6 | 0     | 234.2 | K-Cl extruder |
| **abts-1**  | 569.5 | 206.5  | 223.9 | 569.5 | 206.5 | 223.9 | Na-Cl/HCO₃ extruder |
| **nkcc-1**  | 0     | 0      | 0     | 0     | 0     | 0     | Cl loader (immature) |
| **eat-6**   | 1346  | 157    | 388   | 1346  | 157   | 388   | Na/K-ATPase α |
| **mca-3**   | 478   | 95     | 253   | 478   | 95    | 253   | PMCA |
| **sca-1**   | 87    | 53     | 23    | 0     | 53    | 0     | SERCA |
| **pmr-1**   | 33    | 28     | 21    | 0     | 0     | 0     | Golgi Ca-ATPase |
| **ncx-1**   | 100   | 0      | 10    | 100   | 0     | 0     | Na/Ca exchanger SLC8 |
| **ncx-4**   | 177   | 0      | 82    | 0     | 0     | 82    | Na/Ca exchanger SLC24 |

CeNGEN data is per-class (no L/R split), so AVAL and AVAR inherit AVA
values; AIYL/AIYR inherit AIY; RIML/RIMR inherit RIM.

**Implications:**

1. KCC-2 + ABTS-1 belong in Layer 1 for all three cells; NKCC-1 deferred.
2. **Two distinct Cl-clearance regimes:**
   - **AVA + RIM**: KCC-2 and ABTS-1 in **roughly equal proportions**
     (AVA 49% / 51%; RIM 49% / 51%). High combined clearance capacity.
   - **AIY**: ABTS-1 **dominant at 88.5%** with low KCC-2 expression.
     Combined clearance ~5× lower than AVA (233 vs 1168 TPM).
3. Substantive biological prediction: AIY's Cl dynamics will be
   **pH-sensitive** in a way AVA's/RIM's are not, because ABTS-1
   carries Cl via HCO₃ exchange. Layer 1 v1's electroneutral lumping
   loses this AIY-specific biology — see §6.4 for v1 limitation +
   v1.5 refactor trigger.
4. After GABA-A/GluCl perturbation, AIY's intracellular Cl will return
   to baseline ~5× slower than AVA's (scaling with combined TPM).
   Whether this matters phenotypically depends on perturbation
   timescale — Layer 1 validation characterizes this.
5. Na/K-ATPase (eat-6) confirmed in all three; relative density scales
   with TPM: AVA ~3.4× RIM ~8.6× AIY.
6. Plasma membrane Ca extrusion: mca-3 (PMCA) primary; ncx-1 + ncx-4
   secondary in AVA + RIM. **AIY has no NCX** → PMCA-only Ca clearance
   biology (see §6.2 for v1 lumping → v1.5 refactor trigger).
7. SERCA (sca-1) present in all three but ER not modeled in Layer 1
   single-compartment — treat as lumped Ca clearance with PMCA + NCX.
8. The local `public/data/cengen-panel.json` should be extended to
   include these transporters for Layer 1 v2 broadcast to remaining
   cells. Out of Layer 1 scope.

### 3.3 Q2 — Na/K-ATPase density

**Nicoletti 2024 does not publish pump parameters** (confirmed via direct
fetch of PMC10980225; Table 3 has capacitance only; ion homeostasis
mechanisms not in the paper). Phase F's analytic ATP layer treats
Na/K-ATPase as a steady-state consumption rate without explicit current.

No published Na/K-ATPase pump current density measurements exist
specifically for C. elegans neurons. Reference values:

- Mammalian neurons: 30-70% of cellular ATP consumption is Na/K-pump;
  max pump current density ~10-100 μA/cm²; surface density ~10⁵-10⁷
  pumps per cell.
- C. elegans-specific: eat-6 (α-subunit) mutations disrupt excitable
  cell function (Davis 1995 PMID 7905262); no direct pump current
  measurements located.

**Layer 1 approach:** Single `I_NaK_max` parameter per cell, scaled by
relative eat-6 CeNGEN TPM. Hill-ATP dependence with K_d_ATP ≈ 0.1 mM
(mammalian default). Steady-state at rest balances passive K leak +
Na leak. Validation criterion: rest condition reproduces stable
[K]_in / [Na]_in / V.

### 3.4 Q3 — OpenWorm/ChannelWorm

OpenWorm ChannelWorm repository archived 2018-08-27 (read-only). Contains
ion channel Hodgkin-Huxley parameterizations only — **no pump/transporter
models**. c302 multi-scale framework is alive but at LIF-cell level for
the connectome simulation, not per-cell biophysics.

**No competing parameterization to consume.** Layer 1 defines its own
pump kinetics with explicit "assumed-default" labels.

### 3.5 Q5 — Cell volumes

**No published volumetric measurements located for AVAL/AVAR/AIY/RIM.**
WormAtlas neuron pages are qualitative (cell body diameter "a few
microns"). The vEM pipeline (Mulcahy 2018 PMC6262311) reconstructs
geometry but specific volumes for the target cells aren't published.
Witvliet 2021 / Nemanode dataset has the geometric data in principle but
volumes per cell aren't in standard outputs.

**Capacitance-derived surface area (well-constrained):**

| cell | C_m (Nicoletti) | surf @ specific_Cm = 0.86 μF/cm² |
|---|---:|---:|
| AVAL | 9.66 pF  | 1124 μm² |
| AVAR | 8.43 pF  |  980 μm² |
| AIY  | 1.05 pF  |  122 μm² |
| RIM  | 1.55 pF  |  180 μm² |

**Volume requires geometric assumption** — see §6.1 ambiguity. Cylindrical
compartment per Nicoletti's NeuroMorpho approach: vol = surf × r/2 where
r is effective compartment radius. For r in {0.25, 0.5, 1.0 μm}, AVAL
volume spans {140, 280, 560 fL} — a 4× range that directly translates to
4× range in concentration-dynamics timescale.

---

## 4 · Architecture (concrete equations)

### 4.1 Per-cell membrane + ion balance

Per cell, four ion concentration state variables `[K]_in, [Na]_in, [Cl]_in,
[Ca]_in` plus V plus ATP-related state (Layer 1 keeps ATP analytic from
Phase F; see §4.4):

```
dV/dt = -(I_Na_chan + I_K_chan + I_Cl_chan + I_Ca_chan
          + I_leak + I_pump_NaK_membrane + I_pump_Ca_membrane
          + I_KCC2_membrane + I_ext) / C_m

d[K]_in/dt  = -(I_K_chan + I_leak_K + 2·I_pump_NaK_cycle + I_KCC2_K) / (1 · F · vol)
d[Na]_in/dt = -(I_Na_chan + I_leak_Na - 3·I_pump_NaK_cycle - I_ABTS1_Na) / (1 · F · vol)
d[Cl]_in/dt = -(I_Cl_chan_through_GluCl_GABAA + I_KCC2_Cl + I_ABTS1_Cl) / (-1 · F · vol)
d[Ca]_in/dt = -(I_Ca_chan + I_pump_Ca - I_PMCA_NCX_lumped) / (2 · F · vol · (1 + κ_B))
```

Nernst per dt:
```
E_X = (R · T / (z_X · F)) · ln([X]_out / [X]_in)
```

with R = 8.314 J/(mol·K), F = 96485 C/mol, T = 293 K (20°C, C. elegans
standard).

### 4.2 Na/K-ATPase

Three-state Hill-ATP:
```
v_NaK = I_NaK_max · ([Na]_in / (K_Na + [Na]_in))³
                  · ([K]_out / (K_K + [K]_out))²
                  · (ATP / (K_ATP + ATP))^n_Hill
I_NaK_membrane = F · v_NaK  (net +1 charge out per cycle = hyperpolarizing)
I_pump_NaK_cycle = v_NaK    (cycles/sec/cell; ×3 for Na flux, ×2 for K flux)
```

Parameters:
- `I_NaK_max`: per-cell scaling, indexed by eat-6 CeNGEN TPM (AVA 1346,
  AIY 157, RIM 388 → relative {1.0, 0.117, 0.288}). Absolute magnitude
  tuned to maintain rest gradients.
- `K_Na ≈ 10 mM`, `K_K ≈ 1.5 mM` (mammalian default).
- `K_ATP ≈ 0.1 mM`, `n_Hill = 1` (mammalian default; Phase F uses softer
  K_ATP_HALF = 0.05 fraction-of-WT — different parameterization, see
  §6.3 ambiguity).

### 4.3 Ca extrusion (lumped PMCA + NCX + SERCA in Layer 1)

```
I_PMCA_NCX_lumped = I_Ca_clear_max · ([Ca]_in / (K_Ca_clear + [Ca]_in))
I_pump_Ca_membrane = 2 · F · v_PMCA_only  (electrogenic from PMCA component)
```

Parameters:
- `I_Ca_clear_max`: tuned per cell to give [Ca]_in resting in 50-100 nM
  range and ~1s recovery timescale after Ca spike.
- `K_Ca_clear ≈ 0.5 μM` (mammalian PMCA Km).

Layer 1 lumps PMCA + NCX + SERCA into one clearance term. ER as separate
compartment + SERCA-specific dynamics deferred to Layer 4. NCX
electrogenic contribution lumped into membrane current via approximate
proportion (Layer 1 v2 if validation surfaces residuals).

### 4.4 KCC-2 (electroneutral K-Cl symport)

```
v_KCC2 = I_KCC2_max · ([K]_in · [Cl]_in - [K]_out · [Cl]_out) / (K_d_KCC2² + ...)
I_KCC2_K = v_KCC2       (K out per cycle)
I_KCC2_Cl = v_KCC2      (Cl out per cycle, same direction as K)
I_KCC2_membrane = 0     (electroneutral)
```

Standard thermodynamic driving force: equilibrium when [K]_in · [Cl]_in =
[K]_out · [Cl]_out (Payne 1997 mammalian model). For standard mammalian
gradients (140·10 = 1400 in; 4·110 = 440 out), KCC-2 drives Cl out.
`I_KCC2_max` indexed by kcc-2 CeNGEN TPM.

### 4.5 ABTS-1 (electroneutral Na-Cl/HCO3 exchange)

Layer 1 treatment: lumped with KCC-2 as electroneutral Cl extruder
parameterized by **per-cell combined kcc-2 + abts-1 TPM**:

| cell | kcc-2 T2 | abts-1 T2 | combined | ABTS-1 % | rel. to AVA |
|---|---:|---:|---:|---:|---:|
| AVA | 598.6    | 569.5     | 1168.1   | 48.8%    | 1.00× |
| AIY |  26.9    | 206.5     |  233.4   | 88.5%    | 0.20× (5× lower) |
| RIM | 234.2    | 223.9     |  458.1   | 48.9%    | 0.39× (2.5× lower) |

`I_lumped_Cl_extruder_max[cell] = I_lumped_Cl_max_base · (combined_TPM[cell] / 1168.1)`

with `I_lumped_Cl_max_base` calibrated against AVA's expected Cl recovery
timescale at rest. The 5× lower AIY clearance is the substantive
biological prediction of this scaling.

**Explicit HCO₃/pH coupling deferred to Layer 1 v2.** Documented
v1 limitation specifically for AIY (88.5% ABTS-1-dominant — pH dependence
is load-bearing biology there); for AVA/RIM the lumped approximation
is well-defensible (KCC-2 and ABTS-1 are roughly co-equal so neither
dominates the dynamics). See §6.4 resolution log.

### 4.6 ATP coupling to Phase F

See §6.3 ambiguity for the architectural choice. Recommended path:
Layer 1 produces per-cell pump-current sum → translates to ATP
consumption rate (1 ATP per Na/K cycle, 1 ATP per Ca cycle) → feeds
into a rebuilt dynamic version of Phase F's ATP balance. Phase F's
analytic version (`phase_f_metabolic_layer.py`) retained as offline
calibration target; new dynamic version exposed via the same Phase G
hook surface.

### 4.7 Module structure

```
scripts/brain/wave2/
    ion_dynamics.py            # NEW — ion state vars, pump kinetics, Nernst
    pumps/
        na_k_atpase.py         # NEW — Hill-ATP Na/K pump
        ca_clearance.py        # NEW — lumped PMCA+NCX+SERCA
        kcc2.py                # NEW — KCC-2 K-Cl symport
        abts1.py               # NEW — ABTS-1 Cl extruder (lumped)
    cells/
        option_alpha_ava_cell.py   # MODIFIED — wire ion dynamics
        option_alpha_avar_cell.py  # MODIFIED
        option_alpha_aiy_cell.py   # MODIFIED
        option_alpha_rim_cell.py   # MODIFIED
    artifacts/
        layer1_validation/         # NEW — validation outputs
```

Plus `docs/substrate_redesign_roadmap.md` (umbrella doc for Layers 1-7,
created at commit time).

---

## 5 · Validation panel

### 5.1 Per-cell rest stability (each cell, 60s simulation, no stim)

- |Δ[K]_in| < 1 mM over 60s
- |Δ[Na]_in| < 0.5 mM over 60s
- |Δ[Cl]_in| < 0.5 mM over 60s
- |Δ[Ca]_in| < 10 nM over 60s
- V drift < 1 mV over 60s

### 5.2 Nernst-bound voltage

Total V trajectory satisfies `min(E_K, E_Cl) ≤ V ≤ max(E_Na, E_Ca)`
within numerical tolerance at all timesteps. Any violation → indicates
ion mass conservation bug or pump electrogenicity sign error.

### 5.3 GHK resting prediction

Computed `V_GHK = (RT/F) · ln((P_K[K]_out + P_Na[Na]_out + P_Cl[Cl]_in) /
(P_K[K]_in + P_Na[Na]_in + P_Cl[Cl]_out))` within ±5 mV of measured V at
rest. Tests that the channel permeability mix produces the expected
resting V from ion gradients alone.

### 5.4 Current-injection response (per-cell, 7-point sweep)

- AVAL: +10 pA → ~+80 mV plateau (Nicoletti envelope)
- AVAR: +10 pA → ~+40 mV plateau (Nicoletti envelope)
- AIY: graded depolarization across sweep, no spikes
- RIM: graded with low-threshold Ca contribution from CCA-1

Tolerance: phenotype categories preserved (plateau / graded / spiking);
peak amplitudes within ±15% of Nicoletti reported envelope.

### 5.5 ATP balance

At rest: ATP_consumed_per_sec ≈ ATP_produced_per_sec (within 1% for
analytical model coupled to Layer 1 pumps). Under 1× clinical-EC50
Complex I block: ATP_eff drops to predicted level matching Phase F's
analytic prediction within 10%.

### 5.6 Recovery from perturbation

Inject +50 pA for 1s → ion gradients perturb → after stim off, gradients
recover to ±5% of rest within 10s (ion timescale) and ATP balance
recovers within 30s.

### 5.7 Cl perturbation (Phase G dry run)

Open all GluCl conductance to 10× baseline for 5s → measure [Cl]_in
trajectory. Without KCC-2/ABTS-1, [Cl]_in should accumulate
unboundedly. With KCC-2/ABTS-1 wired, [Cl]_in equilibrates and
returns to rest after stim removal. This is the Layer 1 dress
rehearsal for Phase G's GluCl-mediated mechanisms.

---

## 6 · Resolutions log (all ambiguities authorized 2026-05-12)

Each subsection retains the original framing (recommendation + options)
plus an `AUTHORIZED` block with Rohit's resolution and any additions.
Implementation in §7 follows from these resolutions verbatim.

### 6.1 Cell volume effective radius

Capacitance-derived surface area is well-constrained (§3.5). Converting
to volume requires assuming an effective compartment radius for the
cylindrical-compartment geometry Nicoletti uses.

| effective r | AVAL vol | AVAR vol | AIY vol | RIM vol |
|---:|---:|---:|---:|---:|
| 0.25 μm | 140 fL | 122 fL | 15 fL | 23 fL |
| 0.50 μm | 281 fL | 245 fL | 30 fL | 45 fL |
| 1.00 μm | 562 fL | 490 fL | 61 fL | 90 fL |

**Recommendation:** Central default r_eff = 0.5 μm (between thin-process
0.25 μm and soma 1.0 μm); document as assumed value; sensitivity sweep
{0.25, 0.5, 1.0} in Layer 1 validation. Concentration-dynamics
timescale scales inversely with volume, so a 4× volume range = 4×
timescale range — load-bearing for Phase G dose-response.

**AUTHORIZED 2026-05-12:** r_eff = 0.5 μm default + sensitivity sweep
{0.25, 0.5, 1.0} as part of the validation suite (not a separate work
block). Pair with explicit validation criterion: phenotype robustness
across the r_eff range is the success metric — if rest stability,
voltage-clamp envelopes, and recovery timescales all survive across
{0.25, 0.5, 1.0}, volume uncertainty doesn't propagate to phenotype
uncertainty. If validation breaks at extreme volumes, we learn which
r_eff range is biophysically constrained. Either outcome is informative.

Honest framing per §2.8: r_eff is the substrate's free parameter; 0.5 μm
is the central estimate; phenotype robustness across plausible range is
the rigor claim.

### 6.2 Ca clearance lumping vs explicit channels

Three options for how to handle PMCA + NCX + SERCA in Layer 1:

(i) Single `I_Ca_clear_max` lumped term tuned to give physiological
    [Ca]_in(t). Clean, one parameter per cell; doesn't distinguish
    plasma-membrane vs ER extrusion. (Recommended for Layer 1.)

(ii) Separate I_PMCA (electrogenic, mca-3-weighted), I_NCX (electrogenic,
     ncx-1/ncx-4-weighted), I_SERCA (electroneutral, sca-1-weighted but
     ER not modeled so the K_d_ER becomes a fitting parameter). Three
     parameters per cell; more biology-faithful; SERCA effectively
     becomes a "Ca buffer" since no ER compartment exists.

(iii) Defer NCX + SERCA entirely to Layer 1.5; Layer 1 ships PMCA-only.
      Cleanest but loses the NCX electrogenicity and SERCA's
      contribution to AIY/RIM Ca handling.

**Recommendation:** (i) for Layer 1; refactor to (ii) in Layer 1 v2 if
validation surfaces residuals. The CeNGEN data supports differential
expression (AIY has no NCX, RIM has strong ncx-4) but Layer 1 single-
compartment doesn't have enough structural detail to distinguish PMCA
from NCX contributions cleanly.

**AUTHORIZED 2026-05-12:** (i) lumped uniformly across cells with explicit
validation-driven refactor trigger. Lumping is defensible for AVA/RIM
(multiple clearance pathways co-express); for AIY it represents
PMCA-only biology with a lumped parameter. Refactor trigger:
**if AIY's Ca dynamics diverge meaningfully from AVA/RIM in the
validation suite** (specifically, recovery timescales materially
different from what a pure-PMCA pathway predicts vs the lumped
parameterization), refactor to explicit single-pathway PMCA for AIY +
PMCA+NCX+SERCA for AVA/RIM. This is a **v1 → v1.5 deferral**, not v2.

Documented v1 limitation: AIY's clearance is biologically PMCA-only;
v1 models it with the same lumped parameter as AVA/RIM, scaling by
mca-3 TPM rather than by individual pathway weights.

### 6.3 ATP coupling architecture — Phase F restructure

Phase F's current implementation (`phase_f_metabolic_layer.py`) is an
**analytic dose-response model**: anesthetic dose → Complex I block
factor → ATP_steady_state → K-ATP open fraction → membrane V shift.
It's calibrated against Morgan & Sedensky 1995 gas-1 hypersensitivity
and **shipped + validated** at the published level.

Layer 1 makes pump current explicit per cell, which means ATP
consumption becomes a dynamic quantity rather than an analytic
parameter. Two architectural options:

(i) **Phase F becomes a Layer 1 consumer.** Phase F's K_BASE_CONSUMPTION
    parameter (currently lumped) gets replaced by an explicit
    `Σ_cells (I_pump_NaK + I_pump_Ca) / ATP_yield_per_cycle` driven by
    Layer 1's dynamic pumps. Phase F's analytic version retained as an
    offline calibration target; new dynamic Phase F exposed via same
    Phase G hook interface. Cleanest architectural cut; preserves Phase
    F's validated Morgan&Sedensky anchor as a calibration check.

(ii) **Parallel paths.** Phase F's analytic model preserved as-is; Layer
     1 adds a parallel `phase_f_dynamic.py` for the explicit version;
     Phase G hook chooses which to consume. Backward-compatible but
     creates a permanent dual-implementation maintenance burden.

**Recommendation:** (i). The analytic Phase F was a stand-in for the
substrate redesign anyway — the whole point of Layer 1 is to make the
ATP balance load-bearing on real biophysics. Cleaner cut now than
later. Worth Rohit's explicit go-ahead given Phase F is shipped.

**AUTHORIZED 2026-05-12 with explicit preservation strategy:**

1. **Phase F analytic version preserved** as `phase_f_analytic.py` with
   original parameter-locked behavior intact. The Morgan & Sedensky
   2.48× hypersensitivity prediction remains the documented validation
   anchor for the analytic version.
2. **New `phase_f_layer1.py`** becomes the Layer 1 consumer, with
   `K_BASE_CONSUMPTION` replaced by an explicit
   `Σ_cells (I_pump_NaK + I_pump_Ca) / ATP_yield_per_cycle` driven by
   Layer 1's dynamic pumps.
3. **Validation question:** does Phase F Layer 1's Morgan & Sedensky
   prediction land within the analytic version's range? If yes,
   dynamic ≥ analytic. If no, that's a finding — either Layer 1 pump
   parameters need refinement, or the analytic version's parameter-lock
   was masking real cell-class-specific biology (e.g., the 4.5×
   metabolic spread across cells that uniform K_BASE was averaging out).
4. **Phase G hook surface** re-pointed to `phase_f_layer1.py`. Backward
   compatibility for the analytic version preserved at the file level.
5. **Public-facing documentation** (anesthesia-pipeline web page)
   updated to reference both versions with explicit framing:
   "Phase F v1 (analytic) shipped first; Phase F v2 (Layer 1 consumer)
   refines via cell-class-specific metabolic dynamics."

The preservation strategy protects the shipped public deliverable while
enabling the substrate redesign's biological refinement.

### 6.4 ABTS-1 treatment in Layer 1

CeNGEN shows abts-1 strongly co-expressed with kcc-2 in all three cells
(§3.2). ABTS-1 is biologically a Na-Cl/HCO3 exchanger — proper modeling
requires intracellular HCO3 + pH state variables.

Two options for Layer 1:

(i) **Lumped with KCC-2 as electroneutral Cl extruder**, parameterized
    by abts-1 TPM. Effectively treats ABTS-1 as a "second KCC-2"
    without HCO3/pH coupling. Simplification.

(ii) **Defer ABTS-1 entirely to Layer 1 v2** when HCO3/pH layer lands.
     Layer 1 ships with KCC-2-only Cl extrusion, accepting that adult
     Cl homeostasis is under-extruded.

**Recommendation:** (i). KCC-2-only Cl extrusion under-represents the
adult Cl clearance machinery by a factor that depends on relative
expression — for AIY especially, where abts-1 (206 TPM) dominates
kcc-2 (27 TPM), KCC-2-only would be a substantial under-representation.
Lumped electroneutral approximation is biologically defensible at the
Layer 1 abstraction level; HCO3/pH coupling refinement marked as Layer
1 v2 deliverable.

**AUTHORIZED 2026-05-12 with corrected per-cell scaling:**

Per-cell combined kcc-2 + abts-1 TPM (authoritative values from
`021821_medium_threshold2.csv`):

- **AVA**: kcc-2 (598.6) + abts-1 (569.5) = **1168.1 combined**;
  ABTS-1 dominance 48.8% (co-equal).
- **AIY**: kcc-2 (26.9) + abts-1 (206.5) = **233.4 combined**;
  ABTS-1 dominance **88.5%** (dominant).
- **RIM**: kcc-2 (234.2) + abts-1 (223.9) = **458.1 combined**;
  ABTS-1 dominance 48.9% (co-equal).

Revised biological framing: cells fall into two distinct Cl-clearance
regimes — AVA + RIM with co-equal parallel pathways and high combined
clearance, vs AIY with ABTS-1-dominant low-clearance machinery (~5×
lower than AVA). Under anesthetic perturbation, AIY's Cl dynamics will
be more pH-sensitive than AVA/RIM because ABTS-1's HCO₃ dependence
becomes load-bearing.

This strengthens the case for **Layer 1 v2 HCO₃/pH coupling as a real
refinement target rather than a deferred niceity**: AIY's Cl dynamics
specifically depend on it. The lumped approximation is well-defensible
for AVA + RIM but loses dominant biology for AIY. Documented v1
limitation: if validation surfaces AIY-specific Cl dynamics issues
not appearing in AVA/RIM, that's the v1 → v1.5 trigger for AIY-specific
HCO₃/pH modeling.

### 6.5 [Cl]_in mammalian default value

Mammalian default [Cl]_in is commonly cited as 10 mM, but adult
inhibitory-mature neurons run lower ([Cl]_in ~5 mM is standard for
KCC-2-dominant cells; Payne 1997). C. elegans-specific value not
directly measured.

Two options:

(i) [Cl]_in = 5 mM (low-Cl adult default; E_Cl ≈ −78 mV with [Cl]_out =
    110 mM at 20°C).

(ii) [Cl]_in = 10 mM (standard mammalian textbook; E_Cl ≈ −60 mV at
     20°C).

**Recommendation:** (i) — low-Cl adult default is most consistent with
KCC-2 + ABTS-1 dominance in the target cells (all three are mature
interneurons with strong adult Cl extruders). Documented as assumed
value pending C. elegans-specific empirical refinement.

**AUTHORIZED 2026-05-12:** [Cl]_in = 5 mM default with explicit
**"approximation from mammalian KCC-2-dominant neurons (Payne 1997 rat
CA1 pyramidal); awaiting empirical C. elegans-specific refinement"**
labeling. If Layer 1 validation surfaces phenotype issues that trace
back to Cl baseline, the right move is sensitivity analysis on [Cl]_in
default — not assuming Payne's value generalizes.

---

## 7 · Implementation plan (POST-AUTHORIZATION)

Sequence assuming §6.1-6.5 resolved per recommendations:

### 7.1 Foundation (1 work block)

- `scripts/brain/wave2/ion_dynamics.py`: per-cell ion state variable
  Brian2 equations + Nernst-from-concentrations helper + per-cell
  vol/surf bookkeeping.
- Unit tests: pure ion-balance under fixed currents, Nernst against
  hand-calc, mass conservation under zero net current.

### 7.2 Pump module (1 work block)

- `pumps/na_k_atpase.py`: Hill-ATP Na/K pump.
- `pumps/ca_clearance.py`: lumped PMCA+NCX+SERCA per §6.2(i).
- `pumps/kcc2.py`: K-Cl symport.
- `pumps/abts1.py`: lumped Cl extruder per §6.4(i).
- Unit tests: each pump in isolation under voltage clamp; sign
  conventions; stoichiometry.

### 7.3 Cell integration (1 work block per cell)

- AVAL first (most parameter data available). Wire ion dynamics +
  pumps; validate per §5 panel.
- Then AVAR (mostly mirror of AVAL).
- Then RIM, then AIY.

### 7.4 Phase F restructure (1 work block)

- New `phase_f_dynamic.py` reading Layer 1 per-cell pump currents.
- Preserve `phase_f_metabolic_layer.py` as offline calibration anchor.
- Phase G hook re-pointed to dynamic version.

### 7.5 Validation gauntlet (1 work block)

- Full §5 panel on all four cells.
- Sensitivity sweep on r_eff per §6.1 (3 values).
- Cl perturbation Phase G dry run (§5.7).

### 7.6 Commit + roadmap doc (1 work block)

- Conventional-commit chunks with honest scope labels.
- `docs/substrate_redesign_roadmap.md` created (Layers 1-7 overview).
- Layer 1 status: shipped with documented uncertainties; Layer 2 ready.

**Total Layer 1 estimate: 6-8 work blocks.**

---

## 8 · Files of record

- This document: `docs/layer1_design_decisions.md`
- CeNGEN downloads cached at `/tmp/cengen/` (will move to
  `data/cengen_full/` at implementation start; not committed — data
  redistribution per CeNGEN license requires citation, not
  re-hosting):
  - `021821_medium_threshold2.csv` (13.5 MB, threshold 2)
  - `021821_stringent_threshold4.csv` (9.8 MB, threshold 4)
  - Source: https://www.cengen.org/downloads/
- Phase G CALIBRATION_GAP: `AnestheticSimulator/artifacts/phase_g/CALIBRATION_GAP.md`
- Phase F current: `AnestheticSimulator/src/phase_f_metabolic_layer.py`
- Wave 2 cell builders: `scripts/brain/wave2/option_alpha_*_cell.py`

---

## 8 · Inherited parameter audit — methodology (surfaced from §7.3, 2026-05-12)

### 8.1 The Nicoletti 2024 E_Ca finding

Layer 1 §7.3 integrated ion dynamics + pumps + Nicoletti channels per cell.
All four cells (AVAL, AVAR, RIM, AIY) failed ±2% rest stability with
severities AVAL << AVAR << RIM ~ AIY. Root cause: **Nicoletti's channel
parameterization assumes fixed E_Ca = 60 mV**, which under physiology
(`[Ca]_out = 2 mM`) requires `[Ca]_in ≈ 17 μM` — **340× higher than the
mammalian-default 50 nM** authorized for Layer 1 (§6.5).

Nicoletti's fit is internally consistent: channels match published voltage-
clamp traces because the simulations also use E_Ca = 60 mV. The fit
encodes the implicit assumption that [Ca]_in is at the value giving
E_Ca = 60 mV. This is invisible until the substrate makes [Ca]_in an
explicit state variable.

Under Layer 1's physiological E_Ca = 134 mV (at [Ca]_in = 50 nM), the same
gbar values produce **driving forces 70% larger** than Nicoletti's
calibration assumed. EGL-19, CCA-1, and UNC-2 each contribute proportionally
more Ca influx; the lumped Ca-clearance pump cannot match it; [Ca]_in
accumulates into the μM range; positive feedback through Ca-channel
activation depolarizes V; cells with multiple Ca channels (RIM = 3, AIY
v1 = 1, AVAR = 1, AVAL = 1) compound the failure proportionally.

### 8.2 Quantitative per-cell evidence (5s rest, Layer 1 §7.3)

| cell  | V_rest (mV) | [Ca]_in steady-state | E_Ca (mV)  | EGL-19 g (S/cm²) | Ca channel count |
|-------|------------:|---------------------:|-----------:|-----------------:|-----------------:|
| AVAL  | −53         | 2.3 μM              | +85.5      | 9.29e-6          | 1 (EGL-19)       |
| AVAR  | −37         | 55 μM               | +45.5      | 5.74e-6          | 1 (EGL-19)       |
| RIM   | +14         | 666 μM              | +13.9      | 3.20e-4          | 3 (EGL19+CCA1+UNC2) |
| AIY   | +29         | 200 μM              | +29.1      | 1.52e-4          | 1 (EGL-19, v1 set) |

Each cell's [Ca]_in evolves toward where the Ca-clearance pump (at saturating
delta) just matches Ca influx through channels. Higher channel density →
higher steady-state Ca. RIM's 3 Ca channels at high density → catastrophic
accumulation. AIY drifts severely because of its tiny volume (faster
concentration dynamics) compounded by its 5× lower pump TPM ratio.

### 8.3 General principle

**Inherited fits carry implicit state-variable assumptions that become
visible only under explicit-state substrates.** The fit is correct
against its own assumed reference; the assumptions are wrong against the
physical substrate. Resolution requires either (a) accepting the inherited
fit's assumptions as substrate constraints (non-physiological [Ca]_in),
(b) refitting under physical state, or (c) treating the fit as a methods
contribution rather than a state-of-the-art parameterization.

### 8.4 Recurring failure mode for §2.8 epistemic labeling

This finding adds a new category to the §2.8 labeling system:
**"biophysically derived under assumptions inconsistent with substrate."**
Distinguishing this from straight "biophysically derived" matters because
the user of the parameter must know whether it's physically transferable.
Channel parameters inherited from Nicoletti currently carry this label
until §7.3.5 audits and refits.

### 8.5 Forward-looking parameter audit flags

Other inherited parameter sets likely to surface similar inconsistencies:

- **Wicks 1996 graded-release Boltzmann** — derived from Ascaris ventral
  cord recordings under specific saline conditions. Implicit intracellular
  Cl, Mg, ATP, and resting V state. Used in WB3 cross-coupling. **Audit
  required before any Layer 3+ reuse.** Likely needs σ-V_half refit under
  Layer 1's emergent V_rest.
- **Nicoletti calcium-pool dynamics** (cadiff, caintra1) — separate from
  channel parameters; assume specific buffer kinetics + ER coupling.
  Audit before any Layer 4 ER compartment integration.
- **Peptide release rate-coupling constants** — derived from imaging under
  specific calcium-imaging conditions. Audit before Layer 5+ neuromodulation
  integration.
- **Loer & Rand 2022 neurotransmitter table** — categorical (NT identity
  per neuron), not directly parameter-fit, so likely not affected. Verify
  before assuming.

The general rule: **before composing an inherited parameter set into the
substrate, run a brief audit pass surveying what implicit state the fit
assumes; document; refit if inconsistent.** This becomes step zero of any
inherit-and-compose work block going forward.

### 8.6 Uniqueness audit methodology — surfaced from §7.3.5 Phase 5 (2026-05-12)

§7.3.5 Path 2 Phase 5 surfaced a **second methodology contribution**
distinct from §8's state-variable audit. Direct query of Nicoletti 2024
(PMC10980225) on FRAC-flagged parameters in Phase 5 pre-flight confirmed:

- **NO error bars / confidence intervals reported** for any fitted gbar
- **NO sensitivity analyses** documented
- Authors **explicitly acknowledge "non-uniqueness of the set of
  parameters"** suggesting parameter space degeneracy without formal
  characterization

**Implication:** Nicoletti's specific per-cell gbars are point estimates
from local optima of under-constrained fits. Multiple combinations
likely reproduce the same I-V curves equivalently. "Match Nicoletti's
specific values" is therefore the WRONG validation criterion for derived
parameters — it asks methodology to reproduce one of multiple valid
solutions arbitrarily selected by Nicoletti's optimization procedure.

**Methodology contribution:** Treating inherited parameter fits as
ground truth without checking fit uniqueness produces brittle validation
criteria. The substrate redesign now requires **two audits** before
inheriting any parameter set:

1. **State-variable audit (§7.3):** does the fit's implicit ionic state
   match the current substrate?
2. **Uniqueness audit (NEW, §7.3.5):** does the fit have error bars,
   sensitivity analyses, or other evidence of unique determination by
   the underlying data?

**Failing EITHER audit weakens the case for using specific values as
validation anchors.** Failing both (Nicoletti channels) requires
fundamental reframing: validate against underlying MEASURED data
(I-V curves with SEM where reported) rather than against fitted point
estimates.

### 8.7 New §2.8 epistemic label category

Per `docs/channel_parameter_derivation_methodology.md` §6.0.1, §2.8
extends with a new category:

- **biophysically derived under non-unique parameter fits** — derived
  from data fitting but the fit problem is under-constrained; multiple
  parameter combinations produce equivalent fit quality; specific
  point estimates lack uncertainty quantification

Channels inherited from Nicoletti 2024 fall into **BOTH** "inconsistent
substrate" (§8.1-8.5) AND "non-unique parameter fits" (§8.6) categories.

### 8.8 Forward-looking dual-audit application

The dual audit (state-variable + uniqueness) becomes standing methodology
step for Layers 2-7:

- **Wicks 1996 graded release Boltzmann** (Layer 3+): both audits required
  before WB3-equivalent reuse. State-variable: Ascaris saline conditions
  vs substrate state. Uniqueness: Wicks fit error bars, sensitivity.
- **Nicoletti Ca pool dynamics** (Layer 4): both audits required before
  ER compartment integration.
- **Peptide release rate-coupling** (Layer 5+): both audits required
  before neuromodulation integration.
- **Loer & Rand 2022 neurotransmitter table:** categorical assignments
  not parameter fits; uniqueness audit doesn't strictly apply but
  per-neuron NT identity confidence should be documented.

This is the substrate redesign's **second transferable methodology
contribution** (state-variable audit was the first, from §7.3). Both
become step zero of any inherit-and-compose work block in subsequent
substrate redesign layers.

### 8.9 Resolution scope (revised from 8.6)

§7.3.5 (Layer 1.5) Path 2 ships infrastructure (Phase 1-5 deliverables)
plus the dual-audit methodology contribution. Phase 6 proceeds under
**reframed validation criteria** per §8.6:

- Validate against cell-level rest stability + measured I-V envelope
  match, NOT against Nicoletti's specific gbar values
- Channel kinetic parameters preserved from Nicoletti (out of scope for
  Path 2 v1; subject to dual audit in separate work block if Layer 2
  surfaces issues)
- Cross-cell consistency check as additional validation axis

If Phase 6 passes under reframed criteria, Path 2 v1 ships as substrate
redesign's first major demonstration of "biology-derived parameters
under measurement-data validation." If Phase 6 fails, failure pattern
informs Option β targeted refinement (per-cell-family C_global,
γ_IRK refit).

§7.3.5 BLOCKS §7.4 (Phase F restructure depends on correct Ca dynamics
feeding back into ATP consumption via Ca-ATPase load).

---

## 9 · Status

§6.1-6.5 all authorized 2026-05-12. Layer 1 §7.1 + §7.2 v2 SHIPPED with
acceptance criteria met. §7.3 SHIPPED infrastructure but surfaces the
inherited-parameter inconsistency finding (§8); acceptance criteria
**unmet pending §7.3.5 channel refit**.

**Subsequent work blocks** per §7 sequencing:
- §7.3.5 — **NEW**: Channel-substrate consistency audit + refit (BLOCKS §7.4)
- §7.4 — Phase F restructure (analytic preserved + Layer 1 consumer added)
- §7.5 — Validation gauntlet + r_eff sensitivity sweep
- §7.6 — Commit + roadmap doc complete state

Total Layer 1 estimate revised: 10-13 work blocks through commit (was
6-8; the §7.3.5 audit + refit work adds 3-5 blocks). The increase is
substantive methodology contribution, not scope creep — the audit
methodology becomes a transferable pattern for Layers 2-7.
