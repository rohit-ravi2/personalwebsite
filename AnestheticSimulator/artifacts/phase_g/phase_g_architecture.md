# Phase G — Network perturbation architecture

**Date:** 2026-04-28
**Status:** Architecture v1; implementation skeleton in `src/phase_g_network_perturbation.py`
**Substrate:** Wave 2 LIFBrain (Cook 2019 connectome × Loer & Rand 2022 NT signs); minimal Brian2 demo for unit-level validation

---

## Purpose

Phase G is the network perturbation layer that consumes `wave2_overlay_v2.json` (CP7-corrected occupancies + per-target kinetic shifts) and applies anesthetic-specific parameter modifications to a Brian2 LIF network. Phase G is the load-bearing test of whether Wave P's binding pipeline produces network-level phenotypes consistent with anesthesia biology — specifically whether perturbed brain dynamics map onto known anesthesia phenotypes (suppressed locomotion, suppressed touch reversal, mutant-specific differential sensitivity).

Phase G does NOT modify Wave 2 brain code. It layers perturbation hooks externally as a wrapper class, preserving the brain's baseline calibration.

---

## Inputs

- `artifacts/kinetics/wave2_overlay_v2.json` — per-(anesthetic, target) kinetic shifts and corrected occupancies
  - 6 anesthetics × 30 targets = 180 entries
  - Each entry has `mechanism_class`, `occupancy_1xEC50` (CP7-corrected), `parameters.{n_Ca_delta, rate_factor, ...}`
- Channel-to-neuron expression mapping (CeNGEN, simplified for v1)
- Substrate brain object: `LIFBrain` (Brian2-backed, 300 neurons from Cook 2019)

## Outputs

- Per-(anesthetic, dose) perturbation vector: dict mapping (neuron_or_synapse, parameter_name) → multiplicative shift
- Perturbed brain instance ready for simulation
- Predicted phenotype: state distribution, firing rates, behavioral proxies

---

## Mechanism class → perturbation hook mapping

`wave2_overlay_v2.json` already groups targets by mechanism class. Phase G applies one of five hook types depending on class:

| mechanism_class | hook | parameter modified | direction |
|---|---|---|---|
| `gaba_potentiation` | enhance GABA-A inhibitory synapses | W_chem (post-UNC-49 edges) | × (1 + occ × multiplier) → stronger inhibition |
| `glucl_potentiation` | enhance GluCl inhibitory synapses | W_chem (post-AVR-14/15, GLC-1/2/3/4 edges) | × (1 + occ × multiplier) |
| `k2p_potentiation` | add hyperpolarizing K-leak current | I_ext on TWK-18/TWK-29 expressing neurons | additive negative current ∝ occ |
| `nachr_antagonism` | reduce nAChR excitatory synapses | W_chem (post-ACR-16/UNC-29/38/63 edges) | × (1 - occ × block_efficacy) |
| `complex_i_block` | reduce ATP → open K-ATP channels | I_ext on all neurons (uniform K-ATP coupling) | additive negative current via Phase F coupling |
| `snare_cooperativity` | reduce release probability | W_syn global scale × Phase E fold-change | × Phase E predicted release-p reduction |

For halothane (multi-target binder), all five hooks fire in parallel with different occupancies per target.

## Channel-to-neuron expression mapping (v1, CeNGEN-derived)

For each anesthetic target, identify the *C. elegans* neurons in the LIFBrain roster that express the channel. v1 uses a simplified hand-curated mapping; v2 should integrate CeNGEN's full expression matrix.

| Target | Expressing neurons (v1 short list) | Source |
|---|---|---|
| UNC-49 (GABA-A) | All neurons receiving GABAergic input — practically determined by NT identity at presynaptic side (sign = -1 in connectome) | Loer & Rand 2022 + Bamber 1999 PMID 9986093 |
| AVR-14, AVR-15 (GluCl) | Motor neurons + AVA, AVB, AIB | Cook 2019 + Dent 1997 PMID 9027382 |
| GLC-1/2/3/4 (GluCl) | Pharyngeal + body wall muscle, motor neurons | CeNGEN |
| ACR-16 (nAChR α7-like) | Muscle, AVA, AVD, AIA | Touroutine 2005 PMID 15837794 |
| UNC-29/38/63 (nAChR α) | Body wall muscle | Richmond 1999 PMID 10570485 |
| TWK-18 (K2P) | AVA, AVD, body wall muscle | Singaram 2011 PMID 22137475 |
| TWK-29 (K2P) | Broadly expressed | CeNGEN |
| KCNK2 (homolog target — not in worm; reference only) | — | — |
| GAS-1 / NDUFS2 (Complex I) | All neurons (mitochondrial) | Kayser 2001 PMID 11278828 |
| UNC-64 (SNARE syntaxin-1A) | All chemical synapses (presynaptic) | Saifee 1998 PMID 9697860 |
| RIC-4 (SNAP-25) | All chemical synapses | Hwang 2007 PMID 17988642 |
| SNB-1 (synaptobrevin) | All chemical synapses | Nonet 1998 PMID 9786969 |

Phase G v1 uses the connectome's NT-sign vector to identify GABAergic vs cholinergic vs glutamatergic synapses; this gives a robust first-order mapping without needing per-cell channel expression data.

---

## API contract

```python
class AnestheticPerturbation:
    """Phase G perturbation manager.

    Loads wave2_overlay_v2.json and produces parameter modification
    profiles per (anesthetic, dose). Designed to apply to LIFBrain
    or any Brian2 NeuronGroup/Synapses-based brain.
    """

    def __init__(self, overlay_path: str | Path, channel_expression: dict | None = None):
        """Load v2 overlay and channel-expression mapping."""

    def list_anesthetics(self) -> list[str]:
        """Return anesthetics available in the overlay."""

    def compute_perturbation_vector(self, anesthetic: str, dose_multiplier: float = 1.0) -> dict:
        """Return per-target perturbation magnitudes at the given dose.

        Returns dict with structure:
        {
          "anesthetic": str,
          "dose_multiplier": float,
          "per_class": {
              "gaba_potentiation": [{"target": str, "occupancy": float,
                                       "perturbation_magnitude": float}, ...],
              "k2p_potentiation": [...], ...
          },
          "summary": {
              "n_classes_engaged": int,
              "max_class_occupancy": float,
              "mean_class_occupancy": float,
          }
        }
        """

    def apply_to_brain(self, brain, anesthetic: str, dose_multiplier: float = 1.0) -> dict:
        """Apply perturbations to a Brian2-backed brain instance.

        Modifies in-place: brain.W_chem (synaptic weights),
        brain.neurons.I_ext (per-neuron static current).

        Returns a `revert_handle` dict with original values for use
        by `revert(brain, revert_handle)`.
        """

    def revert(self, brain, revert_handle: dict) -> None:
        """Restore brain parameters to baseline using revert_handle."""

    def predict_phenotype(self, brain, scenario: str, anesthetic: str,
                          dose_multiplier: float = 1.0,
                          duration_ms: float = 1000) -> dict:
        """Run a Brian2 simulation under perturbation, return readouts."""
```

---

## Phenotype readouts

Phase G v1 produces three readouts from the perturbed brain:

1. **Aggregate firing rate** — mean spikes/sec across the network. Anesthesia broadly suppresses cortical activity → expect monotonic decrease with dose.
2. **Command interneuron activity** — AVA, AVB, AIB, AIY mean firing rates. Forward locomotion correlates with AVB > AVA; reversal with AVA > AVB. Halothane suppression should compress both.
3. **State entropy** — proxy for behavioral repertoire diversity. Anesthesia should reduce entropy.

Future readouts (deferred):
- Touch cascade response (requires Phase δ touch closure)
- Locomotor parameter sweep (requires muscle driver)
- Mutant differential sensitivity (gas-1, twk-18 — requires substrate variants)

---

## Validation phenotypes (test plan)

| # | Test | Expected outcome | Reference |
|---|---|---|---|
| 1 | Halothane dose-response on aggregate firing rate | Monotonic decrease 0.1× → 5× EC50; ~50% suppression at 1× | Crowder 1996 PMID 8873562 |
| 2 | Etomidate vs halothane firing-rate suppression | Etomidate dominates GABA-A → uniform suppression; halothane multi-target → stronger suppression at clinical dose | Belelli 1997 PMID 9298537 |
| 3 | gas-1 differential sensitivity | gas-1 mutant immobilizes at lower halothane dose (Phase F prediction) | Morgan & Sedensky 1994 PMID 7943840 |
| 4 | twk-18 GoF differential sensitivity | K2P-gf hypersensitive (CP6 corrected direction) | Singaram 2011 PMID 22137475 |
| 5 | Hexafluoroethane null perturbation | Despite engaging targets at 1 mM, network response should be muted because the integration of partial occupancies on multiple weak targets fails to cross behavioral threshold | Eger 2001 (non-immobilizer prediction) |

---

## Integration with Wave 2 LIFBrain

LIFBrain (`scripts/brain/lif_brain.py`) is the Wave 2 substrate. Integration approach:

1. **Construction:** `brain = LIFBrain(...)` (uses connectome.npz, default parameters)
2. **Perturbation:** `pert = AnestheticPerturbation(overlay_v2_path); pert.apply_to_brain(brain, "halothane", dose_multiplier=1.0)`
3. **Hook implementation:**
   - `gaba_potentiation` → multiply rows of `brain._W_chem_runtime` where presynaptic NT == "GABA"
   - `glucl_potentiation` → multiply rows where presynaptic NT == "Glu" AND postsynaptic neuron is in GluCl-expressing list
   - `nachr_antagonism` → multiply rows where presynaptic NT == "ACh" AND postsynaptic in nAChR-expressing list (subtractive, since ACh is excitatory)
   - `k2p_potentiation` → set `brain.neurons.I_ext[i]` to negative current for K2P-expressing neurons
   - `complex_i_block` → set `brain.neurons.I_ext[i]` to negative current uniformly (K-ATP coupling)
   - `snare_cooperativity` → multiply `brain.W_syn` by Phase E fold-change

4. **Simulation:** `brain.net.run(duration*ms)` → produces SpikeMonitor data
5. **Readout:** extract per-neuron firing rates, aggregate into state distribution

LIFBrain integration is implemented but deferred for execution to a separate work block where it can be tested with full Brian2 codegen and timing budget. The minimal demo network in CP B.4 below uses the same perturbation logic on a small Brian2 LIF substrate to validate the dose-response shape and perturbation hooks before LIFBrain integration.

---

## Architectural decisions (documented)

1. **Wrapper, not modification.** Phase G layers perturbation as a wrapper class that mutates brain parameters in-place and provides a revert handle. No changes to lif_brain.py or production simulator code.

2. **Channel expression mapping is simplified for v1.** Uses connectome NT-sign vector + hand-curated short lists per target. Full CeNGEN expression matrix integration is deferred to v2.

3. **CP7-corrected occupancies are consumed directly.** Each target's `occupancy_1xEC50` from wave2_overlay_v2.json drives the perturbation magnitude. This propagates the f_allo=2.50× correction into network-level predictions.

4. **Dose multiplier is applied as Hill-equation scaling.** dose_multiplier=k means concentration is k × clinical EC50 → occupancy(k) = k × occ_1x / (1 + (k - 1) × occ_1x) for the linear approximation, or proper Hill `c×k / (c×k + Kd)` if Kd is recoverable from occupancy.

5. **No tunable Phase G parameters.** Phase G has zero hand-tuned scaling factors; all magnitudes derive from wave2_overlay_v2.json + Phase E/F mechanisms. This is by design — Phase G must be falsifiable without parameter adjustment, otherwise the same parameter-tuning critique that hit Phase F applies.

6. **Brian2-only.** No NEURON, no NEST, no custom integrator. Uses LIFBrain's existing Brian2 substrate.

---

## Failure modes and mitigations

- **NaN/inf in firing rates:** Brian2 sometimes produces NaN under extreme perturbation. Mitigation: clip perturbation_magnitude to [-0.95, +5.0] range; reject simulation runs that produce NaN.
- **Network freezes (no spikes):** under aggressive perturbation, network may go silent. Mitigation: detect zero-spike runs and report as "fully suppressed" rather than simulation error.
- **Network saturates (all neurons at refractory ceiling):** under reverse perturbation (e.g., reduced inhibition), network may saturate. Mitigation: cap aggregate firing rate at 100 Hz/neuron; report as "saturated."
- **Brain construction failure:** if connectome.npz missing or version-incompatible, fall back to minimal Brian2 demo network for unit-level testing.

---

## Future work (out of overnight scope)

- LIFBrain integration smoke test with full 300-neuron substrate
- Mutant variant integration (gas-1: scale Complex I rate; twk-18: scale K2P leak)
- Touch cascade integration when Phase δ closes ALM/AVM/AIB pathway
- CeNGEN expression matrix v2 (per-cell channel expression for sharper hook localization)
- Dose-response sweeps for all 6 anesthetics
