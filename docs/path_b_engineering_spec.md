# Path B engineering specification — FSM/classifier recalibration under per-edge

**Status:** specification, 2026-04-25 (overnight work block by Session 1).
**Scope:** comprehensive engineering spec for path B of the per-edge resolution.
**Dependencies:** path A T0 closure (Session 2's override registry); does not depend on Layer 2 NT classification or Layer 3 peptidergic extension.

This document specifies what path B requires: calcium-kernel revision, classifier-bank retraining, metric-mapping update, and verification. Implementation is not in scope; this spec produces an implementation-ready plan and is grounded in empirical calibration data from this session.

---

## 1. Empirical foundation (data this spec is grounded in)

Three diagnostic blocks, all from the per-edge baseline scenario_traces (n=10 seeds × 6 scenarios × 60s each, generated earlier this session) and the Atanas 2023 dataset (10 worms, parsed to `atanas_worm_*.npz`):

### 1.1 Calcium kernel saturation (Task 1)

Existing kernel:
- Two-exponential GCaMP7f-like: `(1 - exp(-t/0.1)) * exp(-t/0.5)`
- Sample dt = 0.6 s (Atanas sampling rate)
- Per-spike amplitude scaling = 0.5
- Calibration claim: "single spike → peak ΔF/F ≈ 0.5 (matches Atanas range 0-4)"

**Observed saturation under per-edge synthetic ΔF/F vs Atanas-measured ΔF/F (per readout cell, p99):**

| cell | Atanas p99 | per-edge p99 (worst scenario) | ratio | status |
|---|---|---|---|---|
| AVEL | 1.13 | 51.05 (osmotic) | **45×** | SATURATED |
| AVER | 1.12 | 57.13 (osmotic) | **51×** | SATURATED |
| AIBL | 1.17 | 24.71 (osmotic) | **21×** | SATURATED |
| RMER | 0.28 | 3.70 (aerotaxis) | **13×** | SATURATED |
| IL2DL | 0.19 | 1.97 (chemo/aero) | **10×** | SATURATED |
| M3L | 0.27 | 2.46 (food) | **9×** | SATURATED |
| AUAL | 1.33 | 14.52 (food) | **11×** | SATURATED |
| URXL | 0.87 | 10.05 (aerotaxis) | **12×** | SATURATED |
| CEPDL | 0.56 | 2.15 (osmotic) | **4×** | SATURATED |
| I3 | 0.56 | 3.87 (food) | **7×** | SATURATED |
| NSML | 0.84 | 3.35 (food) | **4×** | SATURATED |
| NSMR | 0.78 | 3.10 (food) | **4×** | SATURATED |
| OLQDL | 1.27 | 2.37 | 2× | borderline |
| OLQDR | 1.14 | 1.77 | 1.6× | borderline |
| OLQVL | 1.24 | 2.37 | 2× | borderline |
| ASEL | 1.13 | 6.12 (chemotaxis) | 5× | saturated only on chemotaxis |
| SMDVL | 1.15 | 0.51 | 0.4× | UNDER-CALIBRATED (cell silenced) |

**16/18 cells exceed 2× Atanas p99 in at least one scenario.** SMDVL is the inverse problem (silenced). The kernel as currently specified produces synthetic ΔF/F values that do not match Atanas-trained distributions for the rate regime per-edge produces.

### 1.2 Classifier event probabilities under per-edge synthetic Ca (Task 2)

Running the existing 8-event classifier bank on per-edge synthetic ΔF/F:

| event | spontaneous | touch | osmotic | food | chemotaxis | aerotaxis |
|---|---|---|---|---|---|---|
| reversal_onset | **0.011** | **0.003** | **0.029** | **0.009** | **0.014** | **0.007** |
| reversal_offset | **0.995** | **0.995** | **0.995** | **0.995** | **0.994** | **0.989** |
| forward_run_onset | 0.045 | 0.022 | 0.022 | 0.023 | 0.055 | 0.038 |
| forward_run_offset | 0.032 | 0.021 | 0.021 | 0.018 | 0.039 | 0.023 |
| omega_onset | 0.000 | 0.000 | 0.009 | 0.000 | 0.000 | 0.000 |
| pirouette_entry | 0.060 | 0.021 | 0.040 | 0.011 | 0.079 | 0.015 |
| **quiescence_onset** | **0.706** | **0.882** | **0.915** | **0.734** | **0.815** | **0.603** |
| speed_burst_onset | 0.039 | 0.021 | 0.021 | 0.025 | 0.042 | 0.035 |

**Two classifiers are pinned at near-1.0 across all scenarios:**
- `reversal_offset` (~0.995): constantly says "not in reversal"
- `quiescence_onset` (~0.7-0.9): constantly says "entering quiescence"

**One classifier is essentially silent across all scenarios:**
- `reversal_onset` (0.003-0.029): never crosses 0.5 even on touch where biology predicts strong reversal

This is the mechanism behind the dREV → dPIR/QUI channel shift Session 1 observed in modulator audits. The classifier's logistic regression weights were trained on Atanas data with rate distributions where AVA peaks at ~5-10 Hz; under per-edge mode AVA peaks at 100+ Hz. The synthetic calcium pushes 25-57× past Atanas p99, and the logits saturate.

### 1.3 Behavioral trajectory under per-edge (Task 4)

Using `body_xy` from per-edge baseline scenario_traces (already saved during the overnight compute):

| scenario | net displacement (mm) | path length (mm) | path efficiency | d-to-target (start → end) | min d-to-target |
|---|---|---|---|---|---|
| spontaneous | 692 ± 55 | 34838 | 0.020 | n/a | n/a |
| touch | 812 ± 56 | 28999 | 0.028 | n/a | n/a |
| osmotic_shock | 457 ± 59 | 19079 | 0.024 | n/a | n/a |
| food | 439 ± 47 | 34927 | 0.013 | n/a | n/a |
| **chemotaxis** | 860 ± 77 | 37037 | 0.023 | **84 → 835** (got farther) | 11 mm |
| **aerotaxis** | 909 ± 69 | 37312 | 0.024 | **87 → 880** (got farther) | 12 mm |

**Path efficiency is uniformly ~0.02-0.03 across all scenarios** — essentially random-walk regardless of scenario. The body moves but does not navigate.

**For chemotaxis:** worm starts 84 mm from food, ends 835 mm away (10× farther). Got close (11 mm) at some point. **Chemotaxis trajectory phenotype fails under per-edge — sensors fire correctly but behavioral output doesn't navigate.**

**For aerotaxis:** same pattern. Worm starts 87 mm from preferred O2 zone, ends 880 mm away. Got close (12 mm) at some point. **Aerotaxis trajectory phenotype fails under per-edge.**

(Note: path lengths in the 30+ meter range over 60 seconds suggest the MuJoCo body/drag scaling produces motion ~10× faster than biological worms. This is a separate simulator-scaling issue not in path B's scope but flagged for the calibration-debt list.)

---

## 2. Path B scope (three sub-problems)

### 2.1 Sub-problem A — Calcium kernel revision

**Problem:** the spike-to-calcium convolution accumulates without bound at high firing rates. At 100 Hz, 600ms-binned spike count = 60; convolution with a kernel that peaks at ~1.0 and per-spike amp 0.5 produces ΔF/F up to 60 × 0.5 × 1.0 = 30 (vs Atanas measured ΔF/F that never exceeds ~3 for any cell).

**The kernel is missing biological saturation.** Real GCaMP indicators saturate at high Ca²⁺ levels. The current linear-convolution approach is correct for low rates (where Atanas was trained) but breaks down at the rates per-edge produces.

**Specified fix:** add a saturating nonlinearity on top of the linear convolution, parametrized to match Atanas distributions when fed Atanas-equivalent rates:

```python
# Current implementation (neural_classifier_bank.py:88)
def spikes_to_calcium(spike_trains, dt=0.6):
    kern = calcium_kernel(dt)
    out = np.zeros_like(spike_trains)
    for i in range(N):
        conv = np.convolve(spike_trains[i], kern, mode="full")[:T]
        out[i] = 0.5 * conv  # ← linear, no saturation
    return out

# Specified replacement
def spikes_to_calcium(spike_trains, dt=0.6):
    kern = calcium_kernel(dt)
    out = np.zeros_like(spike_trains)
    for i in range(N):
        conv = np.convolve(spike_trains[i], kern, mode="full")[:T]
        # Hill-saturation on the linear convolution
        # F_max set per Atanas global max ≈ 4.0
        # K_d set so single-spike-equivalent (~0.5 linear) maps to ~0.4 ΔF/F
        F_MAX = 4.0
        K_D = 2.0   # half-max at 2.0 linear convolution units
        N_HILL = 2  # cooperative-binding exponent
        out[i] = F_MAX * conv**N_HILL / (K_D**N_HILL + conv**N_HILL)
    return out
```

**Verification protocol for the kernel revision:**

1. Run the revised kernel on Atanas-derived spike trains (back out spike rates from Atanas ΔF/F via deconvolution). Expected: revised kernel reproduces Atanas ΔF/F distributions within ±20%.
2. Run the revised kernel on per-edge baseline rasters. Expected: per-cell p99 ΔF/F falls within 2× of Atanas p99 for all 18 readout cells (currently 16/18 fail this).
3. Pre-specified saturation curve check: at convolution input = 2× Atanas-equivalent peak, output should compress to ~F_MAX × 0.85 (so 100+ Hz rates don't push output past 4.0).

**Files to modify:**
- `scripts/brain/neural_classifier_bank.py:88-104` — `spikes_to_calcium` function body, ~15 lines.
- Add 3 module-level constants (`CALCIUM_F_MAX`, `CALCIUM_K_D`, `CALCIUM_N_HILL`) ~3 lines.

**Risk:** classifier weights stored in `classifier_bank.npz` were trained against the linear kernel. Replacing the kernel without retraining the bank breaks the classifier. **Kernel revision and classifier retraining must happen together in one commit, not staged separately.**

### 2.2 Sub-problem B — Classifier bank retraining

**Problem:** The 18-readout classifier was trained on Atanas worm 1-8 (train) and 9-10 (test) in default-mode firing-rate distributions. Under per-edge mode, those distributions shift dramatically (see Section 1.1).

**Three structural issues:**

1. **SMDVL is functionally silenced under per-edge** (firing rate 0.00-0.13 Hz across all 6 scenarios). One of 18 readout cells is dead. Bank retraining without SMDVL produces a 17-cell readout. Or SMDVL gets replaced.

2. **NSM (NSML/NSMR) fires at 1.6-2.8 Hz across all scenarios** — much lower than Atanas's NSM rates (which peak at 4 Hz with feeding-pump-aligned firing). This broke 5HT phenotype detection in today's modulator audit. Bank retraining needs to either accept NSM at lower rates or address the NSM-silencing root cause (Layer 3 peptidergic work).

3. **Cascade cells (AVEL, AVER, AIBL) fire at 100-160+ Hz under per-edge stim**, vs Atanas peaks at ~10 Hz. The classifier weights for these cells are specifically miscalibrated. After kernel revision (saturating to ~4.0 max), the synthetic ΔF/F will be ~4 instead of 25-57, but the temporal dynamics still differ.

**Specified retraining methodology:**

1. **Generate per-edge synthetic-Ca training data:** run scenario sweeps under per-edge with revised calcium kernel (n=10 seeds × 6 scenarios × 60s = same as overnight characterization, ~3 hr wall). For each (scenario, seed), produce 18-readout synthetic-Ca traces aligned with Atanas-style 600ms sampling.

2. **Behavioral labels:** the existing FSM was trained on Atanas behavior labels. Retraining under per-edge requires LABELS for per-edge data. Three options:
   - (a) Use the existing FSM/classifier on default-mode equivalent runs to LABEL per-edge runs at corresponding moments. Requires assumption that default-mode FSM correctly labels behaviors.
   - (b) Re-derive labels from per-edge body_xy trajectories using behavioral-state heuristics (velocity thresholds, curvature thresholds, etc.). Requires defining heuristic for each event class.
   - (c) Use Atanas-derived labels directly and accept that per-edge dynamics produce different label distributions — train classifier to match per-edge synthetic-Ca to Atanas-style labels. This perpetuates the channel shift but at least produces consistent classification.
   - **Recommendation:** option (b) — derive labels from kinematics directly. Velocity/curvature/duration-based heuristics for FORWARD/REVERSE/OMEGA/PIR/QUI are well-defined in the worm-behavior literature (Stephens et al. 2008, Helms et al. 2019).

3. **Training procedure:** preserve logistic regression as the model class (interpretable; matches existing infrastructure). Re-train per event with the same feature_set strings (`features_<event>` in `classifier_bank.npz`). Cross-validate on held-out seeds (e.g., train on seeds 42-47, validate on 48-51).

4. **Specific decisions:**
   - **SMDVL replacement candidate:** RMDVL (post_sign_glu shared, similar circuit role) or AVEL (already in 18-readout, less interesting). **Recommendation:** drop SMDVL → 17-cell readout for now; add a replacement only if classifier accuracy degrades >0.10 AUC.
   - **NSM:** keep in readout (its low firing IS the signal — Mode 2 dependence on NSM reflects readout-trivial classification). Don't replace.

**Files to create:**
- `scripts/brain/build_classifier_bank_v2.py` — new training script for per-edge regime. ~300 LOC, modeled on `neural_classifier_bank.py:train_bank()`.
- `scripts/brain/derive_behavioral_labels.py` — new module for kinematics-based labeling. ~200 LOC.

**Files to modify:**
- `scripts/brain/neural_classifier_bank.py:180-260` — `train_bank` function: parameterize on input data path so v2 trainer can reuse the model-fitting logic.

**Output artifacts:**
- `classifier_bank_v2_per_edge.npz` (alongside existing `classifier_bank.npz`)
- `ClosedLoopEnv` adds `classifier_bank_path` parameter to allow swap-in.

**Risk:**
- Behavioral label derivation is the highest-risk component. Bad labels → bad classifier. Pre-spec verification: derived labels for spontaneous scenario should match published worm spontaneous-state distributions (~70% forward, ~10% reversal at 22°C; Flavell 2013).
- 5HT/NSM Mode 2 reproduction may not transfer cleanly — that's a known calibration-debt item, not a classifier failure.

### 2.3 Sub-problem C — Metric mapping update

**Problem:** the existing audit infrastructure assumes specific metrics (ΔREV, ΔQUI) track specific neural manipulations (AVA ablation → ΔREV; RIS ablation → ΔQUI). Under per-edge, these mappings shift channels (AVA → dPIR; modulator audits today showed ΔREV signal weakens, ΔPIR strengthens).

**Specified update:** produce a documented mapping table that future audits can use:

| ablation | primary channel default | primary channel per-edge | biological interpretation |
|---|---|---|---|
| AVA | ΔREV (−0.49) | ΔPIR (−0.117) | reversal phenotype, expressed as pirouette-circuit shift under per-edge |
| RIS | ΔQUI (−0.24) | (suppressed at baseline; pending Layer 3) | quiescence — needs RIS rescue |
| NSM (5HT) | ΔREV (−0.61) + ΔQUI (+0.52) | ΔPIR (−0.335) + ΔQUI (+0.55) | 5HT/dwelling phenotype, more PIR-weighted under per-edge |
| AVB (PDF-1) | ΔREV (−0.17) | ΔREV (−0.083) + ΔPIR (−0.092) | forward command, both channels under per-edge |
| AVK (FLP-1) | flat (Mode 1) | flat (Mode 1) | readout-blind, both modes |
| AIA (FLP-2) | ΔREV (−0.59) | flat | sign-convention artifact — Mode 3 → Mode 1 under per-edge |
| FLP-1, NLP-12, OA, TA | Mode 1 | (untested under per-edge but expected Mode 1) | readout-blind, sign-mode-independent |
| DA | Mode 2 (CEPDL in readout) | (untested, expected Mode 2 with channel shift like 5HT) | direct readout effect |

**Files to modify:**
- New: `docs/per_edge_metric_mapping.md` — narrative document.
- `scripts/brain/phase0_audit.py` — add `--primary-channel` flag for ablation runs that defaults per ablation type (use mapping table above).

**Risk:** low. Mapping table can be amended as more ablations are tested.

---

## 3. Verification protocol

### 3.1 Pre-specified verification outcomes (before implementation)

After path B implementation lands (kernel revision + classifier retraining + metric mapping):

**Outcome α′:** Classifier bank reproduces phenotype validations under per-edge.
- AVA ablation → reversal_onset prob > 0.5 during touch peri-window in ≥ 8/10 seeds.
- RIS-equivalent quiescence (post-Layer 3 rescue if available) → quiescence_onset > 0.5 during food settled-window.
- 5HT (NSM) ablation → reversal phenotype detectable on dPIR or recovered on dREV.

**Outcome β′:** Calcium kernel revision matches Atanas distributions.
- Per-cell synthetic ΔF/F p99 within 2× Atanas p99 for ≥ 16/18 readout cells.
- Atanas-back-derived spike rates passed through revised kernel reproduce Atanas ΔF/F within ±20%.

**Outcome γ′:** Trajectory phenotypes recover.
- Chemotaxis: median min-d-to-target < 50 mm AND end-d-to-target < start-d-to-target (worm approaches food more often than not).
- Aerotaxis: same metric for preferred-O2 zone.

**Outcome δ′:** No regressions in Mode 1 modulator audits.
- FLP-1, NLP-12, OA, TA still classified as Mode 1 under per-edge after path B (matches today's overnight finding).

**Outcome ε′:** Failure mode — classifier accuracy doesn't recover above 0.7 AUC on held-out per-edge seeds. Indicates the per-edge dynamic regime is too far from Atanas training distribution for any classifier-on-Ca approach to work, and the project needs activity-FSM (P1 #4) as the readout instead of classifier-FSM. This is a fallback; would supersede path B's classifier-retraining strategy.

### 3.2 Test suite (concrete invocations)

```bash
# Test 1: kernel + classifier integration smoke
python scripts/brain/test_classifier_bank_v2_smoke.py
# Expected: classifier loads, predicts on per-edge synthetic Ca,
# event-probability distributions are within 0.05-0.95 (no saturation).

# Test 2: phenotype reproduction
python scripts/brain/phase0_audit.py --mode phenotype --ablations AVA \
    --tier default --use-per-edge-glu \
    --classifier-bank classifier_bank_v2_per_edge.npz \
    --output-prefix phase0_path_b_verify
# Expected: ΔREV more negative than path-A baseline; >= 8/10 negative seeds.

# Test 3: trajectory verification (chemotaxis)
python scripts/brain/phase0_audit.py --mode scenario --tier default \
    --use-per-edge-glu --scenarios chemotaxis \
    --classifier-bank classifier_bank_v2_per_edge.npz
# Then run: python scripts/brain/phase0_path_b_diagnostic.py
# Expected: trajectory metrics show worm approaches target more often than baseline.

# Test 4: Mode 1 regression check
python scripts/brain/phase0_modulator_d1.py --modulators FLP-1 NLP-12 OA TA \
    --use-per-edge-glu --output-dir <new path>
# Expected: 4/4 still Mode 1.
```

### 3.3 Rollback plan

- All changes touch new files (`build_classifier_bank_v2.py`, `derive_behavioral_labels.py`, `classifier_bank_v2_per_edge.npz`) plus parameterizing existing `train_bank()` and adding constants.
- Default `ClosedLoopEnv` continues to load `classifier_bank.npz` (the original Atanas-trained bank). v2 is opt-in via `classifier_bank_path` parameter.
- Rollback = drop the v2 npz and the path parameter override.

---

## 4. Effort + dependencies

### Effort estimate

| sub-problem | effort | compute |
|---|---|---|
| A — Kernel revision | 0.5 day (15 lines + 3 constants) | none |
| B — Classifier retraining | 3-5 days (label derivation, training pipeline, verification) | 4-6 hr (per-edge sweep at finer time resolution for label derivation) |
| C — Metric mapping doc | 0.5 day | none |
| Verification | 1 day | 4 hr (phenotype audits) |
| **Total** | **5-7 days** | **~10 hr compute** |

### Dependencies

**Blockers:**
- Path A T0 closure (Session 2's override registry must land) — needed because path B's verification runs use `--use-per-edge-glu` with curated overrides via the registry.

**Not blockers:**
- Layer 2 NT classification (Session 3's NT fix work) — path B's verification touches scenarios where NT classification matters (food, chemotaxis), but path B can proceed with current connectome and re-verify later.
- Layer 3 peptidergic extension (Session 3's RIS/ALA rescue) — RIS phenotype verification will fail until peptidergic rescue lands, but path B can verify other phenotypes.

**Sequencing recommendation:**

1. **Path A closure first** (Session 2 tonight).
2. **Path B sub-problem A (kernel revision)** can land standalone — produces immediate diagnostic improvement on cell saturation without breaking anything.
3. **Path B sub-problems B + C land together** as v2 classifier release. Requires ~1 week of focused work.
4. **Layer 2 + Layer 3 work** can proceed in parallel with path B sub-problem B once path A lands. Path B's phenotype verifications would then re-run after Layer 3 ships to capture rescued RIS.

### Sequencing argument: path B before vs after Layer 2/3

**Argument for path B FIRST:**
- Path B's kernel revision is independent and immediately useful.
- Path B's classifier retraining methodology is independent of NT classification or peptidergic rescue.
- Doing path B first means Layer 2/3 work can use a CORRECT classifier to verify that NT/peptidergic fixes produce the right behavioral outputs.

**Argument for Layer 2/3 FIRST:**
- Layer 3 rescue of RIS (and possibly ALA, NSM) would change the per-edge baseline firing distributions that path B trains on. Training on a baseline that doesn't include rescued cells produces a classifier that doesn't reflect the rescued circuit.
- Bank retraining is expensive (5-7 days) and re-doing it after Layer 3 lands would waste effort.

**Recommendation:** path B sub-problem A (kernel revision) lands first standalone. Sub-problems B+C wait until Layer 3 peptidergic rescue lands. This avoids re-training the classifier after RIS/ALA rescue changes baseline distributions.

---

## 5. What's NOT in scope for path B

- **Atanas-style spontaneous-state classifier validation against held-out worms.** That's a longer-term audit task; path B's verification uses synthetic per-edge data, not Atanas held-out worms.
- **Multi-time-window classifier (e.g., GRU/LSTM-based).** Logistic regression is preserved as the model class; switching to RNN-based classifier is a separate research direction.
- **Activity-FSM as alternative.** P1 #4 ActivityFSM exists and reads command-neuron rates directly. If path B fails (Outcome ε′), the project pivots to activity-FSM as the readout. But activity-FSM is not what path B is.
- **Defecation rhythm reproduction.** AVL is silenced under per-edge; defecation phenotype is a Layer 3 issue (peptidergic / pacemaker), not classifier-readout.
- **MuJoCo body scaling.** Trajectory analysis surfaced ~10× over-fast worm motion, but that's a separate body-scaling debt item.

---

## 6. Calibration debt summary (beyond path B)

For tomorrow's scoping decisions, items surfaced by this diagnostic that are NOT in path B's scope:

1. **MuJoCo body scaling:** path lengths 30+ meters over 60 seconds suggest body or drag scaling is ~10× too fast. Trajectory metrics need rescaling before chemotaxis/aerotaxis verification means anything quantitatively.
2. **Spontaneous worm baseline behavior distribution:** per-edge produces FWD 0.23 / REV 0.39 in spontaneous (Atanas-ish: ~70% FWD / 10% REV). State distribution is shifted; this is downstream of the classifier issue and should resolve when path B lands.
3. **Sensor-to-behavior mapping for chemotaxis/aerotaxis:** sensors fire correctly under per-edge but trajectories don't navigate. This may be classifier-broken (path B fixes) or may be sensor-to-motor cascade broken (separate investigation).
4. **AWC/AFD weakly responsive:** AWC peaks at 1.5 Hz across all scenarios; AFD at 1.3 Hz. Both are key sensors with stronger published response. May need sensor-cascade calibration (T2-#4) — independent of path B.
5. **Synthetic-Ca for ALL 300 neurons (not just 18):** Atanas data exists for ~134 cells per worm; the classifier uses the 18-cell strict intersection. A wider readout would be possible — but that's the readout-expansion item from earlier audit, not path B.

---

## 7. Implementation files (concrete)

### Created
- `scripts/brain/build_classifier_bank_v2.py` (new, ~300 LOC)
- `scripts/brain/derive_behavioral_labels.py` (new, ~200 LOC)
- `scripts/brain/test_classifier_bank_v2_smoke.py` (new, ~50 LOC)
- `docs/per_edge_metric_mapping.md` (new, narrative)
- `classifier_bank_v2_per_edge.npz` (new artifact, output of v2 training)

### Modified
- `scripts/brain/neural_classifier_bank.py:88-104` — `spikes_to_calcium` body, ~15 lines
- `scripts/brain/neural_classifier_bank.py:74-76` — add `CALCIUM_F_MAX`, `CALCIUM_K_D`, `CALCIUM_N_HILL` constants
- `scripts/brain/neural_classifier_bank.py:180-260` — `train_bank` parameterize input data path
- `scripts/brain/closed_loop_env.py:99-106` — add `classifier_bank_path: Path | None = None` constructor kwarg
- `scripts/brain/closed_loop_env.py:175` — pass classifier_bank_path to `ClassifierBank()` init
- `scripts/brain/phase0_audit.py` — add `--classifier-bank` CLI flag

### Reused as-is
- `scripts/brain/phase0_postvolt_peredge_baseline_analyze.py` — for rate distribution comparison post-implementation
- `scripts/brain/phase0_path_b_diagnostic.py` — for kernel saturation re-check post-implementation
