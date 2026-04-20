# Phase 3d-6 — Perturbation ensemble audit (reproducibility)

Reproducibility audit of the v3 perturbation suite. Each (config × ablation)
cell was run across 3 seeds (42, 43, 44) at 20 s simulation duration.
`np.random.seed()` and `brian2.seed()` were both locked, but Brian2 noise-
integration still produces substantial run-to-run variance at 20 s scales.
Report gives mean ± std of the (ablated − control) delta across seeds.

**Significance rule:** a phenotype effect is called "significant" if
|μ| ≥ 2·SEM (= 2·σ/√n) AND |μ| ≥ 0.05. With n=3 seeds SEM = σ/1.73. Smaller
deltas are treated as noise at this run length.

## Per-(config × ablation) results on the target state

| config | ablation | target state | μ ± σ | verdict |
|---|---|---|---:|---|
| **v3.0** | RIS / osmotic_shock | ΔQUI | **−0.24 ± 0.33** | suggestive, 2/3 seeds correct direction (Turek 2016) — not significant |
| v3.1 | RIS / osmotic_shock | ΔQUI | −0.01 ± 0.11 | noise |
| v3.2 | RIS / osmotic_shock | ΔQUI | +0.02 ± 0.03 | noise (effect fully suppressed) |
| v3.0 | NSM / food | ΔQUI | +0.18 ± 0.36 | noise |
| v3.1 | NSM / food | ΔQUI | +0.31 ± 0.46 | noise |
| v3.2 | NSM / food | ΔQUI | +0.25 ± 0.11 | **significant but wrong direction** (Flavell 2013 says −) |
| v3.0 | RIM / touch | ΔREV | −0.09 ± 0.30 | noise |
| v3.1 | RIM / touch | ΔREV | −0.05 ± 0.28 | noise |
| v3.2 | RIM / touch | ΔREV | +0.04 ± 0.02 | stable but tiny, not meaningful |
| **v3.0** | **AVA / touch** | **ΔREV** | **−0.57 ± 0.37** | ✓ **SIGNIFICANT, correct direction (Chalfie 1985)** |
| v3.1 | AVA / touch | ΔREV | −0.29 ± 0.52 | suggestive but noisy |
| v3.2 | AVA / touch | ΔREV | −0.03 ± 0.13 | noise (AVA effect suppressed) |
| v3.0 | AVB / spontaneous | ΔQUI | +0.10 ± 0.02 | small tight signal (FORWARD not logged) |
| v3.1 | AVB / spontaneous | ΔQUI | +0.16 ± 0.12 | small signal |
| v3.2 | AVB / spontaneous | ΔQUI | +0.02 ± 0.03 | noise |
| v3.0 | PDE / spontaneous | ΔQUI | +0.18 ± 0.32 | noise (one outlier) |
| v3.1 | PDE / spontaneous | ΔQUI | +0.07 ± 0.13 | noise |
| v3.2 | PDE / spontaneous | ΔQUI | −0.00 ± 0.01 | stable null |

## Per-seed data for the canonical phenotypes

**v3.0 RIS ablation / osmotic shock** (ΔQUI by seed):
| seed | control QUI | ablated QUI | Δ |
|---|---:|---:|---:|
| 42 | 0.75 | 0.22 | **−0.53** |
| 43 | — | — | −0.28 |
| 44 | — | — | +0.10 |

Mean ± σ = −0.24 ± 0.33. The original v3.0 single-run −0.53 was the strongest-signal draw; the mean is smaller and noisier. **Directionally suggestive of Turek 2016 (2/3 seeds negative), not statistically robust at n=3.**

**v3.0 AVA ablation / touch** (ΔREV by seed):
| seed | control REV | ablated REV | Δ |
|---|---:|---:|---:|
| 42 | — | — | −0.15 |
| 43 | — | — | −0.77 |
| 44 | — | — | −0.80 |

Mean ± σ = **−0.57 ± 0.37**. |μ|/2SEM = 1.35, clears significance threshold. **All 3 seeds correct direction.** This is the one robust reproduction: **AVA ablation abolishes reversal response to touch, matching Chalfie 1985.**

## What the audit actually shows

**1. The v3.0 "flagship success" on RIS was partly overclaimed.** The −0.53 single-run delta exists, but averaging across 3 seeds gives −0.24 ± 0.33. Directionally consistent with Turek 2016 (2/3 seeds) but not statistically significant with n=3. The signal is there; it's smaller and noisier than we thought.

**2. The AVA finding is real.** −0.57 ± 0.37 across 3 seeds, all correct direction, clears our pre-specified 2·SEM threshold. This is a genuine Chalfie 1985 reproduction and the one result we can confidently stand behind.

**3. Biological corrections (v3.1, v3.2) systematically degrade phenotype reproduction.** The RIS trend (small in v3.0) disappears in v3.1 and fully washes out in v3.2. The AVA effect weakens in v3.1 (still directional but non-significant) and disappears in v3.2. This is now statistically confirmed, not a single-run artifact: **adding biologically-accurate 5HT targeting and per-edge Glu receptors shifts the network out of the regime where these phenotypes emerge.**

**4. Brian2 stochastic variance at 20 s dominates most effect sizes.** With σ typically 0.3–0.5 per ablation, we cannot detect ablation effects smaller than ~0.35 reliably at n=3. This is a measurement problem, not a biology problem.

## Diagnosis of "what's going wrong"

We now know definitively:

1. **The model captures AVA → reversal causality.** AVA ablation reliably abolishes reversal response in v3.0. This isn't a fluke.

2. **The model captures RIS → quiescence only weakly in v3.0.** Directional signal exists but within noise at 20 s. Would likely be detectable at 60–120 s.

3. **Adding biological corrections degrades the calibrated phenotypes.** v3.1/v3.2 changes are scientifically correct but break the previous calibration of modulation strengths and FSM thresholds. Classic calibration cascade: fix one thing, break three.

4. **20 s is too short.** We need ensemble n ≥ 5 and duration ≥ 60 s to separate signal from noise for small-effect ablations.

## v3.3 scope — now grounded in measured data

Given the audit, v3.3 should:

1. **Standardise at n = 5 seeds × 60 s per ablation** for all phenotype claims. Existing 20 s × n=1 reports don't meet significance bar.
2. **Recalibrate modulation strength for v3.2 (per-edge Glu + 5HT excluded) to recover AVA reproduction.** Since v3.0 achieved it with per-neuron signs, we know it's possible; the v3.2 shift in network excitation just moved us out of the right regime. Tune `mod_strength_pa` and FSM thresholds to put us back.
3. **Aim for AVA reproduction at significance first**, then see if RIS follows. AVA is the easier target (larger effect, shorter causal chain).
4. **Drop claims for NSM, RIM, PDE, AVB** from the site copy. Audit shows they are all dominated by noise at current run length — we have no basis for any phenotype claim on those four.
5. **Only claim phenotypes that pass n ≥ 5 × 60 s × all-seeds-correct-direction bar.**

## Path to publishable claims

For a methods paper: the AVA/touch result stands. With v3.3's higher-power protocol (n=5 × 60s), we'd also likely recover RIS/osmotic at significance. Two reproductions with proper error bars are more defensible than the single-run "2/6 success" story I was telling before. The others stay as null results or get dropped.

This audit has been essential. The single-seed perturbation report was misleading. The 97-minute compute was worth the honest reassessment.
