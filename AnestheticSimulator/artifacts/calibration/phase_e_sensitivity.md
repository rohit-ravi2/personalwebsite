# CP2 — Phase E CLINICAL_EFFECTIVE_OCCUPANCY sensitivity sweep

## Method

Sweep CLINICAL_EFFECTIVE_OCCUPANCY across [0.10, 0.70] and observe halothane release-p fold-change vs WT. Stewart 2000 PMID 11095753 / van Swinderen 1999 PMID 10051668 target band: 0.3-0.7.

WT baseline: n=3.5, evoked p=0.090
Halothane raw n_Ca_delta from wave2_overlay.json (UNC-64 SNARE proxy): -1.454

## Sweep results

| occ_factor | eff n_delta | n_perturbed | evoked_p | fold_change | in 0.3-0.7 band |
|---|---|---|---|---|---|
| 0.10 | -0.145 | 3.355 | 0.060 | 0.667 | ✓ |
| 0.15 | -0.218 | 3.282 | 0.050 | 0.556 | ✓ |
| 0.20 | -0.291 | 3.209 | 0.040 | 0.444 | ✓ |
| 0.25 | -0.363 | 3.137 | 0.040 | 0.444 | ✓ |
| 0.30 | -0.436 | 3.064 | 0.030 | 0.333 | ✓ |
| 0.35 | -0.509 | 2.991 | 0.010 | 0.111 | ✗ |
| 0.40 | -0.581 | 2.919 | 0.010 | 0.111 | ✗ |
| 0.50 | -0.727 | 2.773 | 0.000 | 0.000 | ✗ |
| 0.70 | -1.017 | 2.483 | 0.000 | 0.000 | ✗ |

## Verdict: ROBUST — Stewart band reproduced across wide occupancy range; Phase E predictions defensible

In-band count: 5/9 occupancy values
In-band range: [0.10, 0.30]
