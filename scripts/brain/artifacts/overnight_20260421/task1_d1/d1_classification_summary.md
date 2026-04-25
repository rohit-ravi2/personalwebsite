# Task 1 — D1 modulator Mode classification

Generated: 2026-04-21 12:15:09

Empirical Mode classification for each of the 9 existing v3 
modulators based on control vs releaser-ablated behavioral 
state distributions (scenario-matched, n=3 seeds × 60s).

## Per-modulator classification

| modulator | scenario | ΔFWD | ΔREV | ΔOMG | ΔPIR | ΔQUI | observed Mode | rationale |
|---|---|---|---|---|---|---|---|---|
| **5HT** | touch | +0.09 | -0.61 | +0.14 | -0.13 | +0.52 | Mode 2 (readout-trivial) | releaser in readout (['NSML', 'NSMR']); signal lik |
| **DA** | touch | +0.00 | -0.16 | +0.00 | +0.00 | +0.16 | Mode 2 (readout-trivial) | releaser in readout (['CEPDL']); signal likely via |
| **FLP-1** | osmotic_shock | +0.00 | -0.00 | +0.00 | +0.00 | +0.00 | Mode 1 (readout-blind) | no behavioral signal; mechanism may operate below  |
| **FLP-11** | osmotic_shock | +0.01 | -0.01 | +0.00 | +0.02 | -0.01 | Mode 1 (readout-blind) | no behavioral signal; mechanism may operate below  |
| **FLP-2** | osmotic_shock | +0.09 | -0.59 | +0.00 | +0.08 | +0.42 | Mode 3 (readout-cascade) | releaser outside readout but signal present; propa |
| **NLP-12** | osmotic_shock | -0.00 | +0.00 | +0.00 | -0.02 | +0.01 | Mode 1 (readout-blind) | no behavioral signal; mechanism may operate below  |
| **OA** | touch | -0.00 | +0.01 | +0.00 | -0.02 | +0.01 | Mode 1 (readout-blind) | no behavioral signal; mechanism may operate below  |
| **PDF-1** | touch | -0.00 | -0.17 | +0.00 | -0.05 | +0.23 | Mode 3 (readout-cascade) | releaser outside readout but signal present; propa |
| **TA** | touch | -0.01 | +0.03 | +0.00 | -0.03 | +0.01 | Mode 1 (readout-blind) | no behavioral signal; mechanism may operate below  |

## Summary

- **Mode 1 (readout-blind)**: 5 modulators
- **Mode 2 (readout-trivial)**: 2 modulators
- **Mode 3 (readout-cascade)**: 2 modulators

## Per-modulator state distributions

### 5HT (touch, releasers: NSML;NSMR;HSNL;HSNR;ADFL;ADFR)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.03 ± 0.01 | 0.12 ± 0.03 | +0.09 |
| REVERSE | 0.71 ± 0.06 | 0.10 ± 0.07 | -0.61 |
| OMEGA | 0.02 ± 0.00 | 0.15 ± 0.06 | +0.14 |
| PIROUETTE | 0.13 ± 0.06 | 0.00 ± 0.00 | -0.13 |
| QUIESCENT | 0.12 ± 0.01 | 0.63 ± 0.14 | +0.52 |

### DA (touch, releasers: PDEL;PDER;ADEL;ADER;CEPDL;CEPDR;CEPVL;CEPVR)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.03 ± 0.01 | 0.03 ± 0.01 | +0.00 |
| REVERSE | 0.71 ± 0.06 | 0.55 ± 0.23 | -0.16 |
| OMEGA | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.00 |
| PIROUETTE | 0.13 ± 0.06 | 0.13 ± 0.06 | +0.00 |
| QUIESCENT | 0.12 ± 0.01 | 0.27 ± 0.19 | +0.16 |

### FLP-1 (osmotic_shock, releasers: AVKL;AVKR)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.02 ± 0.01 | 0.02 ± 0.00 | +0.00 |
| REVERSE | 0.85 ± 0.04 | 0.85 ± 0.02 | -0.00 |
| OMEGA | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.00 |
| PIROUETTE | 0.03 ± 0.05 | 0.03 ± 0.02 | +0.00 |
| QUIESCENT | 0.07 ± 0.01 | 0.08 ± 0.01 | +0.00 |

### FLP-11 (osmotic_shock, releasers: RIS)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.01 |
| REVERSE | 0.87 ± 0.02 | 0.86 ± 0.05 | -0.01 |
| OMEGA | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.00 |
| PIROUETTE | 0.02 ± 0.02 | 0.03 ± 0.02 | +0.02 |
| QUIESCENT | 0.07 ± 0.01 | 0.06 ± 0.02 | -0.01 |

### FLP-2 (osmotic_shock, releasers: AIAL;AIAR;RID)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.02 ± 0.01 | 0.11 ± 0.01 | +0.09 |
| REVERSE | 0.85 ± 0.04 | 0.26 ± 0.10 | -0.59 |
| OMEGA | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.00 |
| PIROUETTE | 0.03 ± 0.05 | 0.12 ± 0.02 | +0.08 |
| QUIESCENT | 0.07 ± 0.01 | 0.49 ± 0.12 | +0.42 |

### NLP-12 (osmotic_shock, releasers: DVA)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.02 ± 0.01 | 0.02 ± 0.00 | -0.00 |
| REVERSE | 0.85 ± 0.04 | 0.86 ± 0.02 | +0.00 |
| OMEGA | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.00 |
| PIROUETTE | 0.03 ± 0.05 | 0.02 ± 0.02 | -0.02 |
| QUIESCENT | 0.07 ± 0.01 | 0.09 ± 0.02 | +0.01 |

### OA (touch, releasers: RICL;RICR)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.03 ± 0.01 | 0.03 ± 0.00 | -0.00 |
| REVERSE | 0.71 ± 0.06 | 0.72 ± 0.11 | +0.01 |
| OMEGA | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.00 |
| PIROUETTE | 0.13 ± 0.06 | 0.11 ± 0.04 | -0.02 |
| QUIESCENT | 0.12 ± 0.01 | 0.13 ± 0.07 | +0.01 |

### PDF-1 (touch, releasers: AVBL;AVBR)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.03 ± 0.01 | 0.03 ± 0.01 | -0.00 |
| REVERSE | 0.71 ± 0.06 | 0.54 ± 0.09 | -0.17 |
| OMEGA | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.00 |
| PIROUETTE | 0.13 ± 0.06 | 0.08 ± 0.05 | -0.05 |
| QUIESCENT | 0.12 ± 0.01 | 0.34 ± 0.03 | +0.23 |

### TA (touch, releasers: RIML;RIMR)

| state | CONTROL | ABLATED | Δ |
|---|---|---|---|
| FORWARD | 0.03 ± 0.01 | 0.02 ± 0.00 | -0.01 |
| REVERSE | 0.71 ± 0.06 | 0.74 ± 0.09 | +0.03 |
| OMEGA | 0.02 ± 0.00 | 0.02 ± 0.00 | +0.00 |
| PIROUETTE | 0.13 ± 0.06 | 0.09 ± 0.03 | -0.03 |
| QUIESCENT | 0.12 ± 0.01 | 0.13 ± 0.07 | +0.01 |

## Interpretation

The three-Mode taxonomy now has empirical classification 
for the full v3 modulator set. Each Mode has a distinct 
expected signature in behavioral readout:

- **Mode 1 (readout-blind):** |Δ| < 0.15 across all states. 
  Behavioral null despite mechanism operation. Required 
  molecular audit to detect the underlying signal.
- **Mode 2 (readout-trivial):** strong Δ driven by direct 
  readout-neuron zeroing. Signal is real but not biology — 
  it's the classifier responding to having its inputs cut.
- **Mode 3 (readout-cascade):** Δ signal via synaptic 
  propagation from non-readout ablated neuron to readout 
  neurons. Direction of the effect may match biology but 
  mechanism does not.

This completes the paper's empirical basis for the 
4-layer falsification framework at Layer 1 (classifier 
readout correctness).