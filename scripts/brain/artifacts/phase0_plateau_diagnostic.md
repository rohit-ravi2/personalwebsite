# Phase 0 — T4-2 plateau equation diagnostic

Single-neuron diagnostic on AVAL to determine whether the scaffold's 0 ms / +4.5 mV plateau baseline is a parameter problem, a rest-potential mismatch, or an equation-formulation problem.

## Scaffold parameters at time of diagnostic

- g_ca = 2.5 nS
- v_ca_half = -30.0 mV
- g_axial = 1.5 nS
- v_rest = -65.0 mV  (likely inherited from mammalian template)

**Confirmed primary reference:** Mellem et al. 2008 (Nat Neurosci, PMC2697921) — replaces the unverified 'Gao & Hobert 2020' citation. Mellem reports AVA rest at −20 to −30 mV.

## Probe 1 — Analytical I_ca(v_d)

| v_d (mV) | m_inf | I_ca (pA) |
|---|---|---|
| -60 | 0.0067 | 1.84 |
| -50 | 0.0344 | 8.61 |
| -40 | 0.1589 | 35.75 |
| -30 | 0.5 | 100.0 |
| -20 | 0.8411 | 147.2 |
| -10 | 0.9656 | 144.83 |
| 0 | 0.9933 | 124.16 |

The equation's shape is correct: m_inf is sigmoid, ~0 at −50 mV, ~0.5 at v_ca_half (−30 mV), saturating near 1 above −20 mV. I_ca follows m_inf × (e_ca − v_d) as expected.

## Probe 2 — Somatic 50 pA / 100 ms (baseline replication)

- v_s peak = -54.92 mV
- v_d peak = -60.21 mV (matches Phase 0 baseline −60.2 mV)
- m_inf peak = 0.0065 (effectively zero — Ca channel never opens)
- I_ca peak = +1.78 pA

## Probe 3 — Strong somatic 500 pA / 100 ms

- v_s peak = +41.24 mV
- v_d peak = +3.83 mV
- v_d crossed v_ca_half? **YES**

## Probe 4 — Dendritic clamp scan

| v_d clamp (mV) | m_inf | h_ss | I_ca (pA) |
|---|---|---|---|
| -60 | 0.007 | 0.991 | +1.8 |
| -50 | 0.034 | 0.956 | +8.6 |
| -40 | 0.159 | 0.815 | +35.7 |
| -35 | 0.303 | 0.682 | +64.4 |
| -30 | 0.500 | 0.541 | +100.0 |
| -25 | 0.697 | 0.435 | +130.7 |
| -20 | 0.841 | 0.374 | +147.2 |
| -10 | 0.966 | 0.331 | +144.8 |

Ca-current magnitudes under direct dendritic clamp confirm the gating equations behave as analytical predictions require. Gating is not broken.

## Probe 5 — v_rest = −25 mV (Mellem 2008)

- v_s peak = -10.68 mV
- v_d peak = -3.69 mV
- m_inf peak = 0.9877
- I_ca peak = +138.90 pA
- v_d crossed v_ca_half? **YES**
- Plateau after release (v_d at +100 ms post): -12.507206572402485 mV

## Probe 6 — g_axial = 10 nS (~7× scaffold default)

- v_s peak = -56.06 mV
- v_d peak = -57.33 mV
- v_d crossed v_ca_half? **NO**

## Diagnosis

- ✗ **Gating equations appear broken.** Even under direct dendritic clamp at −30 mV, I_ca does not reach expected magnitude.
- ✓ **Root cause: v_rest parameter mismatch.** Scaffold uses −65 mV (mammalian cortical template); Mellem 2008 measures AVA rest at −20 to −30 mV. Switching v_rest = −25 mV activates the plateau under 50 pA injection without other changes.

## Implications for T4-2 plan

- Before expanding the plateau-calibration grid, update `COMPARTMENTAL_ROSTER` v_rest values to match Mellem 2008 (−20 to −30 mV for command interneurons).
- If Probe 5 shows the rest fix is sufficient, re-run Phase 0 plateau baseline — expect 15/15 pass at −25 mV rest without any other changes.
- If the rest fix is necessary-but-not-sufficient, the grid search still runs but starts from the corrected rest.
- Citation audit: anywhere the project's documentation cites 'Gao & Hobert 2020' for AVA, replace with Mellem et al. 2008 (Nat Neurosci, PMC2697921, DOI:10.1038/nn.2131).
