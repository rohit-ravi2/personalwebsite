# CP3 — DCE concentration sweep diagnostic

## Method

Reuse existing Vina poses for cis-1,2-dichloroethylene (anesthetic) and trans-1,2-dichloroethylene (non-anesthetic per Eger 2001) against the 30 Tier-1 C. elegans targets. Compute engagement count at varying aqueous concentrations using Hill occupancy (no K_p amplification — DCE K_p data unverified).

**Diagnostic claim:** if pipeline distinguishes cis (anesthetic) from trans (non-anesthetic), it's measuring target-specific shape fitting, not bulk lipophilicity. If they engage similarly across concentrations, pipeline lacks conformational specificity.

## Sweep results

| conc (µM) | cis engaged / 30 | trans engaged / 30 | gap (cis - trans) |
|---|---|---|---|
| 100 | 0 | 0 | 0 |
| 300 | 0 | 0 | 0 |
| 1000 | 9 | 12 | -3 |
| 3000 | 29 | 29 | 0 |
| 10000 | 30 | 30 | 0 |
| 30000 | 30 | 30 | 0 |

## Verdict: **FAIL — no conformational specificity (pipeline responds to bulk lipophilicity, not shape)**

- Max gap (cis − trans): 0
- Min gap: -3
- Eger 2001 anesthetic concentration range (1-10 mM aqueous):
  - 1000 µM: cis 9, trans 12, gap -3
  - 3000 µM: cis 29, trans 29, gap 0
  - 10000 µM: cis 30, trans 30, gap 0
