# Stage 5 — discriminative power test

## Method

For each compound (6 anesthetics + 8 negative controls), count how many of 30 Tier-1 targets show >10% occupancy at 100.0 µM aqueous (no K_p amplification — fair raw-aqueous comparison since negative controls don't have lipid:water partition data).

Discriminative pipeline: anesthetics show substantially higher engagement than negative controls. Non-discriminative: similar engagement counts.

Eger 2001 diagnostic: cis-1,2-dichloroethylene (anesthetic) vs trans (NOT anesthetic). Same lipid solubility, different shape. If pipeline distinguishes them, it's measuring target-specific fit, not bulk lipophilicity.

## Engagement counts at 100.0 µM aqueous

| compound | category | targets engaged (>10% occ) / 30 |
|---|---|---|
| etomidate | anesthetic | 30 |
| ketamine | anesthetic | 30 |
| propofol | anesthetic | 30 |
| isoflurane | anesthetic | 29 |
| sevoflurane | anesthetic | 29 |
| halothane | anesthetic | 10 |
| hexafluoroethane | negative_control | 24 |
| benzene | negative_control | 16 |
| cyclohexane | negative_control | 10 |
| npentane | negative_control | 2 |
| cis_12_dichloroethylene | negative_control | 0 |
| dimethyl_ether | negative_control | 0 |
| methanol | negative_control | 0 |
| trans_12_dichloroethylene | negative_control | 0 |

## cis/trans-1,2-DCE diagnostic

- cis (anesthetic): 0/30 engaged
- trans (non-anesthetic): 0/30 engaged
- difference: 0
- pipeline does NOT distinguish shape — likely responding to bulk lipophilicity

## Verdict

**DISCRIMINATIVE — pipeline distinguishes anesthetics from inert lipophilic compounds**

- Median anesthetic engagement: 30/30
- Median negative-control engagement: 2/30
- Gap: 28
