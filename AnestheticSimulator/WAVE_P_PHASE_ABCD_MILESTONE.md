# Wave P — Phases A+B+C+D consolidated milestone

## Pipeline state

- Vina dockings completed: 540
- (anesthetic, target) occupancy rows: 180
- Kinetic-shift rows for Wave 2 perturbation: 180

## Gate C.1 — multi-target framing falsifiability check

Verdict: **PASS**

(See `artifacts/occupancy/gate_c1_summary.md` for the engaged-target list and
top-15 (anesthetic, target) occupancy ranking.)

## Drop-in artifacts for Wave 2 perturbation runs

- `artifacts/kinetics/wave2_overlay.json` — by_anesthetic → by_target →
  parameter shifts (g_max factor, τ_decay factor, n_Ca delta, rate factor)
  with evidence-grade tags (LITERATURE / ANALOGY / CONSERVATIVE / DEFERRED).
- `artifacts/occupancy/occupancy_matrix.csv` — wide-form gene × anesthetic
  occupancy at 1× clinical EC50.
- `artifacts/occupancy/best_pocket_per_target.csv` — long form with pocket id,
  predicted Kd, and per-dose occupancy at 0.5×/1×/2×/5× EC50.

## Per-stage stdout snapshots (last invocation)

### scan_pose_affinities

```
Pose files found: 540
Druggability entries from log: 540
Wrote: /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/binding/vina_results_from_poses.csv (540 rows)
Wrote compat schema (consumable by phase_c_occupancy.py): /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/binding/vina_results.csv (540 rows)

Targets covered: 30
Genes (count of dockings):
  ACR-16      18 dockings  best ΔG=-6.00
  ACR-2       18 dockings  best ΔG=-6.60
  AVR-14      18 dockings  best ΔG=-6.60
  AVR-15      18 dockings  best ΔG=-6.90
  EXP-1       18 dockings  best ΔG=-5.90
  GAS-1       18 dockings  best ΔG=-6.00
  GLC-1       18 dockings  best ΔG=-5.70
  GLC-2       18 dockings  best ΔG=-5.90
  LEV-1       18 dockings  best ΔG=-5.90
  MEV-1       18 dockings  best ΔG=-6.70
  NCA-2       18 dockings  best ΔG=-6.20
  NLF-1       18 dockings  best ΔG=-5.80
  NUO-1       18 dockings  best ΔG=-6.90
  NUO-2       18 dockings  best ΔG=-7.00
  NUO-3       18 dockings  best ΔG=-7.70
  NUO-4       18 dockings  best ΔG=-7.50
  RIC-4       18 dockings  best ΔG=-5.50
  SNB-1       18 dockings  best ΔG=-4.90
  SNT-1       18 dockings  best ΔG=-5.50
  TWK-18      18 dockings  best ΔG=-6.90
  TWK-29      18 dockings  best ΔG=-6.20
  TWK-7       18 dockings  best ΔG=-6.60
  UNC-13      18 dockings  best ΔG=-6.50
  UNC-18      18 dockings  best ΔG=-7.30
  UNC-29      18 dockings  best ΔG=-5.60
  UNC-38      18 dockings  best ΔG=-6.10
  UNC-49      18 dockings  best ΔG=-6.50
  UNC-63      18 dockings  best ΔG=-6.20
  UNC-64      18 dockings  best ΔG=-5.70
  UNC-79      18 dockings  best ΔG=-6.40

```

### phase_c_occupancy

```
Vina rows read: 540
Unique (anesthetic, gene) pairs with valid affinity: 180
Best-pocket-per-target table: /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/occupancy/best_pocket_per_target.csv
Occupancy matrix at 1x EC50: /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/occupancy/occupancy_matrix.csv

============================================================
GATE C.1 — multi-target framing falsifiability check
============================================================
Targets with >10% occupancy at 1x EC50 (≥1 anesthetic): 30
  PASS — multi-target framing supported (≥5 targets engaged)

Engaged targets: ACR-16, ACR-2, AVR-14, AVR-15, EXP-1, GAS-1, GLC-1, GLC-2, LEV-1, MEV-1, NCA-2, NLF-1, NUO-1, NUO-2, NUO-3, NUO-4, RIC-4, SNB-1, SNT-1, TWK-18, TWK-29, TWK-7, UNC-13, UNC-18, UNC-29, UNC-38, UNC-49, UNC-63, UNC-64, UNC-79

Top 15 (anesthetic, target) pairs by 1x-EC50 occupancy:
anesthetic   gene            ΔG      Kd_uM   occ@1x
ketamine     ACR-2        -6.60      14.45    1.000
ketamine     AVR-14       -6.40      20.25    1.000
ketamine     AVR-15       -6.70       12.2    1.000
ketamine     MEV-1        -6.70       12.2    1.000
ketamine     NUO-1        -6.90      8.704    1.000
ketamine     NUO-2        -7.00      7.352    1.000
ketamine     NUO-3        -7.40      3.741    1.000
ketamine     NUO-4        -7.50       3.16    1.000
ketamine     TWK-18       -6.90      8.704    1.000
ketamine     UNC-79       -6.40      20.25    1.000
ketamine     ACR-16       -5.90      47.11    0.999
ketamine     EXP-1        -5.90      47.11    0.999
ketamine     GAS-1        -6.00      39.79    0.999
ketamine     GLC-1        -5.70      66.03    0.999
ketamine     GLC-2        -5.90      47.11    0.999

Summary md: /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/occupancy/gate_c1_summary.md

```

### phase_d_kinetic_shifts

```
Loaded 180 occupancy rows; 32 target→class mappings
Per-row kinetic shifts:  /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/kinetics/kinetic_shifts_at_1xEC50.csv  (180 rows)
Wave 2 overlay JSON:     /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/kinetics/wave2_overlay.json

=== Phase D summary ===
Total (anesthetic × target × parameter) shifts: 180
By mechanism class:
  nachr_antagonism           36 rows
  snare_cooperativity        36 rows
  complex_i_block            30 rows
  glucl_potentiation         24 rows
  nca_block                  18 rows
  k2p_potentiation           18 rows
  gaba_potentiation          12 rows
  complex_ii_block           6 rows
By evidence grade:
  LITERATURE       150 rows
  ANALOGY          24 rows
  CONSERVATIVE     6 rows

Top 12 largest-magnitude shifts at 1× EC50:
anesthetic   gene       class                     param                   value grade     
  ketamine    EXP-1      gaba_potentiation         tau_decay_factor       3.9970 LITERATURE
  ketamine    UNC-49     gaba_potentiation         tau_decay_factor       3.9970 LITERATURE
  isoflurane  EXP-1      gaba_potentiation         tau_decay_factor       3.9700 LITERATURE
  halothane   EXP-1      gaba_potentiation         tau_decay_factor       3.9670 LITERATURE
  halothane   UNC-49     gaba_potentiation         tau_decay_factor       3.9430 LITERATURE
  sevoflurane EXP-1      gaba_potentiation         tau_decay_factor       3.9100 LITERATURE
  isoflurane  UNC-49     gaba_potentiation         tau_decay_factor       3.9070 LITERATURE
  sevoflurane UNC-49     gaba_potentiation         tau_decay_factor       3.8950 LITERATURE
  propofol    UNC-49     gaba_potentiation         tau_decay_factor       3.8560 LITERATURE
  propofol    EXP-1      gaba_potentiation         tau_decay_factor       3.6820 LITERATURE
  ketamine    TWK-18     k2p_potentiation          g_max_factor           3.0000 LITERATURE
  ketamine    TWK-29     k2p_potentiation          g_max_factor           2.9980 LITERATURE

Markdown summary: /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/kinetics/phase_d_summary.md

```
