# CeNGEN TPM inventory — Layer 1 channels (Phase 3)

**Status:** Phase 3 of §7.3.5 Path 2. Per-(channel, cell) TPM inventory
for the 9 channels in current Layer 1 cells, plus heteromer check for
KQT-1/KQT-3 + paralog aggregation for IRK and NCA families.

**Date:** 2026-05-12

**Reference:** `docs/channel_parameter_derivation_methodology.md` §2 +
`docs/channel_gamma_inventory.md` Phase 2 deliverable.

**Data source:** `cengen.org/storage/021821_medium_threshold2.csv` (CeNGEN
T2 integrated bulk+scRNA-seq, recommended-default threshold). Cells AVAL
and AVAR inherit the CeNGEN "AVA" class TPM (CeNGEN doesn't L/R-split
AVA, AIY, or RIM).

---

## 1 · Per-(gene, cell) TPM table (CeNGEN T2)

```
gene          AVA       AIY       RIM    role
---------  -------   -------   -------   ---------------------------
egl-19        89.5      30.3     132.9   Cav1 L-type Ca
cca-1        109.3       0.0      36.3   Cav3 T-type Ca
unc-2        203.9      93.9      57.2   Cav2 P/Q-type Ca
shl-1          0.0       0.0     153.1   Kv4 / Shal A-type K
kqt-1          0.0      63.4       0.0   KCNQ M-current K
kqt-3          0.0       0.0       5.9   KCNQ paralog (heteromer-check)
egl-2         64.5       0.0      65.8   EAG / Kv10 K
unc-103       46.1       0.0     112.2   hERG / Kv11 K
irk-1         78.2       0.0       0.0   Kir2 paralog
irk-2         68.5       0.0      87.6   Kir2 paralog
irk-3         18.9       0.0      32.7   Kir2 paralog
nca-1     NOT FOUND IN CeNGEN T2          NALCN pore-forming paralog (below threshold)
nca-2        153.2      29.2      88.0   NALCN pore-forming paralog
unc-77       165.2       0.0      93.3   NCA-family (excluded — auxiliary per §2.4)
unc-80       839.4     147.0     291.7   NCA channelosome regulator (excluded)
```

Source: `021821_medium_threshold2.csv` queried 2026-05-12.

---

## 2 · Per-channel aggregated TPMs (under methodology §2.4 rules)

Wave 2 cell builders use single-channel-module parameterizations for some
channels that map to multiple CeNGEN genes (IRK family; NCA pore-forming
paralogs). Per methodology §2.4 paralog-separate default, **each paralog
forms its own channel; their gbars sum at the membrane.** Wave 2's single
channel module represents the sum.

**Aggregation rule for v1:**
- Single-gene channels: `TPM_channel = TPM_gene`
- Multi-gene paralog channels (IRK, NCA): `TPM_channel = sum(TPM_pore_forming_paralogs)`
- Auxiliary subunits (unc-77, unc-80): excluded from density per §2.4
- Heteromeric channels (none confirmed in scope): would use min-across-pore-forming (KQT-1 heteromer check below)

**Aggregated TPMs for Layer 1 derivation:**

| channel module | TPM_AVA | TPM_AIY | TPM_RIM | aggregation |
|---|---:|---:|---:|---|
| EGL-19 | 89.5 | 30.3 | 132.9 | single gene egl-19 |
| CCA-1 | 109.3 | 0.0 | 36.3 | single gene cca-1 |
| UNC-2 | 203.9 | 93.9 | 57.2 | single gene unc-2 |
| SHL-1 | 0.0 | 0.0 | 153.1 | single gene shl-1 |
| KQT-1 | 0.0 | 63.4 | 0.0 | single gene kqt-1 (heteromer ruled out — see §4) |
| EGL-2 | 64.5 | 0.0 | 65.8 | single gene egl-2 |
| UNC-103 | 46.1 | 0.0 | 112.2 | single gene unc-103 |
| IRK | **165.6** | **0.0** | **120.3** | sum(irk-1, irk-2, irk-3) |
| NCA | **153.2** | **29.2** | **88.0** | nca-2 alone (nca-1 below T2 threshold; unc-77 excluded as auxiliary) |

---

## 3 · Substantive findings

### 3.1 SHL-1 in AIY — Nicoletti-vs-CeNGEN discrepancy (FINDING)

Wave 2 AIY cell builder includes SHL-1 (Kv4 A-type K) with gbar ≈
7.59e-4 S/cm² (= 0.5 nS / 65.89e-8 cm²). **CeNGEN T2 reports
TPM_shl-1_AIY = 0** (below threshold 2).

Under Path 2 derivation: `gbar_SHL1_AIY = γ × TPM × E × C_global = 6 pS × 0 × ... = 0`.
Path 2 predicts no SHL-1 current in AIY, contradicting Nicoletti.

**Possible explanations:**
1. SHL-1 is expressed below CeNGEN T2 threshold but functionally present
   biologically (T2 false negative); Nicoletti's AIY parameterization
   captured a real but low-expression channel
2. Nicoletti's AIY model uses SHL-1 phenomenologically (parameterization
   based on whole-cell I-V fit, not AIY-specific expression data); the
   "AIY SHL-1" current may actually come from a different K channel that
   Nicoletti folded into SHL-1's kinetic profile
3. CeNGEN T2 doesn't sample AIY's SHL-1 expression well enough to detect
   it (limited single-cell coverage)

**Resolution path:** Document as Phase 3 finding. Phase 5 validation will
test whether AIY can achieve stable rest with SHL-1 gbar = 0. If yes,
Path 2 derivation rejects Nicoletti's SHL-1 inclusion in AIY as a
non-biology-grounded artifact of the fitting process. If no (AIY rest
fails without SHL-1), substantive question opens about how to reconcile
Path 2 derivation with Nicoletti's parameterization.

This is exactly the kind of finding Path 2 was designed to surface per
methodology §1: "Methodology is validated by checking whether derived
gbar values reproduce Nicoletti's published data within tolerance.
Failure to match is a substantive finding... not a failure of fitting."

### 3.2 nca-1 below CeNGEN T2 threshold — NCA channel uses nca-2 alone

CeNGEN T2 contains no `nca-1` entry for any neuron. WBGene00003502
(nca-1 WormBase ID) doesn't match any T2 row. Likely below threshold
broadly. The NCA channel-family pore-forming subunits in CeNGEN T2 are
nca-2 only.

Under §2.4 paralog-separate + auxiliary-excluded rule:
- NCA channel TPM = TPM_nca-2 alone (nca-1 absence → contributes 0)
- Wave 2 cell builder's single "NCA" module parameterized by nca-2 TPM

This is consistent with `AnestheticSimulator/equation_validation/
cengen_coupling/cengen_channel_inventory.csv` which mapped Wave 2's NCA
channel to {nca-2, unc-77} (no nca-1).

**v1 decision:** NCA channel TPM = TPM_nca-2 alone. unc-77 excluded as
auxiliary per §2.4. If Phase 5 surfaces NCA discrepancy, alternative
aggregation (include unc-77 as pore-forming) is a candidate refinement —
unc-77 status as pore vs auxiliary is debated in NALCN-family literature
(Yeh 2008 vs Lu 2009 inconsistent).

### 3.3 KQT-1/KQT-3 heteromer hypothesis — REJECTED (paralog-separate confirmed)

Per Phase 2 §4.2 flag, Phase 3 checked KQT-3 expression in AIY relative
to KQT-1. Per the decision rule:
- AIY: kqt-1 = 63.4 TPM, kqt-3 = **0.0 TPM** (below T2 threshold)
- kqt-3 / kqt-1 ratio in AIY = **0%** (far below 20% paralog-separate
  threshold)

**Decision:** Paralog-separate confirmed for KQT-1 in AIY. KQT-1 alone
determines KQT density. Methodology §2.4 KQT-family flag resolved.
Heteromer aggregation NOT applied.

### 3.4 IRK family — sum-of-paralogs aggregation under §2.4 paralog-separate

Three IRK paralogs (irk-1, irk-2, irk-3) all carry K current with similar
unitary conductance (Kir2.x ~25 pS) — methodology §2.4 paralog-separate
treats them as separate channels in parallel. Wave 2 cell builder uses
single IRK module with one gbar.

Under sum-of-paralogs interpretation (paralog-separate at gene level,
sum at channel module): **AVA TPM_IRK = 165.6 (78.2+68.5+18.9)**;
**RIM TPM_IRK = 120.3 (0+87.6+32.7)**; **AIY TPM_IRK = 0**.

**Per-cell paralog dominance:**
- AVA: irk-1 dominant (47% of total IRK); irk-2 41%; irk-3 12%
- RIM: irk-2 dominant (73%); irk-3 27%; irk-1 absent
- AIY: no IRK expression detected at T2

**v1 decision:** sum-of-paralogs for the single Wave 2 IRK channel
module. Documented for transparency. If Phase 5 surfaces IRK discrepancy
specifically in cells with paralog imbalance (AVA vs RIM), alternative
aggregations (dominant-paralog-only, weighted by Kir2.x homolog-specific
γ) are refinement candidates.

### 3.5 Other zero-TPM cases in Layer 1 cell channel sets

Per the per-cell channel inventory (Phase 2 §3.5 / progress summary):
checked each cell's Wave 2 channels against CeNGEN T2 TPMs:

| cell | Wave 2 channels | TPM>0 in CeNGEN? | discrepancies |
|---|---|---|---|
| AVAL/AVAR | EGL-19, IRK, NCA, UNC-103 | all ✓ | none |
| AIY (v1) | EGL-19, KQT-1, SHL-1, NCA | EGL-19 ✓, KQT-1 ✓, **SHL-1 ✗**, NCA ✓ | **SHL-1 = 0** (§3.1) |
| RIM | EGL-19, SHL-1, IRK, CCA-1, UNC-2, EGL-2 | all ✓ | none |

**One discrepancy total:** AIY SHL-1. Most cell-channel pairs are consistent
between Nicoletti's parameterization and CeNGEN expression. This is a
methodologically positive sign — Path 2 derivation will mostly reproduce
Nicoletti at the channel-presence level even if gbar magnitudes shift.

---

## 4 · Heteromer/paralog decision summary (methodology §2.4 status)

| family | scope | decision rule (Phase 3) | outcome |
|---|---|---|---|
| EGL-19 | AVAL/AVAR/AIY/RIM | single-gene | (default) |
| CCA-1 | RIM | single-gene | (default) |
| UNC-2 | RIM | single-gene | (default) |
| SHL-1 | AIY/RIM | single-gene | (default; AIY=0 TPM is §3.1 finding) |
| KQT-1 | AIY | paralog-separate (heteromer flag from Phase 2 §4.2) | **heteromer rejected — kqt-3/kqt-1 ratio = 0% < 20%; KQT-1 alone** |
| EGL-2 | RIM | single-gene | (default) |
| UNC-103 | AVAR/RIM | single-gene | (default) |
| IRK | AVAL/AVAR/RIM | paralog-separate → sum-of-paralogs at channel module | sum(irk-1, irk-2, irk-3) per cell |
| NCA | AVAR/AIY/RIM | paralog-separate (pore); aux excluded | nca-2 alone (nca-1 below T2; unc-77 excluded) |

---

## 5 · Translation efficiency (E_translation) v1 assumption

Per Decision 3 (pre-authorized 2026-05-12): **E_translation = 1.0
uniformly** for all channels in Layer 1 v1.

Epistemic label per §2.8: "free parameter with sensitivity sweep (initially
uniform)." Documented v2 refinement candidate if Phase 5 surfaces
systematic per-family residuals.

---

## 6 · Per-cell surface area verification (from Layer 1 §7.1 substrate)

Surface areas from `scripts/brain/wave2/ion_dynamics.py`
(`NICOLETTI_CAPACITANCE_pF`) confirmed against Wave 2 cell builders:

| cell | Cm (pF) | specific Cm (μF/cm²) | A_cell (cm²) | A_cell (μm²) |
|---|---:|---:|---:|---:|
| AVAL | 9.66 | 0.859551 | 1.124e-5 | 1124 |
| AVAR | 8.43 | 0.751761 | 1.122e-5 | 1122 |
| AIY  | 1.05 | 1.6      | 6.59e-7  |  65.9 |
| RIM  | 1.55 | 1.5      | 1.03e-6  | 103 |

(Specific Cm values are cell-specific from Nicoletti's neuromorpho EM
reconstructions, not the textbook 1 μF/cm². Methodology document §1.3
note about "1 μF/cm² standard" should be corrected — this is a Phase 3
documentation fix flagged for Phase 7 cleanup.)

Surface area does NOT enter the Path B intensive formula (per Phase 1
pushback Item 1 resolution). A_cell is documented here for completeness
and downstream cell-builder consistency only.

---

## 7 · Local CeNGEN panel coverage gap

Audit of `public/data/cengen-panel.json` (the local panel used by Layer 1
dashboard + earlier analysis):

- **Channel genes IN local panel (6/15):** egl-19, cca-1, unc-2, nca-2,
  unc-77, unc-80
- **Channel genes NOT in local panel (9/15):** shl-1, kqt-1, kqt-3,
  egl-2, unc-103, irk-1, irk-2, irk-3, nca-1

This is consistent with `docs/substrate_redesign_roadmap.md` cross-cutting
track "CeNGEN panel extension" — the local panel needs extension for
Layer 1 v2 broadcast to remaining cells AND for downstream Path 2 work
(receptors, transporters, channels not in current Wave 2 set).

For Phase 3 deliverable, all required TPMs were pulled directly from the
full CeNGEN T2 CSV; local panel limitation doesn't block Phase 3 but is
flagged for the cross-cutting track. Local panel extension is a separate
bounded work block (estimated <1 work block).

---

## 8 · Phase 3 acceptance criteria status

Per methodology / roadmap:

- [x] Every (channel, cell) combination has TPM value documented (9
      channels × 3 cells = 27 derivation entries)
- [x] Per-cell surface area verified against §7.1 substrate
- [x] Translation efficiency assumption (E_translation = 1.0) documented
- [x] TPM gaps explicitly identified (nca-1 below threshold; AIY SHL-1=0)
- [x] KQT-3 heteromer hypothesis resolved (paralog-separate confirmed)
- [x] IRK paralog aggregation rule applied (sum-of-paralogs at channel module)

**Phase 3 SHIPPED.** Ready for Phase 4 (`C_global` calibration).

---

## 9 · Files of record

- This document: `docs/channel_tpm_inventory.md`
- Methodology reference: `docs/channel_parameter_derivation_methodology.md` §2.4
- γ inventory (Phase 2): `docs/channel_gamma_inventory.md`
- Progress tracking: `scripts/brain/wave2/artifacts/path2_progress_summary.md`
- Phase 3 checkpoint: `scripts/brain/wave2/artifacts/path2_phase3_checkpoint.json`
- CeNGEN T2 source: `cengen.org/storage/021821_medium_threshold2.csv`
  (cached at `/tmp/cengen/` during this work block)
