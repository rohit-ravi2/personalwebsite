# Path 2 Phase 1 — Pre-flight pushback (2026-05-12)

**Status:** Phase 1 paused before methodology doc write. Four items surface
ambiguity not resolved by Rohit's five pre-authorized decisions. Per Phase 1
failure-mode instructions, surfacing to this file and standing by for
direction.

Two items (#1, #2) are **blockers** for Section 2 of the methodology doc —
they affect every downstream phase. Two items (#3, #4) are **flags** that
I could proceed past with caveats if you prefer.

---

## Item 1 (BLOCKER) — Unit pipeline in Section 2 formula

The user-specified formula is:

```
gbar[channel][cell] = γ[channel] × N[channel][cell]
N[channel][cell] = TPM × E_translation × A[cell] × C_global
```

With A in the formula, N has units of **total channels per cell** (extensive),
and `gbar = γ × N` produces **total cell conductance** (e.g., pS, not S/cm²).

But the Wave 2 cell builders use **intensive** conductance density
(`gbar_egl19_Scm2`, S/cm²). The cell builders divide by area implicitly via
the membrane equation `I = i_mAcm2 × surf_cm2`.

This creates a conversion question that affects the calibration math:

**Path A** — gbar is extensive (Siemens per cell). The methodology produces
per-cell gbar values, and a final step converts to S/cm² for the cell
builder. Calibration in Phase 4 uses extensive gbar = γ × N_Nicoletti × …
where N_Nicoletti is back-derived from intensive Nicoletti gbar × Nicoletti
surface area.

**Path B** — gbar is intensive (S/cm², matches cell builder). Formula
should not contain A (it cancels). The "biological" intuition is then:
density of channels per cm² ∝ TPM × C_global; `γ × density × ΔV` gives
current density directly. Calibration uses intensive gbar throughout.

Both are mathematically defensible. **Path B is cleaner** (no extensive/
intensive conversion at the methodology-doc boundary; matches existing cell
builder convention). But the user's spec explicitly puts A in N, suggesting
Path A.

Recommended Path B with A removed from formula, OR Path A with explicit
"extensive gbar → intensive S/cm²" conversion documented in Section 2 and
applied in Phase 6 cell-builder integration.

**Need direction on which path.**

## Item 2 (BLOCKER) — Multi-gene channel TPM aggregation

Some channel modules in Wave 2 cell builders map to **multiple genes**:

- **NCA channel** ↔ pore-forming nca-1 + nca-2 + unc-77, auxiliary unc-80
- **IRK channel** ↔ irk-1 + irk-2 + irk-3 (the three Kir-family genes; Wave 2 uses one channel parameterization)
- **SHL-1** is single-gene (shl-1) — clean
- **EGL-19** is single-gene (egl-19) — clean

For multi-gene channels, the TPM for "the channel" must aggregate gene TPMs.
Three options:

**(a) Sum TPMs** across the gene family. Treats all genes as interchangeable
pore-forming subunits. Simple, defensible if expression patterns are
correlated. Risk: double-counts in cells expressing multiple homologs.

**(b) Max TPM** across genes. Conservative — picks the dominant pore-former.
Loses information about heteromeric channels.

**(c) Weighted sum** with per-gene stoichiometric weights (e.g., NCA pore
subunits weighted 1.0, auxiliary subunits 0.0). Most biology-faithful but
requires explicit subunit-weighting decisions per channel.

Recommended **(a) sum**, with documentation that this is a v1 approximation
and (c) is v2 refinement if validation surfaces issues. But this is a
substantive methodology choice that affects derived gbar for any
multi-gene channel.

**Need direction on aggregation rule.**

## Item 3 (FLAG) — C_global "biophysically plausible range"

The Phase 4 hard-stop condition says:

> Calibration produces C_global outside biophysically plausible range
> (orders of magnitude off from naive estimates)

The user's Phase 4 workflow gives "plausible range probably 1e-6 to 1e3
depending on γ units (pS or S)." A back-of-envelope estimate using:

- AVAL EGL-19 intensive gbar = 9.29e-6 S/cm² (Nicoletti)
- AVAL surface area = 1.124e-5 cm²
- Extensive G_total = 1.04e-10 S ≈ 104 pS
- γ_EGL19 ≈ 4–20 pS (Cav1 family range)
- Implied total channels per cell ≈ 5–25
- AVAL EGL-19 TPM = 89.5
- C_global = N_total / (TPM × E_translation × A) = ~25 / (89.5 × 1 × 1.124e-5)
  ≈ 24,800 channels per (cm² · TPM)

That's ~10^4, outside the 1e-6 to 1e3 range cited. I think the cited range
is an order-of-magnitude rough estimate, and the actual biophysically
plausible range is in the 10^3 to 10^5 channels/(cm²·TPM) regime. **Worth
either updating the hard-stop range to ~1e2 to 1e6, or specifying the
plausibility criterion from first principles** (max channel density × min
TPM × min surface area as lower bound; min channel density × max TPM × max
surface area as upper bound).

I can proceed past this with caveats — I'll document the computed C_global
with explicit unit accounting and biophysical sanity-check argument, and
hard-stop only if the value is clearly nonsensical (e.g., negative,
infinity, predicts <1 channel/cell). Confirm this is acceptable, or
specify a tighter range.

## Item 4 (FLAG) — Mellem 2008 vs Liu 2020 reference

The user's "Key reference files" lists:

> Mellem 2008 (for AVA voltage range empirical anchor)

But per the Wave 2 Mellem investigation
(`scripts/brain/wave2/artifacts/mellem_investigation_pushback.md`), Mellem
2008 characterizes **RMD plateau dynamics, NOT AVA**. The primary-source
quote from Mellem 2008 is "we never observed action potentials in AVA."
The correct AVA experimental anchor is **Liu/Chen/Wang 2020 Nat Commun
11:5076** (DOI 10.1038/s41467-020-18893-9), which is Nicoletti 2024's
reference [29] for raw AVA voltage-clamp + current-clamp recordings. This
is the C-45 catalog entry, classified Direct (verified 2026-05-02).

This is likely a session-history artifact in the prompt. **I'll proceed
with Liu 2020 as the AVA voltage-range anchor where it matters
(particularly Phase 5 I-V validation against AVA voltage-clamp data),
and document this in the methodology doc footnotes.** Confirm or override.

---

## Summary of asks

- **Item 1**: which unit pipeline (Path A extensive with conversion, or
  Path B intensive throughout)?
- **Item 2**: which multi-gene TPM aggregation (sum / max / weighted sum)?
- **Item 3**: proceed past with my own sanity-check, or specify tighter
  hard-stop range?
- **Item 4**: confirm Liu 2020 replaces Mellem 2008 as AVA voltage anchor?

I won't write the methodology document until Items 1 and 2 are resolved.
Items 3 and 4 I can proceed past with documented caveats if you'd rather
not loop. Standing by.

---

## Files of record

- This document: `docs/channel_parameter_derivation_methodology_pushback.md`
- Phase 1 spec: in the §7.3.5 Path 2 prompt
- Methodology pre-authorized decisions: 5 (calibration architecture, γ
  labels, E_translation, reference, refinement triggers) — none cover
  Items 1–4 above
