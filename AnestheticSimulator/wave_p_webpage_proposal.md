# Wave P web page proposal — `/projects/anesthesia-pipeline`

**Date:** 2026-04-28 (Work Block 2, CP1 of webpage build)
**Status:** PAUSE FOR REVIEW. No code or assets written; no Astro routes added.
**Reference page:** `src/content/projects/c-elegans-multimodal.mdx` (100 lines MDX) + `src/components/react/CelegansDashboard.tsx` (4163 lines TSX).

---

## Pre-flight findings (all clean)

1. **Astro project structure verified.** `src/content/projects/<slug>.mdx` is the project-page pattern (12 existing). `src/components/react/<Name>.tsx` is the React component pattern. Astro 4.16 + React 18 + Tailwind 3.4. URL pattern `/projects/<slug>` resolves via `src/pages/projects/[...slug].astro`.

2. **Wave P data is web-shippable.** `wave2_overlay_v2.json` is 86 KB; calibration tables are small CSVs; case studies are MDs. Total budget for `public/data/anesthesia/` JSONs target ~500 KB — well within static-site limits. Large artifacts (PDB structures, Vina poses) are gitignored and stay backend-only.

3. **SMILES → prediction interactivity model.** No RDKit-js or chemistry library installed; adding one would bloat the bundle. **Recommendation:** the playground operates over a curated panel of 14 cached compounds (6 anesthetics + 8 negative controls) with full pre-rendered binding profiles. For custom SMILES input, the page accepts the string, validates basic syntax client-side (no chemistry validation), shows a clear "Custom compound predictions require a backend pipeline run; this page shows the 14 cached compounds" message, and offers a "find nearest cached compound by name search" affordance. Honest about what the static deploy can and cannot do.

4. **URL `/projects/anesthesia-pipeline` is consistent** with existing slug conventions (kebab-case, descriptive).

5. **Visual standard is high.** CelegansDashboard at 4163 lines covers 5 scenarios × 300-neuron brain × 9 modulators × FSM timeline × interactive arena. The new page does not need to match that complexity but should match the visual quality (Tailwind-styled, rich SVG/Canvas viz, honest inline framing). Target: a well-polished single-component playground at ~800-1500 lines.

---

## Page structure (proposed)

```
/projects/anesthesia-pipeline
│
├── Hero / thesis
│     "From molecular binding to network perturbation —
│      a digital pharmacology pipeline for general anesthesia,
│      calibrated against published electrophysiology and
│      bounded by honest scope labels."
│
├── The story in one paragraph
│     What Wave P does: docks 6 anesthetics + 8 negative controls
│     against 30 C. elegans Tier-1 anesthetic targets; predicts
│     binding profile + kinetic shifts + network perturbation.
│     Calibrated against mammalian-homolog functional EC50; rigor-
│     tightened with allosteric correction; documented boundary
│     where binding alone is insufficient (Eger non-immobilizer).
│
├── Pipeline architecture diagram
│     Phase A (AlphaFold) → B (Vina) → C (occupancy) → D (kinetic
│     shifts) → E/F (Markov synapse + ATP) → G (network perturbation)
│     With honest scope labels per phase: SHIPPED / SCAFFOLD / IN
│     PROGRESS / DEFERRED.
│
├── Interactive: anesthesia pipeline playground
│   ├── Compound selector (14 cached + custom SMILES with caveat)
│   ├── Predicted binding profile heatmap (anesthetic × 30 targets)
│   ├── Per-target details: predicted Kd, occupancy, kinetic shift,
│   │   four-category verdict label (VERIFIED / HOMOLOG / AWAITING /
│   │   UNCALIBRATED)
│   ├── Mechanism class engagement summary (8 classes)
│   └── Predicted network effect (Phase G dose-response)
│
├── Calibration story
│     The CP1-CP8 rigor pass — what got downgraded, what survived.
│     Phase F structural parameter-lock; Kd-vs-EC50 conflation;
│     allosteric correction f_allo = 2.50× (76% → 94% within 10×).
│     Four-category verdict structure replacing 5/5 PASS.
│     Numbers + 1-2 small SVG charts.
│
├── Boundary findings
│     Eger non-immobilizer puzzle — CP3 cis/trans-DCE FAIL +
│     CP7 hexafluoroethane FAIL. Documented as boundary, not bug.
│     Honest framing: binding pipeline is lipophilic-pocket-fit
│     detector; anesthetic-vs-non-immobilizer discrimination is
│     a network-level question for Phase G to attempt.
│
├── Methodology rigor framing
│     Pre-flight pushback as systematic methodology (>37 citation
│     issues + 1 parameter-lock + 1 direction-inversion caught).
│     Link to case studies for readers who want depth.
│
├── Status / what's shipped vs in-progress
│     Phase A-D: SHIPPED. Phase E/F: SCAFFOLDED + sensitivity-
│     verified at CP2. Phase G: architecture + perturbation
│     manager + dose-response demo SHIPPED; LIFBrain integration
│     PENDING. Phase H/I/J: SCAFFOLDED.
│
└── Sources & attribution (collapsible fold)
     Primary sources used in calibration; Rohit's role; computational
     scope ($0 external spend, RTX 4060 Ti). Honest note about
     citation corrections during rigor pass.
```

Length target: ~150-250 MDX lines, comparable to c-elegans-multimodal.mdx (100 lines).

---

## Data summary JSONs to generate

Target dir: `public/data/anesthesia/` (new). All export scripts ship in `AnestheticSimulator/src/web_export_*.py` and reproduce JSONs from artifacts.

### 1. `binding_profile.json` (~30-50 KB)

Per-(anesthetic, target) full record. Consumed by the heatmap + per-target detail.

```json
{
  "anesthetics": ["halothane", "isoflurane", "sevoflurane", "propofol",
                  "etomidate", "ketamine"],
  "targets": [
    {"gene": "ACR-16", "uniprot": "P48180", "mechanism_class": "nachr_antagonism",
     "human_homolog": "CHRNA4", "expression_class": "command_interneuron"},
    ...
  ],
  "predictions": {
    "halothane": {
      "ACR-16": {
        "vina_dG_kcal_per_mol": -4.5,
        "predicted_Kd_uM_v1": 500.98,
        "predicted_Kd_uM_v2_corrected": 199.99,
        "occupancy_at_1xEC50_v2": 0.987,
        "kinetic_shift_param": "n_Ca_delta",
        "kinetic_shift_value": -1.4535,
        "verdict_category": "VERIFIED",
        "verdict_confidence": "HIGH",
        "calibration_log_err_post": 0.001,
        "comment": "TREK-1 mammalian homolog; canonical halothane K2P calibration"
      },
      ...
    },
    ...
  }
}
```

### 2. `negative_controls.json` (~10-15 KB)

Predictions for the 8 negative-control ligands (benzene, methanol, n-pentane, cyclohexane, dimethyl ether, hexafluoroethane, cis-1,2-DCE, trans-1,2-DCE) on the same 30 targets. Powers the negative-control comparison panel.

### 3. `calibration_summary.json` (~5 KB)

Roll-up of CP1-CP8 verdict structure. Per-anchor four-category label, per-class metrics, allosteric correction factor, the 7+1+5+3+2 verdict counts. Consumed by the calibration story panel.

### 4. `dose_response.json` (~2 KB)

Phase G halothane dose-response curve from `phase_g_halothane_dose_response.csv`. Consumed by network-effect chart.

### 5. `pipeline_meta.json` (~1 KB)

High-level pipeline state: which phases are shipped, scaffolded, deferred. Driven by the pipeline architecture diagram. Single source of truth for the status banners across the page.

### 6. `case_studies.json` (~3 KB)

Title + one-paragraph summary + word count + repo link for each of the 5 methodology case studies. Powers the "methodology rigor" sidebar with links into the GitHub repo (since the case studies are in `artifacts/methodology_paper/`).

**Total ship size:** ~50-80 KB — below the c-elegans-multimodal data dir (which is several MB across multiple scenarios).

---

## Component architecture (proposed)

```
src/components/react/AnesthesiaPipeline.tsx     [main orchestrator, ~600-900 lines]
│
├── PipelineDiagram           [SVG: A → B → C → D → E/F → G with status bands]
├── CompoundSelector          [Dropdown over 14 cached + SMILES input]
├── BindingProfileHeatmap     [SVG: anesthetic × 30 targets, color-coded by occupancy]
├── TargetDetailPanel         [On heatmap-cell click: predicted Kd, kinetic shift,
│                              verdict category, calibration log_err, anchor PMID]
├── MechanismSummary          [8-class engagement bar chart for selected anesthetic]
├── DoseResponseChart         [SVG line chart: Phase G dose-response with caveat]
├── CalibrationStoryPanel     [Per-class metrics table + allosteric correction
│                              before/after numbers + 7+1+5+3+2 verdict tiles]
├── BoundaryFindingsCard      [CP3 + CP7 FAIL summaries with honest framing]
└── SourcesFold               [Collapsible: primary sources, citation corrections,
                                computational scope, Rohit's role]
```

All Tailwind-styled. Native SVG/Canvas for charts (no Recharts dep added). Lucide-react icons. Existing project uses React 18 hooks (useState, useMemo, useCallback) + framer-motion is NOT installed; just CSS transitions or @radix-ui's animation primitives.

Single-file `AnesthesiaPipeline.tsx` for simplicity (matching CelegansDashboard's mono-file pattern); sub-components as internal `function X() {}` definitions, lazy-load JSON data via fetch on mount.

---

## Playground interactivity model

### Default mode (cached compounds)

User selects one of 14 compounds from a dropdown. The page renders:

- **Binding profile heatmap:** 1 row × 30 columns. Each cell shows occupancy at 1× clinical EC50 (color-mapped). Click → TargetDetailPanel opens.
- **Mechanism class engagement:** 8 bars showing aggregate occupancy per mechanism class (gaba_potentiation, k2p_potentiation, complex_i_block, etc.). Reveals the multi-target binding profile.
- **Dose-response card:** Phase G's halothane dose-response (other anesthetics show "Phase G calibration in progress; halothane is the canonical example").
- **Verdict status:** the four-category label per target with a confidence chip (HIGH / MEDIUM / LOW / —).

### Custom SMILES mode

User pastes a SMILES string. The page:

- Validates basic SMILES syntax client-side (regex-level, not chemistry-level — atoms, bonds, parens, brackets balance).
- **Does NOT show a fake binding profile.** Instead shows a clear card:
  > "Custom compound predictions require a backend pipeline run (Phase A AlphaFold + Phase B Vina + Phase D kinetic shifts; ~30 minutes on RTX 4060 Ti). This static deployment ships predictions for 14 cached compounds. Use the dropdown to explore the cached panel, or run the pipeline locally — see `AnestheticSimulator/` in the GitHub repo."
- Offers a "find nearest cached anesthetic by name match" affordance — accepts a string like "halothane" or "propofol" and suggests the matching cached compound.

This is the honest framing per the work block prompt. No fake predictions for novel compounds.

### Comparison mode (added bonus)

Toggle to compare two compounds side-by-side. Halothane vs hexafluoroethane is the prepared compelling demo (binding pipeline cannot distinguish the anesthetic from the Eger non-immobilizer — boundary finding made tangible).

---

## Honest inline framing throughout the UI

Pulled from CP6/CP8 verdict structure and CP3/CP7 boundary findings:

- Heatmap header: "Predicted occupancy at 1× clinical EC50, post-allosteric-correction (CP5 f_allo = 2.50×)."
- TargetDetailPanel verdict line: "VERIFIED · HIGH confidence · log_err 0.001 vs Patel & Honoré 1999 PMID 10321245."
- Dose-response card: "Phase G halothane dose-response on minimal LIF demo. 50%-firing-rate suppression at 0.01× clinical EC50 — 100× tighter than Crowder 1996 PMID 8873562 behavioral EC50. Behavioral threshold calibration is in-progress; current curve shape is qualitatively correct, quantitative calibration pending LIFBrain integration."
- Boundary card: "CP3: pipeline cannot distinguish cis-1,2-DCE (anesthetic) from trans-1,2-DCE (non-anesthetic). CP7: hexafluoroethane (Eger non-immobilizer) engages 30/30 targets vs cis-DCE 22/30. The binding pipeline is a lipophilic-pocket-fit detector, not an anesthetic-specificity detector. Anesthetic-specificity emerges at the network level — Phase G is the next bet."

All numbers traceable to artifacts under `AnestheticSimulator/artifacts/`.

---

## Estimated time per CP

| CP | Stage | Hours (estimate) | Notes |
|---|---|---|---|
| CP1 | This proposal | done | ~25 min elapsed |
| CP2 | Generate 6 data summary JSONs + ship export scripts | 2-3 | 6 export scripts + JSON validation |
| CP3 | React component (single-file ~800-1200 lines) | 4-6 | The bulk of the build |
| CP4 | mdx project page + integration | 1-2 | Mostly prose + component embed |
| CP5 | Cover image + projects index update | 30-60 min | Cover via existing artifact image or simple SVG |
| CP6 | Build verification + manual review (Rohit walks through) | 1 | Pause for Rohit |
| CP7 | Commit + push | 30 min | 4 logical commits per prompt |
| **Total** | | **~10-13 hours** | |

Tight against 12-hour budget; if Stage CP3 (the component build) overruns, surface as anomaly and pause for review rather than push through.

---

## Cover image

Two options:
- **Option A (recommended):** Generate a simple SVG cover from the binding profile heatmap data — visually arresting, honest representation of the pipeline output. ~30 min; in-page consistent.
- **Option B:** Use an existing artifact (e.g., a screenshot of a docking pose). Requires producing the screenshot (which depends on having PyMOL/VMD or similar — adds setup time).

Default to A unless Rohit prefers a particular image style.

---

## Out-of-scope explicit reminders

Per the prompt:
- ❌ Live in-browser Vina docking (impossible without WASM port)
- ❌ Real-time Brian2 perturbation simulation
- ❌ Phase G LIFBrain integration (not yet shipped — page reflects current state)
- ❌ Novel compound predictions (handled honestly with the "backend run required" message)
- ❌ Modifications to existing C. elegans dashboard

---

## What the user is asked to approve

**Question 1 (load-bearing):** approve the page structure (sections + ordering) shown above?

**Question 2 (load-bearing):** approve the component architecture (single-file `AnesthesiaPipeline.tsx` ~800-1200 lines, mono-component pattern matching CelegansDashboard, native SVG/Canvas viz, no new deps)?

**Question 3 (load-bearing):** approve the playground interactivity model — 14 cached compounds default + honest "backend run required" caveat for custom SMILES?

**Question 4:** approve the URL `/projects/anesthesia-pipeline` and cover-image approach (Option A: SVG from binding profile)?

**Question 5:** approve the data summary JSON structure (6 files, ~50-80 KB total under `public/data/anesthesia/`)?

If revisions wanted to any of these, flag and I'll re-propose. If approved, I'll proceed to CP2 (data exports) and CP3 (component build) without further pause until CP6 (build verification, manual review).

---

## What happens after approval

CP2: ship 6 export scripts + JSONs (~2-3 hours)
CP3: build `AnesthesiaPipeline.tsx` (~4-6 hours)
CP4: write mdx page + embed component (~1-2 hours)
CP5: cover image + projects index update (~30-60 min)
CP6: `npm run build` + `npm run preview` + manual walk-through → **PAUSE FOR REVIEW** before push
CP7: 4 logical commits + push to main (Vercel auto-deploys)

CP6 is the next pause point after CP1. Between CP1 and CP6, execution proceeds without further pauses since the prompt explicitly defers manual review to that stage.
