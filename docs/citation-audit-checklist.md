# Citation audit checklist

*Triggered by Phase 0 finding: "Gao & Hobert 2020" referenced in 6+
locations across the project could not be verified via web search.
The replacement anchor for AVA plateau data is Mellem et al. 2008
(Nat Neurosci, PMC2697921, DOI:10.1038/nn.2131).*

**Scope:** every biological citation that drives a calibration target,
parameter value, or architectural decision. Citations that are purely
background / context-setting don't need this level of scrutiny, but
load-bearing ones do.

**Verification standard:** DOI or PubMed Central ID (PMCID). If
neither can be confirmed via https://doi.org or https://pubmed.ncbi.nlm.nih.gov,
flag for replacement or removal.

## Files to audit

### Load-bearing (parameter/calibration citations)

- [ ] `scripts/brain/compartmental_neurons.py` — `COMPARTMENTAL_ROSTER`
      notes field. Cites Chalfie 1985, Wicks 1996, Wang 2020, Kawano 2011,
      Faumont 2011, Chalasani 2007, Macosko 2009, Van Buskirk 2007,
      Turek 2016, Li 2006. Verify each; confirm which are DOI-accessible.
- [ ] `scripts/brain/lif_brain.py` — sign override comments cite
      Chalasani 2007, Piggott 2011, Hart 2006, Ortiz 2009, Shinkai 2011,
      Guo 2015, Beverly 2011, Bargmann 2006, Li 2014. Verify.
- [ ] `scripts/brain/modulation_layer.py` + `build_modulator_tables.py` —
      receptor sign references (Brockie 2001, Maricq 1995, Ranganathan 2000,
      Tsalik 2003, Chase 2004, Suo 2003, Sanyal 2004, Alkema 2005,
      Donnelly 2013, Roeder 2005, Turek 2016, Bhattacharya 2014,
      Oranth 2018, Janssen 2008). Verify.
- [ ] `scripts/brain/sensory_transduction.py` — cascade citations
      (Bargmann 1993, Suzuki 2008, Thiele 2009, Chalasani 2007,
      Zaslaver 2015, Colbert 1997, Hilliard 2005, Kahn-Kirby 2004,
      Komatsu 1996, Garrity 2010, Chalfie & Sulston 1981, O'Hagan 2005).
      Verify. These drive T2-#4 calibration targets.
- [ ] `scripts/brain/environment.py` — aerotaxis citations (Gray 2004,
      Cheung 2005, Zimmer 2009, Hallem & Sternberg 2008, Laurent 2015). Verify.
- [ ] `scripts/brain/build_motor_innervation.py` — White 1986, Pereira 2015,
      Cook 2019. Verify.
- [ ] `scripts/brain/build_modulator_tables_v4.py` — **new INS refs pending
      verification**: Pierce 2001, Cornils 2011, Chen 2013, Tomioka 2006,
      Li 2003. The INS-22 reference is already flagged as unverified
      in the file's meta. Verify all.

### Narrative documentation

- [ ] `docs/claude-chat-context.md` — survey of phase history and
      biological grounding. Verify all paper citations.
- [ ] `docs/project-history.md` — same.
- [ ] `docs/new-session-primer.md` — references "Gao & Hobert 2020" in
      primer facts (point #2 of "Tier 2 item #2"). **Replace with
      Mellem 2008.**
- [ ] `docs/tier2-4-execution-plan.md` — references "Gao & Hobert 2020".
      **Replace with Mellem 2008.**
- [ ] `docs/current-state-summary.md` — already flags the citation
      replacement; audit the rest.

### Artifacts (derived, lower priority but included for completeness)

- [ ] `scripts/brain/artifacts/ensemble_report.md` — EXPECTED phenotype
      signs cite Turek 2016, Flavell 2013, Chalfie 1985, Alkema 2005 /
      Gordus 2015, Chase 2004. Verify.
- [ ] `scripts/brain/artifacts/perturbation_report.md` — same.
- [ ] `scripts/brain/artifacts/t0_run_report.md` — already under active
      review; verify.
- [ ] `scripts/brain/artifacts/v33_audit_report.md` — same.
- [ ] `scripts/brain/references/*/meta.json` — each meta.json has a
      citation; confirmed ones are the T2-#4 targets (Thiele 2009,
      Chalasani 2007, Hilliard 2005, Clark 2006, O'Hagan 2005) plus
      Mellem 2008 (already flagged as citation replacement in
      `gao_hobert_2020_ava/meta.json`).

## Verification workflow per citation

1. Search author + year + journal in PubMed (`https://pubmed.ncbi.nlm.nih.gov/?term=...`).
2. If found: record PMID or DOI. Compare title/abstract to the claim
   made in our file. Does the paper actually support that claim?
3. If not found: try Google Scholar as fallback.
4. If still not found: **flag for replacement or removal**.
5. If found but the claim doesn't match: **flag for correction**.

## Known unverified

| citation | appears in | replacement candidate |
|---|---|---|
| Gao & Hobert 2020 (AVA plateau) | compartmental_neurons.py notes, docs/new-session-primer.md, docs/tier2-4-execution-plan.md, `references/gao_hobert_2020_ava/meta.json` | **Mellem et al. 2008** (PMC2697921, DOI:10.1038/nn.2131) |

Add rows here as audit proceeds.

## Estimated effort

~2 hours human time. Parallelizable across files. Worth more than it
sounds because phantom citations compound — each re-use adds
credibility by repetition. One two-hour pass prevents an embarrassing
finding at paper review time.
