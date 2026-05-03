# V6 cross-phylum data tables (Mus musculus, generic LIF substrate)

V6 M1 deliverable. Four CSVs paralleling worm V3 + fly V4 structure with
mouse-specific anchors and a generic random-graph substrate.

## Files

- `mouse_anesthetic_perturbation_table.csv` — same Hill curves as worm/fly tables; the EC50 anchors are mammalian electrophys to begin with (Mihic 1997, Patel 1999, Forman 1996, Hanley 2002, Stewart 2000, Lu 2007). GluCl class DROPPED (no mammalian ortholog). Adds `desflurane` rows (mammalian-only volatile).
- `mouse_immobilization_anchors.csv` — gold-standard MAC + LRR EC50 anchors. Halothane MAC ~350 µM aqueous (essentially identical to worm Crowder 1996 and fly van Swinderen 1999 — striking conservation across phyla).
- `mouse_directional_mutants.csv` — 14 directional anchors with PMIDs. RESISTANT-rich (10 RESISTANT vs 4 HYPER) — opposite direction skew from worm V3.
- `mouse_mutant_baseline_perturbations.csv` — LIF entry points per mutant. V1 approximation: Type A mutants (NDUFS4 cKO, Stx1A, Vamp2) use baseline shifts cleanly; Type B receptor-insensitivity mutants (β3(N265M), α1(H101R), K2P KOs, GIRK KO) approximated via wsyn_global_factor > 1 — predicts RESIST direction but not per-anesthetic specificity. V2 schema extension would add per-class engagement factors per mutant.

## Substrate decision (per V5 M2 finding)

V6 uses a **generic LIF random graph** at ~3,000 neurons, not a mammalian connectome. Per V5 M2, connectome topology beyond cell-type aggregates is not load-bearing in the architecture (fly result transfers to random graphs, worm needs only cell-type-block-preserving connectivity). Building a mammalian connectome of comparable scale to Cook 2019 / Winding 2023 was infeasible; per V5 M2 it's also unnecessary.

**Documented caveat**: V6 tests only the LRR / immobilization phenotype — the most "invertebrate-like" component of mammalian anesthesia. Higher-order mammalian features (cortical EEG burst suppression, NREM-like slow oscillations, gamma suppression, consciousness disruption) are NOT testable in this architecture and must NOT be claimed by V6.

## Cross-phylum transfer logic

What changes (mouse vs fly):
1. Substrate: random graph instead of Winding 2023
2. NT identity: mammalian E:I ratio ~80:20 (vs fly heuristic that gave 95:5)
3. GluCl class dropped from perturbation table
4. Drops desflurane and ether anchors added
5. Mutant set RESISTANT-rich (mammalian receptor-binding-site genetics)

What stays the same:
1. Mihic / Patel / Forman / Hanley / Stewart / Lu EC50 anchors (mammalian source)
2. Hill curve framework
3. Network-state metric (quiescent fraction on locomotor command set)
4. Single-anchor calibration on halothane MAC
5. ~340-350 µM halothane MAC across all three organisms

## Cross-phylum MAC conservation

The single most striking pre-flight finding:

```
worm halothane EC50  (Crowder 1996)        ~340 µM aqueous
fly halothane MAC    (van Swinderen 1999)  ~340 µM aqueous
mouse halothane MAC  (Sonner 1999)         ~350 µM aqueous
```

Three organisms across two phyla, same effective concentration of halothane at immobilization. Either same conserved targets are integrating to produce the same threshold (the conserved-substrate hypothesis), or this is an extraordinary coincidence. The V6 result tests whether the SAME perturbation table can recover all three with a single calibration parameter per organism.

## V1 limitations (honest)

1. **No mammalian connectome** — generic random graph; the architecture's overdetermination per V5 M2 makes this acceptable but the substrate is biologically unmotivated.
2. **Receptor-insensitivity mutants approximated** — β3(N265M), TREK-1 KO, etc. modeled via baseline shift rather than per-class engagement. Predicts direction but not per-anesthetic specificity.
3. **No cortical-thalamic dynamics** — mammalian anesthesia features beyond LRR are out of scope.
4. **No NMDA class** — ketamine is anchored only to nAChR + K2P engagement (the same as in worm/fly tables); a future addition would let ketamine be more meaningfully tested in V6.

These limitations are documented; V6 results will be reported with these constraints explicit.
