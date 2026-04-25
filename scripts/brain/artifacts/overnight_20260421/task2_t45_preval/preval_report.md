# Task 2 — T4-5 candidate pre-validation smoke test

Generated: 2026-04-21 12:22:30

Baseline-gate check that the simulator runs cleanly for each 
candidate's target scenario. NOT a phenotype test. The actual 
peptide-injection happens during T4-5 implementation proper.

| candidate | scenario | status | frames | mean rate (Hz) | issues | wall |
|---|---|---|---|---|---|---|
| **FLP-13** | osmotic_shock | FLAG | 600 | 51.26 | runaway firing (mean rate 51.3 Hz) | 86.7s |
| **FLP-18** | touch | PASS | 600 | 16.52 | none | 88.3s |
| **FLP-21** | spontaneous | PASS | 600 | 17.96 | none | 77.5s |
| **NLP-40** | spontaneous | PASS | 600 | 17.96 | none | 77.5s |
| **DAF-28** | food | PASS | 600 | 24.5 | none | 101.8s |

Total wall time: 7.2 min

## Verdict

- **4/5** candidates pass smoke gate
- Flagged candidates need inspection before T4-5 start.