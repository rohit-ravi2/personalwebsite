# Track C2 — Molecular baseline for 9 modulators

For each modulator in the v3 layer, extract peak modulator 
concentration from D1 CONTROL runs. Classify as 
MECHANISM_OPERATING if mean peak > 10% of concentration cap, 
MECHANISM_INERT otherwise.

| modulator | status | mean peak | std | time-to-peak (s) | n seeds |
|---|---|---|---|---|---|
| **5HT** | MECHANISM_OPERATING | 2.955 | 0.12 | 55.82 | 3 |
| **DA** | MECHANISM_OPERATING | 2.193 | 0.046 | 47.38 | 3 |
| **FLP-1** | MECHANISM_INERT | 0.127 | 0.045 | 14.78 | 3 |
| **FLP-11** | MECHANISM_OPERATING | 10.0 | 0.0 | 11.97 | 3 |
| **FLP-2** | MECHANISM_OPERATING | 20.0 | 0.0 | 10.07 | 3 |
| **NLP-12** | MECHANISM_OPERATING | 10.0 | 0.0 | 25.23 | 3 |
| **OA** | MECHANISM_INERT | 0.362 | 0.037 | 51.42 | 3 |
| **PDF-1** | MECHANISM_OPERATING | 40.261 | 0.139 | 59.35 | 3 |
| **TA** | MECHANISM_OPERATING | 9.618 | 0.121 | 55.68 | 3 |

Operating: 7/9
Inert: 2/9

## Classification reason per modulator

- **5HT**: peak 2.95 > threshold 1.0
- **DA**: peak 2.19 > threshold 1.0
- **FLP-1**: peak 0.13 < threshold 1.0
- **FLP-11**: peak 10.00 > threshold 1.0
- **FLP-2**: peak 20.00 > threshold 1.0
- **NLP-12**: peak 10.00 > threshold 1.0
- **OA**: peak 0.36 < threshold 1.0
- **PDF-1**: peak 40.26 > threshold 1.0
- **TA**: peak 9.62 > threshold 1.0