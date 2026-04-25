# Track C1 — Peptide receptor pharmacology audit

Generated: 2026-04-22 15:22:24

Receptor-level functional annotation for 9 existing modulators + 
5 T4-5 candidates. Sources: literature consensus (cited inline), 
Ripoll-Sánchez 2023 where accessible. Entries marked UNVERIFIED 
are annotations I couldn't confirm from primary literature in the 
available fetch window — flag for manual review.

## Access notes

- Ripoll-Sánchez 2023 supplementary tables: NOT FETCHED directly.
  Cited in source column where annotation matches their published 
  receptor-ligand interaction table (last night's Task 8 web fetch 
  confirmed paper existence and FLP-18 → NPR-5 specifically).
- WormBase: not queried gene-by-gene; annotations compiled from 
  primary-literature citations.

## Per-modulator receptor pharmacology

| modulator | receptor | type | sign | G-protein | source |
|---|---|---|---|---|---|
| FLP-11 | npr-1 | metabotropic | inhibitory | Gi | de Bono 1998 + Rogers 2003 |
| FLP-11 | npr-22 | metabotropic | inhibitory | Gi | literature_consensus |
| FLP-11 | dmsr-1 | metabotropic | inhibitory | Gi/Go | Turek 2016 + Beets 2023 |
| FLP-11 | dmsr-7 | metabotropic | inhibitory | Gi/Go (inferred) | literature_consensus UNVERIFIED |
| FLP-11 | npr-11 | metabotropic | inhibitory | Gi | literature_consensus |
| FLP-1 | npr-4 | metabotropic | inhibitory | Gi | Cohen 2009 UNVERIFIED for specific |
| FLP-1 | npr-5 | metabotropic | inhibitory | Gi | Cohen 2009 |
| FLP-1 | npr-11 | metabotropic | inhibitory | Gi | Bhattacharya 2014 |
| FLP-2 | npr-30 | metabotropic | inhibitory | Gi (inferred) | Oranth 2018 UNVERIFIED specific G-protein |
| FLP-2 | frpr-18 | metabotropic | inhibitory | Gi | Oranth 2018 |
| NLP-12 | ckr-1 | metabotropic | excitatory | Gq | Hu 2011 + Janssen 2008 |
| NLP-12 | ckr-2 | metabotropic | excitatory | Gq | Hu 2011 + Janssen 2008 |
| PDF-1 | pdfr-1 | metabotropic | excitatory | Gs | Janssen 2008 + Flavell 2013 |
| 5HT | mod-1 | ionotropic | inhibitory | N/A (Cl- channel) | Ranganathan 2000 |
| 5HT | ser-1 | metabotropic | excitatory | Gq | Tsalik 2003 |
| 5HT | ser-4 | metabotropic | inhibitory | Gi | Tsalik 2003 |
| 5HT | ser-5 | metabotropic | excitatory | Gq | Hamdan 1999 |
| 5HT | ser-6 | metabotropic | excitatory | Gs | literature_consensus |
| 5HT | ser-7 | metabotropic | excitatory | Gs | Hobson 2006 |
| DA | dop-1 | metabotropic | excitatory | Gq | Sanyal 2004 |
| DA | dop-2 | metabotropic | inhibitory | Gi | Chase 2004 |
| DA | dop-3 | metabotropic | inhibitory | Gi | Chase 2004 |
| DA | dop-4 | metabotropic | inhibitory | Gi (inferred) | Sugiura 2005 UNVERIFIED |
| TA | tyra-2 | metabotropic | excitatory | Gq | Alkema 2005 |
| TA | tyra-3 | metabotropic | inhibitory | Gi | Alkema 2005 |
| TA | ser-2 | metabotropic | inhibitory | Gi | Donnelly 2013 |
| TA | lgc-55 | ionotropic | excitatory | N/A (cation channel) | Ringstad 2009 |
| OA | octr-1 | metabotropic | inhibitory | Gi | Wragg 2007 |
| OA | ser-3 | metabotropic | excitatory | Gq | Mills 2012 |
| OA | ser-6 | metabotropic | excitatory | Gs | Mills 2012 |
| FLP-13 | dmsr-1 | metabotropic | inhibitory | Gi/Go | Nath 2016 + Beets 2023 |
| FLP-13 | dmsr-2 | metabotropic | inhibitory | Gi/Go | Nath 2016 |
| FLP-18 | npr-1 | metabotropic | inhibitory | Gi | Cohen 2009 (partial) |
| FLP-18 | npr-4 | metabotropic | inhibitory | Gi | Cohen 2009 |
| FLP-18 | npr-5 | metabotropic | inhibitory | Gi | Cohen 2009 + Ripoll-Sánchez 2023 |
| FLP-21 | npr-1 | metabotropic | inhibitory | Gi | Rogers 2003 |
| NLP-40 | aex-2 | metabotropic | excitatory | Gs | Wang 2013 + Mahoney 2008 |
| DAF-28 | daf-2 | tyrosine_kinase_rcpt | variable (DAF-16/AKT pathway) | N/A (TK receptor) | Li 2003 + Pierce 2001 |

## Effect-type summary per modulator

| modulator | n_receptors | sign mix | type mix |
|---|---|---|---|
| **FLP-11** | 5 | inhibitory | metabotropic |
| **FLP-1** | 3 | inhibitory | metabotropic |
| **FLP-2** | 2 | inhibitory | metabotropic |
| **NLP-12** | 2 | excitatory | metabotropic |
| **PDF-1** | 1 | excitatory | metabotropic |
| **5HT** | 6 | excitatory, inhibitory | ionotropic, metabotropic |
| **DA** | 4 | excitatory, inhibitory | metabotropic |
| **TA** | 4 | excitatory, inhibitory | ionotropic, metabotropic |
| **OA** | 3 | excitatory, inhibitory | metabotropic |
| **FLP-13** | 2 | inhibitory | metabotropic |
| **FLP-18** | 3 | inhibitory | metabotropic |
| **FLP-21** | 1 | inhibitory | metabotropic |
| **NLP-40** | 1 | excitatory | metabotropic |
| **DAF-28** | 1 | variable (DAF-16/AKT pathway) | tyrosine_kinase_rcpt |

## Flagged UNVERIFIED entries (4)

- **FLP-11 → dmsr-7** (literature_consensus UNVERIFIED)
- **FLP-1 → npr-4** (Cohen 2009 UNVERIFIED for specific)
- **FLP-2 → npr-30** (Oranth 2018 UNVERIFIED specific G-protein)
- **DA → dop-4** (Sugiura 2005 UNVERIFIED)
