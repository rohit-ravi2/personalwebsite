export type Verdict = "ruled-out" | "kept" | "partial";

export type InvestigationEntry = {
  n: number;
  /** Optional date label for timeline pages, e.g. "2026-09-21" or "Sep 2026". */
  date?: string;
  q: string;
  verdict: Verdict;
  tag: string;
  tried: string;
  found: string;
  ruled: string;
  why: string;
  next: string;
};
