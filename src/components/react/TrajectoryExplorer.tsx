import * as React from "react";
import { useEffect, useMemo, useState } from "react";

type Prediction = {
  rank: number;
  mutation: string;
  rerank_score: number;
  lgbm_score: number | null;
  transformer_logprob: number | null;
  is_canonical_amr: boolean;
  mut_freq_global: number;
  is_actual_observed: boolean;
};

type Transition = {
  transition_id: string;
  drug_class: string;
  drug: string;
  step_number: number;
  from_genotype: string;
  actual_next_mutation: string;
  is_canonical_target: boolean;
  lab_or_clinical: string;
  source: string;
  top10_predictions: Prediction[];
  top3_hit: boolean;
};

type Lookup = {
  drug_classes: string[];
  total_transitions: number;
  headline_metrics: Record<string, number>;
  by_drug_class: Record<string, Transition[]>;
};

const LOOKUP_URL = "/models/amr-mutation-trajectory/transitions_lookup.json";

function MutationBadge({ tag }: { tag: string }) {
  const colors: Record<string, string> = {
    canonical: "bg-emerald-500/15 text-emerald-700 border-emerald-500/40",
    acquired: "bg-blue-500/15 text-blue-700 border-blue-500/40",
    point: "bg-slate-500/15 text-slate-700 border-slate-500/40",
  };
  const c = colors[tag] ?? colors.point;
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border ${c} font-mono`}>
      {tag}
    </span>
  );
}

export function TrajectoryExplorer() {
  const [lookup, setLookup] = useState<Lookup | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selectedClass, setSelectedClass] = useState<string>("carbapenem");
  const [selectedIdx, setSelectedIdx] = useState(0);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch(LOOKUP_URL);
        const j = (await r.json()) as Lookup;
        if (!alive) return;
        setLookup(j);
      } catch (e: any) {
        setErr(String(e?.message || e));
      }
    })();
    return () => { alive = false; };
  }, []);

  // Reset transition selection when class changes
  useEffect(() => {
    setSelectedIdx(0);
  }, [selectedClass]);

  const transitions = useMemo(() => {
    if (!lookup) return [];
    return lookup.by_drug_class[selectedClass] ?? [];
  }, [lookup, selectedClass]);

  const current = transitions[selectedIdx];
  const classStats = useMemo(() => {
    if (!transitions.length) return null;
    const total = transitions.length;
    const hits = transitions.filter((t) => t.top3_hit).length;
    const canonicalHits = transitions.filter(
      (t) => t.top3_hit && t.is_canonical_target
    ).length;
    const nCanon = transitions.filter((t) => t.is_canonical_target).length;
    return {
      total,
      hit_rate: hits / total,
      canonical_hit_rate: nCanon ? canonicalHits / nCanon : 0,
    };
  }, [transitions]);

  if (err) {
    return (
      <div className="my-6 rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm">
        Lookup failed to load: {err}
      </div>
    );
  }
  if (!lookup) {
    return (
      <div className="my-6 rounded-lg border p-4 text-sm text-muted-foreground">
        Loading trajectory data…
      </div>
    );
  }

  // Sort drug classes by n descending
  const classes = lookup.drug_classes
    .filter((c) => (lookup.by_drug_class[c]?.length ?? 0) > 0)
    .sort((a, b) => (lookup.by_drug_class[b]?.length ?? 0) - (lookup.by_drug_class[a]?.length ?? 0));

  return (
    <div className="my-6 flex flex-col gap-4">
      {/* Drug-class picker */}
      <div className="rounded-lg border p-4 bg-card">
        <div className="flex items-baseline justify-between mb-2">
          <h4 className="text-sm font-semibold">Drug class</h4>
          <span className="text-xs text-muted-foreground">
            {transitions.length} test transitions
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {classes.map((c) => {
            const n = lookup.by_drug_class[c]?.length ?? 0;
            return (
              <button
                key={c}
                onClick={() => setSelectedClass(c)}
                className={`rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${
                  selectedClass === c
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-muted hover:border-primary/40"
                }`}
              >
                {c} <span className="opacity-60">({n})</span>
              </button>
            );
          })}
        </div>
        {classStats && (
          <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
            <div>
              <div className="text-muted-foreground">test transitions</div>
              <div className="font-mono font-bold text-lg">{classStats.total}</div>
            </div>
            <div>
              <div className="text-muted-foreground">top-3 hit rate</div>
              <div className="font-mono font-bold text-lg">
                {(classStats.hit_rate * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-muted-foreground">canonical-AMR hit rate</div>
              <div className="font-mono font-bold text-lg">
                {(classStats.canonical_hit_rate * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}
      </div>

      {transitions.length > 0 && current && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Transition list */}
          <div className="rounded-lg border p-3 bg-card lg:max-h-[440px] lg:overflow-y-auto">
            <div className="text-xs font-medium text-muted-foreground mb-2 sticky top-0 bg-card pb-1">
              Pick a transition
            </div>
            <div className="space-y-1">
              {transitions.slice(0, 50).map((t, i) => (
                <button
                  key={t.transition_id}
                  onClick={() => setSelectedIdx(i)}
                  className={`w-full text-left rounded-md border px-2 py-1.5 text-xs transition-colors ${
                    i === selectedIdx
                      ? "border-primary bg-primary/10"
                      : "border-muted hover:border-primary/30"
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-mono text-[10px] opacity-70">
                      step {t.step_number}
                    </span>
                    <div className="flex items-center gap-1">
                      {t.is_canonical_target && (
                        <span title="canonical AMR target" className="text-emerald-600">★</span>
                      )}
                      {t.top3_hit && (
                        <span title="top-3 hit" className="text-blue-600">✓</span>
                      )}
                    </div>
                  </div>
                  <div className="font-mono truncate" title={t.actual_next_mutation}>
                    → {t.actual_next_mutation}
                  </div>
                </button>
              ))}
            </div>
            {transitions.length > 50 && (
              <p className="text-[10px] text-muted-foreground mt-2 italic">
                Showing first 50 of {transitions.length}
              </p>
            )}
          </div>

          {/* Selected transition + top-10 predictions */}
          <div className="lg:col-span-2 rounded-lg border p-4 bg-card">
            <div className="mb-3">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">
                Transition · step {current.step_number}
              </div>
              <div className="font-mono text-sm mt-1">
                <span className="opacity-60">{current.from_genotype || "wildtype"}</span>
                <span className="mx-2 text-primary">→</span>
                <span className="font-bold">{current.actual_next_mutation}</span>
              </div>
              <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                <span>drug: <span className="font-mono">{current.drug}</span></span>
                <span>•</span>
                <span>{current.lab_or_clinical}</span>
                {current.is_canonical_target && (
                  <>
                    <span>•</span>
                    <span className="text-emerald-600 font-medium">canonical-AMR target</span>
                  </>
                )}
                {current.source && (
                  <>
                    <span>•</span>
                    <span className="font-mono">{current.source}</span>
                  </>
                )}
              </div>
            </div>

            <div className="text-xs font-semibold mb-2 flex items-center justify-between">
              <span>Top-10 reranker predictions</span>
              <span className="text-muted-foreground">
                top-3 = {current.top3_hit ? <span className="text-emerald-600 font-bold">HIT</span> : <span className="text-red-600">miss</span>}
              </span>
            </div>
            <div className="space-y-1.5">
              {current.top10_predictions.map((p) => {
                const isHit = p.is_actual_observed;
                const hitInTop3 = isHit && p.rank <= 3;
                return (
                  <div
                    key={p.mutation + p.rank}
                    className={`rounded-md border px-2 py-1.5 text-xs ${
                      hitInTop3
                        ? "border-emerald-500/50 bg-emerald-500/10"
                        : isHit
                        ? "border-amber-500/40 bg-amber-500/5"
                        : "border-muted"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="font-mono w-6 text-right opacity-60">
                          #{p.rank}
                        </span>
                        <span className="font-mono truncate" title={p.mutation}>
                          {p.mutation}
                        </span>
                        {p.is_canonical_amr && (
                          <span className="text-emerald-600 text-[10px]" title="canonical AMR mutation">★</span>
                        )}
                        {isHit && (
                          <span className="text-emerald-600 font-bold">← actual</span>
                        )}
                      </div>
                      <div className="flex gap-3 text-[10px] opacity-70 font-mono shrink-0">
                        <span title="reranker score">{p.rerank_score.toFixed(2)}</span>
                        {p.transformer_logprob !== null && (
                          <span title="transformer log-prob">
                            tf {p.transformer_logprob.toFixed(1)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      <details className="rounded-lg border p-3 text-xs bg-card">
        <summary className="cursor-pointer font-medium">How the reranker is built</summary>
        <p className="mt-2 text-muted-foreground leading-relaxed">
          Stage 1 — a LightGBM LambdaRank scores all candidates from a 14,006-mutation
          vocabulary using per-candidate features (frequency priors, ESM2 projector,
          DMS fitness). Stage 2 — keep its top-10 candidates; an autoregressive
          transformer scores the same candidates using ordered mutation history. Stage 3 —
          a second LightGBM reranker over 23 features combines stage-1 score, stage-2 log
          probability, frequency priors, and drug-class one-hots. The autoregressive
          context contribution from stage 2 (a +0.37 top-3 lift in the Day-2 ablation) is
          what unlocks the canonical-AMR signal that pure LGBM ranking (top-3 = 0.054 on
          full vocabulary) misses entirely.
        </p>
      </details>

      <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-xs leading-relaxed">
        <strong>Honest read on the demo.</strong> The reranker's published canonical-AMR
        top-3 = 0.403 was measured on the 72 transitions where the full feature pipeline
        ran end-to-end. The full 571-transition test split shown here includes transitions
        where the actual mutation isn't in the candidate set (a vocabulary-coverage gap, not
        a model failure). Per-class hit rates are lower than the headline number for that
        reason. This is documented in REPORT_day3.md §9 #1 as a Day-4 priority.
      </div>
    </div>
  );
}

export default TrajectoryExplorer;
