import * as React from "react";
import { useState } from "react";
import type { InvestigationEntry, Verdict } from "../../data/investigations/types";
import { SPECIFIED_INFORMATION } from "../../data/investigations/specified-information";

/**
 * InvestigationLog — a project's investigation as a sequence of tests.
 * Entries live in src/data/investigations/; the default is the specified-information arc.
 *
 * A detective story of systematic elimination. Each entry: what we tried, what we found,
 * what it ruled out, why, and how that pointed us to the next attempt. Plain language;
 * numbers only where they carry the verdict. Every result is real (from the analysis arc).
 */

const V: Record<Verdict, { chip: string; dot: string; label: string }> = {
  "ruled-out": { chip: "bg-rose-500/15 text-rose-300", dot: "bg-rose-400", label: "ruled out" },
  kept: { chip: "bg-emerald-500/15 text-emerald-300", dot: "bg-emerald-400", label: "kept" },
  partial: { chip: "bg-amber-500/15 text-amber-300", dot: "bg-amber-400", label: "partial" },
};

export default function InvestigationLog({
  entries = SPECIFIED_INFORMATION,
}: {
  entries?: InvestigationEntry[];
}) {
  const [open, setOpen] = useState<number[]>([1]);
  const toggle = (n: number) =>
    setOpen((o) => (o.includes(n) ? o.filter((x) => x !== n) : [...o, n]));

  return (
    <div className="not-prose my-8">
      <div className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-slate-500">
        <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-emerald-400" /> kept</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-amber-400" /> partial</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-rose-400" /> ruled out</span>
        <span className="ml-auto">tap any step to expand</span>
      </div>

      <ol className="relative space-y-3 border-l border-slate-700/60 pl-5">
        {entries.map((e) => {
          const v = V[e.verdict];
          const isOpen = open.includes(e.n);
          return (
            <li key={e.n} className="relative">
              <span className={`absolute -left-[26px] top-3 h-3 w-3 rounded-full ring-4 ring-slate-900 ${v.dot}`} />
              <button
                onClick={() => toggle(e.n)}
                className="w-full rounded-xl border border-slate-700/60 bg-slate-900/60 p-4 text-left transition hover:border-slate-600 hover:bg-slate-900/90"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs font-mono text-slate-500">{String(e.n).padStart(2, "0")}</span>
                    <span className="text-sm font-semibold text-slate-100">{e.q}</span>
                  </div>
                  <span className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${v.chip}`}>
                    {v.label}
                  </span>
                </div>
                <div className="mt-1 pl-6 text-[11px] uppercase tracking-wide text-sky-400/70">
                  {e.date && <span className="mr-2 font-mono normal-case text-slate-500">{e.date}</span>}
                  {e.tag}
                </div>

                {isOpen && (
                  <div className="mt-3 space-y-2.5 pl-6 text-xs leading-relaxed text-slate-300">
                    <p><span className="font-semibold text-slate-400">We tried — </span>{e.tried}</p>
                    <p><span className="font-semibold text-slate-400">We found — </span>{e.found}</p>
                    <p><span className="font-semibold text-slate-400">Ruled out — </span>{e.ruled}</p>
                    <p><span className="font-semibold text-slate-400">Why — </span>{e.why}</p>
                    <p className="text-slate-400"><span className="font-semibold text-sky-400/80">So next → </span>{e.next}</p>
                  </div>
                )}
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
