import * as React from "react";
import { useState } from "react";

/**
 * InvestigationLog — the journey of "where is the specified information in the connectome?"
 *
 * A detective story of systematic elimination. Each entry: what we tried, what we found,
 * what it ruled out, why, and how that pointed us to the next attempt. Plain language;
 * numbers only where they carry the verdict. Every result is real (from the analysis arc).
 */

type Verdict = "ruled-out" | "kept" | "partial";

type Entry = {
  n: number;
  q: string;
  verdict: Verdict;
  tag: string;
  tried: string;
  found: string;
  ruled: string;
  why: string;
  next: string;
};

const ENTRIES: Entry[] = [
  {
    n: 1,
    q: "Does the wiring diagram predict its own reliable connections?",
    verdict: "kept",
    tag: "Contact geometry",
    tried:
      "Use physical contact — which neurons touch, and how much surface they share — to predict which connections show up reliably across different worms.",
    found:
      "It works remarkably well: contact geometry predicts reliable wiring about as well as one animal predicts another (AUC 0.84–0.91). That upper limit is the 'noise ceiling' — no model can beat it.",
    ruled:
      "Nothing yet — this is the anchor. Contact is the dominant factor in reliable wiring.",
    why:
      "You can't wire two neurons that don't touch, and where they touch a lot they usually connect. Most of the wiring is set by who is physically next to whom.",
    next:
      "If contact explains almost everything up to the noise ceiling, is the small leftover a hidden molecular 'address code'? We went looking for it.",
  },
  {
    n: 2,
    q: "Do neurons with similar genes wire together?",
    verdict: "ruled-out",
    tag: "Molecular address code",
    tried:
      "Test whether neurons with similar gene-expression profiles are more likely to connect, on top of contact.",
    found:
      "No. Similar-expression added essentially nothing (+0.0004), and it failed a proper statistical control.",
    ruled:
      "The simplest molecular-matching idea — 'like cells wire to like cells.'",
    why:
      "Randomly chosen genes predicted wiring just as well as the 'real' ones. It was picking up cell-type identity, not a wiring code.",
    next:
      "Maybe it isn't similarity but a lock-and-key pairing — a specific source→target code. Test for that instead.",
  },
  {
    n: 3,
    q: "Is there a hidden lock-and-key pairing code?",
    verdict: "ruled-out",
    tag: "Relational code",
    tried:
      "Let a model learn any source→target compatibility pattern it wanted (a flexible latent code) on top of contact.",
    found:
      "No. The learned code collapsed to zero — the model found nothing worth using beyond contact.",
    ruled:
      "A hidden relational / combinatorial wiring code.",
    why:
      "If a pairwise 'grammar' existed, a flexible model would have latched onto it. It didn't.",
    next:
      "One specific pairing still looked real, though — the sender's neurotransmitter. Zoom in on that.",
  },
  {
    n: 4,
    q: "Does the sender's neurotransmitter bias its wiring?",
    verdict: "partial",
    tag: "Neurotransmitter bias",
    tried:
      "Test whether a neuron's transmitter (glutamate, acetylcholine, GABA…) affects which of its contacts become reliable synapses.",
    found:
      "A small but genuinely real effect (+0.015, survives every control): glutamate senders over-wire, acetylcholine/GABA senders under-wire. But the fancier 'pairwise compatibility' pattern mostly vanished once we accounted for when the neurons are born.",
    ruled:
      "A rich transmitter 'grammar.' We kept only a small, one-sided sender bias.",
    why:
      "Most of the apparent pairing turned out to be developmental timing in disguise — birth order, not a molecular handshake.",
    next:
      "Timing kept absorbing our signals. That was a clue: maybe development, not molecules, is the real carrier. Follow the timing.",
  },
  {
    n: 5,
    q: "Does the wiring diagram predict how signals actually flow?",
    verdict: "ruled-out",
    tag: "Structure → function",
    tried:
      "Use the connectome to predict measured signal propagation — real function — not just which wires exist.",
    found:
      "No. Structure barely predicts how strongly signals travel (correlation ≈ 0.10). Even simulating dynamics on the wiring did worse than the plain static features.",
    ruled:
      "A clean bridge from wiring structure to function.",
    why:
      "Real signaling is dominated by extrasynaptic effects and by whether a cell can respond at all — not by the wiring diagram alone.",
    next:
      "Function is downstream and messy. So go upstream instead — to the developmental program that builds the wiring in the first place.",
  },
  {
    n: 6,
    q: "Can a gene-regulatory network generate each cell's machinery?",
    verdict: "ruled-out",
    tag: "Regulatory network",
    tried:
      "Use a measured gene-regulatory network to predict each neuron's ion-channel and receptor repertoire — the parts a biophysical model currently has to hand-fit.",
    found:
      "No. The network's specific structure added nothing — randomly chosen genes worked as well as the real regulators of each gene.",
    ruled:
      "A regulatory network acting as a generative layer beneath the wiring.",
    why:
      "The only predictable part was generic cell identity, not regulatory logic. A real network would have beaten random genes; it didn't.",
    next:
      "Push even further upstream — to the cell lineage, the exact family tree of divisions that builds every neuron.",
  },
  {
    n: 7,
    q: "Does the cell lineage specify fate and wiring?",
    verdict: "ruled-out",
    tag: "Lineage",
    tried:
      "Use the worm's exact, invariant lineage (which cell divides into which) to predict neuron identity and wiring.",
    found:
      "Lineage predicts broad type (sensory / interneuron / motor) but not transmitter identity — and it adds essentially nothing to wiring beyond contact and timing (+0.0018).",
    ruled:
      "The lineage as a wiring blueprint.",
    why:
      "Equivalent neurons often come from different branches of the tree; final identity is set at the end, not inherited from the family tree. And physical contact isn't lineage-derived either.",
    next:
      "Maybe the lineage itself is just the output of a hidden, compact program. Test whether such a program exists.",
  },
  {
    n: 8,
    q: "Is there a compact hidden program behind the lineage?",
    verdict: "ruled-out",
    tag: "Hidden generator",
    tried:
      "Ask whether a short, reusable rulebook could regenerate the lineage — one meaningfully shorter than just listing every division.",
    found:
      "Barely. The lineage is only 1.23× more compressible than a random tree, and its large branches are all unique.",
    ruled:
      "A compact hidden developmental generator.",
    why:
      "The worm's lineage is famously near-fixed and mosaic — each large branch is individually specified. The list essentially is the shortest description.",
    next:
      "Two places left to check: developmental gene expression, and bioelectric fields.",
  },
  {
    n: 9,
    q: "Does developmentally-timed gene expression carry hidden wiring info?",
    verdict: "ruled-out",
    tag: "Developmental expression",
    tried:
      "Use embryonic gene expression (not the adult snapshot) to predict contact and wiring.",
    found:
      "No. It didn't beat adult expression, wasn't specific to the relevant genes, and its apparent gain was a data-leakage artifact.",
    ruled:
      "A developmental-timing molecular code hiding in early expression.",
    why:
      "Same trap as before: what looked like signal was generic cell state, not a wiring instruction.",
    next:
      "One exotic possibility remained — that information is written into a bioelectric field rather than molecules.",
  },
  {
    n: 10,
    q: "Could a bioelectric field write information beyond local rules?",
    verdict: "ruled-out",
    tag: "Bioelectric field",
    tried:
      "Build a grounded bioelectric tissue and ask whether a non-local field could push it into a target pattern that its local rules can't reach on their own.",
    found:
      "No. The local rules already sufficed; the non-local field just smoothed the tissue out rather than writing anything new.",
    ruled:
      "An irreducible 'global' or bioelectric blueprint.",
    why:
      "Every stable pattern the field produced was one the local rules already allowed.",
    next:
      "With the hidden-code hypotheses exhausted, we turned back to what actually survived.",
  },
  {
    n: 11,
    q: "So what is actually left standing?",
    verdict: "kept",
    tag: "The synthesis",
    tried:
      "Add up everything that survived its controls and ask what it says together.",
    found:
      "Developmental birth-timing mediates the reliable 'strong core' of wiring (+0.061 beyond contact); that core is predictable right up to the noise ceiling; and the weak, variable connections behave like tolerated noise.",
    ruled:
      "The idea that the connectome hides a compact code. It doesn't — at least not one recoverable from any data we have.",
    why:
      "Every place a hidden code could live, a proper control dissolved it into contact, timing, or plain cell identity.",
    next:
      "The honest conclusion: the genome specifies a physical-and-temporal scaffold, not an edge-by-edge wiring table. See the ledger below.",
  },
];

const V: Record<Verdict, { chip: string; dot: string; label: string }> = {
  "ruled-out": { chip: "bg-rose-500/15 text-rose-300", dot: "bg-rose-400", label: "ruled out" },
  kept: { chip: "bg-emerald-500/15 text-emerald-300", dot: "bg-emerald-400", label: "kept" },
  partial: { chip: "bg-amber-500/15 text-amber-300", dot: "bg-amber-400", label: "partial" },
};

export default function InvestigationLog() {
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
        {ENTRIES.map((e) => {
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
                <div className="mt-1 pl-6 text-[11px] uppercase tracking-wide text-sky-400/70">{e.tag}</div>

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
