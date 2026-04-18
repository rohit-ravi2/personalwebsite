import * as React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

type Role = "sensory" | "inter" | "motor" | "other";

type Node = {
  id: string;
  role: Role;
  category: string;
  // runtime fields populated by the force graph
  x?: number;
  y?: number;
  __color?: string;
};

type Edge = {
  source: string | Node;
  target: string | Node;
  weight: number;
  type: "chemical" | "gap";
};

type Meta = {
  source: string;
  nodes: number;
  edges: number;
  chemical_edges: number;
  gap_edges: number;
};

type Graph = { nodes: Node[]; edges: Edge[]; meta: Meta };

// Light mode palette — warm accents, high contrast on cream/white cards
const LIGHT = {
  sensory: "#b8520a",   // warm orange
  inter: "#1a2a4a",     // deep navy (matches favicon)
  motor: "#2f6b3c",     // deep green
  other: "#6b7280",     // grey
  edgeChemical: "rgba(26,42,74,0.18)",
  edgeGap: "rgba(184,82,10,0.22)",
  edgeHighlight: "rgba(26,42,74,0.9)",
  bg: "transparent",
  text: "#1a2a4a",
  dim: "rgba(156,163,175,0.25)",
};

const DARK = {
  sensory: "#f59e42",
  inter: "#93c5fd",
  motor: "#86efac",
  other: "#9ca3af",
  edgeChemical: "rgba(147,197,253,0.22)",
  edgeGap: "rgba(245,158,66,0.24)",
  edgeHighlight: "rgba(247,237,211,0.95)",
  bg: "transparent",
  text: "#f2ead3",
  dim: "rgba(148,163,184,0.22)",
};

function useDarkMode() {
  const [dark, setDark] = useState<boolean>(() =>
    typeof document !== "undefined" && document.documentElement.classList.contains("dark"),
  );
  useEffect(() => {
    if (typeof window === "undefined") return;
    const root = document.documentElement;
    const obs = new MutationObserver(() => setDark(root.classList.contains("dark")));
    obs.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return dark;
}

function useElementSize(): [React.RefObject<HTMLDivElement>, { w: number; h: number }] {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 520 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (width > 0) setSize({ w: Math.round(width), h: Math.max(360, Math.round(height)) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

export function ConnectomeExplorer() {
  const [graph, setGraph] = useState<Graph | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showGap, setShowGap] = useState(true);
  const [showChemical, setShowChemical] = useState(true);
  const dark = useDarkMode();
  const palette = dark ? DARK : LIGHT;
  const [wrapRef, { w, h }] = useElementSize();
  const fgRef = useRef<any>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/data/connectome.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: Graph) => { if (!cancelled) setGraph(d); })
      .catch((e) => { if (!cancelled) setErr(String(e)); });
    return () => { cancelled = true; };
  }, []);

  // Build adjacency set for neighborhood highlighting.
  const neighborhood = useMemo(() => {
    if (!graph) return new Map<string, Set<string>>();
    const m = new Map<string, Set<string>>();
    for (const n of graph.nodes) m.set(n.id, new Set([n.id]));
    for (const e of graph.edges) {
      const s = typeof e.source === "string" ? e.source : e.source.id;
      const t = typeof e.target === "string" ? e.target : e.target.id;
      m.get(s)?.add(t);
      m.get(t)?.add(s);
    }
    return m;
  }, [graph]);

  const filteredData = useMemo(() => {
    if (!graph) return { nodes: [], links: [] };
    const links = graph.edges.filter((e) => (e.type === "gap" ? showGap : showChemical));
    return { nodes: graph.nodes, links };
  }, [graph, showGap, showChemical]);

  const activeSet = selected ? neighborhood.get(selected) ?? new Set([selected]) : null;

  const resetView = () => {
    setSelected(null);
    fgRef.current?.zoomToFit?.(400, 40);
  };

  if (err) {
    return (
      <div className="my-6 rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm">
        Connectome data failed to load: {err}
      </div>
    );
  }

  if (!graph) {
    return (
      <div className="my-6 flex h-[400px] w-full items-center justify-center rounded-lg border text-sm text-muted-foreground">
        Loading connectome…
      </div>
    );
  }

  const roleCounts = {
    sensory: graph.nodes.filter((n) => n.role === "sensory").length,
    inter: graph.nodes.filter((n) => n.role === "inter").length,
    motor: graph.nodes.filter((n) => n.role === "motor").length,
  };

  return (
    <div className="my-6 flex flex-col gap-3">
      {/* Legend + controls */}
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-full" style={{ background: palette.sensory }} />
          <span>Sensory ({roleCounts.sensory})</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-full" style={{ background: palette.inter }} />
          <span>Inter ({roleCounts.inter})</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-full" style={{ background: palette.motor }} />
          <span>Motor ({roleCounts.motor})</span>
        </div>
        <span className="mx-2 text-muted-foreground">·</span>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={showChemical} onChange={(e) => setShowChemical(e.target.checked)} />
          <span>Chemical ({graph.meta.chemical_edges})</span>
        </label>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={showGap} onChange={(e) => setShowGap(e.target.checked)} />
          <span>Gap junction ({graph.meta.gap_edges})</span>
        </label>
        {selected && (
          <>
            <span className="mx-2 text-muted-foreground">·</span>
            <span className="font-medium">Focused: {selected}</span>
            <button
              onClick={resetView}
              className="ml-1 rounded border px-2 py-0.5 text-xs hover:bg-accent"
            >
              Clear
            </button>
          </>
        )}
      </div>

      {/* Graph canvas */}
      <div
        ref={wrapRef}
        className="relative h-[560px] w-full overflow-hidden rounded-lg border bg-background"
        onClick={(e) => {
          // click on empty canvas area clears selection
          if ((e.target as HTMLElement).tagName === "CANVAS" && e.currentTarget === e.currentTarget) {
            // nothing — react-force-graph handles its own click, but we use onBackgroundClick below
          }
        }}
      >
        <ForceGraph2D
          ref={fgRef}
          graphData={filteredData as any}
          width={w}
          height={h}
          backgroundColor={palette.bg}
          nodeRelSize={4}
          nodeCanvasObject={(node: any, ctx, globalScale) => {
            const n = node as Node & { x: number; y: number };
            const dimmed = activeSet && !activeSet.has(n.id);
            const color = dimmed ? palette.dim : palette[n.role as Role] ?? palette.other;
            const r = selected === n.id ? 6 : 4;
            ctx.beginPath();
            ctx.arc(n.x, n.y, r, 0, 2 * Math.PI, false);
            ctx.fillStyle = color;
            ctx.fill();
            if (selected === n.id) {
              ctx.lineWidth = 1.5 / globalScale;
              ctx.strokeStyle = palette.text;
              ctx.stroke();
            }
            // Draw labels above a zoom threshold, or always if highlighted.
            if (globalScale >= 1.6 || selected === n.id || (activeSet && activeSet.has(n.id))) {
              const fontSize = Math.max(3, 10 / globalScale);
              ctx.font = `${fontSize}px sans-serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillStyle = dimmed ? palette.dim : palette.text;
              ctx.fillText(n.id, n.x, n.y + r + 1);
            }
          }}
          nodePointerAreaPaint={(node: any, color, ctx) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(node.x, node.y, 6, 0, 2 * Math.PI, false);
            ctx.fill();
          }}
          linkColor={(link: any) => {
            const s = typeof link.source === "string" ? link.source : link.source.id;
            const t = typeof link.target === "string" ? link.target : link.target.id;
            if (activeSet) {
              if (activeSet.has(s) && activeSet.has(t)) return palette.edgeHighlight;
              return palette.dim;
            }
            return link.type === "gap" ? palette.edgeGap : palette.edgeChemical;
          }}
          linkWidth={(link: any) => {
            const s = typeof link.source === "string" ? link.source : link.source.id;
            const t = typeof link.target === "string" ? link.target : link.target.id;
            if (activeSet && activeSet.has(s) && activeSet.has(t)) return 1.5;
            return Math.min(2, 0.3 + Math.log10(1 + link.weight) * 0.45);
          }}
          linkDirectionalArrowLength={(link: any) => (link.type === "chemical" ? 2.5 : 0)}
          linkDirectionalArrowRelPos={1}
          linkDirectionalArrowColor={(link: any) => {
            const s = typeof link.source === "string" ? link.source : link.source.id;
            const t = typeof link.target === "string" ? link.target : link.target.id;
            if (activeSet) {
              return activeSet.has(s) && activeSet.has(t) ? palette.edgeHighlight : palette.dim;
            }
            return palette.edgeChemical;
          }}
          cooldownTicks={180}
          onEngineStop={() => fgRef.current?.zoomToFit?.(400, 40)}
          onNodeClick={(node: any) => setSelected((prev) => (prev === node.id ? null : node.id))}
          onBackgroundClick={() => setSelected(null)}
          onNodeHover={(node: any) => {
            const el = wrapRef.current;
            if (el) el.style.cursor = node ? "pointer" : "default";
          }}
        />
      </div>

      <p className="text-xs text-muted-foreground">
        {graph.nodes.length} neuron classes, {graph.meta.chemical_edges} chemical +
        {" "}{graph.meta.gap_edges} gap-junction connections. Source: Cook et al. 2019 (hermaphrodite cell-class adjacency, corrected July 2020).
        Click a neuron to highlight its immediate neighborhood; click the background to clear.
      </p>
    </div>
  );
}

export default ConnectomeExplorer;
