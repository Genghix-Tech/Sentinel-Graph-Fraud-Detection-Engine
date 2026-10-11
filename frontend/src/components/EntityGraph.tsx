import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { ForceGraphMethods } from "react-force-graph-2d";
import { GraphEdge, GraphNode } from "../api";

type Kind = GraphNode["kind"];
const KIND_LABEL: Record<Kind, string> = { account: "Account", device: "Device", ip: "IP address", card: "Card", address: "Shipping address" };
const KIND_VAR: Record<Kind, string> = {
  account: "--series-1", device: "--series-2", ip: "--series-3", card: "--series-4", address: "--series-5",
};

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Each entity kind has its own shape, so identity never relies on colour alone.
function drawShape(ctx: CanvasRenderingContext2D, kind: Kind, x: number, y: number, r: number) {
  ctx.beginPath();
  if (kind === "account") ctx.arc(x, y, r, 0, 2 * Math.PI);
  else if (kind === "device") ctx.rect(x - r, y - r, 2 * r, 2 * r);
  else if (kind === "ip") { ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.2, y); ctx.lineTo(x, y + r * 1.2); ctx.lineTo(x - r * 1.2, y); ctx.closePath(); }
  else if (kind === "card") ctx.roundRect(x - r * 1.4, y - r * 0.85, r * 2.8, r * 1.7, r * 0.3);
  else { ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.15, y + r * 0.9); ctx.lineTo(x - r * 1.15, y + r * 0.9); ctx.closePath(); }
}

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onSelect?: (n: GraphNode | null) => void;
}

export default function EntityGraph({ nodes, edges, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const fg = useRef<ForceGraphMethods>();
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<GraphNode | null>(null);

  useEffect(() => {
    if (!box.current) return;
    setWidth(box.current.clientWidth || 600);
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);

  const data = useMemo(
    () => ({ nodes: nodes.map((n) => ({ ...n })), links: edges.map((e) => ({ ...e })) }),
    [nodes, edges],
  );
  const colors = useMemo(() => {
    const c: Record<string, string> = {
      critical: cssVar("--status-critical"), serious: cssVar("--status-serious"), text: cssVar("--text-secondary"),
      link: cssVar("--axis"), surface: cssVar("--surface-2"), primary: cssVar("--text-primary"),
    };
    (Object.keys(KIND_VAR) as Kind[]).forEach((k) => (c[k] = cssVar(KIND_VAR[k])));
    return c;
  }, [nodes]);

  useEffect(() => {
    const t = setTimeout(() => fg.current?.zoomToFit(400, 40), 700);
    return () => clearTimeout(t);
  }, [data, width]);

  return (
    <div>
      <div className="graph-box" ref={box}>
        <ForceGraph2D
          ref={fg}
          graphData={data}
          width={width}
          height={460}
          backgroundColor="rgba(0,0,0,0)"
          cooldownTicks={120}
          linkColor={(l: any) => (l.source?.in_transaction || l.target?.in_transaction) && (l.source?.focus || l.target?.focus) ? colors.primary : colors.link}
          linkWidth={(l: any) => Math.min(4, 0.8 + Math.log2(1 + l.count))}
          onNodeHover={(n: any) => setHover(n ?? null)}
          onNodeClick={(n: any) => onSelect?.(n)}
          nodeLabel={() => ""}
          nodeCanvasObject={(n: any, ctx, scale) => {
            const node = n as GraphNode & { x: number; y: number };
            const r = node.kind === "account" ? 7 : 5.5;
            const fill = node.kind === "account"
              ? node.known_fraud ? colors.critical : node.alerts ? colors.serious : colors.account
              : colors[node.kind];
            drawShape(ctx, node.kind, node.x, node.y, r);
            ctx.fillStyle = fill;
            ctx.fill();
            ctx.lineWidth = 2 / scale;
            ctx.strokeStyle = colors.surface;  // surface ring separates overlapping marks
            ctx.stroke();
            if (node.focus || node.in_transaction) {
              drawShape(ctx, node.kind, node.x, node.y, r + 3.5);
              ctx.lineWidth = 2 / scale;
              ctx.strokeStyle = colors.primary;
              ctx.stroke();
            }
            if (node.kind === "account" || scale > 1.6 || node === hover) {
              ctx.font = `${11 / Math.max(scale, 0.8)}px system-ui, sans-serif`;
              ctx.textAlign = "center";
              ctx.fillStyle = colors.text;
              ctx.fillText(node.label, node.x, node.y + r + 11 / Math.max(scale, 0.8));
            }
          }}
          nodePointerAreaPaint={(n: any, color, ctx) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(n.x, n.y, 12, 0, 2 * Math.PI);
            ctx.fill();
          }}
        />
        {hover && (
          <div className="chart-tip" style={{ position: "absolute", top: 10, left: 10, pointerEvents: "none" }}>
            <b>{KIND_LABEL[hover.kind]} · <span className="mono">{hover.label}</span></b>
            {hover.kind === "account" ? (
              <>
                <div className="line">{hover.known_fraud ? "Confirmed fraud" : hover.alerts ? `${hover.alerts} alert(s)` : "No alerts"}</div>
                {hover.focus && <div className="line">Account under investigation</div>}
              </>
            ) : (
              <div className="line">Used by {hover.shared_by} account{hover.shared_by === 1 ? "" : "s"}</div>
            )}
            {hover.in_transaction && <div className="line">Used in the flagged transaction</div>}
          </div>
        )}
      </div>
      <div className="legend">
        {(Object.keys(KIND_LABEL) as Kind[]).map((k) => (
          <span key={k}>
            <svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden>
              <ShapeIcon kind={k} color={`var(${KIND_VAR[k]})`} />
            </svg>
            {KIND_LABEL[k]}
          </span>
        ))}
        <span><i style={{ background: "var(--status-serious)", borderRadius: "50%" }} />Account with alerts</span>
        <span><i style={{ background: "var(--status-critical)", borderRadius: "50%" }} />Confirmed fraud</span>
        <span><i style={{ border: "2px solid var(--text-primary)", borderRadius: "50%" }} />In this transaction</span>
      </div>
    </div>
  );
}

function ShapeIcon({ kind, color }: { kind: Kind; color: string }) {
  if (kind === "account") return <circle r="5.5" fill={color} />;
  if (kind === "device") return <rect x="-5" y="-5" width="10" height="10" fill={color} />;
  if (kind === "ip") return <polygon points="0,-6 6,0 0,6 -6,0" fill={color} />;
  if (kind === "card") return <rect x="-6.5" y="-4" width="13" height="8" rx="1.5" fill={color} />;
  return <polygon points="0,-6 6,5 -6,5" fill={color} />;
}
