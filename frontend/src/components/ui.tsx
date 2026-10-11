import { ReactNode, useEffect, useRef, useState } from "react";
import { AlertStatus, Severity, fmtPct } from "../api";

const SEV_LABEL: Record<Severity, string> = { critical: "Critical", high: "High", medium: "Medium" };
const SEV_ICON: Record<Severity, string> = { critical: "▲", high: "◆", medium: "●" };
export const SEV_COLOR: Record<Severity, string> = {
  critical: "var(--status-critical)",
  high: "var(--status-serious)",
  medium: "var(--status-warning)",
};

export function SeverityBadge({ severity }: { severity: Severity }) {
  return (
    <span className={`badge sev-${severity}`}>
      <span style={{ color: SEV_COLOR[severity], fontSize: 10 }} aria-hidden>
        {SEV_ICON[severity]}
      </span>
      {SEV_LABEL[severity]}
    </span>
  );
}

const STATUS_LABEL: Record<AlertStatus, string> = {
  open: "Open",
  confirmed_fraud: "✓ Confirmed fraud",
  dismissed: "✕ Dismissed",
};
export function StatusBadge({ status }: { status: AlertStatus }) {
  return <span className={`badge status-${status}`}>{STATUS_LABEL[status]}</span>;
}

export function riskColor(score: number) {
  if (score >= 0.8) return "var(--status-critical)";
  if (score >= 0.5) return "var(--status-serious)";
  if (score >= 0.3) return "var(--status-warning)";
  return "var(--status-good)";
}

export function RiskMeter({ score }: { score: number | null }) {
  if (score === null) return <span className="muted">—</span>;
  return (
    <span className="risk num">
      <span className="bar">
        <i style={{ width: `${Math.max(4, score * 100)}%`, background: riskColor(score) }} />
      </span>
      {fmtPct(score)}
    </span>
  );
}

export function Card({ title, sub, action, children, className = "" }: {
  title?: ReactNode; sub?: ReactNode; action?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {sub && <p>{sub}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function Kpi({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card kpi">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function ChartTip({ active, payload, label, fmt }: {
  active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string;
  fmt?: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <b>{label}</b>
      {payload.map((p) => (
        <div className="line num" key={p.name}>
          <i style={{ background: p.color }} />
          {p.name}: <span style={{ color: "var(--text-primary)" }}>{fmt ? fmt(p.value) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

/** Poll an async loader on an interval; re-runs when deps change. */
export function usePoll<T>(load: () => Promise<T>, ms: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loader = useRef(load);
  loader.current = load;
  const reload = () =>
    loader.current().then(
      (d) => { setData(d); setError(null); },
      (e: Error) => setError(e.message),
    );
  useEffect(() => {
    reload();
    if (!ms) return;
    const id = setInterval(reload, ms);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, reload };
}

export function Toast({ children, onClose }: { children: ReactNode; onClose: () => void }) {
  useEffect(() => {
    const id = setTimeout(onClose, 6000);
    return () => clearTimeout(id);
  }, [onClose]);
  return <div className="toast" role="status">{children}</div>;
}
