import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, fmtMoney, fmtPct, fmtTime, PATTERN_LABEL, Scenario, titleCase } from "../api";
import { Card, ChartTip, Kpi, RiskMeter, SEV_COLOR, Toast, usePoll } from "../components/ui";

const SCENARIOS: { id: Scenario; label: string; hint: string }[] = [
  { id: "ring", label: "Fraud ring", hint: "6–9 linked accounts reusing a known fraudster's device" },
  { id: "ato", label: "Account takeover", hint: "Established customer hijacked from abroad" },
  { id: "card_testing", label: "Card testing", hint: "Burst of micro-charges across stolen cards" },
  { id: "legit", label: "Normal traffic", hint: "25 ordinary purchases by existing customers" },
];

export default function Overview() {
  const { data, error, reload } = usePoll(api.overview, 5000);
  const [busy, setBusy] = useState<Scenario | null>(null);
  const [toast, setToast] = useState<React.ReactNode>(null);
  const navigate = useNavigate();

  const simulate = async (s: Scenario) => {
    setBusy(s);
    try {
      const r = await api.simulate(s);
      setToast(
        <div>
          <b>{SCENARIOS.find((x) => x.id === s)!.label} injected</b>
          <div className="secondary" style={{ marginTop: 4 }}>
            {r.transactions} transactions scored · {r.alerts} alert{r.alerts === 1 ? "" : "s"} raised · max risk {fmtPct(r.max_score)}
          </div>
          {r.alert_ids.length > 0 && (
            <button className="btn primary" style={{ marginTop: 10 }} onClick={() => navigate(`/alerts/${r.alert_ids[r.alert_ids.length - 1]}`)}>
              Investigate →
            </button>
          )}
        </div>,
      );
      reload();
    } catch (e) {
      setToast(<span>Simulation failed: {(e as Error).message}</span>);
    } finally {
      setBusy(null);
    }
  };

  const toggleStream = async () => {
    if (!data) return;
    await api.setStream(!data.stream.enabled);
    reload();
  };

  if (error) return <div className="error">Can't reach the API: {error}. Is the backend running and seeded?</div>;
  if (!data) return <div className="empty">Loading…</div>;
  const { kpis, model } = data;
  const m = model?.metrics;
  const base = m?.models.tabular_xgboost;
  const ens = m?.models.ensemble;

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Fraud operations overview</h1>
          <p>Live scoring with graph features, supervised + unsupervised models, and explainable alerts.</p>
        </div>
        <div className="row">
          <span className={`toggle ${data.stream.enabled ? "on" : ""}`} onClick={toggleStream} role="switch" aria-checked={data.stream.enabled}>
            <span className="track" /> Live stream
            {data.stream.enabled && <span className="pulse" />}
          </span>
        </div>
      </div>

      <Card title="Inject a scenario" sub="Generate events, score them in real time and watch alerts appear.">
        <div className="row">
          {SCENARIOS.map((s) => (
            <button key={s.id} className={`btn ${s.id === "legit" ? "" : "primary"}`} disabled={busy !== null} title={s.hint} onClick={() => simulate(s.id)}>
              {busy === s.id ? "Scoring…" : s.label}
            </button>
          ))}
        </div>
      </Card>

      {m && base && ens && (
        <section className="card hero">
          <div className="big num">{m.hidden_fraud.count}</div>
          <div>
            <h2>Hidden fraud found that a standard transaction classifier missed</h2>
            <p>
              On held-out data at the same alert volume ({fmtPct(m.alert_budget)} of traffic), the graph-aware ensemble caught{" "}
              {m.hidden_fraud.count} fraudulent transactions ({fmtMoney(m.hidden_fraud.amount)}) that a tabular XGBoost
              baseline let through. Fraud-ring recall: <b>{fmtPct(base.recall_by_pattern.ring ?? 0)}</b> →{" "}
              <b>{fmtPct(ens.recall_by_pattern.ring ?? 0)}</b>. PR-AUC {base.pr_auc.toFixed(3)} → {ens.pr_auc.toFixed(3)}.{" "}
              <Link to="/models" style={{ color: "var(--accent)" }}>See the evaluation →</Link>
            </p>
          </div>
        </section>
      )}

      <div className="grid cols-4">
        <Kpi label="Open alerts" value={<span className="num">{kpis.open_alerts}</span>}
          sub={`${kpis.open_by_severity.critical} critical · ${kpis.open_by_severity.high} high · ${kpis.open_by_severity.medium} medium`} />
        <Kpi label="Amount at risk (open)" value={<span className="num">{fmtMoney(kpis.amount_at_risk)}</span>}
          sub={`${fmtMoney(kpis.amount_prevented)} confirmed & blocked`} />
        <Kpi label="Transactions scored" value={<span className="num">{kpis.scored.toLocaleString()}</span>}
          sub={`${kpis.transactions.toLocaleString()} in the graph`} />
        <Kpi label="Analyst decisions" value={<span className="num">{kpis.confirmed_fraud + kpis.dismissed}</span>}
          sub={`${kpis.confirmed_fraud} fraud · ${kpis.dismissed} false positives`} />
      </div>

      <div className="grid cols-3-2">
        <Card title="Alerts per day" sub="By severity, last 3 weeks plus live traffic">
          <div style={{ height: 260 }}>
            <ResponsiveContainer>
              <BarChart data={data.daily_alerts.map((d) => ({ ...d, label: d.day.slice(5) }))} barCategoryGap={4}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="label" tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
                <YAxis allowDecimals={false} tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} width={30} />
                <Tooltip content={<ChartTip />} cursor={{ fill: "var(--surface-2)" }} />
                <Legend iconType="square" iconSize={9} wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
                {(["medium", "high", "critical"] as const).map((s, i) => (
                  <Bar key={s} dataKey={s} name={titleCase(s)} stackId="a" fill={SEV_COLOR[s]} stroke="var(--surface-1)" strokeWidth={1}
                    radius={i === 2 ? [4, 4, 0, 0] : 0} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Suspected fraud rings" sub="Linked-account clusters with open alerts">
          {data.hot_clusters.length === 0 ? (
            <div className="empty">No multi-account clusters with open alerts.</div>
          ) : (
            <table>
              <thead><tr><th>Cluster</th><th className="r">Accounts</th><th className="r">Open alerts</th><th className="r">Known fraud</th></tr></thead>
              <tbody>
                {data.hot_clusters.map((c) => (
                  <tr key={c.id} className="click" onClick={() => navigate(`/alerts/${c.top_alert_id}`)}>
                    <td className="mono">{c.id}</td>
                    <td className="r num">{c.size}</td>
                    <td className="r num">{c.open_alerts}</td>
                    <td className="r num">{c.known_fraud > 0 ? <span className="badge status-confirmed_fraud">{c.known_fraud}</span> : 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </div>

      <div className="grid cols-3-2">
        <Card title="Live transaction feed" sub="Most recent transactions through the engine">
          <div className="table-wrap">
            <table>
              <thead><tr><th>Time</th><th>Account</th><th>Category</th><th className="r">Amount</th><th>Risk</th></tr></thead>
              <tbody>
                {data.recent_transactions.map((t) => (
                  <tr key={t.id}>
                    <td className="muted num">{fmtTime(t.timestamp)}</td>
                    <td className="mono">{t.account_id}</td>
                    <td>{titleCase(t.merchant_category)}</td>
                    <td className="r num">{fmtMoney(t.amount)}</td>
                    <td><RiskMeter score={t.risk_score} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Risk score distribution" sub="All scored transactions (log scale)">
          <div style={{ height: 260 }}>
            <ResponsiveContainer>
              <BarChart data={data.score_histogram.map((b) => ({ ...b, label: `${Math.round(b.bucket * 100)}%`, plot: b.count || null }))} barCategoryGap={2}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="label" interval={3} tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
                <YAxis scale="log" domain={[1, "auto"]} allowDataOverflow tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} width={40} />
                <Tooltip content={<ChartTip />} cursor={{ fill: "var(--surface-2)" }} />
                <Bar dataKey="plot" name="Transactions" fill="var(--series-1)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {model && <p className="muted" style={{ margin: "8px 0 0", fontSize: 12 }}>
            Model v{model.version} · alert threshold {fmtPct(model.metrics.thresholds.alert)} · GNN backend: {model.gnn_backend}
          </p>}
        </Card>
      </div>
      <p className="muted" style={{ fontSize: 12 }}>
        Data is synthetic. Injected patterns: {Object.values(PATTERN_LABEL).slice(0, 3).join(", ")}.
      </p>
      {toast && <Toast onClose={() => setToast(null)}>{toast}</Toast>}
    </div>
  );
}
