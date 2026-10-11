import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, fmtMoney, fmtPct, fmtTime, MODEL_LABEL, PATTERN_LABEL } from "../api";
import { Card, ChartTip, Kpi, usePoll } from "../components/ui";

const MODEL_ORDER = ["tabular_xgboost", "isolation_forest", "graphsage_gnn", "graph_xgboost", "ensemble"];
const PATTERNS = ["ring", "ato", "card_testing"];

export default function Models() {
  const { data, error, reload } = usePoll(api.models, 3000);
  const [selected, setSelected] = useState<number | null>(null);
  const [starting, setStarting] = useState(false);

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div className="empty">Loading…</div>;
  const active = data.versions.find((v) => v.is_active) ?? data.versions[0];
  const v = data.versions.find((x) => x.version === selected) ?? active;
  if (!v) return <div className="empty">No models trained yet. Run the seed command.</div>;
  const m = v.metrics;
  const base = m.models.tabular_xgboost;
  const ens = m.models.ensemble;

  const retrain = async () => {
    setStarting(true);
    try { await api.retrain(); } finally { setStarting(false); reload(); }
  };

  const patternData = PATTERNS.filter((p) => p in ens.recall_by_pattern).map((p) => ({
    pattern: PATTERN_LABEL[p],
    "Tabular baseline": base.recall_by_pattern[p] ?? 0,
    "SentinelGraph ensemble": ens.recall_by_pattern[p] ?? 0,
  }));
  const importance = v.feature_importance.slice(0, 12).map((f) => ({ ...f, name: f.label }));

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Models & evaluation</h1>
          <p>Time-based split with point-in-time graph features. Every model is compared on the same held-out window at the same alert volume.</p>
        </div>
        <div className="row">
          <span className="secondary">{data.feedback_since_last_train} new analyst label{data.feedback_since_last_train === 1 ? "" : "s"} since v{active.version}</span>
          <button className="btn primary" onClick={retrain} disabled={data.retrain.running || starting}>
            {data.retrain.running ? "Retraining… (~1 min)" : "Retrain with feedback"}
          </button>
        </div>
      </div>
      {data.retrain.error && <div className="error">Retrain failed: {data.retrain.error}</div>}

      <div className="grid cols-4">
        <Kpi label="Ensemble PR-AUC (test)" value={<span className="num">{ens.pr_auc.toFixed(3)}</span>}
          sub={`Tabular baseline: ${base.pr_auc.toFixed(3)}`} />
        <Kpi label={`Recall at ${fmtPct(m.alert_budget)} alert budget`} value={<span className="num">{fmtPct(ens.recall_at_budget)}</span>}
          sub={`Baseline: ${fmtPct(base.recall_at_budget)}`} />
        <Kpi label="Hidden fraud found" value={<span className="num">{m.hidden_fraud.count}</span>}
          sub={`${fmtMoney(m.hidden_fraud.amount)} missed by the baseline`} />
        <Kpi label="Production threshold" value={<span className="num">{fmtPct(m.production_threshold.precision)}</span>}
          sub={`precision · ${fmtPct(m.production_threshold.recall)} recall · ${fmtPct(m.production_threshold.alert_rate, 1)} alerted`} />
      </div>

      <div className="grid cols-2">
        <Card title="Recall by fraud pattern" sub={`Share of each pattern's test transactions inside the top ${fmtPct(m.alert_budget)} of scores`}>
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={patternData} barGap={2} barCategoryGap="28%">
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="pattern" tick={{ fill: "var(--text-secondary)", fontSize: 12 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
                <YAxis domain={[0, 1]} tickFormatter={(x) => fmtPct(x)} tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} width={44} />
                <Tooltip content={<ChartTip fmt={(x) => fmtPct(x, 1)} />} cursor={{ fill: "var(--surface-2)" }} />
                <Legend iconType="square" iconSize={9} wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Tabular baseline" fill="var(--series-2)" radius={[4, 4, 0, 0]}>
                  <LabelList dataKey="Tabular baseline" position="top" formatter={(x: number) => fmtPct(x)} style={{ fill: "var(--text-secondary)", fontSize: 11 }} />
                </Bar>
                <Bar dataKey="SentinelGraph ensemble" fill="var(--series-1)" radius={[4, 4, 0, 0]}>
                  <LabelList dataKey="SentinelGraph ensemble" position="top" formatter={(x: number) => fmtPct(x)} style={{ fill: "var(--text-secondary)", fontSize: 11 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="What drives the model" sub="Mean |SHAP| on held-out transactions">
          <div style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={importance} layout="vertical" margin={{ left: 10, right: 16 }} barCategoryGap={3}>
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="name" width={220} tick={{ fill: "var(--text-secondary)", fontSize: 11.5 }} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTip fmt={(x) => x.toFixed(3)} />} cursor={{ fill: "var(--surface-2)" }} />
                <Bar dataKey="importance" name="Mean |SHAP|" radius={[0, 4, 4, 0]}>
                  {importance.map((f) => <Cell key={f.feature} fill={f.group === "graph" ? "var(--series-1)" : "var(--series-2)"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="legend">
            <span><i style={{ background: "var(--series-1)", borderRadius: 2 }} />Graph feature</span>
            <span><i style={{ background: "var(--series-2)", borderRadius: 2 }} />Behavioural feature</span>
          </div>
        </Card>
      </div>

      <Card title={`Model comparison · v${v.version}`} sub={`Test window from ${fmtTime(m.split.test_start)} · ${m.split.test.toLocaleString()} transactions · ${m.hidden_fraud.test_fraud_total} fraudulent`}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th className="r">PR-AUC</th><th className="r">ROC-AUC</th><th className="r">Recall @ budget</th><th className="r">Precision @ budget</th>
                {PATTERNS.map((p) => <th key={p} className="r">{PATTERN_LABEL[p]}</th>)}</tr>
            </thead>
            <tbody>
              {MODEL_ORDER.map((k) => {
                const r = m.models[k];
                return (
                  <tr key={k} style={k === "ensemble" ? { fontWeight: 600 } : undefined}>
                    <td>{MODEL_LABEL[k]}</td>
                    <td className="r num">{r.pr_auc.toFixed(3)}</td>
                    <td className="r num">{r.roc_auc.toFixed(3)}</td>
                    <td className="r num">{fmtPct(r.recall_at_budget, 1)}</td>
                    <td className="r num">{fmtPct(r.precision_at_budget, 1)}</td>
                    {PATTERNS.map((p) => <td key={p} className="r num">{p in r.recall_by_pattern ? fmtPct(r.recall_by_pattern[p]) : "—"}</td>)}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ fontSize: 12, margin: "10px 0 0" }}>
          Ensemble weights (tuned on validation): XGBoost {m.weights.xgb} · GNN {m.weights.gnn} · anomaly {m.weights.anomaly}. GNN backend: {v.gnn_backend}. Trained in {m.train_seconds}s.
        </p>
      </Card>

      <Card title="Version history" sub="Each retrain folds analyst decisions into the training labels; reviewed transactions leave the test set, so later versions are scored on the remaining unreviewed traffic. Click a version to inspect it.">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Version</th><th>Trained</th><th className="r">Ensemble PR-AUC</th><th className="r">Hidden fraud</th><th className="r">Test fraud txns</th><th className="r">Train rows</th><th className="r">Analyst labels</th><th>Notes</th><th /></tr></thead>
            <tbody>
              {data.versions.map((x) => (
                <tr key={x.version} className="click" onClick={() => setSelected(x.version)}
                  style={x.version === v.version ? { background: "var(--accent-soft)" } : undefined}>
                  <td><b>v{x.version}</b></td>
                  <td className="muted num">{fmtTime(x.created_at)}</td>
                  <td className="r num">{x.metrics.models.ensemble.pr_auc.toFixed(3)}</td>
                  <td className="r num">{x.metrics.hidden_fraud.count}</td>
                  <td className="r num">{x.metrics.hidden_fraud.test_fraud_total}</td>
                  <td className="r num">{x.n_train.toLocaleString()}</td>
                  <td className="r num">{x.n_feedback}</td>
                  <td className="secondary">{x.notes}</td>
                  <td>{x.is_active && <span className="badge status-dismissed">Active</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
