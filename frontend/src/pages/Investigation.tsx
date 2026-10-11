import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, AlertDetail, fmtMoney, fmtPct, fmtTime, PATTERN_LABEL, titleCase } from "../api";
import EntityGraph from "../components/EntityGraph";
import { Card, RiskMeter, SeverityBadge, StatusBadge, usePoll } from "../components/ui";

function Factors({ factors }: { factors: AlertDetail["factors"] }) {
  const max = Math.max(...factors.map((f) => Math.abs(f.contribution)), 0.01);
  return (
    <div>
      {factors.map((f) => {
        const w = (Math.abs(f.contribution) / max) * 50;
        const up = f.contribution > 0;
        return (
          <div className="factor" key={f.feature}>
            <div className="what">
              <span className={`badge ${f.group === "graph" ? "graph" : ""}`} style={{ minWidth: 72, justifyContent: "center" }}>
                {f.group === "graph" ? "Graph" : "Behaviour"}
              </span>
              <span title={f.label}>{f.description}</span>
            </div>
            <div className="track" title={`SHAP ${up ? "+" : ""}${f.contribution.toFixed(2)} log-odds`}>
              <i style={{
                left: up ? "50%" : `${50 - w}%`, width: `${w}%`,
                background: up ? "var(--status-critical)" : "var(--status-good)",
              }} />
            </div>
          </div>
        );
      })}
      <p className="muted" style={{ fontSize: 12, margin: "8px 0 0" }}>
        Bars show each factor's SHAP contribution to the supervised model: right (red) raises risk, left (green) lowers it.
      </p>
    </div>
  );
}

function Decision({ alert, onDone }: { alert: AlertDetail; onDone: () => void }) {
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const decide = async (label: "fraud" | "legit") => {
    setBusy(true);
    try {
      const r = await api.feedback(alert.id, label, notes);
      setMsg(`Saved. ${r.feedback_since_last_train} new analyst label(s) are waiting for the next retrain.`);
      setNotes("");
      onDone();
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card title="Analyst decision" sub="Your verdict becomes a training label and immediately marks linked entities for live scoring.">
      {alert.feedback.length > 0 && (
        <ul style={{ margin: "0 0 12px", paddingLeft: 18 }} className="secondary">
          {alert.feedback.map((f, i) => (
            <li key={i}><b>{f.label === "fraud" ? "Fraud" : "Legitimate"}</b> by {f.analyst} · {fmtTime(f.created_at)}{f.notes ? ` — ${f.notes}` : ""}</li>
          ))}
        </ul>
      )}
      <textarea placeholder="Notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)} />
      <div className="row" style={{ marginTop: 10 }}>
        <button className="btn danger" disabled={busy} onClick={() => decide("fraud")}>✓ Confirm fraud</button>
        <button className="btn good" disabled={busy} onClick={() => decide("legit")}>✕ Not fraud</button>
      </div>
      {msg && <p className="secondary" style={{ margin: "10px 0 0", fontSize: 13 }}>{msg}</p>}
    </Card>
  );
}

export default function Investigation() {
  const id = Number(useParams().id);
  const { data: alert, error, reload } = usePoll(() => api.alert(id), 0, [id]);
  const { data: graph } = usePoll(() => api.alertGraph(id), 0, [id]);

  if (error) return <div className="error">{error}</div>;
  if (!alert) return <div className="empty">Loading investigation…</div>;
  const t = alert.transaction;
  const truth = alert.ground_truth;

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <Link to="/alerts" className="muted">← Alert queue</Link>
          <h1 style={{ marginTop: 6 }} className="row">
            Alert #{alert.id}
            <SeverityBadge severity={alert.severity} />
            <StatusBadge status={alert.status} />
          </h1>
          <p>
            {fmtMoney(t.amount)} · {titleCase(t.merchant_category)} · {t.channel} · {fmtTime(t.timestamp)} ·{" "}
            <span className="mono">{t.id}</span>
          </p>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="muted" style={{ fontSize: 12 }}>Risk score</div>
          <div style={{ fontSize: 34, fontWeight: 700 }} className="num">{fmtPct(alert.risk_score)}</div>
        </div>
      </div>

      <div className="grid cols-2-1">
        <div className="stack">
          <Card title="Why this was flagged">
            <p className="narrative">{alert.narrative}</p>
            <div style={{ marginTop: 14 }}><Factors factors={alert.factors} /></div>
          </Card>

          <Card title="Linked entities" sub={`${alert.cluster_size}-account cluster connected through shared devices, cards and addresses. Hover for details; drag to explore.`}>
            {graph ? <EntityGraph nodes={graph.nodes} edges={graph.edges} /> : <div className="graph-box" />}
          </Card>
        </div>

        <div className="stack">
          <Decision alert={alert} onDone={reload} />

          <Card title="Model signals" sub={`Ensemble of three detectors · model v${alert.model_version}`}>
            <div className="signals">
              <div className="signal"><div className="l">Supervised (XGBoost)</div><div className="v num">{fmtPct(alert.scores.xgb)}</div></div>
              <div className="signal"><div className="l">Graph neural net</div><div className="v num">{fmtPct(alert.scores.gnn)}</div></div>
              <div className="signal"><div className="l">Anomaly detector</div><div className="v num">{fmtPct(alert.scores.anomaly)}</div></div>
            </div>
          </Card>

          <Card title="Context">
            <dl className="kv">
              <dt>Account</dt><dd className="mono">{alert.account.id}</dd>
              <dt>Opened</dt><dd>{fmtTime(alert.account.created_at)}</dd>
              <dt>Country / KYC</dt><dd>{alert.account.country} · level {alert.account.kyc_level}</dd>
              <dt>Device</dt><dd className="mono">{t.device_id}</dd>
              <dt>IP</dt><dd className="mono">{t.ip} ({t.ip_country})</dd>
              <dt>Card</dt><dd className="mono">{t.card_id}</dd>
              <dt>Ship to</dt><dd className="mono">{t.address_id}</dd>
              <dt>Ground truth</dt>
              <dd>
                <span className={`badge ${truth.is_fraud ? "status-confirmed_fraud" : "status-dismissed"}`}>
                  {truth.is_fraud ? PATTERN_LABEL[truth.pattern] ?? truth.pattern : "Legitimate"}
                </span>{" "}
                <span className="muted" style={{ fontSize: 12 }}>synthetic label, demo only</span>
              </dd>
            </dl>
          </Card>

          <Card title="Investigation timeline">
            <ul className="timeline">
              {alert.timeline.map((e, i) => (
                <li key={i}>
                  <span className="when num">{fmtTime(e.timestamp)}</span>
                  <span className={`pin ${e.severity} ${e.kind === "alert" ? "alert" : ""}`} />
                  <span>
                    <div className="t">{e.title}</div>
                    {e.detail && <div className="d">{e.detail}</div>}
                  </span>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </div>
      <div className="row muted" style={{ fontSize: 12 }}>
        Other alerts on this account: <Link to={`/alerts?account_id=${alert.account_id}&status=all`} style={{ color: "var(--accent)" }}>view all</Link>
        <span>·</span> Transaction risk <RiskMeter score={t.risk_score} />
      </div>
    </div>
  );
}
