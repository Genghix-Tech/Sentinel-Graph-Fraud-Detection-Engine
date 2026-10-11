import { useNavigate, useSearchParams } from "react-router-dom";
import { api, fmtMoney, fmtTime, titleCase } from "../api";
import { Card, RiskMeter, SeverityBadge, StatusBadge, usePoll } from "../components/ui";

const PAGE = 25;

export default function Alerts() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const status = params.get("status") ?? "open";
  const severity = params.get("severity") ?? "";
  const sort = params.get("sort") ?? "risk";
  const account = params.get("account_id") ?? "";
  const page = Number(params.get("page") ?? 0);

  const { data, error } = usePoll(
    () => api.alerts({ status: status === "all" ? undefined : status, severity, sort, account_id: account,
      limit: PAGE, offset: page * PAGE }),
    5000,
    [status, severity, sort, account, page],
  );

  const set = (k: string, v: string) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v); else next.delete(k);
    if (k !== "page") next.delete("page");
    setParams(next);
  };

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Alert queue</h1>
          <p>Every alert carries SHAP-based reasons. Rows tagged <span className="badge graph">Graph</span> were driven mainly by relationship signals.</p>
        </div>
      </div>
      <div className="row">
        <div className="seg" role="tablist">
          {[["open", "Open"], ["confirmed_fraud", "Confirmed"], ["dismissed", "Dismissed"], ["all", "All"]].map(([v, l]) => (
            <button key={v} className={status === v ? "on" : ""} onClick={() => set("status", v)}>{l}</button>
          ))}
        </div>
        <select value={severity} onChange={(e) => set("severity", e.target.value)} aria-label="Severity">
          <option value="">All severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
        </select>
        <select value={sort} onChange={(e) => set("sort", e.target.value)} aria-label="Sort">
          <option value="risk">Highest risk first</option>
          <option value="recent">Most recent first</option>
        </select>
        {account && (
          <button className="btn" onClick={() => set("account_id", "")}>Account {account} ✕</button>
        )}
        {data && <span className="muted">{data.total} alert{data.total === 1 ? "" : "s"}</span>}
      </div>

      {error && <div className="error">{error}</div>}
      <Card>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th><th>Time</th><th>Severity</th><th>Risk</th><th>Account</th><th className="r">Amount</th>
                <th>Category</th><th>Main reason</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {data?.items.map((a) => (
                <tr key={a.id} className="click" onClick={() => navigate(`/alerts/${a.id}`)}>
                  <td className="muted num">{a.id}</td>
                  <td className="muted num" style={{ whiteSpace: "nowrap" }}>{fmtTime(a.created_at)}</td>
                  <td><SeverityBadge severity={a.severity} /></td>
                  <td><RiskMeter score={a.risk_score} /></td>
                  <td className="mono">{a.account_id}</td>
                  <td className="r num">{fmtMoney(a.amount)}</td>
                  <td>{titleCase(a.merchant_category)}</td>
                  <td style={{ maxWidth: 360 }}>
                    <div className="row" style={{ flexWrap: "nowrap" }}>
                      {a.graph_driven && <span className="badge graph">Graph</span>}
                      <span className="secondary" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.top_factor}</span>
                    </div>
                  </td>
                  <td><StatusBadge status={a.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          {data && data.items.length === 0 && <div className="empty">No alerts match these filters.</div>}
        </div>
        {data && data.total > PAGE && (
          <div className="spread" style={{ marginTop: 12 }}>
            <span className="muted">Page {page + 1} of {Math.ceil(data.total / PAGE)}</span>
            <div className="row">
              <button className="btn" disabled={page === 0} onClick={() => set("page", String(page - 1))}>← Prev</button>
              <button className="btn" disabled={(page + 1) * PAGE >= data.total} onClick={() => set("page", String(page + 1))}>Next →</button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
