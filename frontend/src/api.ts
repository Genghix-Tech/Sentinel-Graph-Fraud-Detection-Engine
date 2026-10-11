export type Severity = "critical" | "high" | "medium";
export type AlertStatus = "open" | "confirmed_fraud" | "dismissed";
export type Scenario = "legit" | "ring" | "ato" | "card_testing";

export interface Factor {
  feature: string;
  label: string;
  description: string;
  value: number | null;
  contribution: number;
  group: "graph" | "behavioral";
}

export interface Txn {
  id: string;
  account_id: string;
  timestamp: string;
  amount: number;
  merchant_id: string;
  merchant_category: string;
  channel: string;
  device_id: string;
  ip: string;
  ip_country: string;
  card_id: string;
  address_id: string;
  risk_score: number | null;
  model_version: number | null;
}

export interface AlertSummary {
  id: number;
  transaction_id: string;
  account_id: string;
  created_at: string;
  risk_score: number;
  severity: Severity;
  status: AlertStatus;
  scores: { xgb: number; gnn: number; anomaly: number };
  model_version: number;
  amount: number;
  merchant_category: string;
  channel: string;
  top_factor: string | null;
  graph_driven: boolean;
}

export interface TimelineEvent {
  timestamp: string;
  kind: "account" | "entity" | "linked" | "fraud" | "transaction" | "alert" | "feedback";
  title: string;
  detail: string;
  severity: "info" | "warning" | "critical";
  ref: string;
}

export interface AlertDetail extends AlertSummary {
  factors: Factor[];
  narrative: string;
  transaction: Txn;
  account: { id: string; created_at: string; country: string; kyc_level: number; email_domain: string };
  cluster_size: number;
  timeline: TimelineEvent[];
  feedback: { label: string; notes: string; analyst: string; created_at: string }[];
  ground_truth: { pattern: string; is_fraud: boolean };
}

export interface GraphNode {
  id: string;
  kind: "account" | "device" | "ip" | "card" | "address";
  label: string;
  focus: boolean;
  known_fraud: boolean;
  shared_by?: number;
  alerts?: number;
  in_transaction: boolean;
}
export interface GraphEdge {
  source: string;
  target: string;
  kind: string;
  count: number;
  first_seen: string;
}

export interface ModelMetrics {
  pr_auc: number;
  roc_auc: number;
  recall_at_budget: number;
  precision_at_budget: number;
  recall_by_pattern: Record<string, number>;
}

export interface VersionMetrics {
  alert_budget: number;
  models: Record<string, ModelMetrics>;
  hidden_fraud: { count: number; amount: number; by_pattern: Record<string, number>; test_fraud_total: number };
  production_threshold: { alert_rate: number; precision: number; recall: number };
  split: { train: number; validation: number; test: number; fraud_rate: number; test_start: string };
  weights: { xgb: number; gnn: number; anomaly: number };
  thresholds: { alert: number; high: number; critical: number };
  train_seconds: number;
}

export interface ModelVersion {
  version: number;
  created_at: string;
  is_active: boolean;
  metrics: VersionMetrics;
  feature_importance: { feature: string; label: string; importance: number; group: string }[];
  n_train: number;
  n_feedback: number;
  gnn_backend: string;
  notes: string;
}

export interface Overview {
  kpis: {
    transactions: number;
    scored: number;
    alerts_total: number;
    open_alerts: number;
    open_by_severity: Record<Severity, number>;
    confirmed_fraud: number;
    dismissed: number;
    amount_at_risk: number;
    amount_prevented: number;
  };
  daily_alerts: { day: string; medium: number; high: number; critical: number }[];
  score_histogram: { bucket: number; count: number }[];
  hot_clusters: { id: string; size: number; open_alerts: number; known_fraud: number; flagged_accounts: number; top_alert_id: number }[];
  recent_transactions: Txn[];
  model: { version: number; gnn_backend: string; metrics: VersionMetrics; n_feedback: number } | null;
  stream: { enabled: boolean };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  overview: () => request<Overview>("/overview"),
  alerts: (params: Record<string, string | number | undefined>) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && q.set(k, String(v)));
    return request<{ total: number; items: AlertSummary[] }>(`/alerts?${q}`);
  },
  alert: (id: number) => request<AlertDetail>(`/alerts/${id}`),
  alertGraph: (id: number) => request<{ nodes: GraphNode[]; edges: GraphEdge[] }>(`/alerts/${id}/graph`),
  feedback: (id: number, label: "fraud" | "legit", notes: string) =>
    request<{ status: AlertStatus; feedback_since_last_train: number }>(`/alerts/${id}/feedback`, {
      method: "POST",
      body: JSON.stringify({ label, notes }),
    }),
  models: () =>
    request<{
      versions: ModelVersion[];
      feedback_since_last_train: number;
      retrain: { running: boolean; error: string | null; version: number | null };
    }>("/models"),
  retrain: () => request("/models/retrain", { method: "POST" }),
  simulate: (scenario: Scenario) =>
    request<{ scenario: string; transactions: number; alerts: number; alert_ids: number[]; max_score: number }>(
      "/simulate",
      { method: "POST", body: JSON.stringify({ scenario }) },
    ),
  setStream: (enabled: boolean) =>
    request<{ enabled: boolean }>("/stream", { method: "POST", body: JSON.stringify({ enabled }) }),
};

export const fmtMoney = (v: number) =>
  v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: v >= 1000 ? 0 : 2 });
export const fmtPct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;
export const fmtTime = (iso: string) =>
  new Date(iso + (iso.endsWith("Z") ? "" : "Z")).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
export const titleCase = (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export const PATTERN_LABEL: Record<string, string> = {
  ring: "Fraud ring",
  ato: "Account takeover",
  card_testing: "Card testing",
  legit: "Legitimate",
  unknown: "Unknown",
};

export const MODEL_LABEL: Record<string, string> = {
  tabular_xgboost: "Tabular XGBoost (baseline)",
  graph_xgboost: "Graph-feature XGBoost",
  isolation_forest: "Isolation Forest",
  graphsage_gnn: "GraphSAGE GNN",
  ensemble: "SentinelGraph ensemble",
};
