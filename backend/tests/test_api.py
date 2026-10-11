def test_health(client):
    r = client.get("/api/health").json()
    assert r["engine_ready"] and r["model_version"] == 1


def test_model_beats_tabular_baseline(client):
    v = client.get("/api/models").json()["versions"][0]
    models = v["metrics"]["models"]
    assert models["graph_xgboost"]["pr_auc"] >= models["tabular_xgboost"]["pr_auc"]
    assert v["feature_importance"]


def test_overview(client):
    r = client.get("/api/overview").json()
    assert r["kpis"]["transactions"] > 1000
    assert len(r["score_histogram"]) == 20


def test_alert_investigation_flow(client):
    alerts = client.get("/api/alerts", params={"status": "open", "sort": "risk", "limit": 5}).json()
    assert alerts["total"] > 0
    alert_id = alerts["items"][0]["id"]

    detail = client.get(f"/api/alerts/{alert_id}").json()
    assert detail["factors"] and detail["narrative"] and detail["timeline"]
    assert all({"description", "contribution", "group"} <= set(f) for f in detail["factors"])

    graph = client.get(f"/api/alerts/{alert_id}/graph").json()
    assert any(n["focus"] for n in graph["nodes"])

    fb = client.post(f"/api/alerts/{alert_id}/feedback", json={"label": "fraud", "notes": "test"}).json()
    assert fb["status"] == "confirmed_fraud"


def test_simulated_ring_is_detected(client):
    r = client.post("/api/simulate", json={"scenario": "ring"}).json()
    assert r["transactions"] > 0
    assert r["alerts"] / r["transactions"] > 0.5


def test_realtime_scoring_endpoint(client):
    r = client.post("/api/transactions", json={
        "account_id": "acc_test_new", "amount": 42.5, "merchant_category": "grocery", "device_id": "dev_test",
        "ip": "10.0.0.1", "ip_country": "US", "card_id": "card_test", "address_id": "addr_test",
    }).json()
    assert 0 <= r["risk_score"] <= 1
    assert r["decision"] in {"approve", "review"}


def test_validation_errors(client):
    assert client.post("/api/simulate", json={"scenario": "nope"}).status_code == 422
    assert client.get("/api/alerts/999999").status_code == 404
