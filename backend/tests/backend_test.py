"""
Backend regression + Q-Learning + Real Inference tests for
Edge-Cloud AI Orchestration research prototype (iteration 2).
"""
import os
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
if not BASE_URL:
    # fallback: read frontend/.env manually
    from pathlib import Path
    envp = Path("/app/frontend/.env")
    if envp.exists():
        for line in envp.read_text().splitlines():
            if line.startswith("REACT_APP_BACKEND_URL="):
                BASE_URL = line.split("=", 1)[1].strip().rstrip("/")
                break

API = f"{BASE_URL}/api"


@pytest.fixture(scope="session")
def s():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


# ------------------------- Decision Engine status ---------------------------
class TestDecisionStatus:
    def test_status_has_four_engines_trained(self, s):
        r = s.get(f"{API}/decisions/status", timeout=30)
        assert r.status_code == 200
        data = r.json()
        for k in ("RuleBased", "DecisionTree", "RandomForest", "QLearning"):
            assert k in data, f"engine {k} missing"
            assert data[k]["trained"] is True, f"{k} not trained"

    def test_qlearning_metrics_shape(self, s):
        r = s.get(f"{API}/decisions/status", timeout=15)
        m = r.json()["QLearning"]["metrics"]
        assert m["n_states"] == 108
        assert m["alpha"] == 0.15
        assert m["gamma"] == 0.9
        assert "episodes" in m
        assert "mean_reward_last_500" in m
        assert isinstance(m["mean_reward_last_500"], (int, float))


# --------------------------- QLearning predict ------------------------------
CTX_ONLINE = {
    "network_latency_ms": 120.0,
    "connectivity": 1,
    "cpu_available": 55.0,
    "memory_available": 60.0,
    "batch_size": 1,
    "priority": 3,
    "cost_budget_usd": 0.08,
    "model_size_mb": 6.2,
}
CTX_OFFLINE = {**CTX_ONLINE, "connectivity": 0}


class TestQLearningPredict:
    def test_predict_ql_returns_valid_decision(self, s):
        r = s.post(f"{API}/decisions/predict",
                   json={"engine": "ql", "context": CTX_ONLINE, "persist": False},
                   timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["route"] in ("edge", "cloud", "hybrid")
        assert 0.0 <= d["confidence"] <= 1.0
        assert isinstance(d["probabilities"], dict)
        s_prob = sum(d["probabilities"].values())
        assert abs(s_prob - 1.0) < 1e-3, f"probs must sum to 1, got {s_prob}"
        assert "state=" in d["reason"]
        assert "Q=[" in d["reason"]
        assert d["latency_us"] > 0
        assert d["engine"] == "QLearning"

    def test_predict_ql_offline_routes_to_edge(self, s):
        r = s.post(f"{API}/decisions/predict",
                   json={"engine": "ql", "context": CTX_OFFLINE, "persist": False},
                   timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["route"] == "edge", (
            f"Offline context should route to edge (learned via -50 penalty); "
            f"got route={d['route']} reason={d['reason']}"
        )


# --------------------------- QLearning train --------------------------------
class TestQLearningTrain:
    def test_qlearning_train_updates_metrics(self, s):
        # capture baseline
        r0 = s.get(f"{API}/decisions/status", timeout=15).json()
        _ = r0["QLearning"]["metrics"]["mean_reward_last_500"]

        r = s.post(f"{API}/decisions/qlearning/train",
                   json={"episodes": 3000}, timeout=120)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        m = body["trained"]
        assert m["episodes"] == 3000
        assert "mean_reward_last_500" in m
        # verify status reflects updated metrics
        r2 = s.get(f"{API}/decisions/status", timeout=15).json()
        new_mean = r2["QLearning"]["metrics"]["mean_reward_last_500"]
        # metric field should be present; usually will differ, but at least reachable
        assert isinstance(new_mean, (int, float))
        assert r2["QLearning"]["metrics"]["episodes"] == 3000

    def test_qtable_endpoint(self, s):
        r = s.get(f"{API}/decisions/qlearning/qtable", params={"top_states": 5}, timeout=15)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["trained"] is True
        rows = body["rows"]
        assert len(rows) == 5
        for row in rows:
            assert "state" in row and isinstance(row["state"], int)
            assert "q_values" in row and isinstance(row["q_values"], list)
            assert len(row["q_values"]) == 3
            assert all(isinstance(v, (int, float)) for v in row["q_values"])
            assert row["best_route"] in ("edge", "cloud", "hybrid")
            assert "max_q" in row


# --------------------------- Real Inference ---------------------------------
class TestRealInference:
    def test_edge_real_inference(self, s):
        r = s.post(f"{API}/inference/run",
                   json={"mode": "edge", "model_size_mb": 6.2, "batch_size": 1},
                   timeout=30)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["success"] is True
        assert d["mode"] == "edge"
        assert d["latency_ms"] > 0
        assert d["memory_mb"] > 0
        assert d["workload_flops"] > 0
        assert "cpu_percent" in d

    def test_cloud_real_inference_respects_network_latency(self, s):
        r = s.post(f"{API}/inference/run",
                   json={"mode": "cloud", "model_size_mb": 22.5,
                         "network_latency_ms": 150, "connectivity": 1},
                   timeout=30)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["success"] is True
        assert d["mode"] == "cloud"
        assert d["latency_ms"] >= 150, f"expected >=150 (real sleep), got {d['latency_ms']}"

    def test_cloud_offline_fails_cleanly(self, s):
        r = s.post(f"{API}/inference/run",
                   json={"mode": "cloud", "model_size_mb": 6.2,
                         "network_latency_ms": 100, "connectivity": 0},
                   timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["success"] is False
        assert "offline" in d["detail"].lower()

    def test_hybrid_mode(self, s):
        r = s.post(f"{API}/inference/run",
                   json={"mode": "hybrid", "model_size_mb": 6.2, "batch_size": 1},
                   timeout=30)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["mode"] == "hybrid"
        assert "cloud audit-replicate" in d["detail"]
        assert d["success"] is True


# --------------------------- Regression: legacy endpoints -------------------
class TestLegacyEndpoints:
    def test_devices(self, s):
        r = s.get(f"{API}/devices", timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list) and len(data) >= 4

    def test_models(self, s):
        r = s.get(f"{API}/models", timeout=15)
        assert r.status_code == 200
        data = r.json()
        names = {m["name"] for m in data}
        assert {"YOLOv8n", "YOLOv8s"}.issubset(names)

    def test_deployments_list(self, s):
        r = s.get(f"{API}/deployments", timeout=15)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_metrics_summary(self, s):
        r = s.get(f"{API}/metrics/summary", timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert "edge" in data and "cloud" in data

    def test_inferences_ingest(self, s):
        payload = {
            "records": [{
                "device_id": "test-dev", "device_name": "TEST_dev",
                "mode": "edge", "latency_ms": 30.0, "cpu": 40.0,
                "memory": 50.0, "fps": 25.0, "cost_usd": 0.0001,
                "objects_detected": 2, "top_label": "person",
                "top_confidence": 0.9, "connected": True,
            }]
        }
        r = s.post(f"{API}/inferences", json=payload, timeout=15)
        assert r.status_code == 200
        assert r.json()["inserted"] == 1
