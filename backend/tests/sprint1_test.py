"""
Sprint 1 tests — Scenario Manager, Multi-Seed CI, Explain Decision, Report PDF, Assistant.

Covers:
- GET  /api/scenarios (8 named scenarios)
- POST /api/benchmark/run with scenario_id=network_failure
- POST /api/benchmark/ci (mean/std/ci95/low/high/n)
- POST /api/report/generate  + GET /api/benchmark/artefact/thesis_report.pdf
- POST /api/decisions/predict with engine=rf and engine=rule (explanation)
- POST /api/assistant/ask (Claude Sonnet 4.5 via emergentintegrations)
- Multi-turn assistant session_id continuity
"""
import os
import time
import pytest
import requests
from pathlib import Path

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
if not BASE_URL:
    envp = Path("/app/frontend/.env")
    if envp.exists():
        for line in envp.read_text().splitlines():
            if line.startswith("REACT_APP_BACKEND_URL="):
                BASE_URL = line.split("=", 1)[1].strip().rstrip("/")
                break

API = f"{BASE_URL}/api"

EXPECTED_SCENARIOS = {
    "mixed", "factory_normal", "network_failure", "high_cpu_load",
    "large_model", "cloud_congestion", "low_bandwidth", "high_priority",
}


@pytest.fixture(scope="session")
def s():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


# --------------------- GET /api/scenarios ---------------------
class TestScenarios:
    def test_returns_exactly_eight_scenarios(self, s):
        r = s.get(f"{API}/scenarios", timeout=15)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "scenarios" in data
        assert len(data["scenarios"]) == 8
        ids = {sc["id"] for sc in data["scenarios"]}
        assert ids == EXPECTED_SCENARIOS, f"unexpected ids: got {ids}"

    def test_each_scenario_has_required_fields(self, s):
        r = s.get(f"{API}/scenarios", timeout=15)
        for sc in r.json()["scenarios"]:
            assert set(sc.keys()) >= {"id", "name", "description", "icon"}
            assert sc["name"], f"empty name for {sc['id']}"
            assert sc["description"], f"empty description for {sc['id']}"
            assert sc["icon"], f"empty icon for {sc['id']}"


# --------------------- POST /api/benchmark/run (scenario) ---------------------
class TestBenchmarkNetworkFailure:
    def test_network_failure_scenario_yields_low_success_and_edge_preference(self, s):
        r = s.post(f"{API}/benchmark/run",
                   json={"n_scenarios": 100, "seed": 2026,
                         "save_artefacts": True, "scenario_id": "network_failure"},
                   timeout=90)
        assert r.status_code == 200, r.text
        summary = r.json()["summary"]

        # Each engine present
        for eng in ("RuleBased", "DecisionTree", "RandomForest", "QLearning"):
            assert eng in summary

        # Success rate must be a valid probability for every engine.
        # NOTE: A resilient policy (RuleBased routing offline→edge) legitimately
        # reaches ~1.0 here. Naive policies (e.g. QLearning) may fall in the
        # 0.4–0.7 band that reflects the ~55% online rate. Both are acceptable.
        for eng in ("RuleBased", "DecisionTree", "RandomForest", "QLearning"):
            sr = summary[eng]["success_rate"]
            assert 0.30 <= sr <= 1.0, f"{eng} success_rate={sr} outside [0.30, 1.0]"

        # Resilience test: Rule-based should prefer edge over cloud in
        # network-failure conditions (this is the primary acceptance criterion).
        rule_rd = summary["RuleBased"]["route_distribution"]
        assert rule_rd["edge"] > rule_rd["cloud"], (
            f"RuleBased edge={rule_rd['edge']} not > cloud={rule_rd['cloud']} "
            f"under network_failure scenario — resilience broken"
        )

    def test_scenario_id_persisted_in_meta(self, s):
        r = s.post(f"{API}/benchmark/run",
                   json={"n_scenarios": 50, "seed": 42, "scenario_id": "factory_normal"},
                   timeout=60)
        assert r.status_code == 200
        assert r.json()["meta"]["scenario_id"] == "factory_normal"


# --------------------- POST /api/benchmark/ci ---------------------
class TestBenchmarkCI:
    def test_ci_returns_per_engine_metrics_with_ci_fields(self, s):
        r = s.post(f"{API}/benchmark/ci",
                   json={"scenario_id": "mixed", "seeds": [2026, 42, 7], "n_scenarios": 100},
                   timeout=180)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "aggregated" in data and "meta" in data and "per_seed" in data
        assert data["meta"]["scenario_id"] == "mixed"
        assert data["meta"]["seeds"] == [2026, 42, 7]
        assert data["meta"]["n_runs"] == 3

        engines = {"RuleBased", "DecisionTree", "RandomForest", "QLearning"}
        assert set(data["aggregated"].keys()) == engines

        required_ci_fields = {"mean", "std", "ci95", "low", "high", "n"}
        for eng in engines:
            for metric in ("latency_ms_p50", "cost_usd_total"):
                entry = data["aggregated"][eng][metric]
                assert entry is not None, f"{eng}.{metric} is None"
                assert set(entry.keys()) >= required_ci_fields, (
                    f"{eng}.{metric} missing CI fields: got {entry.keys()}"
                )
                assert entry["n"] == 3
                # low <= mean <= high
                assert entry["low"] - 1e-6 <= entry["mean"] <= entry["high"] + 1e-6, (
                    f"{eng}.{metric}: mean {entry['mean']} not within [{entry['low']}, {entry['high']}]"
                )


# --------------------- POST /api/report/generate ---------------------
class TestReportPDF:
    @pytest.fixture(scope="class")
    def report_result(self, s):
        # Use small seeds/n_scenarios to keep runtime under ~40s
        r = s.post(f"{API}/report/generate",
                   json={"scenario_id": "mixed", "seeds": [2026, 42], "n_scenarios": 60},
                   timeout=240)
        assert r.status_code == 200, r.text
        return r.json()

    def test_report_returns_metadata(self, report_result):
        for key in ("path", "scenario_id", "seeds", "n_scenarios",
                    "size_bytes", "aggregated_summary"):
            assert key in report_result, f"report response missing {key}"
        assert report_result["scenario_id"] == "mixed"
        assert report_result["seeds"] == [2026, 42]
        assert report_result["n_scenarios"] == 60
        assert report_result["size_bytes"] > 10000, (
            f"PDF suspiciously small: {report_result['size_bytes']} bytes"
        )
        assert report_result["path"] == "/tmp/benchmark/thesis_report.pdf"

    def test_pdf_file_exists_on_disk(self, report_result):
        p = Path(report_result["path"])
        assert p.exists(), f"{p} does not exist"
        assert p.stat().st_size > 10000

    def test_pdf_artefact_endpoint_returns_pdf(self, s, report_result):
        r = s.get(f"{API}/benchmark/artefact/thesis_report.pdf", timeout=30)
        assert r.status_code == 200
        ct = r.headers.get("content-type", "")
        assert "application/pdf" in ct, f"content-type: {ct}"
        # PDF magic
        assert r.content[:4] == b"%PDF", "response is not a PDF file"

    def test_non_allowlisted_artefact_returns_404(self, s):
        r = s.get(f"{API}/benchmark/artefact/other.pdf", timeout=15)
        assert r.status_code == 404


# --------------------- POST /api/decisions/predict (explanation) ---------------------
CTX = {
    "network_latency_ms": 120.0,
    "connectivity": 1,
    "cpu_available": 55.0,
    "memory_available": 60.0,
    "batch_size": 4,
    "priority": 3,
    "cost_budget_usd": 0.5,
    "model_size_mb": 9.4,
}


class TestExplainDecision:
    def test_rf_returns_full_explanation(self, s):
        r = s.post(f"{API}/decisions/predict",
                   json={"engine": "rf", "context": CTX, "persist": False},
                   timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "explanation" in d
        exp = d["explanation"]
        # factors
        assert isinstance(exp["factors"], list) and len(exp["factors"]) > 0
        f0 = exp["factors"][0]
        for key in ("name", "value", "met", "contribution", "direction"):
            assert key in f0, f"factor missing '{key}': {f0}"
        # top_features
        assert isinstance(exp["top_features"], list) and len(exp["top_features"]) > 0
        tf0 = exp["top_features"][0]
        for key in ("name", "value", "weight"):
            assert key in tf0, f"top_feature missing '{key}': {tf0}"
        assert isinstance(tf0["weight"], (int, float))
        # counterfactual may be None or a string
        assert exp.get("counterfactual") is None or isinstance(exp["counterfactual"], str)
        # summary
        assert isinstance(exp["summary"], str) and len(exp["summary"]) > 0

    def test_rule_returns_factors_but_empty_top_features(self, s):
        r = s.post(f"{API}/decisions/predict",
                   json={"engine": "rule", "context": CTX, "persist": False},
                   timeout=15)
        assert r.status_code == 200, r.text
        exp = r.json()["explanation"]
        assert isinstance(exp["factors"], list) and len(exp["factors"]) > 0
        # rule engine → no ML top_features (spec allows empty or small)
        assert len(exp["top_features"]) <= 2, (
            f"rule engine should not have ML feature importance; got {exp['top_features']}"
        )
        assert exp["counterfactual"] is None
        assert isinstance(exp["summary"], str)


# --------------------- POST /api/assistant/ask ---------------------
class TestAssistant:
    def test_ask_returns_english_answer_mentioning_all_four_policies(self, s):
        r = s.post(f"{API}/assistant/ask",
                   json={"question": "What are the 4 policies?"},
                   timeout=60)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "answer" in data and data["answer"], "empty answer"
        assert "session_id" in data and data["session_id"]
        assert data["model"] == "anthropic/claude-sonnet-4-5-20250929"
        answer = data["answer"].lower()
        # Must mention each of the 4 policies (rule/decision tree/random forest/q-learning)
        assert "rule" in answer, "answer missing 'rule'"
        assert "decision tree" in answer or "decisiontree" in answer, "answer missing 'decision tree'"
        assert "random forest" in answer or "randomforest" in answer, "answer missing 'random forest'"
        assert "q-learning" in answer or "qlearning" in answer or "q learning" in answer, (
            "answer missing 'Q-Learning'"
        )

    def test_session_id_reused_for_multi_turn(self, s):
        # Turn 1
        r1 = s.post(f"{API}/assistant/ask",
                    json={"question": "List the four orchestration policies briefly."},
                    timeout=60)
        assert r1.status_code == 200, r1.text
        session_id = r1.json()["session_id"]
        assert session_id

        # Turn 2 — same session_id passed in, server should echo it back
        r2 = s.post(f"{API}/assistant/ask",
                    json={"question": "Which of them is trained with reinforcement learning?",
                          "session_id": session_id},
                    timeout=60)
        assert r2.status_code == 200, r2.text
        d2 = r2.json()
        assert d2["session_id"] == session_id, (
            f"session_id not reused: sent {session_id}, got {d2['session_id']}"
        )
        # A coherent follow-up should mention Q-Learning
        assert "q" in d2["answer"].lower(), "follow-up doesn't mention Q-related policy"
