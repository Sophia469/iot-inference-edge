"""
Benchmark harness tests for the Comparative Experiment feature (iteration 3).
Covers:
- POST /api/benchmark/run — 4-engine summary structure
- GET  /api/benchmark/last — idempotent read
- GET  /api/benchmark/artefact/{name} — PNG/CSV/JSON allow-list + malicious 404
"""
import os
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
if not BASE_URL:
    from pathlib import Path
    envp = Path("/app/frontend/.env")
    if envp.exists():
        for line in envp.read_text().splitlines():
            if line.startswith("REACT_APP_BACKEND_URL="):
                BASE_URL = line.split("=", 1)[1].strip().rstrip("/")
                break

API = f"{BASE_URL}/api"

REQUIRED_FIELDS = {
    "n", "success_rate",
    "latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
    "cost_usd_total", "decision_time_us_mean",
    "reward_mean", "agreement_with_rule", "route_distribution",
}
ENGINES = {"RuleBased", "DecisionTree", "RandomForest", "QLearning"}


@pytest.fixture(scope="session")
def s():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@pytest.fixture(scope="session")
def bench_result(s):
    """Run one benchmark shared across all tests to keep suite fast."""
    r = s.post(
        f"{API}/benchmark/run",
        json={"n_scenarios": 100, "seed": 2026, "save_artefacts": True},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    return r.json()


# --------------------- POST /api/benchmark/run ---------------------
class TestBenchmarkRun:
    def test_status_and_top_level_fields(self, bench_result):
        assert "summary" in bench_result
        assert "meta" in bench_result
        assert bench_result["meta"]["n_scenarios"] == 100
        assert bench_result["meta"]["seed"] == 2026
        # 4 engines * 100 scenarios = 400 records
        assert bench_result["n_records"] == 400

    def test_summary_has_four_engines(self, bench_result):
        summary = bench_result["summary"]
        assert set(summary.keys()) == ENGINES, f"missing engines: {ENGINES - set(summary.keys())}"

    @pytest.mark.parametrize("engine", sorted(ENGINES))
    def test_engine_summary_has_all_required_fields(self, bench_result, engine):
        s_ = bench_result["summary"][engine]
        missing = REQUIRED_FIELDS - set(s_.keys())
        assert not missing, f"{engine} missing fields: {missing}"
        assert s_["n"] == 100
        assert 0.0 <= s_["success_rate"] <= 1.0
        assert 0.0 <= s_["agreement_with_rule"] <= 1.0
        assert isinstance(s_["route_distribution"], dict)
        assert set(s_["route_distribution"].keys()) == {"edge", "cloud", "hybrid"}
        assert sum(s_["route_distribution"].values()) == 100

    def test_rule_agreement_is_one(self, bench_result):
        # Rule-based agrees with itself
        assert bench_result["summary"]["RuleBased"]["agreement_with_rule"] == 1.0

    def test_decision_tree_agreement_high(self, bench_result):
        # DT was trained on rule labels — should mostly agree
        agree = bench_result["summary"]["DecisionTree"]["agreement_with_rule"]
        assert agree > 0.85, f"DT agreement too low: {agree}"

    def test_qlearning_agreement_can_vary(self, bench_result):
        # QL learns from reward — legitimately may disagree; assert reasonable band
        agree = bench_result["summary"]["QLearning"]["agreement_with_rule"]
        assert 0.0 <= agree <= 1.0
        # No hard lower bound (research finding), but should be numeric
        assert isinstance(agree, (int, float))


# --------------------- GET /api/benchmark/last ---------------------
class TestBenchmarkLast:
    def test_last_returns_previous_summary(self, s, bench_result):
        r = s.get(f"{API}/benchmark/last", timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert data["summary"] is not None
        assert set(data["summary"].keys()) == ENGINES
        # Idempotent — matches most recent run
        assert data["meta"]["n_scenarios"] == bench_result["meta"]["n_scenarios"]
        assert data["meta"]["seed"] == bench_result["meta"]["seed"]

    def test_last_is_idempotent(self, s):
        r1 = s.get(f"{API}/benchmark/last", timeout=15).json()
        r2 = s.get(f"{API}/benchmark/last", timeout=15).json()
        assert r1["meta"] == r2["meta"]


# --------------------- GET /api/benchmark/artefact/{name} ---------------------
CHART_NAMES = [
    "chart_latency_boxplot.png",
    "chart_decision_time.png",
    "chart_cost.png",
    "chart_route_distribution.png",
    "chart_reward.png",
]


class TestBenchmarkArtefacts:
    @pytest.mark.parametrize("chart", CHART_NAMES)
    def test_png_chart_returns_png(self, s, bench_result, chart):
        r = s.get(f"{API}/benchmark/artefact/{chart}", timeout=15)
        assert r.status_code == 200, f"{chart}: {r.text[:200]}"
        assert r.headers.get("content-type", "").startswith("image/png"), \
            f"{chart} content-type: {r.headers.get('content-type')}"
        assert len(r.content) > 1000, f"{chart} suspiciously small: {len(r.content)} bytes"
        # PNG magic number
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n", f"{chart} not a valid PNG"

    def test_csv_artefact(self, s, bench_result):
        r = s.get(f"{API}/benchmark/artefact/benchmark_records.csv", timeout=15)
        assert r.status_code == 200
        text = r.text
        first_line = text.splitlines()[0]
        for col in ("engine", "scenario_id", "route",
                    "decision_time_us", "latency_ms", "cost_usd"):
            assert col in first_line, f"CSV header missing '{col}': {first_line}"
        # 4 engines * 100 scenarios + header
        assert len(text.splitlines()) == 401

    def test_json_summary_artefact(self, s, bench_result):
        r = s.get(f"{API}/benchmark/artefact/benchmark_summary.json", timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert set(data.keys()) == ENGINES

    def test_malicious_artefact_returns_404(self, s):
        # Not in allow-list
        r = s.get(f"{API}/benchmark/artefact/malicious.png", timeout=15)
        assert r.status_code == 404

    def test_path_traversal_returns_404(self, s):
        # allow-list should block traversal attempts
        r = s.get(f"{API}/benchmark/artefact/..%2Fetc%2Fpasswd", timeout=15)
        assert r.status_code == 404
