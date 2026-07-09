"""
Named experimental scenarios for the research platform.

Each scenario is a *distribution* over the DecisionContext feature space.
Running the same policy across different scenarios lets us tell a much
richer story than a single "random mix":

  "Under Network Failure, Q-Learning maintained 95% success while
   Rule-Based fell to 40%..."

The user selects one scenario per experiment. Multiple seeds sample DIFFERENT
context vectors from the SAME scenario distribution — this is what powers
the confidence intervals.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Callable

from decision_engine import DecisionContext


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    icon: str  # phosphor icon name for the UI
    sampler: Callable[[random.Random], DecisionContext]

    def sample(self, rng: random.Random) -> DecisionContext:
        return self.sampler(rng)

    def as_meta(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "description": self.description, "icon": self.icon}


# ----- Sampler helpers -----
def _factory_normal(rng: random.Random) -> DecisionContext:
    return DecisionContext(
        network_latency_ms=rng.uniform(40, 180),
        connectivity=1,
        cpu_available=rng.uniform(40, 85),
        memory_available=rng.uniform(45, 85),
        batch_size=rng.choice([1, 2, 4, 8]),
        priority=rng.choice([2, 2, 3, 3, 4]),
        cost_budget_usd=rng.choice([0.5, 1.0, 2.0, 5.0]),
        model_size_mb=rng.choice([6.2, 9.4, 15.1]),
    )


def _network_failure(rng: random.Random) -> DecisionContext:
    # 45% offline, otherwise very high latency
    offline = rng.random() < 0.45
    return DecisionContext(
        network_latency_ms=9999.0 if offline else rng.uniform(400, 900),
        connectivity=0 if offline else 1,
        cpu_available=rng.uniform(30, 80),
        memory_available=rng.uniform(30, 80),
        batch_size=rng.choice([1, 1, 2, 4]),
        priority=rng.choice([2, 3, 3, 4, 5]),
        cost_budget_usd=rng.choice([0.5, 1.0, 2.0]),
        model_size_mb=rng.choice([6.2, 9.4, 15.1, 22.5]),
    )


def _high_cpu_load(rng: random.Random) -> DecisionContext:
    return DecisionContext(
        network_latency_ms=rng.uniform(50, 200),
        connectivity=1,
        cpu_available=rng.uniform(3, 25),
        memory_available=rng.uniform(15, 40),
        batch_size=rng.choice([4, 8, 16]),
        priority=rng.choice([2, 3, 3, 4]),
        cost_budget_usd=rng.choice([0.5, 1.0, 2.0, 5.0]),
        model_size_mb=rng.choice([9.4, 15.1, 22.5]),
    )


def _large_model(rng: random.Random) -> DecisionContext:
    return DecisionContext(
        network_latency_ms=rng.uniform(60, 250),
        connectivity=1,
        cpu_available=rng.uniform(20, 70),
        memory_available=rng.uniform(20, 70),
        batch_size=rng.choice([8, 16, 32]),
        priority=rng.choice([2, 3, 4]),
        cost_budget_usd=rng.choice([1.0, 2.0, 5.0]),
        model_size_mb=rng.choice([22.5, 22.5, 40.0]),
    )


def _cloud_congestion(rng: random.Random) -> DecisionContext:
    return DecisionContext(
        network_latency_ms=rng.uniform(350, 800),
        connectivity=1,
        cpu_available=rng.uniform(40, 90),
        memory_available=rng.uniform(45, 90),
        batch_size=rng.choice([1, 2, 4, 8]),
        priority=rng.choice([2, 3, 3, 4, 5]),
        cost_budget_usd=rng.choice([0.5, 1.0, 2.0]),
        model_size_mb=rng.choice([6.2, 9.4, 15.1]),
    )


def _low_bandwidth(rng: random.Random) -> DecisionContext:
    return DecisionContext(
        network_latency_ms=rng.uniform(200, 500),
        connectivity=1,
        cpu_available=rng.uniform(35, 75),
        memory_available=rng.uniform(40, 75),
        batch_size=rng.choice([1, 1, 2]),  # tight bandwidth -> small batches
        priority=rng.choice([2, 3, 3, 4]),
        cost_budget_usd=rng.choice([0.5, 1.0, 2.0]),
        model_size_mb=rng.choice([6.2, 9.4]),
    )


def _high_priority(rng: random.Random) -> DecisionContext:
    return DecisionContext(
        network_latency_ms=rng.uniform(30, 150),
        connectivity=1,
        cpu_available=rng.uniform(25, 80),
        memory_available=rng.uniform(30, 80),
        batch_size=rng.choice([1, 2, 4, 8]),
        priority=rng.choice([4, 5, 5]),
        cost_budget_usd=rng.choice([2.0, 5.0, 10.0]),
        model_size_mb=rng.choice([9.4, 15.1, 22.5]),
    )


def _mixed(rng: random.Random) -> DecisionContext:
    connectivity = 1 if rng.random() > 0.15 else 0
    return DecisionContext(
        network_latency_ms=(rng.uniform(20, 500) if connectivity else 9999.0),
        connectivity=connectivity,
        cpu_available=rng.uniform(5, 95),
        memory_available=rng.uniform(10, 95),
        batch_size=rng.choice([1, 1, 2, 4, 8, 16, 32]),
        priority=rng.choice([1, 2, 2, 3, 3, 4, 5]),
        cost_budget_usd=rng.choice([0.0, 0.005, 0.05, 0.5, 1.0, 5.0]),
        model_size_mb=rng.choice([6.2, 9.4, 15.1, 22.5, 40.0]),
    )


SCENARIOS: Dict[str, Scenario] = {
    "mixed": Scenario(
        id="mixed",
        name="Mixed Traffic",
        description="Broad random mix. Baseline / stress-test — the default legacy behaviour.",
        icon="Shuffle",
        sampler=_mixed,
    ),
    "factory_normal": Scenario(
        id="factory_normal",
        name="Factory Normal",
        description="Balanced workload, network healthy, moderate CPU/memory. Baseline operating condition.",
        icon="Factory",
        sampler=_factory_normal,
    ),
    "network_failure": Scenario(
        id="network_failure",
        name="Network Failure",
        description="Frequent disconnections (~45% offline) and very high latency otherwise. Tests resilience.",
        icon="WifiSlash",
        sampler=_network_failure,
    ),
    "high_cpu_load": Scenario(
        id="high_cpu_load",
        name="High CPU Load",
        description="Edge devices saturated (CPU 3–25%, mem 15–40%). Forces offloading decisions.",
        icon="Cpu",
        sampler=_high_cpu_load,
    ),
    "large_model": Scenario(
        id="large_model",
        name="Large Model",
        description="Large models (22.5–40 MB) with large batches (8–32). Tests heavy workload routing.",
        icon="Package",
        sampler=_large_model,
    ),
    "cloud_congestion": Scenario(
        id="cloud_congestion",
        name="Cloud Congestion",
        description="Network reachable but slow (RTT 350–800 ms). Cloud becomes an expensive option.",
        icon="CloudSlash",
        sampler=_cloud_congestion,
    ),
    "low_bandwidth": Scenario(
        id="low_bandwidth",
        name="Low Bandwidth",
        description="Moderate-high latency and small batches. Bandwidth-constrained conditions.",
        icon="CellSignalLow",
        sampler=_low_bandwidth,
    ),
    "high_priority": Scenario(
        id="high_priority",
        name="High Priority",
        description="Only priority 4–5 requests with generous cost budgets. Latency-critical missions.",
        icon="Star",
        sampler=_high_priority,
    ),
}


def get(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_id}")
    return SCENARIOS[scenario_id]


def list_scenarios() -> List[Dict[str, Any]]:
    return [s.as_meta() for s in SCENARIOS.values()]


def sample_scenario(scenario_id: str, n: int, seed: int) -> List[DecisionContext]:
    rng = random.Random(seed)
    scen = get(scenario_id)
    return [scen.sample(rng) for _ in range(n)]
