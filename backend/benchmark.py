"""
Benchmark harness — the research instrument for the thesis.

Runs each of the 4 orchestration policies (Rule, DecisionTree, RandomForest,
Q-Learning) under identical, reproducible workloads and measures:

    - decision_time_us     : time the policy took to choose a route (µs)
    - inferred_latency_ms  : simulated end-to-end latency of the chosen route
    - cost_usd             : simulated cost per inference
    - route_taken          : edge/cloud/hybrid
    - success              : whether the route was executable in that context
    - reward               : combined objective (lower latency + lower cost + success)
    - agreement_with_rule  : whether the ML policy matched the rule-based baseline

Same context vectors are fed to all engines (paired-samples design).
Output: pandas.DataFrame (per-sample) + aggregated summary + PNG charts.

Usage as a script:
    python benchmark.py --scenarios 500 --output /tmp/benchmark

Usage as a library (from the API):
    from benchmark import run_benchmark
    result = run_benchmark(n_scenarios=500)
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Matplotlib is imported lazily inside `save_charts` so the module can still be
# used purely for CSV/JSON output without pulling the plotting backend.

from decision_engine import (
    registry as decision_registry,
    rule_based_decide,
    DecisionContext,
    ROUTES,
)
from qlearning_agent import qlearning_agent, simulate_outcome, ROUTES as Q_ROUTES


ENGINES = ["rule", "dt", "rf", "ql"]
ENGINE_LABELS = {"rule": "RuleBased", "dt": "DecisionTree", "rf": "RandomForest", "ql": "QLearning"}


def _sample_scenarios(n: int, seed: int, scenario_id: str = "mixed") -> List[DecisionContext]:
    """Deterministic scenario generator. Uses a named scenario distribution."""
    from scenarios import sample_scenario
    return sample_scenario(scenario_id, n, seed)


def _ensure_trained() -> None:
    if not decision_registry.dt.trained or not decision_registry.rf.trained:
        decision_registry.train_all(n=3000)
    if not qlearning_agent.trained:
        qlearning_agent.train(episodes=5000)


def run_benchmark(n_scenarios: int = 500, seed: int = 2026,
                  output_dir: str | None = None,
                  scenario_id: str = "mixed") -> Dict[str, Any]:
    """Executes the benchmark suite and returns a full result dict."""
    _ensure_trained()
    scenarios = _sample_scenarios(n_scenarios, seed=seed, scenario_id=scenario_id)

    # Baseline (rule) route per scenario, used for agreement metric
    baseline_routes = [rule_based_decide(s).route for s in scenarios]

    records = []
    for engine in ENGINES:
        for i, ctx in enumerate(scenarios):
            t0 = time.perf_counter()
            decision = decision_registry.decide(ctx, engine)
            decision_time_us = (time.perf_counter() - t0) * 1_000_000

            # Simulate outcome to obtain reward + realised latency/cost
            action_idx = ROUTES.index(decision.route)
            reward, success = simulate_outcome(ctx, action_idx)

            # Derive latency/cost from the same environment model, seeded per scenario
            random.seed(seed * 7919 + i)  # deterministic outcome per scenario
            if decision.route == "edge":
                latency_ms = random.uniform(18, 42) + max(0, (80 - ctx.cpu_available)) * 0.3
                cost = 0.00002
            elif decision.route == "cloud":
                if ctx.connectivity == 0:
                    latency_ms = float("nan")
                    cost = 0.0
                else:
                    latency_ms = ctx.network_latency_ms + random.uniform(30, 120)
                    cost = 0.00085 * ctx.batch_size
            else:  # hybrid
                latency_ms = max(35, ctx.network_latency_ms * 0.5) + random.uniform(15, 50)
                cost = 0.0005 * ctx.batch_size

            records.append({
                "engine": ENGINE_LABELS[engine],
                "scenario_id": i,
                "route": decision.route,
                "decision_time_us": round(decision_time_us, 3),
                "latency_ms": round(latency_ms, 3) if not np.isnan(latency_ms) else None,
                "cost_usd": round(cost, 6),
                "success": bool(success),
                "reward": round(reward, 3),
                "agreement_with_rule": decision.route == baseline_routes[i],
                # Context features (for correlation analysis)
                "ctx_network_latency_ms": ctx.network_latency_ms,
                "ctx_connectivity": ctx.connectivity,
                "ctx_cpu_available": ctx.cpu_available,
                "ctx_priority": ctx.priority,
                "ctx_batch_size": ctx.batch_size,
                "ctx_cost_budget_usd": ctx.cost_budget_usd,
                "ctx_model_size_mb": ctx.model_size_mb,
            })

    df = pd.DataFrame.from_records(records)
    summary = _summarise(df)

    result = {
        "meta": {
            "n_scenarios": n_scenarios,
            "seed": seed,
            "scenario_id": scenario_id,
            "engines": [ENGINE_LABELS[e] for e in ENGINES],
            "generated_at": time.time(),
        },
        "summary": summary,
        "records": records,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "benchmark_records.csv", index=False)
        (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
        try:
            save_charts(df, out)
        except ImportError:
            pass

    return result


def _summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Per-engine aggregation."""
    out = {}
    for engine, sub in df.groupby("engine"):
        lat = sub["latency_ms"].dropna()
        successful = sub["success"].mean()
        out[engine] = {
            "n": int(len(sub)),
            "success_rate": round(float(successful), 4),
            "latency_ms_mean": round(float(lat.mean()), 2) if len(lat) else None,
            "latency_ms_p50":  round(float(lat.median()), 2) if len(lat) else None,
            "latency_ms_p95":  round(float(lat.quantile(0.95)), 2) if len(lat) else None,
            "cost_usd_total": round(float(sub["cost_usd"].sum()), 4),
            "cost_usd_mean":  round(float(sub["cost_usd"].mean()), 6),
            "decision_time_us_mean":  round(float(sub["decision_time_us"].mean()), 2),
            "decision_time_us_p95":   round(float(sub["decision_time_us"].quantile(0.95)), 2),
            "reward_mean":            round(float(sub["reward"].mean()), 3),
            "agreement_with_rule":    round(float(sub["agreement_with_rule"].mean()), 4),
            "route_distribution": {
                r: int((sub["route"] == r).sum()) for r in ROUTES
            },
        }
    return out


def save_charts(df: pd.DataFrame, output_dir: Path) -> None:
    """Renders comparative charts as PNG for the thesis appendix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    engines = sorted(df["engine"].unique().tolist())

    # 1. Latency box-plot
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df[df["engine"] == e]["latency_ms"].dropna().values for e in engines]
    ax.boxplot(data, labels=engines, showfliers=False)
    ax.set_title("End-to-End Latency by Policy")
    ax.set_ylabel("Latency (ms)")
    fig.tight_layout()
    fig.savefig(output_dir / "chart_latency_boxplot.png", dpi=140)
    plt.close(fig)

    # 2. Decision time bar
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [df[df["engine"] == e]["decision_time_us"].mean() for e in engines]
    p95s = [df[df["engine"] == e]["decision_time_us"].quantile(0.95) for e in engines]
    x = np.arange(len(engines))
    ax.bar(x - 0.2, means, width=0.4, label="mean")
    ax.bar(x + 0.2, p95s, width=0.4, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(engines)
    ax.set_title("Decision Latency by Policy (Overhead of the Orchestrator)")
    ax.set_ylabel("Decision time (µs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "chart_decision_time.png", dpi=140)
    plt.close(fig)

    # 3. Cost total
    fig, ax = plt.subplots(figsize=(8, 5))
    totals = [df[df["engine"] == e]["cost_usd"].sum() for e in engines]
    ax.bar(engines, totals, color=["#FFCC00", "#00E676", "#0055FF", "#FF3333"])
    ax.set_title("Total Simulated Cost (USD) — lower is better")
    ax.set_ylabel("Cost (USD)")
    fig.tight_layout()
    fig.savefig(output_dir / "chart_cost.png", dpi=140)
    plt.close(fig)

    # 4. Route distribution stacked
    fig, ax = plt.subplots(figsize=(8, 5))
    route_counts = {
        r: [(df[df["engine"] == e]["route"] == r).sum() for e in engines]
        for r in ROUTES
    }
    bottoms = np.zeros(len(engines))
    colors = {"edge": "#00E676", "cloud": "#0055FF", "hybrid": "#FFCC00"}
    for r in ROUTES:
        ax.bar(engines, route_counts[r], bottom=bottoms, label=r, color=colors[r])
        bottoms += np.array(route_counts[r])
    ax.set_title("Route Distribution per Policy")
    ax.set_ylabel("# of decisions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "chart_route_distribution.png", dpi=140)
    plt.close(fig)

    # 5. Reward distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    reward_data = [df[df["engine"] == e]["reward"].values for e in engines]
    ax.violinplot(reward_data, showmeans=True)
    ax.set_xticks(range(1, len(engines) + 1))
    ax.set_xticklabels(engines)
    ax.set_title("Reward Distribution per Policy — higher is better")
    ax.set_ylabel("Reward")
    fig.tight_layout()
    fig.savefig(output_dir / "chart_reward.png", dpi=140)
    plt.close(fig)


def run_benchmark_ci(scenario_id: str = "mixed",
                     seeds: List[int] | None = None,
                     n_scenarios: int = 200,
                     output_dir: str | None = None) -> Dict[str, Any]:
    """Runs the benchmark for each seed in `seeds` and aggregates statistics with
    95% confidence intervals (Student t-distribution) across seeds.
    """
    if seeds is None:
        seeds = [2026, 42, 7, 1234, 9999]
    per_seed = []
    for s in seeds:
        r = run_benchmark(n_scenarios=n_scenarios, seed=s, scenario_id=scenario_id,
                          output_dir=None)
        per_seed.append({"seed": s, "summary": r["summary"]})

    engines = sorted({e for r in per_seed for e in r["summary"].keys()})
    metrics_of_interest = [
        "latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
        "cost_usd_total", "cost_usd_mean",
        "decision_time_us_mean", "reward_mean",
        "agreement_with_rule", "success_rate",
    ]

    from math import sqrt
    T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086}
    dof = len(seeds) - 1
    t_crit = T_95.get(dof, 2.0)

    aggregated: Dict[str, Any] = {}
    for engine in engines:
        entry: Dict[str, Any] = {}
        for m in metrics_of_interest:
            values = [r["summary"].get(engine, {}).get(m) for r in per_seed]
            values = [float(v) for v in values if v is not None]
            if not values:
                entry[m] = None
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci = t_crit * (std / sqrt(len(values))) if len(values) > 1 else 0.0
            entry[m] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95": round(ci, 4),
                "low": round(mean - ci, 4),
                "high": round(mean + ci, 4),
                "n": len(values),
            }
        aggregated[engine] = entry

    result = {
        "meta": {
            "scenario_id": scenario_id,
            "seeds": seeds,
            "n_scenarios_per_seed": n_scenarios,
            "n_runs": len(seeds),
            "generated_at": time.time(),
            "t_critical_95": t_crit,
        },
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_ci.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default="/tmp/benchmark")
    args = parser.parse_args()
    result = run_benchmark(n_scenarios=args.scenarios, seed=args.seed, output_dir=args.output)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nArtefacts written to: {args.output}")
