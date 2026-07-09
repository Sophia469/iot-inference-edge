"""
Complete-experiment orchestrator + master summary generator.

Runs the multi-seed benchmark for ALL scenarios in sequence, saves per-scenario
artefacts under /app/experiments/<Scenario_Name>/ (PDF + CSV + PNGs), then
produces a MASTER SUMMARY that consolidates the 8 scenarios into a single
ranking + comparison PDF for the thesis conclusion chapter.

Layout after run:
    /app/experiments/
        Factory_Normal/
            report.pdf
            results.csv
            benchmark_summary.json
            benchmark_ci.json
            chart_*.png
        Mixed_Traffic/
            ...
        ...
        master_summary.pdf
        master_summary.csv
        master_summary.json
"""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import scenarios as scenarios_mod
from benchmark import run_benchmark_ci, run_benchmark, save_charts, ENGINE_LABELS
from report import generate_thesis_report

EXPERIMENTS_DIR = Path("/app/experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# In-memory job state (single-worker uvicorn assumption)
JOB_STATE: Dict[str, Any] = {
    "status": "idle",
    "current_scenario": None,
    "completed": [],
    "total": 0,
    "started_at": None,
    "finished_at": None,
    "master_summary": None,
}


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def _run_scenario(scenario_id: str, seeds: List[int], n_scenarios: int) -> Dict[str, Any]:
    scen = scenarios_mod.get(scenario_id)
    out_dir = EXPERIMENTS_DIR / _slug(scen.name)
    out_dir.mkdir(exist_ok=True)

    # CI aggregation
    ci_result = run_benchmark_ci(scenario_id=scenario_id, seeds=seeds,
                                 n_scenarios=n_scenarios, output_dir=str(out_dir))
    # Single seed run for the chart PNGs + CSV
    run_benchmark(n_scenarios=n_scenarios, seed=seeds[0],
                  scenario_id=scenario_id, output_dir=str(out_dir))
    # Rename records CSV for clarity
    src_csv = out_dir / "benchmark_records.csv"
    dst_csv = out_dir / "results.csv"
    if src_csv.exists():
        shutil.move(str(src_csv), str(dst_csv))

    # Rename chart PNGs to friendly names
    renames = {
        "chart_latency_boxplot.png":   "latency.png",
        "chart_cost.png":              "cost.png",
        "chart_reward.png":            "reward.png",
        "chart_route_distribution.png":"route_distribution.png",
        "chart_decision_time.png":     "decision_time.png",
    }
    for src, dst in renames.items():
        src_p = out_dir / src
        if src_p.exists():
            shutil.move(str(src_p), str(out_dir / dst))

    # Per-scenario PDF using existing thesis-report generator
    pdf_path = out_dir / "report.pdf"
    generate_thesis_report(scenario_id=scenario_id, seeds=seeds,
                           n_scenarios=n_scenarios, output_path=str(pdf_path))

    # Reproducibility metadata
    metadata = {
        "scenario": scen.name,
        "scenario_id": scenario_id,
        "description": scen.description,
        "seeds": seeds,
        "runs": n_scenarios,
        "total_evaluations": n_scenarios * len(seeds) * 4,
        "policies": ["RuleBased", "DecisionTree", "RandomForest", "QLearning"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artefacts": [
            "results.csv", "report.pdf", "latency.png", "cost.png",
            "reward.png", "route_distribution.png", "decision_time.png",
            "benchmark_summary.json", "benchmark_ci.json", "metadata.json",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # per-scenario ranking based on mean reward
    ranking = sorted(
        ci_result["aggregated"].items(),
        key=lambda kv: -(kv[1].get("reward_mean", {}).get("mean", -1e9)),
    )
    ranked_engines = [name for name, _ in ranking]

    return {
        "scenario_id": scenario_id,
        "scenario_name": scen.name,
        "artefacts_dir": str(out_dir),
        "ranking_by_reward": ranked_engines,
        "aggregated": ci_result["aggregated"],
        "meta": ci_result["meta"],
    }


def _master_summary_json(per_scenario: List[Dict[str, Any]]) -> Dict[str, Any]:
    engines = sorted(ENGINE_LABELS.values())
    # win-count: how many scenarios each engine ranked #1
    wins: Dict[str, int] = {e: 0 for e in engines}
    global_metrics: Dict[str, Dict[str, List[float]]] = {
        e: {"latency_ms_p50": [], "cost_usd_total": [], "reward_mean": [],
            "success_rate": [], "agreement_with_rule": []}
        for e in engines
    }
    per_scenario_table = []
    for sc in per_scenario:
        top = sc["ranking_by_reward"][0]
        wins[top] += 1
        row = {"scenario": sc["scenario_name"], "winner_reward": top}
        for e in engines:
            m = sc["aggregated"].get(e, {})
            for k in global_metrics[e]:
                v = m.get(k)
                if v and "mean" in v:
                    global_metrics[e][k].append(v["mean"])
                    row[f"{e}_{k}"] = v["mean"]
        per_scenario_table.append(row)

    global_avg = {
        e: {k: (round(float(np.mean(vals)), 4) if vals else None)
            for k, vals in metrics.items()}
        for e, metrics in global_metrics.items()
    }
    overall_ranking = sorted(engines, key=lambda e: -(global_avg[e]["reward_mean"] or -1e9))

    return {
        "generated_at": time.time(),
        "n_scenarios": len(per_scenario),
        "engines": engines,
        "wins_by_engine": wins,
        "global_average_metrics": global_avg,
        "overall_ranking_by_reward": overall_ranking,
        "per_scenario_table": per_scenario_table,
    }


def _master_summary_pdf(master: Dict[str, Any], output_path: Path) -> None:
    engines = master["engines"]

    with PdfPages(output_path) as pdf:
        # Cover
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        fig.text(0.5, 0.85, "MASTER SUMMARY", ha="center",
                 fontsize=24, fontweight="bold", family="monospace")
        fig.text(0.5, 0.80, "Edge–Cloud AI Orchestration", ha="center",
                 fontsize=13, family="monospace", color="#555")
        fig.text(0.5, 0.75, "Consolidated evaluation across all scenarios",
                 ha="center", fontsize=11, family="monospace", color="#333")
        lines = [
            ("Scenarios evaluated", str(master["n_scenarios"])),
            ("Engines", " · ".join(engines)),
            ("Overall ranking (reward)",
             " > ".join(master["overall_ranking_by_reward"])),
            ("Generated at", now),
        ]
        y = 0.55
        for label, value in lines:
            fig.text(0.12, y, label, fontsize=10, family="monospace",
                     color="#666", fontweight="bold")
            fig.text(0.42, y, value, fontsize=10, family="monospace")
            y -= 0.04
        pdf.savefig(fig)
        plt.close(fig)

        # Win-count chart
        fig, ax = plt.subplots(figsize=(8.27, 6))
        wins = [master["wins_by_engine"][e] for e in engines]
        colors = {"RuleBased": "#FFCC00", "DecisionTree": "#00E676",
                  "RandomForest": "#0055FF", "QLearning": "#FF3333"}
        ax.bar(engines, wins, color=[colors.get(e, "#888") for e in engines])
        ax.set_title("Scenario Wins by Policy (highest reward)")
        ax.set_ylabel("# scenarios won")
        for i, v in enumerate(wins):
            ax.text(i, v + 0.05, str(v), ha="center", family="monospace")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Global metrics table
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        fig.text(0.05, 0.94, "Global Average Metrics", fontsize=14,
                 fontweight="bold", family="monospace")
        cols = ["Policy", "Latency p50 (ms)", "Cost total ($)", "Reward",
                "Success rate", "Agreement w/ Rule"]
        data = []
        for e in engines:
            m = master["global_average_metrics"][e]
            data.append([
                e,
                f"{m['latency_ms_p50']:.2f}" if m['latency_ms_p50'] is not None else "—",
                f"{m['cost_usd_total']:.3f}" if m['cost_usd_total'] is not None else "—",
                f"{m['reward_mean']:.3f}" if m['reward_mean'] is not None else "—",
                f"{m['success_rate']*100:.1f}%" if m['success_rate'] is not None else "—",
                f"{m['agreement_with_rule']*100:.1f}%" if m['agreement_with_rule'] is not None else "—",
            ])
        table = ax.table(cellText=data, colLabels=cols, cellLoc="center",
                         colLoc="center", loc="center", bbox=[0.02, 0.55, 0.96, 0.30])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor("#ccc")
            if r == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#0055FF")
        fig.text(0.05, 0.50, "Overall Ranking (by mean reward):",
                 fontsize=11, fontweight="bold", family="monospace")
        for i, e in enumerate(master["overall_ranking_by_reward"], 1):
            fig.text(0.10, 0.47 - i * 0.03, f"{i}. {e}",
                     fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # Per-scenario winner heatmap
        fig, ax = plt.subplots(figsize=(11.69, 6))
        table_data = []
        for row in master["per_scenario_table"]:
            table_data.append([
                row["scenario"], row["winner_reward"],
                *[f"{row.get(f'{e}_reward_mean', 0):.2f}" for e in engines]
            ])
        cols = ["Scenario", "Winner"] + engines
        table = ax.table(cellText=table_data, colLabels=cols, cellLoc="center",
                         loc="center", bbox=[0.02, 0.15, 0.96, 0.75])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#ccc")
            if r == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#0055FF")
        ax.axis("off")
        fig.text(0.05, 0.94, "Per-Scenario Reward Ranking",
                 fontsize=14, fontweight="bold", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


def run_complete_experiment(
    seeds: Optional[List[int]] = None, n_scenarios: int = 100,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs the CI experiment for all (or a subset of) scenarios and generates the master summary."""
    if seeds is None:
        seeds = [2026, 42, 7, 1234, 9999]
    all_scenarios = list(scenarios_mod.SCENARIOS.keys())
    to_run = [s for s in all_scenarios if (not include or s in include)]

    JOB_STATE.update({
        "status": "running", "current_scenario": None,
        "completed": [], "total": len(to_run),
        "started_at": time.time(), "finished_at": None,
        "master_summary": None,
    })

    per_scenario_results = []
    for sid in to_run:
        JOB_STATE["current_scenario"] = sid
        try:
            r = _run_scenario(sid, seeds=seeds, n_scenarios=n_scenarios)
            per_scenario_results.append(r)
            JOB_STATE["completed"].append(sid)
        except Exception as e:
            JOB_STATE["completed"].append(f"{sid}:ERROR:{e}")
    JOB_STATE["current_scenario"] = None

    master = _master_summary_json(per_scenario_results)
    (EXPERIMENTS_DIR / "master_summary.json").write_text(
        json.dumps(master, indent=2, default=str))
    # CSV
    df = pd.DataFrame(master["per_scenario_table"])
    df.to_csv(EXPERIMENTS_DIR / "master_summary.csv", index=False)
    # PDF
    _master_summary_pdf(master, EXPERIMENTS_DIR / "master_summary.pdf")

    JOB_STATE.update({
        "status": "done", "finished_at": time.time(),
        "master_summary": master,
    })
    return master


def zip_all_experiments() -> Path:
    """Bundle everything under /app/experiments into a downloadable zip."""
    out = Path("/tmp/experiments_bundle.zip")
    if out.exists():
        out.unlink()
    shutil.make_archive(str(out).replace(".zip", ""), "zip", str(EXPERIMENTS_DIR))
    return out


def get_status() -> Dict[str, Any]:
    st = dict(JOB_STATE)
    if st["started_at"]:
        st["elapsed_s"] = round(
            (st["finished_at"] or time.time()) - st["started_at"], 1)
    return st
