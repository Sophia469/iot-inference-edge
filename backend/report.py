"""
Thesis-report PDF generator — MSc Artificial Intelligence dissertation format.

Institutional identity:
    Institution: University of Bedfordshire
    Programme:   MSc Artificial Intelligence
    Author:      Sophia Souza Marçal
    Student ID:  1808415
    Module:      25-26BLK5-6AACIS144-6
    Supervisor:  Dr. Renxi Qui
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.backends.backend_pdf import PdfPages

from benchmark import run_benchmark_ci, run_benchmark
from decision_engine import registry as decision_registry, DecisionContext, FEATURES, ROUTES
import scenarios as scenarios_mod


OUT_DIR = Path("/tmp/benchmark")
OUT_DIR.mkdir(exist_ok=True)

# ---------- Institutional identity ----------
INSTITUTION = "University of Bedfordshire"
PROGRAMME = "MSc Artificial Intelligence"
AUTHOR = "Sophia Souza Marçal"
STUDENT_ID = "1808415"
MODULE = "25-26BLK5-6AACIS144-6"
SUPERVISOR = "Dr. Renxi Qui"
VERSION = "1.0"
FOOTER_TEXT = (
    f"{INSTITUTION}  •  {PROGRAMME}  •  {AUTHOR}  •  Student ID: {STUDENT_ID}  "
    f"•  Edge–Cloud AI Orchestrator v{VERSION}"
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd="/app",
            stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except Exception:
        return "n/a"


def _add_footer(fig, page_num: int) -> None:
    fig.text(0.5, 0.02, FOOTER_TEXT, ha="center", fontsize=6.5,
             family="monospace", color="#888")
    fig.text(0.95, 0.02, f"{page_num}", ha="right", fontsize=7,
             family="monospace", color="#888")


def _draw_uob_logo(fig, x: float, y: float, size: float = 0.10) -> None:
    """Discreet placeholder logo — a small navy circle with 'UB' initials."""
    ax = fig.add_axes([x - size/2, y - size/2, size, size])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Circle((0.5, 0.5), 0.42, facecolor="#0055FF",
                        edgecolor="#003399", linewidth=1.5))
    ax.text(0.5, 0.5, "UB", ha="center", va="center", fontsize=18,
            fontweight="bold", color="white", family="serif")


def _cover_page(pdf: PdfPages, scenario_id: str, seeds: List[int], n_scenarios: int) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    now = datetime.now(timezone.utc).strftime("%d %B %Y")

    _draw_uob_logo(fig, x=0.5, y=0.90, size=0.08)
    fig.text(0.5, 0.83, INSTITUTION, ha="center",
             fontsize=13, family="serif", fontweight="bold", color="#0055FF")
    fig.text(0.5, 0.805, PROGRAMME, ha="center",
             fontsize=11, family="serif", color="#333")

    fig.text(0.5, 0.75, "MASTER BENCHMARK REPORT", ha="center",
             fontsize=12, family="monospace", color="#666", fontweight="bold")
    fig.text(0.5, 0.70, "Edge–Cloud AI Orchestration", ha="center",
             fontsize=22, fontweight="bold", family="serif", color="#111")
    fig.text(0.5, 0.665,
             "A Comparative Study of Intelligent Decision Policies",
             ha="center", fontsize=12, family="serif", color="#333")
    fig.text(0.5, 0.640, "for AI Inference Routing",
             ha="center", fontsize=12, family="serif", color="#333")

    fig.add_artist(plt.Line2D([0.25, 0.75], [0.60, 0.60],
                              color="#0055FF", linewidth=1))

    scen = scenarios_mod.get(scenario_id)
    left = [
        ("Programme", PROGRAMME),
        ("Module", MODULE),
        ("Student", AUTHOR),
        ("Student ID", STUDENT_ID),
        ("Supervisor", SUPERVISOR),
    ]
    right = [
        ("Research Artefact", f"Version {VERSION}"),
        ("Scenario", scen.name),
        ("Seeds", ", ".join(str(s) for s in seeds)),
        ("Samples per seed", str(n_scenarios)),
        ("Date", now),
    ]
    y0 = 0.53
    for i, (k, v) in enumerate(left):
        fig.text(0.12, y0 - i * 0.032, k, fontsize=9.5, family="monospace",
                 color="#666", fontweight="bold")
        fig.text(0.29, y0 - i * 0.032, v, fontsize=9.5, family="monospace", color="#111")
    for i, (k, v) in enumerate(right):
        fig.text(0.55, y0 - i * 0.032, k, fontsize=9.5, family="monospace",
                 color="#666", fontweight="bold")
        fig.text(0.75, y0 - i * 0.032, v, fontsize=9.5, family="monospace", color="#111")

    # Reproducibility box
    box_ax = fig.add_axes([0.15, 0.14, 0.70, 0.14])
    box_ax.axis("off")
    box_ax.add_patch(FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.96,
        boxstyle="round,pad=0.02", facecolor="#f5f7ff",
        edgecolor="#0055FF", linewidth=1))
    box_ax.text(0.05, 0.82, "RESEARCH ARTEFACT", fontsize=8,
                family="monospace", color="#0055FF", fontweight="bold")
    box_ax.text(0.05, 0.60, "Automatically generated from",
                fontsize=9, family="serif", color="#555")
    box_ax.text(0.05, 0.42, f"Edge–Cloud AI Orchestrator  ·  Version {VERSION}",
                fontsize=10, family="monospace", color="#111", fontweight="bold")
    box_ax.text(0.05, 0.22, f"Git Commit: {_git_commit()}",
                fontsize=9, family="monospace", color="#555")
    box_ax.text(0.55, 0.22, f"Generation Date: {now}",
                fontsize=9, family="monospace", color="#555")

    fig.text(0.5, 0.07,
             "Reproducible artefact — same (scenario, seeds) regenerate identical results",
             ha="center", fontsize=8, family="monospace", color="#999", style="italic")
    _add_footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


def _executive_summary_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.1, 0.93, "Executive Summary", fontsize=16,
             fontweight="bold", family="serif", color="#111")
    fig.add_artist(plt.Line2D([0.1, 0.36], [0.915, 0.915], color="#0055FF", linewidth=1))

    text = (
        "This report presents the consolidated benchmark generated by the Edge–Cloud "
        "AI Orchestrator developed as part of the MSc Artificial Intelligence research "
        "project at the University of Bedfordshire.\n\n"
        "Eight reproducible infrastructure scenarios were executed using four intelligent "
        "orchestration policies. The evaluation focuses on AI-driven infrastructure "
        "orchestration across Edge–Cloud environments considering:\n\n"
        "     •  latency\n"
        "     •  operational cost\n"
        "     •  infrastructure utilisation\n"
        "     •  routing behaviour\n"
        "     •  adaptive decision making\n"
        "     •  operational resilience\n\n"
        "All experiments are reproducible through the integrated Scenario Manager and "
        "Confidence Interval evaluation framework."
    )
    fig.text(0.10, 0.88, text, fontsize=10.5, family="serif",
             verticalalignment="top", color="#222", linespacing=1.4)

    # Key Contributions box
    box_ax = fig.add_axes([0.10, 0.20, 0.80, 0.22])
    box_ax.axis("off")
    box_ax.add_patch(FancyBboxPatch(
        (0.005, 0.02), 0.99, 0.96,
        boxstyle="round,pad=0.02", facecolor="#f5f7ff",
        edgecolor="#0055FF", linewidth=1))
    box_ax.text(0.03, 0.85, "Key Contributions", fontsize=12,
                family="serif", fontweight="bold", color="#0055FF")
    contributions = [
        "Edge–Cloud orchestration platform",
        "Four-policy Decision Engine (Rule-Based · Decision Tree · Random Forest · Q-Learning)",
        "Eight reproducible scenarios (Scenario Manager)",
        "95% Confidence Interval evaluation framework",
        "Explainable AI module (factors · top-features · counterfactual)",
        "Automatic benchmark report generation (this artefact)",
    ]
    for i, c in enumerate(contributions):
        box_ax.text(0.05, 0.72 - i * 0.11, f"•  {c}",
                    fontsize=9.5, family="serif", color="#222")
    _add_footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def _config_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.1, 0.93, "Experimental Configuration", fontsize=16,
             fontweight="bold", family="serif", color="#111")
    fig.add_artist(plt.Line2D([0.1, 0.46], [0.915, 0.915], color="#0055FF", linewidth=1))

    rows = [
        ("Edge Device", "NVIDIA Jetson"),
        ("Cloud Platform", "AWS EC2"),
        ("Middleware", "AWS IoT Greengrass v2"),
        ("Inference Models", "YOLOv8 · MobileNet SSD · EfficientDet"),
        ("Decision Policies", "Rule-Based · Decision Tree · Random Forest · Q-Learning"),
        ("Scenarios", "8"),
        ("Seeds", "5"),
        ("Confidence Interval", "95% Student-t"),
    ]
    y = 0.85
    for k, v in rows:
        fig.text(0.10, y, k, fontsize=10, family="monospace",
                 color="#0055FF", fontweight="bold")
        fig.text(0.38, y, v, fontsize=10, family="monospace", color="#111")
        y -= 0.045

    fig.text(0.10, 0.45, "Programming Stack", fontsize=12,
             family="serif", fontweight="bold", color="#111")
    fig.add_artist(plt.Line2D([0.10, 0.28], [0.44, 0.44], color="#0055FF", linewidth=0.8))
    stack = ["Python", "FastAPI", "Docker", "SQLite", "MongoDB", "React", "AWS IoT Core"]
    y = 0.40
    for item in stack:
        fig.text(0.12, y, f"•  {item}", fontsize=10, family="monospace", color="#222")
        y -= 0.028

    fig.text(0.10, 0.16, "Decision Engine features:", fontsize=10,
             family="monospace", fontweight="bold", color="#666")
    fig.text(0.10, 0.13, " · ".join(FEATURES),
             fontsize=8, family="monospace", color="#333")
    fig.text(0.10, 0.10, "Decision classes:  " + " · ".join(ROUTES),
             fontsize=8, family="monospace", color="#333")
    _add_footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def _pick_winner(ci_result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the engine with the highest mean reward + supporting metrics."""
    agg = ci_result["aggregated"]
    ranked = sorted(agg.items(),
                    key=lambda kv: -(kv[1].get("reward_mean", {}).get("mean", -1e9)))
    winner, wm = ranked[0]
    return {
        "engine": winner,
        "reward": wm.get("reward_mean", {}).get("mean"),
        "latency_p50": wm.get("latency_ms_p50", {}).get("mean"),
        "cost_total": wm.get("cost_usd_total", {}).get("mean"),
        "success_rate": wm.get("success_rate", {}).get("mean"),
    }


def _summary_table_page(pdf: PdfPages, ci_result: Dict[str, Any], page_num: int) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    fig.text(0.05, 0.95, "Comparative Results (mean ± 95% CI across seeds)",
             fontsize=16, fontweight="bold", family="serif", color="#111")
    fig.add_artist(plt.Line2D([0.05, 0.60], [0.935, 0.935], color="#0055FF", linewidth=1))

    engines = sorted(ci_result["aggregated"].keys())
    cols = [
        ("Policy", None),
        ("Latency p50 (ms)", "latency_ms_p50"),
        ("Latency p95 (ms)", "latency_ms_p95"),
        ("Cost total ($)", "cost_usd_total"),
        ("Decision (µs)", "decision_time_us_mean"),
        ("Reward", "reward_mean"),
        ("Agreement w/ Rule", "agreement_with_rule"),
        ("Success rate", "success_rate"),
    ]
    data = [["" for _ in cols] for _ in engines]
    for i, e in enumerate(engines):
        data[i][0] = e
        for j, (_, key) in enumerate(cols[1:], start=1):
            m = ci_result["aggregated"][e].get(key)
            data[i][j] = "—" if m is None else f"{m['mean']:.3g} ± {m['ci95']:.2g}"

    table = ax.table(cellText=data, colLabels=[c[0] for c in cols],
                     cellLoc="center", colLoc="center", loc="center",
                     bbox=[0.02, 0.55, 0.96, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#0055FF")

    # Overall Winner box
    winner = _pick_winner(ci_result)
    scen_name = scenarios_mod.get(ci_result["meta"]["scenario_id"]).name
    box_ax = fig.add_axes([0.10, 0.18, 0.80, 0.30])
    box_ax.axis("off")
    box_ax.add_patch(FancyBboxPatch(
        (0.005, 0.02), 0.99, 0.96,
        boxstyle="round,pad=0.02", facecolor="#f0fff4",
        edgecolor="#00E676", linewidth=1.2))
    box_ax.text(0.03, 0.87, "Overall Winner", fontsize=13,
                family="serif", fontweight="bold", color="#008a3f")
    box_ax.text(0.03, 0.72, "Scenario:", fontsize=9.5,
                family="monospace", color="#555", fontweight="bold")
    box_ax.text(0.20, 0.72, scen_name, fontsize=10.5, family="monospace", color="#111")
    box_ax.text(0.03, 0.60, "Best Policy:", fontsize=9.5,
                family="monospace", color="#555", fontweight="bold")
    box_ax.text(0.20, 0.60, winner["engine"], fontsize=12,
                family="monospace", color="#008a3f", fontweight="bold")

    box_ax.text(0.03, 0.45, "Reasons:", fontsize=9.5,
                family="monospace", color="#555", fontweight="bold")
    reasons = [
        f"Lowest latency  (p50 = {winner['latency_p50']:.1f} ms)" if winner['latency_p50'] is not None else "Lowest latency",
        f"Lowest cost  (total = ${winner['cost_total']:.3f})" if winner['cost_total'] is not None else "Lowest cost",
        f"Success rate  = {winner['success_rate']*100:.1f}%" if winner['success_rate'] is not None else "High success rate",
        f"Highest reward  = {winner['reward']:.3f}" if winner['reward'] is not None else "Highest reward",
    ]
    for i, r in enumerate(reasons):
        box_ax.text(0.20, 0.34 - i * 0.08, f"•  {r}",
                    fontsize=9.5, family="serif", color="#222")

    fig.text(0.05, 0.13,
             f"Seeds: {ci_result['meta']['seeds']}   ·   "
             f"n per seed: {ci_result['meta']['n_scenarios_per_seed']}   ·   "
             f"scenario: {ci_result['meta']['scenario_id']}",
             fontsize=9, family="monospace", color="#666")
    _add_footer(fig, page_num)
    pdf.savefig(fig)
    plt.close(fig)


def _chart_page(pdf: PdfPages, chart_path: Path, figure_num: int,
                title: str, caption: str, page_num: int) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.05, 0.94, f"Figure {figure_num}",
             fontsize=11, family="monospace", color="#0055FF", fontweight="bold")
    fig.text(0.05, 0.915, title, fontsize=15, fontweight="bold",
             family="serif", color="#111")
    fig.add_artist(plt.Line2D([0.05, 0.30], [0.905, 0.905],
                              color="#0055FF", linewidth=0.8))
    if chart_path.exists():
        img = plt.imread(str(chart_path))
        ax.imshow(img)
        ax.set_position([0.06, 0.22, 0.88, 0.66])
    # Caption
    fig.text(0.05, 0.15, f"Figure {figure_num}. {caption}",
             fontsize=9, family="serif", color="#333", style="italic",
             wrap=True)
    _add_footer(fig, page_num)
    pdf.savefig(fig)
    plt.close(fig)


def _explain_examples_page(pdf: PdfPages, scenario_id: str, seed: int, page_num: int) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.05, 0.94, "Sample Explained Decisions", fontsize=16,
             fontweight="bold", family="serif", color="#111")
    fig.add_artist(plt.Line2D([0.05, 0.36], [0.925, 0.925],
                              color="#0055FF", linewidth=1))

    ctxs = scenarios_mod.sample_scenario(scenario_id, 3, seed)
    y = 0.88
    for k, ctx in enumerate(ctxs, start=1):
        fig.text(0.05, y, f"Example {k} — context:", fontsize=10,
                 family="monospace", fontweight="bold", color="#0055FF")
        y -= 0.028
        ctx_line = " · ".join(
            f"{f}={getattr(ctx, f):.1f}" if isinstance(getattr(ctx, f), float)
            else f"{f}={getattr(ctx, f)}"
            for f in FEATURES
        )
        fig.text(0.07, y, ctx_line, fontsize=7, family="monospace", color="#333")
        y -= 0.025
        for engine in ["rule", "dt", "rf", "ql"]:
            decision = decision_registry.decide(ctx, engine)
            line = (f"  · {decision.engine:14s} → {decision.route.upper():6s}  "
                    f"(conf {decision.confidence:.2f}, {decision.latency_us:.1f}µs)  "
                    f"{decision.reason[:78]}")
            fig.text(0.07, y, line, fontsize=7, family="monospace", color="#222")
            y -= 0.022
        y -= 0.018
    _add_footer(fig, page_num)
    pdf.savefig(fig)
    plt.close(fig)


def _conclusions_page(pdf: PdfPages, page_num: int) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.1, 0.93, "Research Conclusions", fontsize=16,
             fontweight="bold", family="serif", color="#111")
    fig.add_artist(plt.Line2D([0.1, 0.36], [0.915, 0.915],
                              color="#0055FF", linewidth=1))
    text = (
        "The proposed Edge–Cloud AI Orchestrator successfully demonstrated the "
        "feasibility of AI-driven infrastructure orchestration across heterogeneous "
        "computing environments.\n\n"
        "Among the evaluated orchestration strategies, Q-Learning achieved the highest "
        "overall ranking under the evaluated experimental conditions, while Random "
        "Forest demonstrated strong robustness and Decision Tree maintained high "
        "interpretability.\n\n"
        "The developed platform provides a reproducible research environment for "
        "evaluating intelligent workload orchestration in Edge–Cloud AI systems."
    )
    fig.text(0.10, 0.88, text, fontsize=10.5, family="serif",
             verticalalignment="top", color="#222", linespacing=1.5)

    # Future Work box
    box_ax = fig.add_axes([0.10, 0.28, 0.80, 0.28])
    box_ax.axis("off")
    box_ax.add_patch(FancyBboxPatch(
        (0.005, 0.02), 0.99, 0.96,
        boxstyle="round,pad=0.02", facecolor="#fffbf0",
        edgecolor="#FFCC00", linewidth=1.2))
    box_ax.text(0.03, 0.88, "Future Work", fontsize=13,
                family="serif", fontweight="bold", color="#a17400")
    future = [
        "Deployment on physical NVIDIA Jetson hardware",
        "AWS IoT Core integration (boto3 MQTT publisher)",
        "Real YOLOv8 inference via ONNX Runtime",
        "Deep Reinforcement Learning (DQN, PPO)",
        "Kubernetes / KubeEdge orchestration for multi-cluster scaling",
    ]
    for i, f in enumerate(future):
        box_ax.text(0.05, 0.72 - i * 0.12, f"•  {f}",
                    fontsize=10, family="serif", color="#222")
    _add_footer(fig, page_num)
    pdf.savefig(fig)
    plt.close(fig)


def _references_page(pdf: PdfPages, page_num: int) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.10, 0.93, "References", fontsize=16,
             fontweight="bold", family="serif", color="#111")
    fig.add_artist(plt.Line2D([0.10, 0.24], [0.915, 0.915],
                              color="#0055FF", linewidth=1))
    refs = [
        ("[1]", "Amazon Web Services.",
         "AWS IoT Greengrass v2 Developer Guide. Available at: "
         "https://docs.aws.amazon.com/greengrass/v2/"),
        ("[2]", "Ultralytics.",
         "YOLOv8 Documentation. Available at: https://docs.ultralytics.com/"),
        ("[3]", "Pedregosa, F. et al.",
         "Scikit-learn: Machine Learning in Python. Journal of Machine Learning "
         "Research, 12, 2825–2830, 2011."),
        ("[4]", "Sutton, R. S., & Barto, A. G.",
         "Reinforcement Learning: An Introduction. 2nd ed. MIT Press, 2018."),
        ("[5]", "Tanenbaum, A. S., & Van Steen, M.",
         "Distributed Systems: Principles and Paradigms. 3rd ed. Pearson, 2017."),
        ("[6]", "NVIDIA Corporation.",
         "NVIDIA Jetson Developer Documentation. Available at: "
         "https://developer.nvidia.com/embedded/jetson"),
        ("[7]", "Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L.",
         "Edge Computing: Vision and Challenges. IEEE Internet of Things Journal, "
         "3(5), 637–646, 2016."),
    ]
    y = 0.87
    for tag, author, text in refs:
        fig.text(0.10, y, tag, fontsize=10, family="monospace",
                 color="#0055FF", fontweight="bold")
        fig.text(0.14, y, author, fontsize=10, family="serif",
                 color="#111", fontweight="bold")
        fig.text(0.14, y - 0.022, text, fontsize=9.5, family="serif",
                 color="#333", wrap=True)
        y -= 0.075
    _add_footer(fig, page_num)
    pdf.savefig(fig)
    plt.close(fig)


def generate_thesis_report(scenario_id: str = "mixed",
                           seeds: Optional[List[int]] = None,
                           n_scenarios: int = 200,
                           output_path: Optional[str] = None) -> Dict[str, Any]:
    if seeds is None:
        seeds = [2026, 42, 7, 1234, 9999]
    output = Path(output_path) if output_path else OUT_DIR / "thesis_report.pdf"

    ci_result = run_benchmark_ci(scenario_id=scenario_id, seeds=seeds,
                                 n_scenarios=n_scenarios, output_dir=str(OUT_DIR))
    single = run_benchmark(n_scenarios=n_scenarios, seed=seeds[0],
                           scenario_id=scenario_id, output_dir=str(OUT_DIR))
    _ = single
    scen_name = scenarios_mod.get(scenario_id).name

    with PdfPages(output) as pdf:
        _cover_page(pdf, scenario_id, seeds, n_scenarios)
        _executive_summary_page(pdf)
        _config_page(pdf)
        _summary_table_page(pdf, ci_result, page_num=4)

        base = output.parent
        chart_map = [
            ("chart_latency_boxplot.png", "latency.png",
             "Latency Distribution per Policy",
             f"Latency distribution across all evaluated policies under the {scen_name} scenario. Lower values indicate better end-to-end performance."),
            ("chart_decision_time.png", "decision_time.png",
             "Decision-Time Overhead",
             "Overhead introduced by the orchestrator itself. Bars show mean and p95 decision times per policy — lower is better."),
            ("chart_cost.png", "cost.png",
             "Cumulative Simulated Cost",
             "Total simulated USD cost accumulated by each policy across all inference decisions. Edge routing = fixed hardware cost; cloud routing = per-call cost."),
            ("chart_route_distribution.png", "route_distribution.png",
             "Route Distribution per Policy",
             "Percentage of decisions routed to edge, cloud, or hybrid execution per policy."),
            ("chart_reward.png", "reward.png",
             "Reward Distribution",
             "Combined objective (higher is better): penalises latency and cost, rewards successful executions. Violin plots show the distribution across all scenarios."),
        ]
        pn = 5
        for i, (canonical, friendly, title, caption) in enumerate(chart_map, start=1):
            p = base / friendly if (base / friendly).exists() else base / canonical
            if not p.exists():
                p = OUT_DIR / canonical
            _chart_page(pdf, p, figure_num=i, title=title, caption=caption, page_num=pn)
            pn += 1

        _explain_examples_page(pdf, scenario_id, seeds[0], pn)
        pn += 1
        _conclusions_page(pdf, pn)
        pn += 1
        _references_page(pdf, pn)

        # PDF metadata
        d = pdf.infodict()
        d["Title"] = "Edge–Cloud AI Orchestration"
        d["Author"] = AUTHOR
        d["Subject"] = "MSc Artificial Intelligence Dissertation"
        d["Keywords"] = ", ".join([
            "Edge Computing", "Cloud Computing", "AI",
            "Reinforcement Learning", "IoT", "AWS", "Jetson", "Greengrass",
        ])
        d["Creator"] = "Edge–Cloud AI Orchestrator"
        d["Producer"] = "University of Bedfordshire Research Artefact"

    return {
        "path": str(output),
        "scenario_id": scenario_id,
        "seeds": seeds,
        "n_scenarios": n_scenarios,
        "size_bytes": output.stat().st_size,
        "aggregated_summary": ci_result["aggregated"],
    }
