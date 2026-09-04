"""
Real Evidence Report
====================

Generates a PDF from REAL orchestration evidence only.

Excluded intentionally:
- synthetic benchmarks
- simulated scenarios
- random values
- confidence-interval benchmark runs
- attributed AWS cost
- derived FPS estimates
- placeholder FLOPs/payload values

Evidence categories:
1. Measured infrastructure telemetry
2. Recorded real orchestration decisions
3. Q-Learning state learned from real executions
4. Real execution measurements, when supplied
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


TITLE = "Experimental Evaluation Report"
SUBTITLE = "AI-Driven Infrastructure Orchestration for Edge-Cloud Environments"

INSTITUTION = "University of Bedfordshire"
PROGRAMME = "MSc Artificial Intelligence"
AUTHOR = "Sophia Souza Marçal"
STUDENT_ID = "1808415"
MODULE = "25-26BLK5-6AACIS144-6"
SUPERVISOR = "Dr. Renxi Qui"


def _clean_text(value: Any) -> str:
    if value is None:
        return "-"
    return (
        str(value)
        .replace("Â·", "·")
        .replace("â€“", "-")
        .replace("â€”", "-")
    )


def _footer(fig, page: int) -> None:
    fig.text(
        0.5,
        0.025,
        f"{INSTITUTION}  |  {PROGRAMME}  |  Student ID: {STUDENT_ID}  |  Page {page}",
        ha="center",
        fontsize=7,
        family="monospace",
    )


def _cover(pdf: PdfPages, page: int) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    now = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    fig.text(
        0.5,
        0.955,
        INSTITUTION,
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.925,
        PROGRAMME,
        ha="center",
        fontsize=10,
    )

    fig.text(
        0.5,
        0.895,
        f"Author: {AUTHOR}   |   Student ID: {STUDENT_ID}",
        ha="center",
        fontsize=9,
    )

    fig.text(
        0.5,
        0.872,
        f"Module: {MODULE}   |   Supervisor: {SUPERVISOR}",
        ha="center",
        fontsize=9,
    )

    fig.text(
        0.5,
        0.82,
        TITLE,
        ha="center",
        fontsize=20,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.77,
        SUBTITLE,
        ha="center",
        fontsize=11,
    )

    fig.text(
        0.5,
        0.68,
        "Evidence source: implemented Edge-Cloud orchestration system",
        ha="center",
        fontsize=10,
    )

    fig.text(
        0.5,
        0.64,
        "No synthetic benchmark or simulated scenario data included",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.55,
        f"Generated: {now}",
        ha="center",
        fontsize=9,
        family="monospace",
    )

    fig.text(
        0.12,
        0.40,
        "Included evidence:\n\n"
        "• Real infrastructure telemetry\n"
        "• Recorded orchestration decisions\n"
        "• Q-Learning real-execution state\n"
        "• Real Edge / Cloud / Hybrid execution measurements when available",
        fontsize=10,
        linespacing=1.7,
    )

    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def _telemetry_page(
    pdf: PdfPages,
    telemetry: Optional[Dict[str, Any]],
    page: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    fig.text(
        0.08,
        0.93,
        "1. Real Infrastructure Telemetry",
        fontsize=15,
        fontweight="bold",
    )

    fig.text(
        0.08,
        0.89,
        "Measured infrastructure values collected from the live system.",
        fontsize=9,
    )

    if not telemetry:
        fig.text(0.08, 0.82, "No live telemetry supplied.", fontsize=10)
    else:
        y = 0.82
        for key, value in telemetry.items():
            fig.text(
                0.10,
                y,
                f"{_clean_text(key)}:",
                fontsize=9,
                fontweight="bold",
                family="monospace",
            )
            fig.text(
                0.43,
                y,
                _clean_text(value),
                fontsize=9,
                family="monospace",
            )
            y -= 0.042
            if y < 0.10:
                break

    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def _decision_page(
    pdf: PdfPages,
    decisions: List[Dict[str, Any]],
    page: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    fig.text(
        0.05,
        0.94,
        "2. Recorded Real Orchestration Decisions",
        fontsize=15,
        fontweight="bold",
    )

    fig.text(
        0.05,
        0.90,
        "Decision latency is policy decision time, not AI model inference latency.",
        fontsize=8.5,
    )

    rows = []

    for d in decisions[:12]:
        ctx = d.get("context") or {}

        rows.append(
            [
                _clean_text(d.get("engine")),
                _clean_text(d.get("route")),
                f'{float(ctx.get("network_latency_ms", 0)):.2f}',
                f'{float(ctx.get("cpu_available", 0)):.1f}',
                f'{float(ctx.get("memory_available", 0)):.1f}',
                f'{float(d.get("latency_us", 0)):.1f}',
            ]
        )

    columns = [
        "Engine",
        "Route",
        "Network RTT\n(ms)",
        "CPU Available\n(%)",
        "Memory Available\n(%)",
        "Decision Time\n(us)",
    ]

    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            bbox=[0.05, 0.20, 0.90, 0.62],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
    else:
        fig.text(0.05, 0.78, "No recorded decisions supplied.", fontsize=10)

    fig.text(
        0.05,
        0.10,
        "Confidence, probabilities and Q-values are algorithm-derived values; "
        "they are not physical infrastructure measurements.",
        fontsize=8,
    )

    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def _qlearning_page(
    pdf: PdfPages,
    qtable: Optional[Dict[str, Any]],
    page: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    fig.text(
        0.05,
        0.94,
        "3. Q-Learning Real-Execution Evidence",
        fontsize=15,
        fontweight="bold",
    )

    if not qtable:
        fig.text(0.05, 0.82, "No Q-Learning data supplied.", fontsize=10)
    else:
        fig.text(
            0.05,
            0.88,
            f"Learning mode: {_clean_text(qtable.get('learning_mode'))}",
            fontsize=10,
            fontweight="bold",
        )

        fig.text(
            0.05,
            0.84,
            f"Real execution updates: {_clean_text(qtable.get('real_updates'))}",
            fontsize=10,
        )

        rows = []
        for row in (qtable.get("rows") or [])[:12]:
            q = row.get("q_values") or []

            q_edge = q[0] if len(q) > 0 else "-"
            q_cloud = q[1] if len(q) > 1 else "-"
            q_hybrid = q[2] if len(q) > 2 else "-"

            rows.append(
                [
                    _clean_text(row.get("state")),
                    _clean_text(q_edge),
                    _clean_text(q_cloud),
                    _clean_text(q_hybrid),
                    _clean_text(row.get("best_route")),
                    _clean_text(row.get("max_q")),
                ]
            )

        columns = [
            "State",
            "Q Edge",
            "Q Cloud",
            "Q Hybrid",
            "Best Route",
            "Max Q",
        ]

        if rows:
            table = ax.table(
                cellText=rows,
                colLabels=columns,
                cellLoc="center",
                colLoc="center",
                bbox=[0.05, 0.22, 0.90, 0.53],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)

        features = qtable.get("features_encoded") or []

        fig.text(
            0.05,
            0.14,
            "Encoded policy features: " + ", ".join(map(str, features)),
            fontsize=8,
        )

        fig.text(
            0.05,
            0.09,
            "Q-values are learned algorithmic signals derived from real execution feedback; "
            "they are not directly measured physical metrics.",
            fontsize=8,
        )

    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def _execution_page(
    pdf: PdfPages,
    executions: Optional[List[Dict[str, Any]]],
    page: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    fig.text(
        0.05,
        0.94,
        "4. Real Execution Measurements",
        fontsize=15,
        fontweight="bold",
    )

    fig.text(
        0.05,
        0.90,
        "Only successful real executions are included.",
        fontsize=9,
    )

    valid = [
        x for x in (executions or [])
        if x and bool(x.get("success", False))
    ]

    rows = []

    for x in valid:
        mode = _clean_text(x.get("mode"))

        execution_label = (
            "hybrid-edge-component"
            if mode.lower() == "hybrid"
            else mode
        )

        rows.append(
            [
                execution_label,
                f'{float(x.get("latency_ms", 0)):.2f}',
                f'{float(x.get("cpu_percent", 0)):.2f}',
                f'{float(x.get("memory_mb", 0)):.2f}',
                _clean_text(x.get("detail")),
            ]
        )

        if mode.lower() == "hybrid" and "cloud_latency_ms" in x:
            rows.append(
                [
                    "hybrid-cloud-component",
                    f'{float(x.get("cloud_latency_ms", 0)):.2f}',
                    f'{float(x.get("cloud_cpu_percent", 0)):.2f}',
                    f'{float(x.get("cloud_memory_mb", 0)):.2f}',
                    "Cloud component of successful Hybrid execution",
                ]
            )

    columns = [
        "Execution",
        "Latency (ms)",
        "CPU (%)",
        "Memory (MB)",
        "Detail",
    ]

    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            bbox=[0.03, 0.23, 0.94, 0.58],
            colWidths=[0.15, 0.13, 0.12, 0.13, 0.47],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
    else:
        fig.text(
            0.05,
            0.78,
            "No successful execution measurements supplied.",
            fontsize=10,
        )

    fig.text(
        0.05,
        0.13,
        "Excluded: derived FPS, placeholder FLOPs/payload, synthetic values "
        "and attributed cloud cost.",
        fontsize=8,
    )

    fig.text(
        0.05,
        0.09,
        "A single execution demonstrates functional operation; repeated runs are "
        "required before making statistical performance claims.",
        fontsize=8,
    )

    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def generate_real_evidence_report(
    decisions: Optional[List[Dict[str, Any]]] = None,
    qtable: Optional[Dict[str, Any]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
    executions: Optional[List[Dict[str, Any]]] = None,
    output_path: str = "/tmp/real_evidence_report.pdf",
) -> Dict[str, Any]:

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    decisions = decisions or []
    executions = executions or []

    with PdfPages(output) as pdf:
        _cover(pdf, 1)
        _telemetry_page(pdf, telemetry, 2)
        _decision_page(pdf, decisions, 3)
        _qlearning_page(pdf, qtable, 4)
        _execution_page(pdf, executions, 5)

        metadata = pdf.infodict()
        metadata["Title"] = TITLE
        metadata["Subject"] = "Real Edge-Cloud orchestration evidence"
        metadata["Creator"] = "Edge-Cloud AI Orchestrator"

    return {
        "ok": True,
        "report_type": "real_evidence_only",
        "output_path": str(output),
        "decision_count": len(decisions),
        "successful_execution_count": sum(
            1 for x in executions if x and x.get("success") is True
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }



