"""
Inference runner for LIVE Edge-Cloud execution.

LIVE execution:
- EDGE workloads are executed remotely on edge-node-01.
- CLOUD workloads are executed remotely on AWS EC2 cloud-node-01.
- Latency, CPU and memory metrics are measured by the node
  that actually executes the workload.
- No artificial network sleep or simulated infrastructure metrics
  are used in this module.

COST:
- Edge AWS attributed cost is zero.
- Cloud cost is an attributed compute cost based on the real measured
  execution time and the configured EC2 On-Demand hourly rate.
- This is NOT presented as an individual AWS billing line item.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any

from edge_client import run_edge_inference
from cloud_client import run_cloud_inference


# ------------------------------------------------------------------
# AWS cost configuration
# ------------------------------------------------------------------

AWS_INSTANCE_TYPE = "t3.micro"
AWS_REGION = "eu-west-2"

# Set this environment variable to the official current Linux/Unix
# On-Demand hourly rate for t3.micro in eu-west-2.
#
# We deliberately do not invent or hard-code a regional price.
AWS_EC2_HOURLY_RATE_USD = float(
    os.environ.get("AWS_EC2_HOURLY_RATE_USD", "0.0")
)


def _attributed_cloud_cost(execution_ms: float) -> float:
    """
    Attribute a proportional share of EC2 compute cost to this execution.

    This is an analytical per-execution allocation:

        hourly_rate * execution_seconds / 3600

    It is NOT the literal AWS invoice charge for an individual inference.
    """
    if execution_ms <= 0 or AWS_EC2_HOURLY_RATE_USD <= 0:
        return 0.0

    execution_seconds = execution_ms / 1000.0

    return AWS_EC2_HOURLY_RATE_USD * (
        execution_seconds / 3600.0
    )


@dataclass
class InferenceResult:
    mode: Literal["edge", "cloud"]
    latency_ms: float
    cpu_percent: float
    memory_mb: float
    fps_estimate: float
    workload_flops: int
    payload_kb: float
    success: bool
    detail: str

    # Monetary cost attributed to this execution.
    cost_usd: float = 0.0
    cost_type: str = "attributed_compute_cost"
    pricing_instance: str | None = None
    pricing_region: str | None = None
    hourly_rate_usd: float = 0.0



def run_edge(
    model_size_mb: float,
    batch_size: int = 1
) -> InferenceResult:
    """
    Execute the workload remotely on the real Edge Compute Node.
    """

    try:
        edge_result = run_edge_inference(
            model_size_mb=model_size_mb,
            batch_size=batch_size
        )

    except Exception as exc:
        return InferenceResult(
            mode="edge",
            latency_ms=0.0,
            cpu_percent=0.0,
            memory_mb=0.0,
            fps_estimate=0.0,
            workload_flops=0,
            payload_kb=0.0,
            success=False,
            detail=f"Edge execution failed: {exc}",
            cost_usd=0.0,
            cost_type="no_aws_compute_cost",
        )

    latency_ms = float(
        edge_result.get("latency_ms", 0.0)
    )

    cpu_percent = float(
        edge_result.get("cpu_percent", 0.0)
    )

    memory_delta_mb = float(
        edge_result.get("memory_delta_mb", 0.0)
    )

    fps = (
        1000.0 / latency_ms
        if latency_ms > 0
        else 0.0
    )

    return InferenceResult(
        mode="edge",
        latency_ms=round(latency_ms, 2),
        cpu_percent=round(cpu_percent, 2),
        memory_mb=round(memory_delta_mb, 2),
        fps_estimate=round(fps, 2),

        # Not estimated/simulated.
        workload_flops=0,
        payload_kb=0.0,

        success=(
            edge_result.get("status")
            == "completed"
        ),

        detail=(
            "Remote execution on edge-node-01 "
            f"({edge_result.get('platform', 'Linux')})"
        ),

        # No AWS EC2 compute is used by an Edge-only execution.
        cost_usd=0.0,
        cost_type="no_aws_compute_cost",
        pricing_instance=None,
        pricing_region=None,
        hourly_rate_usd=0.0,
    )


def run_cloud(
    model_size_mb: float,
    network_latency_ms: float | None = None,
    connectivity: int = 1,
    batch_size: int = 1
) -> InferenceResult:
    """
    Execute the workload remotely on the real AWS EC2 Cloud Compute Node.

    network_latency_ms is retained for API compatibility,
    but it is NOT used to simulate latency.
    """

    if connectivity == 0:
        return InferenceResult(
            mode="cloud",
            latency_ms=0.0,
            cpu_percent=0.0,
            memory_mb=0.0,
            fps_estimate=0.0,
            workload_flops=0,
            payload_kb=0.0,
            success=False,
            detail="Cloud execution unavailable: connectivity=0",
            cost_usd=0.0,
            pricing_instance=AWS_INSTANCE_TYPE,
            pricing_region=AWS_REGION,
            hourly_rate_usd=AWS_EC2_HOURLY_RATE_USD,
        )

    try:
        cloud_result = run_cloud_inference(
            model_size_mb=model_size_mb,
            batch_size=batch_size
        )

    except Exception as exc:
        return InferenceResult(
            mode="cloud",
            latency_ms=0.0,
            cpu_percent=0.0,
            memory_mb=0.0,
            fps_estimate=0.0,
            workload_flops=0,
            payload_kb=0.0,
            success=False,
            detail=f"AWS cloud execution failed: {exc}",
            cost_usd=0.0,
            pricing_instance=AWS_INSTANCE_TYPE,
            pricing_region=AWS_REGION,
            hourly_rate_usd=AWS_EC2_HOURLY_RATE_USD,
        )

    latency_ms = float(
        cloud_result.get("latency_ms", 0.0)
    )

    cpu_percent = float(
        cloud_result.get("cpu_percent", 0.0)
    )

    memory_delta_mb = float(
        cloud_result.get(
            "memory_delta_mb",
            0.0
        )
    )

    fps = (
        1000.0 / latency_ms
        if latency_ms > 0
        else 0.0
    )

    attributed_cost = _attributed_cloud_cost(
        latency_ms
    )

    return InferenceResult(
        mode="cloud",
        latency_ms=round(latency_ms, 2),
        cpu_percent=round(cpu_percent, 2),
        memory_mb=round(memory_delta_mb, 2),

        fps_estimate=round(
            fps,
            2
        ),

        workload_flops=0,
        payload_kb=0.0,

        success=(
            cloud_result.get("status")
            == "completed"
        ),

        detail=(
            "Remote execution on AWS EC2 "
            "cloud-node-01 "
            f"({cloud_result.get('platform', 'Linux')})"
        ),

        cost_usd=round(attributed_cost, 12),
        cost_type="attributed_compute_cost",
        pricing_instance=AWS_INSTANCE_TYPE,
        pricing_region=AWS_REGION,
        hourly_rate_usd=AWS_EC2_HOURLY_RATE_USD,
    )


def run(
    mode: str,
    model_size_mb: float,
    network_latency_ms: float | None = None,
    connectivity: int = 1,
    batch_size: int = 1
) -> Dict[str, Any]:

    if mode == "edge":
        return asdict(
            run_edge(
                model_size_mb,
                batch_size=batch_size
            )
        )

    if mode == "cloud":
        return asdict(
            run_cloud(
                model_size_mb,
                network_latency_ms,
                connectivity,
                batch_size=batch_size
            )
        )

    if mode == "hybrid":
        # Both routes are executed for real.
        edge = run_edge(
            model_size_mb,
            batch_size=batch_size
        )

        cloud = run_cloud(
            model_size_mb,
            network_latency_ms,
            connectivity,
            batch_size=batch_size
        )

        return {
            "mode": "hybrid",

            "success": (
                edge.success
                and cloud.success
            ),

            # Edge measurements
            "latency_ms": edge.latency_ms,
            "cpu_percent": edge.cpu_percent,
            "memory_mb": edge.memory_mb,
            "fps_estimate": edge.fps_estimate,

            # Cloud measurements
            "cloud_latency_ms": cloud.latency_ms,
            "cloud_cpu_percent": cloud.cpu_percent,
            "cloud_memory_mb": cloud.memory_mb,
            "cloud_fps_estimate": cloud.fps_estimate,

            "workload_flops": 0,
            "payload_kb": 0.0,

            # Hybrid uses AWS for its Cloud component.
            "cost_usd": cloud.cost_usd,
            "cost_type": "attributed_compute_cost",
            "pricing_instance": AWS_INSTANCE_TYPE,
            "pricing_region": AWS_REGION,
            "hourly_rate_usd": AWS_EC2_HOURLY_RATE_USD,

            "detail": (
                "Real hybrid execution: "
                "edge-node-01 + AWS cloud-node-01"
            ),
        }

    raise ValueError(
        f"Unknown inference mode: {mode}"
    )
