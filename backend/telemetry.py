import time
from datetime import datetime, timezone

from edge_client import (
    get_edge_telemetry,
    get_edge_health,
)


def measure_edge_latency():
    """
    Measure the real application-level RTT between the Orchestrator
    and the Edge Compute Node.

    The timer includes the HTTP request to the Edge /health endpoint.
    """

    start = time.perf_counter()

    try:
        health = get_edge_health()

        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return {
            "connected": health.get("status") == "healthy",
            "latency_ms": round(latency_ms, 2),
        }

    except Exception:
        return {
            "connected": False,
            "latency_ms": None,
        }


def get_live_telemetry():
    """
    Collect real telemetry from the Edge Compute Node.

    CPU, memory and disk values are measured inside edge-node-01.
    Network latency is measured from the Orchestrator to the Edge API.
    """

    try:
        edge = get_edge_telemetry()

    except Exception:
        return {
            "node": "edge-node-01",
            "platform": "Linux",

            "cpu_percent": None,
            "cpu_available": None,
            "cpu_count": None,

            "memory_percent": None,
            "memory_available": None,
            "memory_total_gb": None,
            "memory_available_gb": None,

            "disk_percent": None,
            "disk_free_gb": None,

            "network_latency_ms": None,
            "connected": False,

            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    network = measure_edge_latency()

    cpu_percent = float(
        edge.get("cpu_percent", 0)
    )

    memory_percent = float(
        edge.get("memory_percent", 0)
    )

    return {
        "node": edge.get(
            "node",
            "edge-node-01"
        ),
        "platform": edge.get(
            "platform",
            "Linux"
        ),

        "cpu_percent": round(
            cpu_percent,
            2
        ),
        "cpu_available": round(
            100.0 - cpu_percent,
            2
        ),
        "cpu_count": edge.get(
            "cpu_count"
        ),

        "memory_percent": round(
            memory_percent,
            2
        ),
        "memory_available": round(
            100.0 - memory_percent,
            2
        ),
        "memory_total_gb": edge.get(
            "memory_total_gb"
        ),
        "memory_available_gb": edge.get(
            "memory_available_gb"
        ),

        "disk_percent": edge.get(
            "disk_percent"
        ),
        "disk_free_gb": edge.get(
            "disk_free_gb"
        ),

        "network_latency_ms": network[
            "latency_ms"
        ],
        "connected": network[
            "connected"
        ],

        "timestamp": edge.get(
            "timestamp",
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }