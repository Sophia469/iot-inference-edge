from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import psutil
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

_MODEL_CACHE: dict[str, YOLO] = {}


def _get_model(model_name: str) -> YOLO:
    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    cache_key = str(model_path)

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = YOLO(str(model_path))

    return _MODEL_CACHE[cache_key]


def run_yolo(
    image_path: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    mode: str = "edge",
) -> Dict[str, Any]:

    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    model = _get_model(model_name)
    process = psutil.Process()

    # ---------------------------------------------------------
    # REAL RESOURCE MEASUREMENT - BEFORE INFERENCE
    # ---------------------------------------------------------

    memory_before = process.memory_info().rss / (1024 * 1024)

    cpu_times_before = process.cpu_times()
    wall_start = time.perf_counter()

    # ---------------------------------------------------------
    # REAL YOLO INFERENCE
    # ---------------------------------------------------------

    results = model.predict(
        source=str(image),
        conf=confidence,
        verbose=False,
    )

    # ---------------------------------------------------------
    # REAL RESOURCE MEASUREMENT - AFTER INFERENCE
    # ---------------------------------------------------------

    wall_end = time.perf_counter()
    cpu_times_after = process.cpu_times()

    latency_ms = (wall_end - wall_start) * 1000.0

    # CPU time actually consumed by the process during
    # the measured YOLO inference window.
    cpu_time_used = (
        (cpu_times_after.user - cpu_times_before.user)
        + (cpu_times_after.system - cpu_times_before.system)
    )

    wall_seconds = max(wall_end - wall_start, 1e-9)

    # Normalise process CPU utilisation across all logical CPUs.
    # This keeps the dashboard metric on a 0-100% scale even
    # when PyTorch/YOLO uses several CPU cores simultaneously.
    cpu_count = psutil.cpu_count(logical=True) or 1

    cpu_percent = (
        (cpu_time_used / wall_seconds) * 100.0
    ) / cpu_count

    cpu_percent = min(max(cpu_percent, 0.0), 100.0)

    memory_after = process.memory_info().rss / (1024 * 1024)

    # ---------------------------------------------------------
    # YOLO DETECTIONS
    # ---------------------------------------------------------

    result = results[0]
    detections = []

    if result.boxes is not None:
        names = result.names

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            coordinates = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": round(score, 4),
                    "bounding_box": {
                        "x1": round(coordinates[0], 2),
                        "y1": round(coordinates[1], 2),
                        "x2": round(coordinates[2], 2),
                        "y2": round(coordinates[3], 2),
                    },
                }
            )

    # ---------------------------------------------------------
    # DERIVED PERFORMANCE METRICS
    # ---------------------------------------------------------

    fps_estimate = 1000.0 / max(latency_ms, 0.001)

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    return {
        "workload_type": "computer_vision",
        "mode": mode,
        "model_name": model_name,
        "image_path": str(image),
        "confidence_threshold": confidence,

        # Real measured performance
        "latency_ms": round(latency_ms, 2),
        "cpu_percent": round(cpu_percent, 2),
        "memory_before_mb": round(memory_before, 2),
        "memory_after_mb": round(memory_after, 2),
        "memory_delta_mb": round(
            memory_after - memory_before,
            2,
        ),

        # Derived throughput
        "fps_estimate": round(fps_estimate, 2),

        # Real YOLO detections
        "detections_count": len(detections),
        "detections": detections,

        "success": True,
        "detail": "real YOLOv8 object detection workload",
    }