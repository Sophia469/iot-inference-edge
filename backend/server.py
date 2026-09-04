from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import json
import logging
import asyncio
import time
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
import uuid
from datetime import datetime, timezone, timedelta
from florence_runner import florence_runner
from telemetry import get_live_telemetry
from cloud_client import get_cloud_health, get_cloud_telemetry, run_cloud_vision, run_cloud_florence
from edge_client import run_edge_vision, run_edge_florence

from decision_engine import (
    registry as decision_registry,
    DecisionContext,
    FEATURES as DECISION_FEATURES,
    ROUTES as DECISION_ROUTES,
)
import decision_db


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="Edge-Cloud AI Orchestration API")
api_router = APIRouter(prefix="/api")


# ---------- Models ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Device(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    location: str
    device_type: str = "NVIDIA Jetson Nano"
    status: Literal["online", "offline", "degraded"] = "online"
    connected: bool = True
    deployed_model_id: Optional[str] = None
    deployed_model_name: Optional[str] = None
    cpu: float = 0.0
    memory: float = 0.0
    last_seen: str = Field(default_factory=_now_iso)


class ModelDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    task: str
    size_mb: float
    accuracy: float
    created_at: str = Field(default_factory=_now_iso)


class Deployment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str
    device_name: str
    model_id: str
    model_name: str
    model_version: str
    status: Literal["queued", "downloading", "installing", "active", "failed"] = "queued"
    started_at: str = Field(default_factory=_now_iso)
    completed_at: Optional[str] = None
    
class DeployRequest(BaseModel):
    device_id: str
    model_id: str


class InferenceRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str
    device_name: str
    mode: Literal["edge", "cloud"]
    latency_ms: float
    cpu: float
    memory: float
    fps: float
    cost_usd: float
    objects_detected: int
    top_label: str
    top_confidence: float
    connected: bool
    timestamp: str = Field(default_factory=_now_iso)


class InferenceBatch(BaseModel):
    records: List[InferenceRecord]


class NetworkToggle(BaseModel):
    connected: bool


# ---------- Seed data ----------
DEFAULT_DEVICES = [
    {"name": "jetson-edge-01", "location": "Factory Floor A"},
    {"name": "jetson-edge-02", "location": "Warehouse North"},
    {"name": "jetson-edge-03", "location": "Loading Dock"},
    {"name": "jetson-edge-04", "location": "Assembly Line B"},
]

DEFAULT_MODELS = [
    {"name": "YOLOv8n", "version": "1.0.0", "task": "object-detection", "size_mb": 6.2, "accuracy": 0.82},
    {"name": "YOLOv8s", "version": "1.2.1", "task": "object-detection", "size_mb": 22.5, "accuracy": 0.88},
    {"name": "MobileNet-SSD", "version": "2.1.0", "task": "object-detection", "size_mb": 9.4, "accuracy": 0.79},
    {"name": "EfficientDet-D0", "version": "1.0.3", "task": "object-detection", "size_mb": 15.1, "accuracy": 0.85},
]


async def seed_data():
    dev_count = await db.devices.count_documents({})
    if dev_count == 0:
        docs = [Device(**d).model_dump() for d in DEFAULT_DEVICES]
        await db.devices.insert_many(docs)
        logger.info("Seeded %d devices", len(docs))

    m_count = await db.models.count_documents({})
    if m_count == 0:
        docs = [ModelDoc(**m).model_dump() for m in DEFAULT_MODELS]
        await db.models.insert_many(docs)
        logger.info("Seeded %d models", len(docs))

    # Assign default model to devices missing one
    default_model = await db.models.find_one({"name": "YOLOv8n"}, {"_id": 0})
    if default_model:
        await db.devices.update_many(
            {"deployed_model_id": None},
            {"$set": {
                "deployed_model_id": default_model["id"],
                "deployed_model_name": f"{default_model['name']}@{default_model['version']}",
            }},
        )


# ---------- Routes ----------
@api_router.get("/")
async def root():
    return {"service": "edge-cloud-ai-orchestrator", "status": "ok"}


@api_router.get("/devices", response_model=List[Device])
async def list_devices():
    docs = await db.devices.find({}, {"_id": 0}).to_list(1000)
    return docs


@api_router.post("/devices/{device_id}/network", response_model=Device)
async def toggle_network(device_id: str, payload: NetworkToggle):
    status = "online" if payload.connected else "offline"
    result = await db.devices.find_one_and_update(
        {"id": device_id},
        {"$set": {"connected": payload.connected, "status": status, "last_seen": _now_iso()}},
        return_document=True,
        projection={"_id": 0},
    )
    if not result:
        raise HTTPException(status_code=404, detail="Device not found")
    return result


@api_router.get("/models", response_model=List[ModelDoc])
async def list_models():
    docs = await db.models.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return docs


@api_router.post("/deployments", response_model=Deployment)
async def create_deployment(req: DeployRequest):
    device = await db.devices.find_one({"id": req.device_id}, {"_id": 0})
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    model = await db.models.find_one({"id": req.model_id}, {"_id": 0})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    dep = Deployment(
        device_id=device["id"],
        device_name=device["name"],
        model_id=model["id"],
        model_name=model["name"],
        model_version=model["version"],
        status="active",
        completed_at=_now_iso(),
    )
    await db.deployments.insert_one(dep.model_dump())

    await db.devices.update_one(
        {"id": device["id"]},
        {"$set": {
            "deployed_model_id": model["id"],
            "deployed_model_name": f"{model['name']}@{model['version']}",
            "last_seen": _now_iso(),
        }},
    )
    return dep


@api_router.get("/deployments", response_model=List[Deployment])
async def list_deployments(limit: int = 20):
    docs = (
        await db.deployments.find({}, {"_id": 0})
        .sort("started_at", -1)
        .to_list(limit)
    )
    return docs


@api_router.post("/inferences")
async def record_inferences(batch: InferenceBatch):
    if not batch.records:
        return {"inserted": 0}
    docs = [r.model_dump() for r in batch.records]
    await db.inferences.insert_many(docs)
    # keep last 2000 records
    total = await db.inferences.count_documents({})
    if total > 2000:
        overflow = total - 2000
        old = await db.inferences.find({}, {"_id": 1}).sort("timestamp", 1).to_list(overflow)
        if old:
            await db.inferences.delete_many({"_id": {"$in": [d["_id"] for d in old]}})
    return {"inserted": len(docs)}


@api_router.get("/metrics/summary")
async def metrics_summary():
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    pipeline = [
        {"$match": {"timestamp": {"$gte": cutoff}}},
        {"$group": {
            "_id": "$mode",
            "avg_latency": {"$avg": "$latency_ms"},
            "avg_cpu": {"$avg": "$cpu"},
            "avg_memory": {"$avg": "$memory"},
            "avg_fps": {"$avg": "$fps"},
            "total_cost": {"$sum": "$cost_usd"},
            "count": {"$sum": 1},
        }},
    ]
    agg = await db.inferences.aggregate(pipeline).to_list(10)
    result = {"edge": None, "cloud": None}
    for r in agg:
        result[r["_id"]] = {
            "avg_latency": round(r["avg_latency"], 2),
            "avg_cpu": round(r["avg_cpu"], 2),
            "avg_memory": round(r["avg_memory"], 2),
            "avg_fps": round(r["avg_fps"], 2),
            "total_cost": round(r["total_cost"], 4),
            "count": r["count"],
        }
    return result


@api_router.get("/metrics/timeseries")
async def metrics_timeseries(limit: int = 60):
    docs = (
        await db.inferences.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .to_list(limit)
    )
    docs.reverse()
    return docs

# ---------- Real Infrastructure Telemetry ----------

@api_router.get("/telemetry/live")
async def telemetry_live():
    """
Returns real telemetry from the registered Edge Compute Node.
No simulated/random infrastructure metrics.
    No simulated/random infrastructure metrics.
    """
    try:
        telemetry = get_live_telemetry()

        return {
            "source": "real",
            "device_type": "edge",
            **telemetry,
        }

    except Exception as exc:
        logger.exception("Failed to collect live telemetry")
        raise HTTPException(
            status_code=500,
            detail=f"Telemetry collection failed: {exc}",
        ) from exc


# ---------- Live AI Decision ----------

@api_router.post("/decisions/predict-live")
async def decisions_predict_live(
    engine: Literal["rule", "dt", "rf", "ql"] = "rule",
    batch_size: int = 1,
    priority: int = 3,
    cost_budget_usd: float = 1.0,
    model_size_mb: float = 6.2,
):
    """
    LIVE orchestration pipeline.

    Uses real Edge telemetry, real AWS availability/RTT, the selected
    orchestration policy, and real execution on the selected compute node.
    No synthetic infrastructure metrics are generated here.
    """
    try:
             # 1. Real Edge telemetry with failover-safe handling
        try:
            edge_telemetry = get_live_telemetry()

            raw_cpu_available = edge_telemetry.get("cpu_available")
            raw_memory_available = edge_telemetry.get("memory_available")

            if raw_cpu_available is None or raw_memory_available is None:
                raise ValueError("Edge telemetry incomplete")

            edge_cpu_available = float(raw_cpu_available)
            edge_memory_available = float(raw_memory_available)
            edge_connected = bool(edge_telemetry.get("connected", True))

        except Exception as edge_exc:
            edge_telemetry = {
                "node": "edge-node-01",
                "status": "unreachable_or_saturated",
                "connected": False,
                "error": str(edge_exc),
            }
            edge_cpu_available = 0.0
            edge_memory_available = 0.0
            edge_connected = False

        # 2. Real Cloud health and Orchestrator -> AWS application RTT
        cloud_start = time.perf_counter()
        try:
            cloud_health = get_cloud_health()
            cloud_rtt_ms = (time.perf_counter() - cloud_start) * 1000.0
            cloud_connected = cloud_health.get("status") == "healthy"
        except Exception as cloud_exc:
            cloud_health = {
                "status": "unreachable",
                "error": str(cloud_exc),
            }
            cloud_rtt_ms = 9999.0
            cloud_connected = False

        # 3. Real Cloud telemetry
        try:
            cloud_telemetry = get_cloud_telemetry()
        except Exception as cloud_exc:
            cloud_telemetry = {
                "node": "cloud-node-01",
                "status": "unreachable",
                "error": str(cloud_exc),
            }

        # 4. Build live decision context
        # network_latency_ms represents the real offload RTT to AWS.
        ctx = DecisionContext(
            network_latency_ms=round(cloud_rtt_ms, 2),
            connectivity=1 if cloud_connected else 0,
            cpu_available=edge_cpu_available,
            memory_available=edge_memory_available,
            batch_size=batch_size,
            priority=priority,
            cost_budget_usd=cost_budget_usd,
            model_size_mb=model_size_mb,
        )

        # 5. Select route with the requested policy
        result = decision_registry.decide(ctx, engine=engine)
        selected_route = result.route 
                # Automatic failover: Edge unavailable -> Cloud
        failover_applied = False

        if not edge_connected and cloud_connected:
            selected_route = "cloud"
            failover_applied = True

        # 6. Execute the selected route for real
        import inference_runner

        execution = inference_runner.run(
            mode=selected_route,
            model_size_mb=model_size_mb,
            network_latency_ms=cloud_rtt_ms,
            connectivity=1 if cloud_connected else 0,
            batch_size=batch_size,
        )
                # Post-execution failover:
        # If Edge or Hybrid execution fails but Cloud is available,
        # retry the workload directly on the Cloud node.
        failover_reason = None

        if (
            not execution.get("success", False)
            and cloud_connected
            and selected_route in ("edge", "hybrid")
        ):
            failover_applied = True
            failover_reason = f"{selected_route}_execution_failed"
            selected_route = "cloud"

            execution = inference_runner.run(
                mode="cloud",
                model_size_mb=model_size_mb,
                network_latency_ms=cloud_rtt_ms,
                connectivity=1,
                batch_size=batch_size,
            )
                    # 7. Real Q-Learning feedback
        # Learn only from the actual execution result.
        q_learning_update = None

        if engine == "ql":
            from qlearning_agent import qlearning_agent

            execution_latency_ms = float(
                execution.get("latency_ms", 0.0)
            )

            execution_success = bool(
                execution.get("success", False)
            )

            execution_cost_usd = float(
                execution.get("cost_usd", 0.0)
            )

            execution_cpu_percent = float(
                execution.get("cpu_percent", 0.0)
            )

            resource_pressure = max(
                0.0,
                execution_cpu_percent - 70.0
            ) * 0.05

            q_learning_update = qlearning_agent.learn_from_real_execution(
                ctx=ctx,
                route=selected_route,
                latency_ms=execution_latency_ms,
                success=execution_success,
                cost_usd=execution_cost_usd,
                resource_pressure=resource_pressure,
                failover_applied=failover_applied,
            )

        # 8. Persist the real decision context
        context_data = {
            "network_latency_ms": round(cloud_rtt_ms, 2),
            "connectivity": 1 if cloud_connected else 0,
            "cpu_available": edge_cpu_available,
            "memory_available": edge_memory_available,
            "batch_size": batch_size,
            "priority": priority,
            "cost_budget_usd": cost_budget_usd,
            "model_size_mb": model_size_mb,
        }

        await decision_db.insert_decision(
            engine=result.engine,
            route=result.route,
            confidence=result.confidence,
            reason=result.reason,
            probabilities=result.probabilities,
            context=context_data,
            latency_us=result.latency_us,
        )

        # 9. Return infrastructure + decision + real execution
        return {
            "source": "real_infrastructure",

            "infrastructure": {
                "edge": edge_telemetry,
                "cloud": {
                    **cloud_telemetry,
                    "rtt_ms": round(cloud_rtt_ms, 2),
                    "connected": cloud_connected,
                },
            },

            "workload": {
                "batch_size": batch_size,
                "priority": priority,
                "cost_budget_usd": cost_budget_usd,
                "model_size_mb": model_size_mb,
            },

            "decision": {
                "policy_route": result.route,
                "route": selected_route,
                "failover_applied": failover_applied,
                "failover_reason": failover_reason,
                "confidence": result.confidence,
                "engine": result.engine,
                "reason": result.reason,
                "probabilities": result.probabilities,
                "decision_latency_us": result.latency_us,
            },

            "execution": execution,
            "learning": q_learning_update,
        }

    except Exception as exc:
        logger.exception("Live orchestration failed")
        raise HTTPException(
            status_code=500,
            detail=f"Live orchestration failed: {exc}",
        ) from exc


# ---------- App wiring ----------
# Include the router in the main app (moved to bottom after all endpoints registered)
# app.include_router(api_router)  # moved

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DecisionRequest(BaseModel):
    engine: Literal["rule", "dt", "rf", "ql"] = "rule"
    context: dict
    persist: bool = True


class TrainRequest(BaseModel):
    n_samples: int = 4000


class QTrainRequest(BaseModel):
    episodes: int = 5000


class RealInferenceRequest(BaseModel):
    mode: Literal["edge", "cloud", "hybrid"] = "edge"
    model_size_mb: float = 6.2
    network_latency_ms: float | None = None
    connectivity: int = 1
    batch_size: int = 1


class YoloInferenceRequest(BaseModel):
    mode: Literal["edge", "cloud", "hybrid"] = "edge"
    image_path: str
    confidence: float = 0.25
    model_name: str = "yolov8n.pt"


class FlorenceInferenceRequest(BaseModel):
    image_path: str
    task: str = "<MORE_DETAILED_CAPTION>"


class BenchmarkRequest(BaseModel):
    n_scenarios: int = 500
    seed: int = 2026
    save_artefacts: bool = True
    scenario_id: str = "mixed"


class BenchmarkCIRequest(BaseModel):
    scenario_id: str = "mixed"
    seeds: List[int] = [2026, 42, 7, 1234, 9999]
    n_scenarios: int = 200


class ReportRequest(BaseModel):
    scenario_id: str = "mixed"
    seeds: List[int] = [2026, 42, 7, 1234, 9999]
    n_scenarios: int = 200


class AssistantRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class CompleteBenchmarkRequest(BaseModel):
    seeds: List[int] = [2026, 42, 7, 1234, 9999]
    n_scenarios: int = 100
    include: Optional[List[str]] = None  # None => all 8 scenarios


LAST_BENCHMARK: dict = {"summary": None, "records_head": [], "meta": None}


@api_router.get("/decisions/status")
async def decisions_status():
    return decision_registry.status()


@api_router.post("/decisions/train")
async def decisions_train(req: TrainRequest):
    result = decision_registry.train_all(n=req.n_samples)
    return {"ok": True, "trained": result}


@api_router.post("/decisions/predict")
async def decisions_predict(req: DecisionRequest):
    try:
        ctx = DecisionContext(**{k: req.context[k] for k in DECISION_FEATURES})
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")
    try:
        result = decision_registry.decide(ctx, req.engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if req.persist:
        await decision_db.insert_decision(
            engine=result.engine, route=result.route, confidence=result.confidence,
            reason=result.reason, probabilities=result.probabilities,
            context=req.context, latency_us=result.latency_us,
        )
    # Enrich with structured explanation
    import explainer as _explainer
    feature_importance = None
    q_values = None
    if req.engine == "dt" and decision_registry.dt.trained:
        feature_importance = decision_registry.dt.metrics.get("feature_importance")
    elif req.engine == "rf" and decision_registry.rf.trained:
        feature_importance = decision_registry.rf.metrics.get("feature_importance")
    elif req.engine == "ql":
        try:
            from qlearning_agent import qlearning_agent as _q, state_index
            if _q.trained:
                q_values = _q.q[state_index(ctx)].tolist()
        except Exception:
            pass
    explanation = _explainer.explain(
        ctx, req.engine, result.route,
        feature_importance=feature_importance, q_values=q_values,
    )
    return {
        "route": result.route,
        "confidence": result.confidence,
        "engine": result.engine,
        "reason": result.reason,
        "probabilities": result.probabilities,
        "latency_us": result.latency_us,
        "explanation": explanation,
    }


@api_router.get("/decisions/history")
async def decisions_history(limit: int = 50, engine: Optional[str] = None):
    return await decision_db.recent_decisions(limit=limit, engine=engine)


@api_router.get("/decisions/distribution")
async def decisions_distribution(engine: Optional[str] = None, minutes: int = 30):
    return await decision_db.route_distribution(engine=engine, minutes=minutes)


@api_router.post("/decisions/qlearning/train")
async def qlearning_train(req: QTrainRequest):
    from qlearning_agent import qlearning_agent
    metrics = qlearning_agent.train(episodes=req.episodes)
    return {"ok": True, "trained": metrics}


@api_router.get("/decisions/qlearning/qtable")
async def qlearning_qtable(top_states: int = 12):
    from qlearning_agent import qlearning_agent
    if not qlearning_agent.trained:
        return {"trained": False, "rows": []}
    return {"trained": True, **qlearning_agent.q_table_summary(top_states=top_states)}


@api_router.post("/inference/run")
async def inference_run(req: RealInferenceRequest):
    """Runs a REAL compute-bound workload and returns measured metrics.

    Backs the academic validity claim: latencies, CPU% and memory are measured
    with time.perf_counter and psutil, not simulated with random numbers.
    """
    import inference_runner
    result = inference_runner.run(
        mode=req.mode,
        model_size_mb=req.model_size_mb,
        network_latency_ms=req.network_latency_ms,
        connectivity=req.connectivity,
        batch_size=req.batch_size,
    )
    return result

@api_router.post("/inference/camera-frame")
async def upload_camera_frame(file: UploadFile = File(...)):
    """Stores a browser camera frame for real inference execution."""
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Camera frame must be JPEG, PNG or WebP.",
        )

    capture_dir = ROOT_DIR / "runtime" / "camera"
    capture_dir.mkdir(parents=True, exist_ok=True)

    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[file.content_type]

    image_path = capture_dir / f"camera_{uuid.uuid4().hex}{suffix}"

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Camera frame is empty.",
        )

    image_path.write_bytes(data)

    return {
        "ok": True,
        "image_path": str(image_path.resolve()),
        "source": "pc_camera",
    }

@api_router.post("/inference/vision")
async def vision_inference(
    file: UploadFile = File(...),
    model: str = "yolo",
    confidence: float = 0.25,
    mode: str = "edge",
    engine: str = "rule",
):
    """Routes transient camera vision workloads to real Edge/Cloud nodes."""

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Image must be JPEG, PNG or WebP.",
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Image is empty.",
        )

    if mode not in {
        "edge",
        "cloud",
        "hybrid",
        "auto",
    }:
        raise HTTPException(
            status_code=400,
            detail="mode must be 'edge', 'cloud', 'hybrid' or 'auto'",
        )

    auto_decision = None
    requested_mode = mode

    if mode == "auto":
        # Real Edge telemetry
        try:
            edge_telemetry = get_live_telemetry()

            raw_cpu_available = edge_telemetry.get("cpu_available")
            raw_memory_available = edge_telemetry.get("memory_available")

            if raw_cpu_available is None or raw_memory_available is None:
                raise ValueError("Edge telemetry incomplete")

            edge_cpu_available = float(raw_cpu_available)
            edge_memory_available = float(raw_memory_available)
            edge_connected = bool(
                edge_telemetry.get("connected", True)
            )

        except Exception:
            edge_cpu_available = 0.0
            edge_memory_available = 0.0
            edge_connected = False

        # Real Orchestrator -> AWS application RTT
        cloud_start = time.perf_counter()

        try:
            cloud_health = get_cloud_health()

            cloud_rtt_ms = (
                time.perf_counter() - cloud_start
            ) * 1000.0

            cloud_connected = (
                cloud_health.get("status") == "healthy"
            )

        except Exception:
            cloud_rtt_ms = 9999.0
            cloud_connected = False

        # Workload-aware model characteristics.
        # YOLOv8n is lightweight; Florence-2 is substantially heavier.
        if model == "florence":
            workload_model_size_mb = 447.0
        else:
            workload_model_size_mb = 6.2

        # Build live decision context using real infrastructure
        # telemetry plus characteristics of the requested AI workload.
        ctx = DecisionContext(
            network_latency_ms=round(cloud_rtt_ms, 2),
            connectivity=1 if cloud_connected else 0,
            cpu_available=edge_cpu_available,
            memory_available=edge_memory_available,
            batch_size=1,
            priority=3,
            cost_budget_usd=1.0,
            model_size_mb=workload_model_size_mb,
        )

        # AI-driven route selection
        auto_decision = decision_registry.decide(
            ctx,
            engine=engine,
        )

        mode = auto_decision.route

        # Pre-execution failover
        if not edge_connected and cloud_connected:
            mode = "cloud"
    if model == "yolo":
        filename = (
            file.filename
            or "camera-frame.jpg"
        )

        content_type = (
            file.content_type
            or "image/jpeg"
        )

        if mode == "edge":
            result = run_edge_vision(
                image_bytes=data,
                filename=filename,
                content_type=content_type,
                confidence=confidence,
            )

            result["source"] = "pc_camera"
            result["stored"] = False
            result["orchestrated_route"] = "edge"
            result["requested_mode"] = requested_mode

            if auto_decision is not None:
                result["decision"] = {
                    "engine": auto_decision.engine,
                    "route": auto_decision.route,
                    "confidence": auto_decision.confidence,
                    "reason": auto_decision.reason,
                    "probabilities": auto_decision.probabilities,
                }

            return result

        if mode == "cloud":
            result = run_cloud_vision(
                image_bytes=data,
                filename=filename,
                content_type=content_type,
                confidence=confidence,
            )

            result["source"] = "pc_camera"
            result["stored"] = False
            result["orchestrated_route"] = "cloud"
            result["requested_mode"] = requested_mode

            if auto_decision is not None:
                result["decision"] = {
                    "engine": auto_decision.engine,
                    "route": auto_decision.route,
                    "confidence": auto_decision.confidence,
                    "reason": auto_decision.reason,
                    "probabilities": auto_decision.probabilities,
                }

            return result

        # HYBRID:
        # execute the same real camera frame on both nodes.
        edge_result = None
        cloud_result = None
        edge_error = None
        cloud_error = None

        try:
            edge_result = run_edge_vision(
                image_bytes=data,
                filename=filename,
                content_type=content_type,
                confidence=confidence,
            )
        except Exception as exc:
            edge_error = str(exc)

        try:
            cloud_result = run_cloud_vision(
                image_bytes=data,
                filename=filename,
                content_type=content_type,
                confidence=confidence,
            )
        except Exception as exc:
            cloud_error = str(exc)

        if edge_result is None and cloud_result is None:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Hybrid vision failed on both nodes. "
                    f"Edge: {edge_error}; "
                    f"Cloud: {cloud_error}"
                ),
            )

        # Prefer Edge detections as the primary visual result
        # to preserve the same response shape expected by the frontend.
        primary = (
            edge_result
            if edge_result is not None
            else cloud_result
        )

        result = {
            **primary,
            "mode": "hybrid",
            "execution": "hybrid",
            "source": "pc_camera",
            "stored": False,
            "orchestrated_route": "hybrid",
            "requested_mode": requested_mode,

            "edge_success": (
                edge_result is not None
            ),
            "cloud_success": (
                cloud_result is not None
            ),

            "edge_latency_ms": (
                edge_result.get(
                    "latency_ms"
                )
                if edge_result
                else None
            ),
            "edge_cpu_percent": (
                edge_result.get(
                    "cpu_percent"
                )
                if edge_result
                else None
            ),
            "edge_memory_delta_mb": (
                edge_result.get(
                    "memory_delta_mb"
                )
                if edge_result
                else None
            ),

            "cloud_latency_ms": (
                cloud_result.get(
                    "latency_ms"
                )
                if cloud_result
                else None
            ),
            "cloud_cpu_percent": (
                cloud_result.get(
                    "cpu_percent"
                )
                if cloud_result
                else None
            ),
            "cloud_memory_delta_mb": (
                cloud_result.get(
                    "memory_delta_mb"
                )
                if cloud_result
                else None
            ),

            "edge_result": edge_result,
            "cloud_result": cloud_result,

            "detail": (
                "Real hybrid YOLO execution: "
                "edge-node-01 + AWS cloud-node-01"
            ),
        }

        if auto_decision is not None:
            result["decision"] = {
                "engine": auto_decision.engine,
                "route": auto_decision.route,
                "confidence": auto_decision.confidence,
                "reason": auto_decision.reason,
                "probabilities": auto_decision.probabilities,
            }

        return result
    elif model == "florence":
        filename = (
            file.filename
            or "camera-frame.jpg"
        )

        content_type = (
            file.content_type
            or "image/jpeg"
        )

        task_prompt = "<MORE_DETAILED_CAPTION>"

        if mode == "edge":
            result = await asyncio.to_thread(
                run_edge_florence,
                image_bytes=data,
                filename=filename,
                content_type=content_type,
                task_prompt=task_prompt,
            )

            result["source"] = "pc_camera"
            result["stored"] = False
            result["orchestrated_route"] = "edge"
            result["requested_mode"] = requested_mode

            if auto_decision is not None:
                result["decision"] = {
                    "engine": auto_decision.engine,
                    "route": auto_decision.route,
                    "confidence": auto_decision.confidence,
                    "reason": auto_decision.reason,
                    "probabilities": auto_decision.probabilities,
                }

            return result

        if mode == "cloud":
            result = await asyncio.to_thread(
                run_cloud_florence,
                image_bytes=data,
                filename=filename,
                content_type=content_type,
                task_prompt=task_prompt,
            )

            result["source"] = "pc_camera"
            result["stored"] = False
            result["orchestrated_route"] = "cloud"
            result["requested_mode"] = requested_mode

            if auto_decision is not None:
                result["decision"] = {
                    "engine": auto_decision.engine,
                    "route": auto_decision.route,
                    "confidence": auto_decision.confidence,
                    "reason": auto_decision.reason,
                    "probabilities": auto_decision.probabilities,
                }

            return result
        # HYBRID:
        # Edge performs fast YOLO detection while Cloud performs
        # richer Florence-2 semantic interpretation.
        edge_result = None
        cloud_result = None
        edge_error = None
        cloud_error = None

        async def _run_hybrid_edge():
            return await asyncio.to_thread(
                run_edge_vision,
                image_bytes=data,
                filename=filename,
                content_type=content_type,
                confidence=confidence,
            )

        async def _run_hybrid_cloud():
            return await asyncio.to_thread(
                run_cloud_florence,
                image_bytes=data,
                filename=filename,
                content_type=content_type,
                task_prompt=task_prompt,
            )

        edge_task = asyncio.create_task(
            _run_hybrid_edge()
        )

        cloud_task = asyncio.create_task(
            _run_hybrid_cloud()
        )

        edge_outcome, cloud_outcome = await asyncio.gather(
            edge_task,
            cloud_task,
            return_exceptions=True,
        )

        if isinstance(edge_outcome, Exception):
            edge_error = str(edge_outcome)
        else:
            edge_result = edge_outcome

        if isinstance(cloud_outcome, Exception):
            cloud_error = str(cloud_outcome)
        else:
            cloud_result = cloud_outcome

        if edge_result is None and cloud_result is None:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Hybrid vision failed on both nodes. "
                    f"Edge YOLO: {edge_error}; "
                    f"Cloud Florence: {cloud_error}"
                ),
            )

        detections = (
            edge_result.get("detections", [])
            if edge_result
            else []
        )

        description = (
            cloud_result.get("description")
            if cloud_result
            else None
        )

        edge_latency = (
            edge_result.get("latency_ms")
            if edge_result
            else None
        )

        cloud_latency = (
            cloud_result.get("latency_ms")
            if cloud_result
            else None
        )

        available_latencies = [
            value
            for value in (
                edge_latency,
                cloud_latency,
            )
            if value is not None
        ]

        hybrid_latency_ms = (
            max(available_latencies)
            if available_latencies
            else None
        )

        result = {
            "status": "completed",
            "success": (
                edge_result is not None
                or cloud_result is not None
            ),
            "node": "edge-node-01 + cloud-node-01",
            "execution": "hybrid",
            "mode": "hybrid",
            "workload_type": "hybrid_vision",
            "source": "pc_camera",
            "stored": False,
            "orchestrated_route": "hybrid",
            "requested_mode": requested_mode,

            "hybrid_strategy": (
                "edge_yolo_plus_cloud_florence"
            ),

            "latency_ms": hybrid_latency_ms,

            "detections_count": len(detections),
            "detections": detections,
            "description": description,

            "edge_success": (
                edge_result is not None
            ),
            "cloud_success": (
                cloud_result is not None
            ),

            "edge_workload": "yolo",
            "cloud_workload": "florence",

            "edge_latency_ms": edge_latency,
            "cloud_latency_ms": cloud_latency,

            "edge_cpu_percent": (
                edge_result.get("cpu_percent")
                if edge_result
                else None
            ),

            "cloud_cpu_percent": (
                cloud_result.get("cpu_percent")
                if cloud_result
                else None
            ),

            "edge_memory_delta_mb": (
                edge_result.get("memory_delta_mb")
                if edge_result
                else None
            ),

            "cloud_memory_delta_mb": (
                cloud_result.get("memory_delta_mb")
                if cloud_result
                else None
            ),

            "edge_result": edge_result,
            "cloud_result": cloud_result,

            "detail": (
                "Real cooperative hybrid vision execution: "
                "YOLO on edge-node-01 + Florence-2 on "
                "AWS cloud-node-01"
            ),
        }

        if auto_decision is not None:
            result["decision"] = {
                "engine": auto_decision.engine,
                "route": auto_decision.route,
                "confidence": auto_decision.confidence,
                "reason": auto_decision.reason,
                "probabilities": auto_decision.probabilities,
            }

        return result

    else:
        raise HTTPException(
            status_code=400,
            detail="model must be 'yolo' or 'florence'",
        )

@api_router.post("/inference/yolo")
async def yolo_inference_run(req: YoloInferenceRequest):
    """Runs a real YOLO computer-vision workload."""
    try:
        from yolo_runner import run_yolo

        result = run_yolo(
            image_path=req.image_path,
            model_name=req.model_name,
            confidence=req.confidence,
            mode=req.mode,
        )
        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logging.exception("YOLO inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"YOLO inference failed: {exc}",
        ) from exc


@api_router.post("/inference/florence")
async def florence_inference_run(req: FlorenceInferenceRequest):
    """Runs a Florence-2 visual assessment."""
    try:
        result = florence_runner.analyse(
            image_path=req.image_path,
            task_prompt=req.task,
        )
        return result

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logging.exception("Florence inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Florence inference failed: {exc}",
        ) from exc


@api_router.post("/benchmark/run")
async def benchmark_run(req: BenchmarkRequest):
    """Runs the comparative benchmark of all 4 orchestration policies.

    The research instrument: same reproducible scenarios fed to all engines.
    Returns per-engine summary + saves CSV + PNG charts under /tmp/benchmark.
    """
    global LAST_BENCHMARK
    from benchmark import run_benchmark
    output_dir = "/tmp/benchmark" if req.save_artefacts else None
    result = run_benchmark(
        n_scenarios=req.n_scenarios,
        seed=req.seed,
        output_dir=output_dir,
        scenario_id=req.scenario_id,
    )
    LAST_BENCHMARK = {
        "summary": result["summary"],
        "meta": result["meta"],
        "records_head": result["records"][:5],
        "artefacts_dir": output_dir,
    }
    return {
        "summary": result["summary"],
        "meta": result["meta"],
        "artefacts_dir": output_dir,
        "n_records": len(result["records"]),
    }


@api_router.get("/benchmark/last")
async def benchmark_last():
    return LAST_BENCHMARK


@api_router.get("/benchmark/artefact/{name}")
async def benchmark_artefact(name: str):
    """Serve a specific benchmark artefact (chart PNG or CSV/JSON)."""
    from fastapi.responses import FileResponse
    allowed = {
        "chart_latency_boxplot.png", "chart_decision_time.png",
        "chart_cost.png", "chart_route_distribution.png", "chart_reward.png",
        "benchmark_records.csv", "benchmark_summary.json",
        "benchmark_ci.json", "thesis_report.pdf",
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="artefact not found")
    path = f"/tmp/benchmark/{name}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="run a benchmark first")
    return FileResponse(path)


@api_router.get("/scenarios")
async def list_scenarios():
    import scenarios as _s
    return {"scenarios": _s.list_scenarios()}


@api_router.post("/benchmark/ci")
async def benchmark_ci(req: BenchmarkCIRequest):
    from benchmark import run_benchmark_ci
    return run_benchmark_ci(
        scenario_id=req.scenario_id,
        seeds=req.seeds,
        n_scenarios=req.n_scenarios,
        output_dir="/tmp/benchmark",
    )


@api_router.post("/report/generate")
async def report_generate(req: ReportRequest):
    from report import generate_thesis_report
    return generate_thesis_report(
        scenario_id=req.scenario_id,
        seeds=req.seeds,
        n_scenarios=req.n_scenarios,
        output_path="/tmp/benchmark/thesis_report.pdf",
    )


@api_router.get("/report/real-evidence")
async def report_real_evidence():
    """
    Generates a PDF using REAL orchestration evidence only.

    Includes:
    - live infrastructure telemetry
    - recorded real decisions
    - Q-Learning real-execution state

    Synthetic benchmark, CI, random simulation and attributed cost
    are intentionally excluded.
    """
    from fastapi.responses import FileResponse
    from real_evidence_report import generate_real_evidence_report
    decisions = await decision_db.recent_decisions(limit=30)

    try:
        from qlearning_agent import qlearning_agent

        if qlearning_agent.trained:
            qtable = {
                "trained": True,
                **qlearning_agent.q_table_summary(top_states=12),
            }
        else:
            qtable = {
                "trained": False,
                "rows": [],
            }
    except Exception as exc:
        qtable = {
            "trained": False,
            "rows": [],
            "detail": str(exc),
        }

    try:
        import asyncio

        telemetry = await asyncio.wait_for(
            asyncio.to_thread(get_live_telemetry),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        telemetry = {
            "status": "unavailable",
            "detail": "Real telemetry timed out after 5 seconds.",
        }
    except Exception as exc:
        telemetry = {
            "status": "unavailable",
            "detail": str(exc),
        }

    output_path = "/tmp/real_evidence_report.pdf"

    # Fresh real execution validation for report evidence.
    # Only successful executions will be rendered by the PDF generator.
    import inference_runner

    executions = []

    for execution_mode in ("edge", "cloud", "hybrid"):
        try:
            execution = inference_runner.run(
                mode=execution_mode,
                model_size_mb=6.2,
                batch_size=1,
            )

            if execution.get("success") is True:
                executions.append(execution)

        except Exception:
            # A failed/unavailable route is not included as a
            # performance measurement in the evidence report.
            pass

    generate_real_evidence_report(
        decisions=decisions,
        qtable=qtable,
        telemetry=telemetry,
        executions=executions,
        output_path=output_path,
    )

    return FileResponse(
        output_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'inline; filename="experimental_evaluation_report.pdf"'
        },
    )

@api_router.post("/assistant/ask")
async def assistant_ask(req: AssistantRequest):
    """Answer questions using real orchestration evidence only."""
    import assistant as _assistant
    import asyncio

    recent = await decision_db.recent_decisions(limit=30)

    try:
        from qlearning_agent import qlearning_agent

        if qlearning_agent.trained:
            qlearning_state = {
                "trained": True,
                **qlearning_agent.q_table_summary(top_states=12),
            }
        else:
            qlearning_state = {
                "trained": False,
                "rows": [],
            }
    except Exception as exc:
        qlearning_state = {
            "trained": False,
            "rows": [],
            "detail": str(exc),
        }

    try:
        live_telemetry = await asyncio.wait_for(
            asyncio.to_thread(get_live_telemetry),
            timeout=5.0,
        )
        live_telemetry = {
            "source": "real",
            "device_type": "edge",
            **live_telemetry,
        }
    except asyncio.TimeoutError:
        live_telemetry = {
            "source": "real",
            "status": "unavailable",
            "detail": "Real telemetry timed out after 5 seconds.",
        }
    except Exception as exc:
        live_telemetry = {
            "source": "real",
            "status": "unavailable",
            "detail": str(exc),
        }

    context = await _assistant.build_context(
        recent_decisions=recent,
        live_telemetry=live_telemetry,
        qlearning_state=qlearning_state,
    )

    result = await _assistant.ask(
        question=req.question,
        context=context,
        session_id=req.session_id,
    )
    return result

_COMPLETE_TASK = None


@api_router.post("/experiments/run_complete")
async def experiments_run_complete(req: CompleteBenchmarkRequest):
    """Kick off a background job that runs all scenarios Ã— CI + master summary."""
    global _COMPLETE_TASK
    import experiments as _exp
    if _exp.JOB_STATE.get("status") == "running":
        raise HTTPException(status_code=409, detail="A complete benchmark is already running")

    def _run():
        _exp.run_complete_experiment(
            seeds=req.seeds, n_scenarios=req.n_scenarios, include=req.include,
        )

    loop = asyncio.get_event_loop()
    _COMPLETE_TASK = loop.run_in_executor(None, _run)
    return {"started": True, **_exp.get_status()}


@api_router.get("/experiments/status")
async def experiments_status():
    import experiments as _exp
    return _exp.get_status()


@api_router.get("/experiments/master_summary.{ext}")
async def experiments_master_download(ext: str):
    from fastapi.responses import FileResponse
    if ext not in {"pdf", "csv", "json"}:
        raise HTTPException(status_code=404, detail="unknown format")
    p = f"/app/experiments/master_summary.{ext}"
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="run a complete benchmark first")
    media = {"pdf": "application/pdf", "csv": "text/csv",
             "json": "application/json"}[ext]
    return FileResponse(p, filename=f"master_summary.{ext}", media_type=media)


@api_router.get("/experiments/download_zip")
async def experiments_download_zip():
    from fastapi.responses import FileResponse
    import experiments as _exp
    p = _exp.zip_all_experiments()
    return FileResponse(str(p), filename="experiments_bundle.zip",
                        media_type="application/zip")


# Include the router in the main app after all endpoints are registered
app.include_router(api_router)
@app.on_event("startup")
async def on_startup():
    await seed_data()
    await decision_db.init_db()

    # Q-Learning now learns only from real Edge-Cloud executions.
    from qlearning_agent import qlearning_agent

    if not qlearning_agent.trained:
        logger.info(
            "Q-Learning has no real-execution experience yet. "
            "Starting with an empty real Q-table."
        )
    else:
        logger.info(
            "Q-Learning real-execution Q-table loaded successfully."
        )
@app.on_event("shutdown")
async def on_shutdown():
    client.close()



























