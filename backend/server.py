from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import json
import logging
import random
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
import uuid
from datetime import datetime, timezone, timedelta

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
    network_latency_ms: float = 100.0
    connectivity: int = 1
    batch_size: int = 1


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


@api_router.post("/assistant/ask")
async def assistant_ask(req: AssistantRequest):
    """Answer a natural-language question about the experimental data."""
    import assistant as _assistant

    # Reuse the in-memory last benchmark + latest CI on disk
    last_ci = None
    ci_path = "/tmp/benchmark/benchmark_ci.json"
    if os.path.exists(ci_path):
        try:
            with open(ci_path) as f:
                last_ci = json.load(f)
        except Exception:
            last_ci = None

    # Get last 30 decisions
    recent = await decision_db.recent_decisions(limit=30)
    context = await _assistant.build_context(
        last_benchmark=LAST_BENCHMARK,
        last_ci=last_ci,
        recent_decisions=recent,
    )
    result = await _assistant.ask(
        question=req.question, context=context, session_id=req.session_id
    )
    return result


_COMPLETE_TASK = None


@api_router.post("/experiments/run_complete")
async def experiments_run_complete(req: CompleteBenchmarkRequest):
    """Kick off a background job that runs all scenarios × CI + master summary."""
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
    # Warm up ML models so first predict is fast
    if not decision_registry.dt.trained or not decision_registry.rf.trained:
        logger.info("Training decision engines on startup...")
        decision_registry.train_all(n=3000)
        logger.info("Decision engines trained.")
    # Warm up Q-Learning agent
    from qlearning_agent import qlearning_agent
    if not qlearning_agent.trained:
        logger.info("Training Q-Learning agent on startup...")
        qlearning_agent.train(episodes=5000)
        logger.info("Q-Learning agent trained.")


@app.on_event("shutdown")
async def on_shutdown():
    client.close()
