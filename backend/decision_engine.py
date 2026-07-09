"""
Decision Engine — Edge/Cloud Orchestration
Given real-time context (network, load, cost, model), decides where to route inference:
 - edge   : run locally on Jetson (low latency, low cost, offline-tolerant)
 - cloud  : offload to AWS EC2 (high accuracy models, heavy load)
 - hybrid : run on edge + async replicate to cloud for logging/re-check

Three engines implemented:
 1. RuleBased      — hand-crafted heuristics (interpretable baseline)
 2. DecisionTree   — sklearn DecisionTreeClassifier
 3. RandomForest   — sklearn RandomForestClassifier

Training data is synthesized from the same rule-based policy + gaussian noise
so the ML models learn a realistic (but non-trivial) policy surface.
"""
from __future__ import annotations
import os
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal, List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


ARTEFACT_DIR = Path(__file__).parent / "artefacts"
ARTEFACT_DIR.mkdir(exist_ok=True)

Route = Literal["edge", "cloud", "hybrid"]
ROUTES: List[Route] = ["edge", "cloud", "hybrid"]
FEATURES = [
    "network_latency_ms",   # 0 = offline (very high in practice)
    "connectivity",         # 0/1 (offline/online)
    "cpu_available",        # 0..100
    "memory_available",     # 0..100
    "batch_size",           # 1..32
    "priority",             # 1..5 (5 = critical, prefers accuracy)
    "cost_budget_usd",      # remaining $ per hour
    "model_size_mb",        # bigger => favor cloud
]


@dataclass
class DecisionContext:
    network_latency_ms: float
    connectivity: int
    cpu_available: float
    memory_available: float
    batch_size: int
    priority: int
    cost_budget_usd: float
    model_size_mb: float

    def as_vector(self) -> np.ndarray:
        return np.array([[getattr(self, f) for f in FEATURES]], dtype=float)


@dataclass
class DecisionResult:
    route: Route
    confidence: float
    engine: str
    reason: str
    probabilities: Dict[str, float]
    latency_us: float


# ---------- 1. Rule-Based ----------
def rule_based_decide(ctx: DecisionContext) -> DecisionResult:
    t0 = time.perf_counter()
    reasons = []
    # Hard rules
    if ctx.connectivity == 0:
        reasons.append("offline → edge only")
        return _wrap("edge", 1.0, "RuleBased", reasons, {"edge": 1.0, "cloud": 0.0, "hybrid": 0.0}, t0)

    if ctx.cost_budget_usd <= 0.001:
        reasons.append("cost budget exhausted → edge only")
        return _wrap("edge", 0.95, "RuleBased", reasons, {"edge": 0.95, "cloud": 0.02, "hybrid": 0.03}, t0)

    # Score cloud
    cloud_score = 0.0
    if ctx.network_latency_ms < 80:
        cloud_score += 0.25
        reasons.append("net<80ms")
    if ctx.cpu_available < 40:
        cloud_score += 0.30
        reasons.append("cpu low")
    if ctx.memory_available < 40:
        cloud_score += 0.20
        reasons.append("mem low")
    if ctx.batch_size >= 8:
        cloud_score += 0.15
        reasons.append("batch≥8")
    if ctx.model_size_mb >= 20:
        cloud_score += 0.15
        reasons.append("model≥20MB")
    if ctx.priority >= 4:
        cloud_score += 0.20
        reasons.append("prio≥4")

    edge_score = 1.0 - cloud_score

    # Hybrid when clearly borderline + connectivity ok + cost ok
    if 0.35 <= cloud_score <= 0.60 and ctx.cost_budget_usd > 0.01:
        route = "hybrid"
        probs = {"edge": edge_score * 0.6, "cloud": cloud_score * 0.6, "hybrid": 0.4}
        _normalize(probs)
        reasons.append("borderline → hybrid replicate")
    elif cloud_score > 0.5:
        route = "cloud"
        probs = {"edge": edge_score, "cloud": cloud_score, "hybrid": 0.0}
        _normalize(probs)
    else:
        route = "edge"
        probs = {"edge": edge_score, "cloud": cloud_score, "hybrid": 0.0}
        _normalize(probs)

    return _wrap(route, probs[route], "RuleBased", reasons, probs, t0)


def _normalize(d: Dict[str, float]) -> None:
    s = sum(d.values())
    if s <= 0:
        for k in d:
            d[k] = 1.0 / len(d)
        return
    for k in d:
        d[k] = d[k] / s


def _wrap(route: Route, conf: float, engine: str, reasons: List[str],
          probs: Dict[str, float], t0: float) -> DecisionResult:
    return DecisionResult(
        route=route,
        confidence=round(float(conf), 4),
        engine=engine,
        reason="; ".join(reasons) if reasons else "default policy",
        probabilities={k: round(float(v), 4) for k, v in probs.items()},
        latency_us=round((time.perf_counter() - t0) * 1_000_000, 2),
    )


# ---------- 2 & 3. ML engines ----------
class MLDecisionEngine:
    """Wraps a sklearn estimator with train/predict/persist."""
    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator
        self.trained = False
        self.metrics: Dict[str, Any] = {}
        self.path = ARTEFACT_DIR / f"{name.lower()}.joblib"

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        self.estimator.fit(X_train, y_train)
        y_pred = self.estimator.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        self.trained = True
        importances = getattr(self.estimator, "feature_importances_", None)
        self.metrics = {
            "accuracy": round(acc, 4),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "trained_at": time.time(),
            "feature_importance": (
                {f: round(float(v), 4) for f, v in zip(FEATURES, importances)}
                if importances is not None else None
            ),
        }
        joblib.dump({"estimator": self.estimator, "metrics": self.metrics}, self.path)
        return self.metrics

    def load(self) -> bool:
        if not self.path.exists():
            return False
        blob = joblib.load(self.path)
        self.estimator = blob["estimator"]
        self.metrics = blob["metrics"]
        self.trained = True
        return True

    def predict(self, ctx: DecisionContext) -> DecisionResult:
        if not self.trained:
            raise RuntimeError(f"{self.name} not trained yet")
        t0 = time.perf_counter()
        x = ctx.as_vector()
        proba = self.estimator.predict_proba(x)[0]
        classes = list(self.estimator.classes_)
        probs = {r: 0.0 for r in ROUTES}
        for cls, p in zip(classes, proba):
            probs[cls] = float(p)
        route = max(probs, key=probs.get)
        _normalize(probs)
        return _wrap(route, probs[route], self.name,
                     [f"top-feature: {max(self.metrics.get('feature_importance', {'?':0}).items(), key=lambda kv: kv[1])[0]}"] if self.metrics.get("feature_importance") else [],
                     probs, t0)


# ---------- Synthetic dataset ----------
def _sample_context() -> DecisionContext:
    connectivity = 1 if random.random() > 0.15 else 0
    return DecisionContext(
        network_latency_ms=(random.uniform(20, 500) if connectivity else 9999.0),
        connectivity=connectivity,
        cpu_available=random.uniform(5, 95),
        memory_available=random.uniform(10, 95),
        batch_size=random.choice([1, 1, 2, 4, 8, 16, 32]),
        priority=random.choice([1, 2, 2, 3, 3, 4, 5]),
        cost_budget_usd=random.choice([0.0, 0.005, 0.05, 0.5, 1.0, 5.0, 10.0]),
        model_size_mb=random.choice([6.2, 9.4, 15.1, 22.5, 40.0]),
    )


def build_synthetic_dataset(n: int = 4000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    X, y = [], []
    for _ in range(n):
        ctx = _sample_context()
        # Label from rule-based policy (with 8% label noise for realism)
        label = rule_based_decide(ctx).route
        if random.random() < 0.08:
            label = random.choice([r for r in ROUTES if r != label])
        X.append(ctx.as_vector().flatten())
        y.append(label)
    return np.array(X), np.array(y)


# ---------- Registry ----------
class DecisionEngineRegistry:
    def __init__(self):
        self.dt = MLDecisionEngine("DecisionTree",
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42))
        self.rf = MLDecisionEngine("RandomForest",
            RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1))
        # try load previously trained
        self.dt.load()
        self.rf.load()

    def train_all(self, n: int = 4000) -> Dict[str, Any]:
        X, y = build_synthetic_dataset(n=n)
        return {
            "DecisionTree": self.dt.train(X, y),
            "RandomForest": self.rf.train(X, y),
            "dataset": {"n_samples": int(len(y)), "features": FEATURES, "classes": ROUTES},
        }

    def decide(self, ctx: DecisionContext, engine: str) -> DecisionResult:
        engine = engine.lower()
        if engine in ("rule", "rule-based", "rulebased"):
            return rule_based_decide(ctx)
        if engine in ("dt", "decisiontree", "decision-tree"):
            if not self.dt.trained:
                self.train_all()
            return self.dt.predict(ctx)
        if engine in ("rf", "randomforest", "random-forest"):
            if not self.rf.trained:
                self.train_all()
            return self.rf.predict(ctx)
        if engine in ("ql", "qlearning", "q-learning"):
            from qlearning_agent import qlearning_agent
            if not qlearning_agent.trained:
                qlearning_agent.train(episodes=5000)
            return qlearning_agent.predict(ctx)
        raise ValueError(f"unknown engine: {engine}")

    def status(self) -> Dict[str, Any]:
        from qlearning_agent import qlearning_agent
        return {
            "RuleBased": {"trained": True, "metrics": {"accuracy": 1.0, "note": "deterministic baseline"}},
            "DecisionTree": {"trained": self.dt.trained, "metrics": self.dt.metrics},
            "RandomForest": {"trained": self.rf.trained, "metrics": self.rf.metrics},
            "QLearning": {"trained": qlearning_agent.trained, "metrics": qlearning_agent.metrics},
            "features": FEATURES,
            "classes": ROUTES,
        }


registry = DecisionEngineRegistry()
