"""
Decision Engine â€” Edge/Cloud Orchestration

LIVE DECISION PATH
------------------
The decision engines receive a DecisionContext supplied by the
orchestrator.

In LIVE mode this context is populated from real infrastructure
measurements collected by server.py:

- real Edge CPU availability
- real Edge memory availability
- real AWS connectivity
- real Orchestrator -> AWS RTT
- real workload parameters

No random infrastructure values are generated during LIVE prediction.

SUPPORTED POLICIES
------------------
1. RuleBased
   Interpretable hand-crafted baseline.

2. DecisionTree
   sklearn DecisionTreeClassifier.

3. RandomForest
   sklearn RandomForestClassifier.

4. QLearning
   Reinforcement-learning orchestration policy.

TRAINING / BENCHMARK NOTE
-------------------------
Decision Tree and Random Forest may be trained using the reproducible
synthetic dataset defined in this module.

That training dataset is deliberately isolated from the LIVE inference
path. Synthetic contexts are never substituted for real telemetry
during live orchestration.

Q-Learning training is implemented in qlearning_agent.py.
"""

from __future__ import annotations

import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Tuple

import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# Configuration
# =========================================================

ARTEFACT_DIR = Path(__file__).parent / "artefacts"
ARTEFACT_DIR.mkdir(exist_ok=True)


Route = Literal["edge", "cloud", "hybrid"]

ROUTES: List[Route] = [
    "edge",
    "cloud",
    "hybrid",
]


FEATURES = [
    "network_latency_ms",
    "connectivity",
    "cpu_available",
    "memory_available",
    "batch_size",
    "priority",
    "cost_budget_usd",
    "model_size_mb",
]


# =========================================================
# Data structures
# =========================================================

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
        """
        Convert the decision context to the feature vector expected
        by the ML policies.
        """

        return np.array(
            [[getattr(self, feature) for feature in FEATURES]],
            dtype=float,
        )


@dataclass
class DecisionResult:
    route: Route
    confidence: float
    engine: str
    reason: str
    probabilities: Dict[str, float]
    latency_us: float


# =========================================================
# Shared helpers
# =========================================================

def _normalize(values: Dict[str, float]) -> None:
    """
    Normalise route probabilities in-place.
    """

    total = sum(values.values())

    if total <= 0:
        uniform = 1.0 / len(values)

        for key in values:
            values[key] = uniform

        return

    for key in values:
        values[key] = values[key] / total


def _wrap(
    route: Route,
    confidence: float,
    engine: str,
    reasons: List[str],
    probabilities: Dict[str, float],
    started_at: float,
) -> DecisionResult:

    return DecisionResult(
        route=route,

        confidence=round(
            float(confidence),
            4,
        ),

        engine=engine,

        reason=(
            "; ".join(reasons)
            if reasons
            else "default policy"
        ),

        probabilities={
            key: round(float(value), 4)
            for key, value in probabilities.items()
        },

        latency_us=round(
            (
                time.perf_counter()
                - started_at
            )
            * 1_000_000,
            2,
        ),
    )


# =========================================================
# 1. Rule-Based Policy
# =========================================================

def rule_based_decide(
    ctx: DecisionContext,
) -> DecisionResult:
    """
    Interpretable baseline policy.

    IMPORTANT:
    This function does not generate infrastructure values.

    It only evaluates the DecisionContext supplied by the caller.
    In LIVE mode those values come from real infrastructure telemetry.
    """

    started_at = time.perf_counter()

    reasons: List[str] = []

    # -----------------------------------------------------
    # Hard constraints
    # -----------------------------------------------------

    if ctx.connectivity == 0:

        reasons.append(
            "cloud unavailable -> edge only"
        )

        return _wrap(
            "edge",
            1.0,
            "RuleBased",
            reasons,
            {
                "edge": 1.0,
                "cloud": 0.0,
                "hybrid": 0.0,
            },
            started_at,
        )

    if ctx.cost_budget_usd <= 0.001:

        reasons.append(
            "cost budget exhausted -> edge only"
        )

        return _wrap(
            "edge",
            0.95,
            "RuleBased",
            reasons,
            {
                "edge": 0.95,
                "cloud": 0.02,
                "hybrid": 0.03,
            },
            started_at,
        )

    # -----------------------------------------------------
    # Cloud suitability score
    # -----------------------------------------------------

    cloud_score = 0.0

    # Low AWS RTT makes cloud execution more attractive.
    if ctx.network_latency_ms < 80:

        cloud_score += 0.25
        reasons.append("cloud RTT < 80 ms")

    # Low Edge CPU availability favours offloading.
    if ctx.cpu_available < 40:

        cloud_score += 0.30
        reasons.append("edge CPU availability < 40%")

    # Low Edge memory availability favours offloading.
    if ctx.memory_available < 40:

        cloud_score += 0.20
        reasons.append("edge memory availability < 40%")

    # Larger batches are better candidates for cloud execution.
    if ctx.batch_size >= 8:

        cloud_score += 0.15
        reasons.append("batch size >= 8")

    # Model footprint represents workload compute demand.
    # Very large models strongly favour cloud execution, while
    # lightweight models retain the latency/cost benefits of Edge.
    if ctx.model_size_mb >= 300:

        cloud_score += 0.65
        reasons.append(
            f"large AI workload ({ctx.model_size_mb:.1f} MB) "
            "strongly favours cloud compute"
        )

    elif ctx.model_size_mb >= 20:

        cloud_score += 0.15
        reasons.append(
            f"model size {ctx.model_size_mb:.1f} MB favours cloud"
        )

    # High-priority workloads can favour cloud processing.
    if ctx.priority >= 4:

        cloud_score += 0.20
        reasons.append("priority >= 4")

    cloud_score = min(
        max(cloud_score, 0.0),
        1.0,
    )

    edge_score = 1.0 - cloud_score

    # -----------------------------------------------------
    # Route selection
    # -----------------------------------------------------

    # Cooperative Hybrid:
    # A very large AI workload can benefit from splitting
    # complementary work across healthy Edge and Cloud nodes.
    #
    # In the vision pipeline:
    #   Edge  -> YOLO object detection
    #   Cloud -> Florence-2 semantic interpretation
    cooperative_hybrid = (
        ctx.model_size_mb >= 300
        and ctx.connectivity == 1
        and ctx.cpu_available >= 50
        and ctx.memory_available >= 40
        and ctx.network_latency_ms <= 150
        and ctx.cost_budget_usd > 0.01
    )

    if cooperative_hybrid:

        route: Route = "hybrid"

        probabilities = {
            "edge": 0.20,
            "cloud": 0.30,
            "hybrid": 0.50,
        }

        reasons.append(
            "cooperative Edge-Cloud execution selected: "
            f"large AI workload ({ctx.model_size_mb:.1f} MB), "
            f"Edge capacity healthy "
            f"(CPU available {ctx.cpu_available:.1f}%, "
            f"memory available {ctx.memory_available:.1f}%), "
            f"AWS RTT {ctx.network_latency_ms:.1f} ms"
        )

    elif (
        0.35 <= cloud_score <= 0.60
        and ctx.cost_budget_usd > 0.01
    ):

        route: Route = "hybrid"

        probabilities = {
            "edge": edge_score * 0.6,
            "cloud": cloud_score * 0.6,
            "hybrid": 0.4,
        }

        _normalize(probabilities)

        reasons.append(
            "borderline conditions -> hybrid"
        )

    elif cloud_score > 0.5:

        route = "cloud"

        probabilities = {
            "edge": edge_score,
            "cloud": cloud_score,
            "hybrid": 0.0,
        }

        _normalize(probabilities)

    else:

        route = "edge"

        probabilities = {
            "edge": edge_score,
            "cloud": cloud_score,
            "hybrid": 0.0,
        }

        _normalize(probabilities)

        reasons.append(
            "edge selected: local resources sufficient "
            f"(CPU available {ctx.cpu_available:.1f}%, "
            f"memory available {ctx.memory_available:.1f}%), "
            f"AWS RTT {ctx.network_latency_ms:.1f} ms, "
            f"cloud suitability score {cloud_score:.2f}"
        )

    return _wrap(
        route,
        probabilities[route],
        "RuleBased",
        reasons,
        probabilities,
        started_at,
    )


# =========================================================
# 2 & 3. Decision Tree / Random Forest
# =========================================================

class MLDecisionEngine:
    """
    Wrapper for a sklearn orchestration estimator.

    Training data may be synthetic/reproducible.

    Prediction itself always uses the DecisionContext supplied
    by the caller.
    """

    def __init__(
        self,
        name: str,
        estimator,
    ):

        self.name = name
        self.estimator = estimator

        self.trained = False

        self.metrics: Dict[str, Any] = {}

        self.path = (
            ARTEFACT_DIR
            / f"{name.lower()}.joblib"
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        self.estimator.fit(
            X_train,
            y_train,
        )

        y_pred = self.estimator.predict(
            X_test
        )

        accuracy = float(
            accuracy_score(
                y_test,
                y_pred,
            )
        )

        self.trained = True

        importances = getattr(
            self.estimator,
            "feature_importances_",
            None,
        )

        self.metrics = {
            "accuracy": round(
                accuracy,
                4,
            ),

            "n_train": int(
                len(X_train)
            ),

            "n_test": int(
                len(X_test)
            ),

            "trained_at": time.time(),

            "training_source":
                "reproducible_synthetic_dataset",

            "feature_importance": (
                {
                    feature: round(
                        float(value),
                        4,
                    )
                    for feature, value
                    in zip(
                        FEATURES,
                        importances,
                    )
                }
                if importances is not None
                else None
            ),
        }

        joblib.dump(
            {
                "estimator":
                    self.estimator,

                "metrics":
                    self.metrics,
            },
            self.path,
        )

        return self.metrics

    def load(self) -> bool:
        """
        Load a previously trained model from disk.
        """

        if not self.path.exists():
            return False

        try:

            blob = joblib.load(
                self.path
            )

            self.estimator = blob[
                "estimator"
            ]

            self.metrics = blob.get(
                "metrics",
                {},
            )

            self.trained = True

            return True

        except Exception:

            self.trained = False

            return False

    def predict(
        self,
        ctx: DecisionContext,
    ) -> DecisionResult:
        """
        Predict a route from the supplied context.

        No random or simulated context is generated here.
        """

        if not self.trained:

            raise RuntimeError(
                f"{self.name} is not trained. "
                "Call /api/decisions/train first."
            )

        started_at = time.perf_counter()

        vector = ctx.as_vector()

        prediction_probability = (
            self.estimator
            .predict_proba(vector)[0]
        )

        classes = list(
            self.estimator.classes_
        )

        probabilities = {
            route: 0.0
            for route in ROUTES
        }

        for cls, probability in zip(
            classes,
            prediction_probability,
        ):

            probabilities[
                str(cls)
            ] = float(
                probability
            )

        _normalize(
            probabilities
        )

        route = max(
            probabilities,
            key=probabilities.get,
        )

        reasons: List[str] = []

        feature_importance = (
            self.metrics.get(
                "feature_importance"
            )
        )

        if feature_importance:

            top_feature = max(
                feature_importance.items(),
                key=lambda item: item[1],
            )[0]

            reasons.append(
                f"top feature: {top_feature}"
            )

        return _wrap(
            route,
            probabilities[route],
            self.name,
            reasons,
            probabilities,
            started_at,
        )


# =========================================================
# Synthetic training dataset
# =========================================================

def _sample_context(
    rng: random.Random,
) -> DecisionContext:
    """
    Generate one reproducible synthetic training context.

    IMPORTANT:
    This function is for model training / benchmark only.

    It is NOT called during LIVE orchestration.
    """

    connectivity = (
        1
        if rng.random() > 0.15
        else 0
    )

    return DecisionContext(

        network_latency_ms=(
            rng.uniform(
                20,
                500,
            )
            if connectivity
            else 9999.0
        ),

        connectivity=connectivity,

        cpu_available=rng.uniform(
            5,
            95,
        ),

        memory_available=rng.uniform(
            10,
            95,
        ),

        batch_size=rng.choice(
            [
                1,
                1,
                2,
                4,
                8,
                16,
                32,
            ]
        ),

        priority=rng.choice(
            [
                1,
                2,
                2,
                3,
                3,
                4,
                5,
            ]
        ),

        cost_budget_usd=rng.choice(
            [
                0.0,
                0.005,
                0.05,
                0.5,
                1.0,
                5.0,
                10.0,
            ]
        ),

        model_size_mb=rng.choice(
            [
                6.2,
                9.4,
                15.1,
                22.5,
                40.0,
            ]
        ),
    )


def build_synthetic_dataset(
    n: int = 4000,
    seed: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Build deterministic/reproducible training data.

    This dataset is isolated from the LIVE execution path.
    """

    rng = random.Random(
        seed
    )

    X = []
    y = []

    for _ in range(n):

        ctx = _sample_context(
            rng
        )

        # Baseline label from interpretable policy.
        label = rule_based_decide(
            ctx
        ).route

        # Small deterministic stochastic label noise.
        # Used only for training realism.
        if rng.random() < 0.08:

            alternative_routes = [
                route
                for route in ROUTES
                if route != label
            ]

            label = rng.choice(
                alternative_routes
            )

        X.append(
            ctx
            .as_vector()
            .flatten()
        )

        y.append(
            label
        )

    return (
        np.asarray(
            X,
            dtype=float,
        ),

        np.asarray(
            y,
        ),
    )


# =========================================================
# Registry
# =========================================================

class DecisionEngineRegistry:
    """
    Registry for all four orchestration policies.

    Supported aliases:

    rule
    dt
    rf
    ql
    """

    def __init__(self):

        from qlearning_agent import (
            qlearning_agent
        )

        # Rule Based does not require a model instance.

        self.dt = MLDecisionEngine(
            name="DecisionTree",
            estimator=DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
            ),
        )

        self.rf = MLDecisionEngine(
            name="RandomForest",
            estimator=RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        )

        self.ql = qlearning_agent

        # Attempt to restore persisted supervised models.
        self.dt.load()
        self.rf.load()

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    def train_all(
        self,
        n: int = 4000,
    ) -> Dict[str, Any]:
        """
        Train DT, RF and Q-Learning.

        DT/RF:
        reproducible synthetic labelled dataset.

        Q-Learning:
        training routine implemented in qlearning_agent.py.
        """

        X, y = build_synthetic_dataset(
            n=n,
            seed=42,
        )

        dt_metrics = self.dt.train(
            X,
            y,
        )

        rf_metrics = self.rf.train(
            X,
            y,
        )

        ql_metrics = self.ql.train(
            episodes=5000
        )

        return {

            "RuleBased": {
                "trained": True,
                "training_required": False,
                "type":
                    "interpretable_baseline",
            },

            "DecisionTree":
                dt_metrics,

            "RandomForest":
                rf_metrics,

            "QLearning":
                ql_metrics,

            "dataset": {
                "n_samples": int(n),

                "features":
                    FEATURES,

                "classes":
                    ROUTES,

                "source":
                    "reproducible_synthetic_training_only",

                "live_prediction_uses_real_context":
                    True,
            },
        }

    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    def decide(
        self,
        ctx: DecisionContext,
        engine: str = "rule",
    ) -> DecisionResult:
        """
        Execute the selected orchestration policy using the
        DecisionContext supplied by the caller.
        """

        selected = (
            engine
            or "rule"
        ).strip().lower()

        # -------------------------
        # Rule Based
        # -------------------------

        if selected in (
            "rule",
            "rulebased",
            "rule-based",
            "baseline",
        ):

            return rule_based_decide(
                ctx
            )

        # -------------------------
        # Decision Tree
        # -------------------------

        if selected in (
            "dt",
            "decisiontree",
            "decision-tree",
            "tree",
        ):

            if not self.dt.trained:

                self.dt.load()

            if not self.dt.trained:

                raise RuntimeError(
                    "DecisionTree is not trained. "
                    "Call POST /api/decisions/train first."
                )

            return self.dt.predict(
                ctx
            )

        # -------------------------
        # Random Forest
        # -------------------------

        if selected in (
            "rf",
            "randomforest",
            "random-forest",
            "forest",
        ):

            if not self.rf.trained:

                self.rf.load()

            if not self.rf.trained:

                raise RuntimeError(
                    "RandomForest is not trained. "
                    "Call POST /api/decisions/train first."
                )

            return self.rf.predict(
                ctx
            )

              # -------------------------
        # Q-Learning
        # -------------------------

        if selected in (
            "ql",
            "qlearning",
            "q-learning",
        ):
            return self.ql.predict(
                ctx
            )

        raise ValueError(
            "Unknown decision engine: "
            f"{engine}. "
            "Supported engines are: "
            "rule, dt, rf, ql."
        )

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    def status(
        self,
    ) -> Dict[str, Any]:

        return {

            "RuleBased": {
                "available": True,
                "trained": True,
                "training_required": False,
            },

            "DecisionTree": {
                "available": True,
                "trained":
                    self.dt.trained,
                "metrics":
                    self.dt.metrics,
            },

            "RandomForest": {
                "available": True,
                "trained":
                    self.rf.trained,
                "metrics":
                    self.rf.metrics,
            },

            "QLearning": {
                "available": True,
                "trained":
                    self.ql.trained,
                "metrics":
                    self.ql.metrics,
            },

            "features":
                FEATURES,

            "classes":
                ROUTES,

            "live_context":
                "real infrastructure telemetry",

            "training_context": {
                "DecisionTree":
                    "synthetic reproducible dataset",

                "RandomForest":
                    "synthetic reproducible dataset",

                "QLearning":
                    "qlearning_agent training environment",

                "RuleBased":
                    "no training required",
            },
        }


# =========================================================
# Global registry
# =========================================================

registry = DecisionEngineRegistry()


