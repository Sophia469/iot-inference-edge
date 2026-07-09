"""
Decision explainer — turns raw model outputs into structured, human-readable
explanations for the "Explain Decision" panel and the thesis report.

For each engine:
 - RuleBased : which rules fired, in which direction
 - DecisionTree / RandomForest : global feature importance restricted to the
   features present in this context, plus a counterfactual sentence when one
   feature clearly flipped the decision
 - QLearning : Q-values per action + the state bucket that was hit

Returns a dict:
{
    "factors": [ {name, value, contribution, direction, met}, ... ],
    "counterfactual": "if cpu_available were > 60, decision would change to cloud",
    "top_features": [ {feature, weight}, ... ],
}
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

from decision_engine import (
    DecisionContext, FEATURES, ROUTES,
    rule_based_decide, registry as _reg,
)


def _rule_factors(ctx: DecisionContext) -> List[Dict[str, Any]]:
    """Enumerate every rule that could influence the decision, marking whether it fired."""
    factors = [
        {"name": "connectivity",
         "value": "online" if ctx.connectivity else "offline",
         "met": ctx.connectivity == 1,
         "contribution": "hard requirement for cloud/hybrid",
         "direction": "edge" if ctx.connectivity == 0 else "any"},
        {"name": "cost_budget_usd",
         "value": ctx.cost_budget_usd,
         "met": ctx.cost_budget_usd > 0.001,
         "contribution": "cloud requires budget > 0.001",
         "direction": "edge" if ctx.cost_budget_usd <= 0.001 else "any"},
        {"name": "network_latency_ms",
         "value": ctx.network_latency_ms,
         "met": ctx.network_latency_ms < 80,
         "contribution": "net < 80ms favours cloud",
         "direction": "cloud"},
        {"name": "cpu_available",
         "value": ctx.cpu_available,
         "met": ctx.cpu_available < 40,
         "contribution": "low CPU favours cloud",
         "direction": "cloud"},
        {"name": "memory_available",
         "value": ctx.memory_available,
         "met": ctx.memory_available < 40,
         "contribution": "low memory favours cloud",
         "direction": "cloud"},
        {"name": "batch_size",
         "value": ctx.batch_size,
         "met": ctx.batch_size >= 8,
         "contribution": "batch ≥ 8 favours cloud",
         "direction": "cloud"},
        {"name": "model_size_mb",
         "value": ctx.model_size_mb,
         "met": ctx.model_size_mb >= 20,
         "contribution": "model ≥ 20MB favours cloud",
         "direction": "cloud"},
        {"name": "priority",
         "value": ctx.priority,
         "met": ctx.priority >= 4,
         "contribution": "priority ≥ 4 favours cloud",
         "direction": "cloud"},
    ]
    return factors


def _tree_factors(ctx: DecisionContext, feature_importance: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
    """Uses per-model feature importance to rank factors for this specific input."""
    imp = feature_importance or {}
    ordered = sorted(imp.items(), key=lambda kv: -kv[1])
    out = []
    for f, w in ordered:
        val = getattr(ctx, f, None)
        out.append({
            "name": f,
            "value": val,
            "weight": round(float(w), 4),
            "contribution": f"model gives this feature weight {round(float(w) * 100, 1)}%",
        })
    return out


def _counterfactual(ctx: DecisionContext, engine_key: str, current_route: str) -> Optional[str]:
    """Try changing one feature to see if the decision would flip. O(features * 3) fast."""
    if engine_key not in ("dt", "rf"):
        return None
    est = _reg.dt if engine_key == "dt" else _reg.rf
    if not est.trained:
        return None
    trials = [
        ("network_latency_ms", 30),
        ("network_latency_ms", 500),
        ("cpu_available", 15),
        ("cpu_available", 80),
        ("cost_budget_usd", 0.0),
        ("priority", 5),
        ("model_size_mb", 6.2),
        ("model_size_mb", 40.0),
        ("batch_size", 1),
        ("batch_size", 32),
    ]
    for feat, new_val in trials:
        original = getattr(ctx, feat)
        setattr(ctx, feat, new_val)
        try:
            pred_route = est.predict(ctx).route
        finally:
            setattr(ctx, feat, original)
        if pred_route != current_route:
            return f"if {feat} were {new_val}, decision would change to {pred_route.upper()}"
    return None


def explain(ctx: DecisionContext, engine_key: str, current_route: str,
            feature_importance: Optional[Dict[str, float]] = None,
            q_values: Optional[List[float]] = None) -> Dict[str, Any]:
    """Main entry — returns a rich explanation dict."""
    engine_key = engine_key.lower()

    if engine_key.startswith("rule"):
        factors = _rule_factors(ctx)
        # count how many "cloud" factors fired
        cloud_fired = [f for f in factors if f["met"] and f["direction"] == "cloud"]
        edge_forcing = [f for f in factors if f["met"] and f["direction"] == "edge"]
        summary = (
            f"{len(edge_forcing)} hard rule(s) forced edge"
            if edge_forcing else
            f"{len(cloud_fired)} cloud-favouring condition(s) met"
        )
        return {"factors": factors, "top_features": [], "counterfactual": None, "summary": summary}

    if engine_key in ("dt", "rf"):
        top = _tree_factors(ctx, feature_importance)[:5]
        cf = _counterfactual(ctx, engine_key, current_route)
        summary = f"top factor: {top[0]['name']}" if top else "no importance data"
        return {"factors": _rule_factors(ctx), "top_features": top,
                "counterfactual": cf, "summary": summary}

    if engine_key.startswith("q") or engine_key == "ql":
        factors = _rule_factors(ctx)
        top = []
        if q_values:
            for i, r in enumerate(ROUTES):
                top.append({"name": r, "value": round(float(q_values[i]), 3),
                            "contribution": "Q-value for this state", "weight": None})
        summary = "policy learned via reinforcement learning; Q-values shown per route"
        return {"factors": factors, "top_features": top, "counterfactual": None, "summary": summary}

    return {"factors": [], "top_features": [], "counterfactual": None, "summary": "unknown engine"}
