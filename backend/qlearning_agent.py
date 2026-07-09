"""
Q-Learning Agent for Edge/Cloud orchestration.

ACADEMIC NOTE (important for the thesis):
This agent implements SINGLE-STEP tabular Q-Learning — formally equivalent to a
tabular *contextual bandit* since each context (state) has no temporal successor.
The update rule collapses from
        Q(s,a) ← Q(s,a) + α·[r + γ·max_a' Q(s',a') − Q(s,a)]
to
        Q(s,a) ← Q(s,a) + α·[r − Q(s,a)]
which is still the canonical *reinforcement-learning-from-reward* formulation
used in orchestration/routing literature. When writing the thesis, cite this as
"tabular contextual-bandit Q-Learning" to be precise. To convert to full multi-
step Q-Learning, chain successive contexts as (s_t, a_t, r_t, s_{t+1}) and
propagate values through episodes of length > 1.

 - State  = discretised bucket of (network_latency, cpu_available, cost_budget, priority, connectivity)
 - Action = {edge, cloud, hybrid}
 - Reward = -α·latency_ms  - β·cost_usd*1000  + γ·success  - δ·resource_pressure

Trained offline over N simulated episodes using the same environment model as
the rule-based policy. The agent learns a policy that MINIMISES latency+cost
while penalising unsuccessful routings (e.g. cloud when offline).
"""
from __future__ import annotations
import random
import time
from pathlib import Path
from typing import Tuple, Dict, Any
import json
import numpy as np

from decision_engine import DecisionContext, ROUTES, _wrap, DecisionResult

ARTEFACT_DIR = Path(__file__).parent / "artefacts"
ARTEFACT_DIR.mkdir(exist_ok=True)
Q_TABLE_PATH = ARTEFACT_DIR / "qlearning.npy"
Q_META_PATH = ARTEFACT_DIR / "qlearning_meta.json"

# ---- State discretisation ----
# 3 buckets net, 3 cpu, 2 cost, 3 priority, 2 connectivity => 3*3*2*3*2 = 108 states
NET_BUCKETS = [80.0, 250.0]           # <80 low, <250 mid, else high (or offline)
CPU_BUCKETS = [30.0, 60.0]            # low <30, mid <60, high
COST_BUCKETS = [0.05]                 # tight <=0.05, loose otherwise
PRI_BUCKETS = [2, 4]                  # low <=2, mid <=4, high otherwise

N_NET = 3
N_CPU = 3
N_COST = 2
N_PRI = 3
N_CONN = 2
N_ACTIONS = len(ROUTES)  # 3


def _bucket(value: float, edges) -> int:
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


def state_index(ctx: DecisionContext) -> int:
    if ctx.connectivity == 0:
        n = 2  # offline -> treat as worst net bucket
    else:
        n = _bucket(ctx.network_latency_ms, NET_BUCKETS)
    c = _bucket(ctx.cpu_available, CPU_BUCKETS)
    co = _bucket(ctx.cost_budget_usd, COST_BUCKETS)
    p = _bucket(ctx.priority, PRI_BUCKETS)
    k = ctx.connectivity
    # multi-dim flatten
    return ((((n * N_CPU + c) * N_COST + co) * N_PRI + p) * N_CONN + k)


N_STATES = N_NET * N_CPU * N_COST * N_PRI * N_CONN


# ---- Environment reward model ----
def simulate_outcome(ctx: DecisionContext, action_idx: int) -> Tuple[float, bool]:
    """Return (reward, success). Success = the action was executable."""
    route = ROUTES[action_idx]
    # If offline and route needs cloud -> failure
    if ctx.connectivity == 0 and route in ("cloud", "hybrid"):
        return -50.0, False
    if ctx.cost_budget_usd <= 0 and route in ("cloud", "hybrid"):
        return -30.0, False

    # Latency model
    if route == "edge":
        latency = random.uniform(18, 42) + max(0, (80 - ctx.cpu_available)) * 0.3
        cost = 0.00002
        resource_pressure = max(0, 70 - ctx.cpu_available) * 0.05
    elif route == "cloud":
        latency = ctx.network_latency_ms + random.uniform(30, 120)
        cost = 0.00085 * ctx.batch_size
        resource_pressure = 0.0
    else:  # hybrid
        latency = max(35, ctx.network_latency_ms * 0.5) + random.uniform(15, 50)
        cost = 0.0005 * ctx.batch_size
        resource_pressure = max(0, 70 - ctx.cpu_available) * 0.03

    # Priority weighting: high priority pays more for lower latency
    latency_penalty = latency * (0.05 + 0.02 * ctx.priority)
    cost_penalty = cost * 1000.0 * (2.0 - 0.15 * ctx.priority)
    reward = -latency_penalty - cost_penalty - resource_pressure + 5.0  # base success bonus
    return reward, True


class QLearningAgent:
    def __init__(self, alpha: float = 0.15, gamma: float = 0.9,
                 eps_start: float = 0.9, eps_end: float = 0.05):
        self.q = np.zeros((N_STATES, N_ACTIONS), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.trained = False
        self.metrics: Dict[str, Any] = {}
        self._load()

    # persistence
    def _load(self) -> bool:
        if Q_TABLE_PATH.exists() and Q_META_PATH.exists():
            self.q = np.load(Q_TABLE_PATH)
            self.metrics = json.loads(Q_META_PATH.read_text())
            self.trained = True
            return True
        return False

    def _save(self) -> None:
        np.save(Q_TABLE_PATH, self.q)
        Q_META_PATH.write_text(json.dumps(self.metrics))

    def train(self, episodes: int = 5000) -> Dict[str, Any]:
        from decision_engine import _sample_context
        rewards_history = []
        route_counts = {r: 0 for r in ROUTES}
        for ep in range(episodes):
            eps = self.eps_end + (self.eps_start - self.eps_end) * max(0.0, 1 - ep / (episodes * 0.7))
            ctx = _sample_context()
            s = state_index(ctx)
            # epsilon-greedy
            if random.random() < eps:
                a = random.randint(0, N_ACTIONS - 1)
            else:
                a = int(np.argmax(self.q[s]))
            reward, _ = simulate_outcome(ctx, a)
            # Terminal-like (single step episode); just update towards reward
            self.q[s, a] += self.alpha * (reward - self.q[s, a])
            rewards_history.append(reward)
            route_counts[ROUTES[a]] += 1

        mean_last_500 = float(np.mean(rewards_history[-500:]))
        self.metrics = {
            "episodes": episodes,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "mean_reward_last_500": round(mean_last_500, 3),
            "route_distribution_training": route_counts,
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "trained_at": time.time(),
            "converged_greedy_ratio": round(float(np.mean(np.max(self.q, axis=1) > 0)), 3),
        }
        self.trained = True
        self._save()
        return self.metrics

    def predict(self, ctx: DecisionContext) -> DecisionResult:
        if not self.trained:
            raise RuntimeError("QLearning agent not trained yet")
        t0 = time.perf_counter()
        s = state_index(ctx)
        q_values = self.q[s]
        a = int(np.argmax(q_values))
        # softmax over Q-values for probability display
        z = q_values - q_values.max()
        exp_z = np.exp(z)
        probs_arr = exp_z / (exp_z.sum() + 1e-9)
        probs = {r: float(round(probs_arr[i], 4)) for i, r in enumerate(ROUTES)}
        confidence = probs[ROUTES[a]]
        reason = f"state={s} · Q={q_values.round(2).tolist()}"
        return _wrap(ROUTES[a], confidence, "QLearning", [reason], probs, t0)

    def q_table_summary(self, top_states: int = 12) -> Dict[str, Any]:
        # returns top-N most-visited (highest max Q) states with their preferred action
        indices = np.argsort(-np.max(self.q, axis=1))[:top_states]
        rows = []
        for idx in indices:
            best_action = int(np.argmax(self.q[idx]))
            rows.append({
                "state": int(idx),
                "q_values": [round(float(v), 2) for v in self.q[idx]],
                "best_route": ROUTES[best_action],
                "max_q": round(float(np.max(self.q[idx])), 3),
            })
        return {"rows": rows, "features_encoded": ["network_latency", "cpu_available", "cost_budget", "priority", "connectivity"]}


qlearning_agent = QLearningAgent()
