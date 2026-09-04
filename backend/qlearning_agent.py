"""
Real-Execution Q-Learning Agent for Edge/Cloud orchestration.

FINAL RESEARCH IMPLEMENTATION
-----------------------------
This agent learns exclusively from REAL Edge-Cloud executions.

Infrastructure:
- Edge  : edge-node-01 (Ubuntu Linux VM / Oracle VirtualBox)
- Cloud : cloud-node-01 (Ubuntu Linux / AWS EC2)

No synthetic infrastructure telemetry, simulated latency, simulated cost,
or simulated execution outcome is used to update the final Q-table.

The agent implements SINGLE-STEP tabular Q-Learning, formally equivalent
to a contextual bandit because each orchestration request is treated as
an independent decision context.

State:
    discretised bucket of:
    - network latency / AWS RTT
    - Edge CPU availability
    - cost budget
    - workload priority
    - Cloud connectivity

Actions:
    - edge
    - cloud
    - hybrid

Learning cycle:
    REAL TELEMETRY
        -> STATE
        -> ACTION
        -> REAL EXECUTION
        -> REAL RESULT
        -> REAL REWARD
        -> Q-TABLE UPDATE

Update rule:

    Q(s,a) <- Q(s,a) + alpha * [reward - Q(s,a)]

The reward is calculated only after an actual execution has occurred.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from decision_engine import (
    DecisionContext,
    DecisionResult,
    ROUTES,
    _wrap,
)


# ============================================================
# Persistence
# ============================================================

ARTEFACT_DIR = Path(__file__).parent / "artefacts"
ARTEFACT_DIR.mkdir(exist_ok=True)

# IMPORTANT:
# Separate artefacts from the old simulated Q-table.
# This prevents simulated historical values from contaminating
# the final real-infrastructure experiment.
Q_TABLE_PATH = ARTEFACT_DIR / "qlearning_real.npy"
Q_META_PATH = ARTEFACT_DIR / "qlearning_real_meta.json"


# ============================================================
# State discretisation
# ============================================================

# 3 network buckets
# 3 CPU buckets
# 2 cost buckets
# 3 priority buckets
# 2 connectivity buckets
#
# 3 * 3 * 2 * 3 * 2 = 108 states

NET_BUCKETS = [80.0, 250.0]
CPU_BUCKETS = [30.0, 60.0]
COST_BUCKETS = [0.05]
PRI_BUCKETS = [2, 4]

N_NET = 3
N_CPU = 3
N_COST = 2
N_PRI = 3
N_CONN = 2

N_ACTIONS = len(ROUTES)


def _bucket(value: float, edges) -> int:
    """
    Convert a continuous value into a discrete bucket.
    """
    for i, edge in enumerate(edges):
        if value < edge:
            return i

    return len(edges)


def state_index(ctx: DecisionContext) -> int:
    """
    Convert a REAL infrastructure context into one of 108 states.
    """

    # Cloud unavailable -> worst network state
    if ctx.connectivity == 0:
        network_bucket = 2
    else:
        network_bucket = _bucket(
            ctx.network_latency_ms,
            NET_BUCKETS,
        )

    cpu_bucket = _bucket(
        ctx.cpu_available,
        CPU_BUCKETS,
    )

    cost_bucket = _bucket(
        ctx.cost_budget_usd,
        COST_BUCKETS,
    )

    priority_bucket = _bucket(
        ctx.priority,
        PRI_BUCKETS,
    )

    connectivity_bucket = int(ctx.connectivity)

    return (
        (
            (
                (
                    network_bucket * N_CPU
                    + cpu_bucket
                )
                * N_COST
                + cost_bucket
            )
            * N_PRI
            + priority_bucket
        )
        * N_CONN
        + connectivity_bucket
    )


N_STATES = (
    N_NET
    * N_CPU
    * N_COST
    * N_PRI
    * N_CONN
)


# ============================================================
# Q-Learning Agent
# ============================================================

class QLearningAgent:
    """
    Q-Learning agent that learns only from real executions.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.9,
        exploration_rate: float = 0.10,
    ):
        self.q = np.zeros(
            (N_STATES, N_ACTIONS),
            dtype=float,
        )

        self.alpha = alpha
        self.gamma = gamma

        # 10% controlled exploration using REAL executions.
        self.exploration_rate = exploration_rate

        self.trained = False

        self.metrics: Dict[str, Any] = {
            "learning_mode": "real_execution_only",
            "real_updates": 0,
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "exploration_rate": self.exploration_rate,
        }

        self._load()


    # ========================================================
    # Persistence
    # ========================================================

    def _load(self) -> bool:
        """
        Load only the REAL-execution Q-table.

        Historical qlearning.npy generated from simulated training
        is deliberately ignored.
        """

        if (
            Q_TABLE_PATH.exists()
            and Q_META_PATH.exists()
        ):
            self.q = np.load(Q_TABLE_PATH)

            self.metrics = json.loads(
                Q_META_PATH.read_text()
            )

            self.trained = (
                int(
                    self.metrics.get(
                        "real_updates",
                        0,
                    )
                )
                > 0
            )

            return True

        return False


    def _save(self) -> None:
        """
        Persist the real Q-table and metadata.
        """

        np.save(
            Q_TABLE_PATH,
            self.q,
        )

        Q_META_PATH.write_text(
            json.dumps(
                self.metrics,
                indent=2,
            )
        )


    # ========================================================
    # Simulated training deliberately disabled
    # ========================================================

    def train(
        self,
        episodes: int = 0,
    ) -> Dict[str, Any]:
        """
        Offline/simulated training is intentionally disabled
        in the final research implementation.

        The agent learns through learn_from_real_execution().
        """

        raise RuntimeError(
            "Simulated Q-Learning training is disabled. "
            "The final agent learns only from real "
            "Edge-Cloud executions."
        )


    # ========================================================
    # Real decision
    # ========================================================

    def predict(
        self,
        ctx: DecisionContext,
    ) -> DecisionResult:
        """
        Select EDGE, CLOUD or HYBRID from the current REAL
        infrastructure context.

        During early learning, controlled epsilon-greedy exploration
        allows the agent to collect real experience from different
        execution routes.
        """

        start = time.perf_counter()

        state = state_index(ctx)

        q_values = self.q[state]

        exploration = False


        # ----------------------------------------------------
        # Cloud unavailable
        # ----------------------------------------------------

        if ctx.connectivity == 0:
            action = ROUTES.index("edge")

            reason_text = (
                f"state={state} · cloud unavailable · "
                f"forcing safe EDGE route · "
                f"Q={q_values.round(2).tolist()}"
            )


        # ----------------------------------------------------
        # Controlled REAL exploration
        # ----------------------------------------------------

        elif random.random() < self.exploration_rate:
            action = random.randint(
                0,
                N_ACTIONS - 1,
            )

            exploration = True

            reason_text = (
                f"state={state} · "
                f"real exploration · "
                f"Q={q_values.round(2).tolist()}"
            )


        # ----------------------------------------------------
        # Exploitation
        # ----------------------------------------------------

        else:
            action = int(
                np.argmax(q_values)
            )

            reason_text = (
                f"state={state} · "
                f"real learned policy · "
                f"Q={q_values.round(2).tolist()}"
            )


        # ----------------------------------------------------
        # Probability display
        # ----------------------------------------------------

        q_shifted = (
            q_values
            - q_values.max()
        )

        exp_q = np.exp(q_shifted)

        probs_array = (
            exp_q
            / (
                exp_q.sum()
                + 1e-9
            )
        )

        probabilities = {
            route: float(
                round(
                    probs_array[i],
                    4,
                )
            )
            for i, route
            in enumerate(ROUTES)
        }


        # During cold start all Q-values are equal.
        # Probability values therefore appear approximately equal.
        confidence = probabilities[
            ROUTES[action]
        ]


        if exploration:
            reason_text += (
                f" · selected={ROUTES[action]}"
            )


        return _wrap(
            ROUTES[action],
            confidence,
            "QLearning",
            [reason_text],
            probabilities,
            start,
        )


    # ========================================================
    # REAL LEARNING
    # ========================================================

    def learn_from_real_execution(
        self,
        ctx: DecisionContext,
        route: str,
        latency_ms: float,
        success: bool,
        cost_usd: float = 0.0,
        resource_pressure: float = 0.0,
        failover_applied: bool = False,
    ) -> Dict[str, Any]:
        """
        Update the Q-table using the observed result of an
        ACTUAL Edge/Cloud/Hybrid execution.

        No random latency.
        No simulated infrastructure.
        No simulated success/failure.
        """

        if route not in ROUTES:
            raise ValueError(
                f"Unknown execution route: {route}"
            )


        state = state_index(ctx)

        action = ROUTES.index(route)


        # ----------------------------------------------------
        # Real reward
        # ----------------------------------------------------

        if success:

            # Higher priority makes latency more important.
            latency_penalty = (
                float(latency_ms)
                * (
                    0.05
                    + 0.02
                    * ctx.priority
                )
            )


            # Actual measured/derived monetary execution cost.
            cost_penalty = (
                float(cost_usd)
                * 1000.0
                * (
                    2.0
                    - 0.15
                    * ctx.priority
                )
            )


            reward = (
                5.0
                - latency_penalty
                - cost_penalty
                - float(resource_pressure)
            )


            # A failover means the original orchestration decision
            # could not be executed as intended.
            if failover_applied:
                reward -= 5.0


        else:

            # Strong penalty for an actual failed execution.
            reward = -50.0


        # ----------------------------------------------------
        # Q update
        # ----------------------------------------------------

        old_q = float(
            self.q[state, action]
        )


        self.q[state, action] += (
            self.alpha
            * (
                reward
                - self.q[state, action]
            )
        )


        new_q = float(
            self.q[state, action]
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        real_updates = (
            int(
                self.metrics.get(
                    "real_updates",
                    0,
                )
            )
            + 1
        )


        self.metrics.update(
            {
                "learning_mode":
                    "real_execution_only",

                "real_updates":
                    real_updates,

                "last_real_reward":
                    round(
                        float(reward),
                        4,
                    ),

                "last_real_state":
                    int(state),

                "last_real_action":
                    route,

                "last_real_latency_ms":
                    round(
                        float(latency_ms),
                        4,
                    ),

                "last_real_success":
                    bool(success),

                "last_real_cost_usd":
                    round(
                        float(cost_usd),
                        8,
                    ),

                "last_real_resource_pressure":
                    round(
                        float(resource_pressure),
                        4,
                    ),

                "last_failover_applied":
                    bool(failover_applied),

                "last_real_update_at":
                    time.time(),

                "n_states":
                    N_STATES,

                "n_actions":
                    N_ACTIONS,

                "alpha":
                    self.alpha,

                "gamma":
                    self.gamma,

                "exploration_rate":
                    self.exploration_rate,
            }
        )


        self.trained = True

        self._save()


        return {
            "learning_mode":
                "real_execution_only",

            "state":
                int(state),

            "action":
                route,

            "reward":
                round(
                    float(reward),
                    4,
                ),

            "old_q":
                round(
                    old_q,
                    4,
                ),

            "new_q":
                round(
                    new_q,
                    4,
                ),

            "real_updates":
                real_updates,

            "success":
                bool(success),

            "latency_ms":
                round(
                    float(latency_ms),
                    4,
                ),

            "cost_usd":
                round(
                    float(cost_usd),
                    8,
                ),

            "failover_applied":
                bool(failover_applied),
        }


    # ========================================================
    # Q-table inspection
    # ========================================================

    def q_table_summary(
        self,
        top_states: int = 12,
    ) -> Dict[str, Any]:
        """
        Return the strongest currently learned REAL states.
        """

        indices = np.argsort(
            -np.max(
                self.q,
                axis=1,
            )
        )[:top_states]


        rows = []

        for index in indices:

            best_action = int(
                np.argmax(
                    self.q[index]
                )
            )

            rows.append(
                {
                    "state":
                        int(index),

                    "q_values":
                        [
                            round(
                                float(value),
                                4,
                            )
                            for value
                            in self.q[index]
                        ],

                    "best_route":
                        ROUTES[
                            best_action
                        ],

                    "max_q":
                        round(
                            float(
                                np.max(
                                    self.q[index]
                                )
                            ),
                            4,
                        ),
                }
            )


        return {
            "learning_mode":
                "real_execution_only",

            "real_updates":
                int(
                    self.metrics.get(
                        "real_updates",
                        0,
                    )
                ),

            "rows":
                rows,

            "features_encoded":
                [
                    "network_latency",
                    "cpu_available",
                    "cost_budget",
                    "priority",
                    "connectivity",
                ],
        }


# ============================================================
# Shared agent instance
# ============================================================

qlearning_agent = QLearningAgent()