"""
Research Assistant â€” natural-language Q&A over experimental data.
Powered by Google Gemini.
Grounded in real data from the SQLite decision log, benchmark results and
experimental artefacts.

Design principles:
 - The LLM never invents numbers. All metrics come from the assembled context.
 - The context is compact JSON so token cost stays low (~2-4k tokens/query).
 - Answers must cite the source (scenario, seed, engine, timestamp when known).
"""
from __future__ import annotations
import os
import json
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from google import genai

load_dotenv()

MODEL_PROVIDER = "google"
MODEL_NAME = "gemini-2.5-flash"

ARTEFACT_DIR = Path("/tmp/benchmark")


SYSTEM_PROMPT = """You are the Research Assistant of the "Edge-Cloud AI Orchestrator" academic platform.

You support analysis of the implemented real Edge-Cloud orchestration system.

The current infrastructure consists of:
- Edge: edge-node-01, Linux VirtualBox VM managed through AWS IoT Greengrass.
- Cloud: cloud-node-01 running on AWS EC2.
- Orchestration policies: Rule-Based, Decision Tree, Random Forest and Q-Learning.
- Execution routes: Edge, Cloud and Hybrid.
- Real AI workloads include YOLOv8 and Florence-2.

STRICT RULES:
1. Base every claim only on the JSON context provided. NEVER invent numbers.
2. Prefer real infrastructure telemetry, real routing decisions and real execution measurements.
3. Do NOT present simulated, synthetic benchmark or synthetic CI results as real experimental evidence.
4. If a requested value is not available from real measurements, state clearly: "Not available in the current real-data context."
5. Clearly distinguish measured values from derived algorithmic values such as Q-values, confidence or reward.
6. When explaining a routing decision, identify the engine, selected route and the real context factors that influenced the decision when available.
7. Never claim statistical significance unless an appropriate statistical test is explicitly present in the context.
8. When writing for a thesis, use concise academic language and avoid claims that exceed the available evidence.
9. Default to English because this assistant is part of an academic research platform.
10. Only respond in another language when the user explicitly writes in that language.
11. When explaining decision factors, clearly separate the data into these categories:
    - Measured infrastructure conditions: network latency/RTT, connectivity, CPU availability and memory availability obtained from the live infrastructure.
    - Workload or policy inputs: batch size, priority, cost budget and model size. These are decision inputs and must NOT be described as measured infrastructure metrics.
    - Algorithm-derived values: Q-values, confidence, state and reward. These are outputs or internal values of the decision algorithm and must NOT be described as physical measurements.
"""

async def build_context(
    recent_decisions: Optional[List[Dict[str, Any]]] = None,
    live_telemetry: Optional[Dict[str, Any]] = None,
    qlearning_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a compact context using real orchestration evidence only."""

    ctx: Dict[str, Any] = {
        "platform": {
            "engines": ["RuleBased", "DecisionTree", "RandomForest", "QLearning"],
            "routes": ["edge", "cloud", "hybrid"],
        },
        "live_telemetry": live_telemetry,
        "qlearning_real_state": qlearning_state,
        "recent_decisions_sample": [],
    }

    # Synthetic benchmark, CI and scenario data are intentionally excluded.

    if recent_decisions:
        ctx["recent_decisions_sample"] = [
            {
                "engine": d.get("engine"),
                "route": d.get("route"),
                "confidence": d.get("confidence"),
                "reason": d.get("reason"),
                "context": d.get("context"),
                "timestamp": d.get("created_at"),
            }
            for d in recent_decisions[:15]
        ]

    return ctx

async def ask(
    question: str,
    context: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a single question to Gemini grounded in the experimental context."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")

    session_id = session_id or f"assistant-{uuid.uuid4().hex[:8]}"

    client = genai.Client(api_key=api_key)

    context_json = json.dumps(context, ensure_ascii=False, default=str, indent=2)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "EXPERIMENTAL CONTEXT (source of truth â€” quote from this, never invent):\n"
        f"```json\n{context_json}\n```\n\n"
        f"QUESTION: {question}"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return {
        "session_id": session_id,
        "question": question,
        "answer": response.text,
        "model": f"{MODEL_PROVIDER}/{MODEL_NAME}",
    }




