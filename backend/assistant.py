"""
Research Assistant — natural-language Q&A over experimental data.
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


SYSTEM_PROMPT = """You are the Research Assistant of the "Edge–Cloud AI Orchestrator" academic platform.

You help the researcher analyse experimental results comparing four orchestration policies
(Rule-Based, Decision Tree, Random Forest, Q-Learning) that route AI inference between
NVIDIA Jetson (edge) and AWS EC2 (cloud).

STRICT RULES:
1. Base every claim on the JSON context you receive. NEVER invent numbers.
2. If the requested data is not in the context, say clearly: "Not available in the current context."
3. When comparing values, always show the exact numbers (with units and, when available, ±CI).
4. Cite the source: scenario name, seed, engine, or SQLite timestamp.
5. Prefer concise, structured answers. Bullet lists or small tables > walls of text.
6. When asked to write for a thesis, produce a short academic paragraph (3-5 sentences),
   third-person, past tense, with metrics inline.
7. The metrics in the context are already aggregated with 95% CI when marked '±'.
8. Default to English for all responses (this is a research platform used in academic defense).
   Only respond in another language if the user explicitly writes to you in that language.
"""


async def build_context(
    last_benchmark: Optional[Dict[str, Any]] = None,
    last_ci: Optional[Dict[str, Any]] = None,
    recent_decisions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble a compact, LLM-friendly context object from all data sources."""
    ctx: Dict[str, Any] = {
        "platform": {
            "engines": ["RuleBased", "DecisionTree", "RandomForest", "QLearning"],
            "routes": ["edge", "cloud", "hybrid"],
            "scenarios": [],
        },
        "last_benchmark": None,
        "last_ci_experiment": None,
        "recent_decisions_sample": [],
    }
    # scenarios list
    try:
        import scenarios as _s
        ctx["platform"]["scenarios"] = _s.list_scenarios()
    except Exception:
        pass

    if last_benchmark and last_benchmark.get("summary"):
        ctx["last_benchmark"] = {
            "meta": last_benchmark.get("meta"),
            "summary": last_benchmark["summary"],
        }

    if last_ci:
        ctx["last_ci_experiment"] = {
            "meta": last_ci.get("meta"),
            "aggregated": last_ci.get("aggregated"),
        }
    else:
        # try loading from disk
        ci_path = ARTEFACT_DIR / "benchmark_ci.json"
        if ci_path.exists():
            try:
                data = json.loads(ci_path.read_text())
                ctx["last_ci_experiment"] = {
                    "meta": data.get("meta"),
                    "aggregated": data.get("aggregated"),
                }
            except Exception:
                pass

    if recent_decisions:
        # send the most recent 15 (keeps token cost low)
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
        "EXPERIMENTAL CONTEXT (source of truth — quote from this, never invent):\n"
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