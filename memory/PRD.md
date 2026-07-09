# PRD — Edge–Cloud AI Orchestration (Research Platform)

## Problem Statement
Research platform for intelligent Edge–Cloud AI orchestration using NVIDIA Jetson and AWS, integrating AWS IoT Greengrass, Docker and pre-trained AI models, plus a four-policy Decision Engine (Rule / DT / RF / Q-Learning).

## Research Question
"Given real-time operational context (network, resources, cost, priority), which orchestration policy delivers the best trade-off between latency, cost, resource usage and reliability?"

## Progression Narrative
Rule-Based → Decision Tree → Random Forest → Q-Learning
(fixed heuristics → supervised → ensemble → reinforcement learning)

## Scope
- IN: integration of existing tech, orchestration policy design/evaluation, reproducible experiments
- OUT: training new CV models, building new cloud infra

## Implemented (Jan 2026)
### Iteration 1 — MVP dashboard
### Iteration 2 — Research reframing (Q-Learning, real inference)
### Iteration 3 — Comparative benchmark harness (CSV + 5 PNG charts)
### Iteration 4 — Sprint 1 wrap-up
- Scenario Manager (8 named scenarios)
- Multi-Seed CI (95%) via Student-t
- Explain Decision (XAI: factors + top_features + counterfactual)
- Thesis Report PDF (10 pages via matplotlib PdfPages)
- Experiment Assistant (Claude Sonnet 4.5 via Emergent LLM Key, "Future Work / Post-TCC Preview")
- SQLite WAL mode + busy_timeout for concurrent safety
- Full English UI

### Iteration 5 — Academic finalisation (Feb 2026)
- 12-page academic PDF report with University of Bedfordshire branding
  (cover, executive summary, config, ±95% CI results, 5 IEEE-style figures,
  sample explained decisions, conclusions, references)
- Embedded PDF metadata (author, supervisor, module code, subject, keywords)
- Experiment Assistant promoted from "Future Work preview" to fully integrated
- System Architecture modal updated to reflect final academic state
  (Research Instrumentation, 12-page report, promoted assistant)
- README.md frozen-version academic header (author, programme, module,
  institution, release tag `v1.0.0-thesis`)
- Backend lint clean-up (25 ruff violations resolved)

## Testing
- iteration_1.json: 100% (MVP)
- iteration_2.json: 100% (Q-Learning + real inference)
- iteration_3.json: 100% (benchmark harness)
- iteration_4.json: 100% (48/48 backend + 16/16 frontend — Sprint 1 wrap)

## Sprint Plan
- Sprint 1 ✅ COMPLETE — dashboard, engines, scenarios, CI, explain, PDF, assistant
- Sprint 2 — Execute experiments per scenario (8 PDFs)
- Sprint 3 — Thesis writing (author-led, plataforma-supported)
- Sprint 4 — Video demo + slides + polish
- Post-TCC — Real YOLO ONNX on Jetson + boto3 IoT Core + full assistant module

## Backlog (Post-TCC)
- Real YOLOv8n ONNX Runtime (replace numpy proxy)
- Real boto3 IoT Core MQTT integration
- Real Jetson + EC2 deployment
- Q-table heatmap visualisation
- Multi-device concurrent simulation
