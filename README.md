# iot-inference-edge

**Edge–Cloud AI Orchestrator**

Research platform for AI-driven infrastructure orchestration across Edge–Cloud environments using NVIDIA Jetson and AWS.

Developed as part of the **MSc Artificial Intelligence** programme at the **University of Bedfordshire**.

---

## Academic Information

**Status:** Frozen dissertation artefact

**Author:** Sophia Souza Marçal

**Student ID:** 1808415

**Programme:** MSc Artificial Intelligence

**Module:** 25-26BLK5-6AACIS144-6 – MSc Project

**Institution:** University of Bedfordshire

**Release:** v1.0.0-thesis

---

## Dissertation

**AI-Driven Infrastructure Orchestration for Edge–Cloud Environments:**

*A Comparative Study of Intelligent Decision Policies for AI Inference Routing*

---

## System

**EDGE-CLOUD // ORCHESTRATOR**

---

## 0. Academic artefact

This repository is the frozen research artefact for the thesis chapter *Option 2 — AWS-Native Edge AI Orchestration using AWS IoT Greengrass*.

A full-stack, container-ready prototype that deploys and manages AI inference workloads (YOLO / object-detection) between an **NVIDIA Jetson** edge device and **AWS EC2**, orchestrated through **AWS IoT Greengrass v2**, with an ML-driven **Decision Engine** that dynamically routes each inference request to `edge`, `cloud`, or `hybrid` execution.

---

## 1. Artefact

Research platform for intelligent Edge–Cloud AI orchestration using NVIDIA Jetson and AWS, integrating AWS IoT Greengrass, Docker and pre-trained AI models. Includes a four-policy Decision Engine (Rule-Based / Decision Tree / Random Forest / Q-Learning) that dynamically routes each inference request to minimise latency and cost while maintaining resilience during connectivity issues.

## 2. Contribution

Implementation and comparative evaluation of four orchestration policies for Edge–Cloud AI workloads deployed via AWS IoT Greengrass, showing how progressively more adaptive strategies (fixed rules → supervised tree → ensemble → RL agent) affect latency, cost, resource utilisation and reliability.

---

## 2.1 Scope & Non-Goals

**In scope**
- Integration of existing technologies (AWS IoT Greengrass, Docker, pre-trained AI models)
- Design and evaluation of orchestration policies (Rule / DT / RF / Q-Learning)
- Measurement of latency, cost, resource usage and resilience
- Reproducible experimental harness (`benchmark.py`) producing CSV + charts

**Explicitly out of scope**
- Training of new computer-vision models from scratch — YOLOv8n/s, MobileNet-SSD and EfficientDet-D0 are used as **pre-trained artefacts**
- Development of new cloud infrastructure — the project **consumes** AWS EC2, Greengrass, S3, CloudWatch as they exist
- Camera/OpenCV production pipeline — the visual pipeline is represented as a compute-bound proxy so that the study can isolate the decision layer

---

## 3. Architecture

```
                            AWS CLOUD
   ┌──────────────────────────────────────────────────────────┐
   │  EC2 (t3.medium)                                          │
   │  ├── FastAPI  ── /api/devices, /api/models,               │
   │  │              /api/deployments, /api/inferences         │
   │  │              /api/decisions/{predict,train,history}    │
   │  ├── Decision Engine  (Rule + DecisionTree + RandomForest)│
   │  ├── SQLite  (decision history, ML artefacts)             │
   │  └── MongoDB (live telemetry, deployments, devices)       │
   │                                                            │
   │  AWS IoT Greengrass v2  ── component deployments          │
   │  S3                    ── model repository                │
   │  CloudWatch            ── metrics & alarms                │
   └──────────────────────────────────────────────────────────┘
                             ▲  MQTT  ▼
   ┌──────────────────────────────────────────────────────────┐
   │  NVIDIA JETSON  (nano / xavier)                           │
   │  ├── Docker runtime                                        │
   │  ├── Greengrass Core nucleus                              │
   │  ├── OpenCV + YOLO inference container                    │
   │  ├── MQTT client → IoT Core                               │
   │  └── Local buffer (offline resilience)                    │
   └──────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack

| Layer              | Tech                                                    |
| ------------------ | ------------------------------------------------------- |
| Backend API        | Python 3.11 · FastAPI · Uvicorn                         |
| Decision Engine    | scikit-learn (DecisionTree, RandomForest) + heuristics  |
| Databases          | SQLite (ML/decisions) + MongoDB (telemetry)             |
| Frontend Dashboard | React 19 · Recharts · @phosphor-icons/react · Tailwind  |
| Edge Runtime       | Docker · OpenCV · YOLO (ONNX)                           |
| Cloud Orchestrator | AWS IoT Greengrass v2                                   |
| Cloud Compute      | AWS EC2                                                 |
| Monitoring         | CloudWatch                                              |
| Comms              | MQTT (IoT Core)                                         |
| Analysis           | Pandas · Matplotlib (offline reports)                   |

---

## 5. Decision Engine

The Decision Engine takes a context vector and returns a routing decision:

**Input features**
```
network_latency_ms, connectivity, cpu_available, memory_available,
batch_size, priority, cost_budget_usd, model_size_mb
```

**Output classes**  `edge` · `cloud` · `hybrid`

**Engines**

| Engine        | Model                                       | Notes                                     |
| ------------- | ------------------------------------------- | ----------------------------------------- |
| Rule-Based    | Hand-crafted heuristics                     | Interpretable baseline · deterministic    |
| DecisionTree  | `sklearn.DecisionTreeClassifier(depth=8)`   | Learns policy surface, easy to visualise  |
| RandomForest  | `sklearn.RandomForestClassifier(n=80)`      | Robust to noise, best generalization      |

Training data (4 000 synthetic samples) is generated from the rule-based policy with 8 % label noise. Models are persisted to `backend/artefacts/*.joblib` and reloaded on startup.

---

## 6. Local Run (Docker Compose)

```bash
docker compose up --build
# Frontend → http://localhost
# Backend  → http://localhost:8001/api/
```

## 7. Local Dev (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn server:app --reload --port 8001

# Frontend (separate shell)
cd frontend
yarn install
yarn start
```

---

## 8. Deploying to AWS EC2 + Greengrass

1. **Provision EC2**  ·  `t3.medium`, Amazon Linux 2023, open ports 80 / 8001.
2. **Install Docker**  ·  `sudo dnf install -y docker && sudo systemctl start docker`.
3. **Clone repo & run**  ·  `git clone ... && cd edge-cloud-orch && docker compose up -d`.
4. **Configure Greengrass v2 on Jetson**
   ```bash
   curl -sSL https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip -o gg.zip
   sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE \
        -jar ./GreengrassInstaller/lib/Greengrass.jar \
        --aws-region us-east-1 --thing-name jetson-edge-01 \
        --thing-group-name edge-fleet --provision true \
        --setup-system-service true
   ```
5. **Publish inference component** to S3 and deploy via Greengrass console → `edge-fleet`.
6. **Point Jetson MQTT client** at the EC2 backend URL.

---

## 9. API Reference (selected)

| Method | Path                                    | Purpose                              |
| ------ | --------------------------------------- | ------------------------------------ |
| GET    | `/api/devices`                          | List edge fleet                      |
| POST   | `/api/devices/{id}/network`             | Toggle connectivity (simulate)       |
| GET    | `/api/models`                           | Model repository                     |
| POST   | `/api/deployments`                      | Deploy model → device                |
| POST   | `/api/inferences`                       | Batch ingest inference telemetry     |
| GET    | `/api/metrics/summary`                  | Edge vs Cloud KPI (10 min window)    |
| POST   | `/api/decisions/predict`                | Route decision from chosen engine    |
| POST   | `/api/decisions/train`                  | Retrain DT + RF                      |
| GET    | `/api/decisions/status`                 | Metrics + feature importance         |
| GET    | `/api/decisions/history`                | Recent decisions from SQLite         |
| GET    | `/api/decisions/distribution`           | Route distribution (last N minutes)  |

---

## 10. Evaluation Plan (for the thesis)

Metrics collected per experiment (30 min run per engine):

- **Latency (p50, p95)** — edge vs cloud vs decision-engine routing
- **Resource usage** — Jetson CPU / memory average
- **Cost** — total USD (edge = fixed HW · cloud = $/inference)
- **Reliability** — success rate during simulated disconnects
- **Decision agreement** — how often DT/RF match the rule baseline

Reports can be exported from `/api/decisions/history` + `/api/metrics/timeseries` and analysed in Pandas / Matplotlib.

---

## 11. Directory Layout

```
/app
├── backend/
│   ├── server.py               # FastAPI app + endpoints
│   ├── decision_engine.py      # Rule + DT + RF engines
│   ├── decision_db.py          # SQLite persistence
│   ├── requirements.txt
│   ├── Dockerfile
│   └── artefacts/              # trained .joblib models + decisions.db
├── frontend/
│   ├── src/pages/Dashboard.js  # Control-room dashboard
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
└── README.md
```
