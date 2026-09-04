# AI-Driven Edge–Cloud Orchestration

Research artefact developed for the MSc Artificial Intelligence dissertation:

**AI-Driven Infrastructure Orchestration for Edge–Cloud Environments:  
A Comparative Study of Intelligent Decision Policies for AI Inference Routing**

University of Bedfordshire

---

## 1. Overview

This repository contains the implemented research artefact for an AI-driven
Edge–Cloud orchestration system designed to investigate dynamic AI workload
placement across heterogeneous computing resources.

The system integrates a React monitoring dashboard, a Python/FastAPI
orchestration backend, an Ubuntu Linux Edge environment connected through
AWS IoT Greengrass, and an Amazon EC2 Cloud node.

AI workloads are routed between three execution alternatives:

- Edge
- Cloud
- Hybrid

Routing decisions are produced using Rule-Based and Q-Learning policies.

---

## 2. Research Purpose

The artefact provides an operational environment for investigating how AI
inference workloads can be dynamically placed across Edge and Cloud resources.

The implementation enables routing behaviour to be examined alongside runtime
conditions including execution latency, CPU utilisation, memory utilisation,
network round-trip time, execution success and policy decision time.

The system is intended as a research prototype and experimental platform rather
than a production deployment.

---

## 3. System Architecture

The implemented architecture consists of:

### React Dashboard

Provides runtime visibility of infrastructure state, AI workloads, routing
decisions, telemetry and orchestration behaviour.

### FastAPI Orchestrator

Coordinates workload requests, collects runtime information, invokes the active
decision policy and dispatches execution to the selected Edge, Cloud or Hybrid
route.

### Edge Environment

The Edge node is implemented as an Ubuntu Linux virtual machine running
concurrently with the Windows development environment.

AWS IoT Greengrass provides the Edge deployment and lifecycle-management layer.

Implemented Greengrass component:

`com.edgecloud.InferenceAgent`

Version:

`1.1.2`

### Cloud Environment

Remote Cloud execution is provided by an Amazon EC2 Linux instance identified
within the experimental environment as:

`cloud-node-01`

### Hybrid Execution

Hybrid execution represents cooperative participation of Edge and Cloud
resources within the same workload workflow.

It is not implemented as neural-network or model partitioning.

---

## 4. Decision Policies

### Rule-Based Policy

Provides deterministic routing according to predefined infrastructure and
workload conditions.

The policy serves as an interpretable orchestration baseline.

### Q-Learning Policy

Provides adaptive routing using learned state–action values.

The available actions are:

- Edge
- Cloud
- Hybrid

Execution feedback is converted into reward information and used to update the
Q-values associated with the observed state and selected action.

Q-values and rewards are algorithm-derived decision information and are analysed
separately from measured infrastructure performance.

---

## 5. AI Workloads

### YOLOv8

YOLOv8 provides object-detection inference and represents a computer-vision
workload relevant to operational visual-analysis scenarios.

### Florence-2

Florence-2 provides multimodal visual inference and enables richer
interpretation of visual inputs than object detection alone.

### Live Camera Input

The physical camera of the Windows host can provide live visual input to the AI
workflow.

Captured input is submitted through the orchestration pipeline and processed by
the selected AI workload, enabling end-to-end validation from data acquisition
to returned inference output.

---

## 6. Telemetry and Runtime Monitoring

The orchestration environment records runtime information associated with
workload execution and routing behaviour.

Evaluation data includes:

- execution latency
- CPU utilisation
- memory utilisation
- network round-trip time (RTT)
- execution success
- policy decision time

For Q-Learning experiments, Q-values and rewards are additionally retained as
evidence of adaptive-policy behaviour.

---

## 7. End-to-End Execution

A typical execution follows the sequence:

1. A workload is submitted through the interface.
2. FastAPI receives the request.
3. Runtime and workload information are obtained.
4. The active decision policy evaluates the available routes.
5. Edge, Cloud or Hybrid execution is selected.
6. The workload is dispatched to the selected environment.
7. AI inference is executed.
8. Runtime telemetry and execution outcome are recorded.
9. Results are returned to the dashboard.
10. Q-Learning values are updated when the adaptive policy is active.

---

## 8. AWS IoT Greengrass Deployment

The Edge environment is integrated with AWS IoT Greengrass.

Deployment lifecycle:

S3 artefact  
→ Greengrass component  
→ Edge deployment  
→ dependency installation  
→ inference service startup  
→ Edge runtime

The implemented Edge inference component is:

`com.edgecloud.InferenceAgent v1.1.2`

---

## 9. Gemini AI Assistant

The dashboard additionally integrates a Gemini-based AI assistant through an
API connection.

The assistant provides a natural-language interaction layer through which users
can query system information and obtain contextual explanations of orchestration
results.

Gemini does not perform workload-placement decisions and is not part of the
Rule-Based or Q-Learning routing mechanism.

---

## 10. Experimental Evaluation

The artefact supports empirical evaluation of Edge, Cloud and Hybrid execution
and comparison of Rule-Based and Q-Learning orchestration.

The principal evaluation dimensions are:

- execution responsiveness
- infrastructure resource utilisation
- network conditions
- execution reliability
- routing-policy decision overhead
- adaptive Q-Learning behaviour
- successful AI inference execution

The experimental results reported in the dissertation are based on observations
obtained from execution of the implemented system.

---

## 11. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React |
| Backend / Orchestration | Python, FastAPI, Uvicorn |
| Edge OS | Ubuntu Linux |
| Edge Deployment | AWS IoT Greengrass |
| Cloud Compute | Amazon EC2 |
| AI Workloads | YOLOv8, Florence-2 |
| Decision Policies | Rule-Based, Q-Learning |
| Persistence | SQLite |
| Live Input | PC Camera |
| AI Assistant | Gemini API |

---

## 12. Repository Structure

The repository contains the implementation supporting the orchestration
workflow, including the frontend interface, FastAPI backend, decision-policy
logic, telemetry handling and supporting deployment material.

Exact directory contents may vary according to the final frozen dissertation
release.

---

## 13. Security

Security-sensitive information is not included in the public research artefact.

The repository must not contain:

- AWS access keys
- SSH private keys
- Gemini API keys
- passwords
- authentication tokens
- `.env` files containing secrets
- environment-specific credentials

Required credentials must be supplied separately through secure environment
configuration.

---

## 14. Academic Use

This repository accompanies an MSc Artificial Intelligence dissertation at the
University of Bedfordshire.

The artefact is provided to support examination, reproducibility and further
research into intelligent AI workload orchestration across heterogeneous
Edge–Cloud environments.