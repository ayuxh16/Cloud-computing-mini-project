# Knowledge Distillation and Deployment of a Lightweight Command-Parsing Language Model

**Course:** Cloud Computing Mini Project  
**Repository:** https://github.com/ayuxh16/Cloud-computing-mini-project

## 1. Project Overview

This project explores the knowledge distillation of a lightweight command-parsing language model and its deployment in a local cloud environment.

A larger teacher model (Phi-3.5, served through Ollama) generates structured responses for natural-language commands. These responses are used as training targets for a smaller Pythia-70M student model.

The model is designed to extract four fields:

- `intent`
- `date`
- `time`
- `place`

The project combines language-model training with object storage, parallel processing, containerization, Kubernetes orchestration, network security, automated testing and performance evaluation.

## 2. System Architecture

The intended workflow is:

1. Store the command pool in S3-compatible object storage.
2. Divide commands between parallel teacher-labelling workers.
3. Save teacher responses and checkpoints to object storage.
4. Validate, clean and deduplicate the labelled dataset.
5. Train a Pythia-70M student model using teacher-generated responses.
6. Package the inference service and model in a Docker image.
7. Deploy the service to a local Kubernetes cluster.
8. Evaluate predictions and inspect performance and resource usage.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Teacher model | Phi-3.5 through Ollama |
| Student model | EleutherAI Pythia-70M |
| ML framework | PyTorch, Transformers |
| Programming language | Python |
| Object storage | SeaweedFS (S3-compatible) |
| API | FastAPI |
| Containerization | Docker |
| Orchestration | Kubernetes, k3d |
| Testing | pytest, custom evaluation scripts |
| Load testing | Locust |
| CI/CD | GitHub Actions |

## 4. Repository Structure

```text
.github/workflows/   CI workflow
cluster/             Local Kubernetes cluster configuration
data/                Command pool, validation data and schemas
k8s/                 Kubernetes manifests
network/             Subnetting design and NACL configuration
prompts/             Teacher-model prompt
report/              Technical report
results/             Evaluation and experiment outputs
scripts/             Dataset and experiment utilities
service/             FastAPI inference application and Dockerfile
tests/               Automated tests
tools/               Scoring, load testing and NACL simulator
training/             Student-model training code
worker/               Parallel teacher-labelling worker
```

## 5. Environment Setup

The project was developed and tested in a Windows environment using PowerShell, Docker Desktop, Python 3.11, Ollama, kubectl and k3d.

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the training dependencies:

```powershell
pip install -r training/requirements.txt
```

Install the inference dependencies:

```powershell
pip install -r service/requirements.txt
```

Additional worker dependencies are listed in `worker/requirements.txt`.

## 6. Dataset and Teacher Labelling

The command pool is stored in `data/commands_pool.jsonl`, and validation examples are stored separately in `data/validation.jsonl`.

The teacher prompt is maintained in `prompts/teacher_prompt.txt`.

The worker implementation is in `worker/worker.py`. It supports sharding, retries and periodic checkpoint uploads to S3-compatible object storage.

The intended finalization process is:

```powershell
python scripts/finalize_task1.py
```

This command requires the completed teacher shards to be available in object storage.

## 7. Student Model Training

The student training implementation is in `training/train.py`.

The pilot experiment used 245 teacher-labelled examples. The pilot model is not the final model and its evaluation results should not be interpreted as final project performance.

The final training run requires the complete cleaned teacher-target dataset.

## 8. Inference Service and Docker

The FastAPI application is implemented in `service/app.py`.

The service provides:

- `GET /health` — model/service health check.
- `POST /parse` — parses a natural-language command.

Build the inference image from the project root:

```powershell
docker build -f service/Dockerfile -t command-parser:pilot .
```

The pilot image includes a pilot model. A final image should be built using the final trained student model.

## 9. Kubernetes

The local cluster configuration is in `cluster/`.

Kubernetes manifests are maintained in `k8s/`. They include the application deployment, service and horizontal pod autoscaler configuration.

The application deployment must be tested with a successfully built image containing the intended model.

## 10. Network Security

The NACL configuration is maintained in `network/nacl.json`.

Run the packet simulation:

```powershell
python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check
```

The subnet allocation documented in the report is a proposed logical design and should not be confused with the actual k3d cluster network.

## 11. Evaluation

The evaluation script is:

```powershell
python tools/score.py PREDICTIONS data/validation.jsonl
```

The pilot evaluation achieved:

| Metric | Pilot result |
|---|---:|
| JSON validity | 99% |
| Intent accuracy | 58% |
| Date accuracy | 54% |
| Time accuracy | 85% |
| Place accuracy | 86% |
| Full-frame accuracy | 21% |

These results were obtained from 100 validation examples using a pilot model trained on 245 examples. They are preliminary, not final results.

## 12. Testing and CI

The project contains a GitHub Actions workflow in `.github/workflows/ci.yml` and tests in `tests/`.

Run the NACL check and available tests before submission:

```powershell
python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check
python -m pytest tests -v
```

The CI workflow should also be checked in the GitHub Actions tab.

## 13. Limitations and Current Status

The pilot model demonstrates the training and evaluation workflow, but its intent and date extraction accuracy need improvement.

The complete teacher-labelling process, final student training, successful inference-container startup, Kubernetes deployment and load testing must be verified before they are described as completed.

Only measured results should be reported. Do not interpret proposed resource costs or subnet allocations as deployed cloud infrastructure.

## 14. Future Work

- Complete the full teacher-labelled dataset.
- Improve student-model intent and slot extraction accuracy.
- Train and evaluate the final student model.
- Complete the Docker and Kubernetes deployment.
- Measure inference latency, throughput and resource usage.
- Validate autoscaling and reliability under load.
- Improve automated tests and reproducibility.

## 15. Author and Repository

**Repository:** https://github.com/ayuxh16/Cloud-computing-mini-project
