# Knowledge Distillation and Deployment of a Lightweight Command-Parsing Language Model on a Local Cloud

**Cloud Computing Mini Project**  
**Student:** ____________________  
**Roll Number:** ____________________  
**Department:** Computer Science and Engineering

---

## Abstract

This project investigates the training and deployment of a lightweight natural-language command-parsing model using knowledge distillation. A larger teacher model, Phi-3.5, is used to generate structured responses for natural-language commands. These responses serve as training targets for a smaller Pythia-70M student model.

The project combines machine-learning workflows with cloud-computing concepts, including S3-compatible object storage, parallel processing, Docker containerization, Kubernetes orchestration, network access control, automated testing and performance evaluation.

A pilot experiment was conducted using 245 teacher-labelled examples and a separate validation set of 100 examples. The pilot achieved 99% JSON validity, but intent and date extraction accuracy remained limited. The results demonstrate the feasibility of the initial workflow while highlighting the need for a larger and more representative training dataset.

## 1. Objectives

The main objectives are:

1. Generate structured command labels using a teacher language model.
2. Store datasets and intermediate outputs using S3-compatible object storage.
3. Process commands using parallel workers with checkpoint recovery.
4. Train a lightweight student model using teacher-generated responses.
5. Develop a REST inference service and package it with Docker.
6. Orchestrate application workloads using Kubernetes.
7. implement subnetting, network ACL simulation, testing and performance analysis.

## 2. System Architecture

The intended processing pipeline consists of the following stages:

**Data input:** A command pool containing 3,000 natural-language commands and a separate validation dataset containing 100 examples.

**Teacher labelling:** Phi-3.5 is accessed through Ollama. Kubernetes worker jobs divide the command pool into shards and generate structured JSON responses.

**Object storage:** SeaweedFS provides an S3-compatible interface. The `datasets` and `models` buckets are used for storing dataset and model artifacts. Workers periodically upload checkpoints to support recovery.

**Data cleaning:** Teacher outputs are checked for JSON validity, required keys, valid intents, slot consistency and duplicate commands.

**Student training:** Pythia-70M is fine-tuned using the teacher-generated responses. The validation dataset is kept separate from training.

**Inference and deployment:** A FastAPI service loads the student model and exposes health and command-parsing endpoints. Docker packages the service and model for deployment to Kubernetes.

## 3. Task 1 — Object Storage and Parallel Teacher Labelling

SeaweedFS was deployed in the local Kubernetes environment with an S3-compatible endpoint. Two buckets, `datasets` and `models`, were created.

The original command pool was uploaded to object storage. Parallel Kubernetes worker jobs were configured to process separate shards using the command's line index modulo the shard count.

The worker implementation includes retry handling and periodic checkpoint uploads. This allows completed work to be recovered after a worker failure.

The finalization script is designed to merge the completed shards, remove invalid or duplicate records, retain the original teacher responses, generate a cleaned training dataset and calculate SHA256 checksums.

**Status at the last recorded checkpoint:** The two workers had processed approximately 280 and 290 records respectively. This is a historical progress snapshot, not proof that the complete 3,000-command labelling process finished.

## 4. Task 2 — Knowledge Distillation and Student Training

The teacher model used was Phi-3.5 through Ollama. The student model was EleutherAI Pythia-70M.

The teacher prompt instructs the model to produce a JSON object containing exactly four fields: `intent`, `date`, `time` and `place`.

The allowed intents are:
- alarm_set
- alarm_query
- calendar_set
- calendar_query
- calendar_remove
- weather_query
- datetime_query
- recommendation_events

The pilot training run used 245 teacher-labelled examples. It was trained for two epochs using the original teacher-generated responses.

The observed average training loss decreased from 0.4667 in the first epoch to 0.1357 in the second epoch.

The validation data was not used for training.

This was a pilot experiment. The complete training run requires the full cleaned teacher-target dataset. The pilot loss alone does not establish generalization or final model quality.

## 5. Task 3 — Inference Service and Docker

A FastAPI application was implemented in `service/app.py`.

The service provides:
- `GET /health` for health checking.
- `POST /parse` for command parsing.

The application loads the model from a local model directory and returns the generated prediction. The response also includes model-version and serving-instance headers.

A Dockerfile installs CPU-based PyTorch and the inference dependencies, copies the application and includes the model files.

A pilot Docker image named `command-parser:pilot` was built. Its reported Docker disk usage was 2.34 GB.

**Limitation:** The pilot container was observed in the `Created` state and had not yet been verified as a running, healthy inference service. A successful API response must be recorded before claiming successful container deployment.

## 6. Task 4 — Kubernetes Deployment, Scaling and Reliability

A local Kubernetes cluster was created using k3d. Separate namespaces were configured for the application, object storage and pipeline workers.

Kubernetes manifests were prepared for the application deployment, service and Horizontal Pod Autoscaler. The application deployment specifies two initial replicas and a CPU-based scaling target.

Readiness and liveness probes are configured to use the service health endpoint.

The teacher-labelling workloads use separate Kubernetes jobs, allowing independent processing of command shards.

The application deployment and HPA configuration require successful runtime verification. The existence of manifests alone does not demonstrate that the final inference service was deployed or scaled successfully.

## 7. Task 5 — Subnetting and Network ACL

A NACL configuration was created to demonstrate ordered inbound and outbound traffic rules with default-deny behaviour.

The sample packet set contains six test cases covering permitted and denied traffic.

The NACL simulator was executed with the `--check` option and passed all six sample decisions.

A proposed logical private network allocation was documented:

| Subnet purpose | CIDR |
|---|---|
| Application | 10.20.1.0/24 |
| Data | 10.20.2.0/24 |
| Pipeline | 10.20.3.0/24 |
| Management | 10.20.4.0/24 |
| Reserved | 10.20.5.0/24 |

These are proposed logical subnets for the assignment. They are not a claim that the k3d cluster was deployed using these exact CIDR ranges.

## 8. Task 6 — Reproducibility, Testing and CI/CD

The repository contains Python scripts for dataset inspection, teacher testing, prediction generation, scoring and NACL simulation.

A GitHub Actions workflow was created to run compilation checks, NACL validation and command-line checks for project utilities.

The project includes test files under `tests/`. The complete test suite and the GitHub Actions workflow must be run and their actual results recorded before submission.

The README documents the environment, source-code organization and commands used to reproduce the available steps.

## 9. Task 7 — Performance, Cost and SLO Analysis

The pilot model was evaluated against 100 validation examples.

| Metric | Pilot result |
|---|---:|
| JSON validity | 99% |
| Intent accuracy | 58% |
| Date accuracy | 54% |
| Time accuracy | 85% |
| Place accuracy | 86% |
| Full-frame accuracy | 21% |

The results indicate that the model learned to produce mostly valid JSON, but its intent classification and date extraction require improvement.

A hypothetical resource rate card was prepared using INR values of ₹3.50 per vCPU-hour and ₹0.45 per GiB memory-hour. Under the assumed limits of one vCPU and one GiB memory per replica, the estimated cost is ₹3.95 per replica-hour, or ₹7.90 per hour for two replicas.

This is a modelled estimate based on assumed resource limits, not an actual cloud invoice. A measured cost per 1,000 requests requires a measured request rate.

A final SLO analysis requires recorded latency, throughput, error rate and availability measurements. These values must not be invented.

## 10. Challenges and Limitations

The teacher model's CPU inference speed limited the rate of labelling. Parallel workers and checkpoint recovery were introduced to reduce lost work and allow independent processing.

The pilot student model showed a large difference between JSON validity and full-frame accuracy. This indicates that syntactically valid output does not necessarily contain the correct intent and slot values.

The Docker image was built, but the pilot container had not yet been verified as running. The final Kubernetes application deployment and load tests therefore remain items requiring verification.

The project also requires final checks of the complete teacher dataset, CI workflow and reproducibility instructions.

## 11. Conclusion and Future Work

The project established an initial workflow combining teacher-model labelling, S3-compatible object storage, parallel workers, student-model training, FastAPI and Docker packaging.

The pilot experiment provided useful evaluation evidence and exposed weaknesses in intent classification and date extraction.

Future work includes completing the full teacher-labelled dataset, training the final student model, improving prediction quality, verifying the inference service and Kubernetes deployment, and measuring latency, throughput, reliability and cost under load.

The final submission should include only results that were actually measured and deployments that were successfully verified.

## References

1. EleutherAI, Pythia language model documentation and model repository.
2. Microsoft, Phi-3.5 model documentation.
3. Ollama documentation.
4. Hugging Face Transformers documentation.
5. PyTorch documentation.
6. Kubernetes documentation.
7. Docker documentation.
8. SeaweedFS documentation.
9. FastAPI documentation.
