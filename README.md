# Titanic Survival Prediction - End-to-End MLOps Pipeline
## 70% Done by me ,Guided by GPT
## Overview

This project demonstrates a complete MLOps workflow by building, deploying, and monitoring a machine learning model for Titanic survival prediction. The focus is on implementing production-grade MLOps practices rather than developing a complex machine learning model.

The project covers the entire machine learning lifecycle, including data versioning, validation, experiment tracking, model registry, automated testing, CI/CD, deployment, monitoring, and drift detection.

---

## Architecture

```text
Data
 │
 ▼
DVC (Data Versioning)
 │
 ▼
Great Expectations (Data Validation)
 │
 ▼
Scikit-learn Training Pipeline
 │
 ▼
MLflow (Experiment Tracking & Model Registry)
 │
 ▼
Pytest Testing
 │
 ▼
Docker Containerization
 │
 ▼
GitHub Actions CI/CD
 │
 ▼
Render Deployment
 │
 ▼
FastAPI Inference Service
 │
 ├── Prometheus Monitoring
 │
 ├── Grafana Dashboards
 │
 └── Evidently AI Drift Detection
```
<img width="685" height="505" alt="image" src="https://github.com/user-attachments/assets/31baa365-4d33-4b7c-89d8-c591a8d54be6" />
<img width="696" height="390" alt="image" src="https://github.com/user-attachments/assets/1e0651ca-5618-49ce-9727-3de6adb57646" />

---

## Features

### Data Versioning

* Dataset tracking using DVC
* Version-controlled datasets
* Reproducible experiments
* Dataset rollback capability

### Data Validation

* Schema validation
* Missing value detection
* Data quality checks
* Automated validation pipeline

### Model Training

* Scikit-learn Random Forest Classifier
* Feature preprocessing pipeline
* Automated training workflow
* Performance evaluation

### Experiment Tracking

* Parameter logging
* Metric tracking
* Artifact management
* Experiment comparison using MLflow

### Model Registry

* Model version management
* Staging and Production lifecycle
* Best model promotion

### API Serving

* FastAPI-based inference service
* Input validation
* Interactive Swagger documentation
* REST API endpoints

### Testing

* Unit tests
* API tests
* Model loading tests
* CI quality checks

### CI/CD

* Automated testing
* Automated model training
* Docker image generation
* Automated deployment to Render

### Monitoring

* Request monitoring
* Latency tracking
* Error monitoring
* Resource utilization tracking

### Drift Detection

* Data drift monitoring
* Feature drift analysis
* Prediction drift detection
* Automated reports

---

## Technology Stack

| Component           | Tool               |
| ------------------- | ------------------ |
| Version Control     | Git, GitHub        |
| Data Versioning     | DVC                |
| Data Validation     | Great Expectations |
| Model Training      | Scikit-learn       |
| Experiment Tracking | MLflow             |
| Model Registry      | MLflow Registry    |
| API Framework       | FastAPI            |
| Testing             | Pytest             |
| Containerization    | Docker             |
| CI/CD               | GitHub Actions     |
| Deployment          | Render             |
| Monitoring          | Prometheus         |
| Visualization       | Grafana            |
| Drift Detection     | Evidently AI       |

---

## Project Structure

```text
titanic-mlops/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── api/
│   └── main.py
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│
├── mlruns/
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dvc.yaml
└── README.md
```

---

## Workflow

### 1. Data Versioning

Track dataset versions using DVC.

```bash
dvc add data/raw/titanic.csv
git add .
git commit -m "Add dataset"
```

### 2. Data Validation

Validate incoming data before training.

```bash
python src/data_validation.py
```

### 3. Model Training

Train and log experiments.

```bash
python src/train.py
```

### 4. Run MLflow

```bash
mlflow ui
```

Access MLflow UI:

```text
http://localhost:5000
```

### 5. Run API

```bash
uvicorn api.main:app --reload
```

Access API:

```text
http://localhost:8000/docs
```

---

## API Example

### Request

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "Fare": 7.25
}
```

### Response

```json
{
  "survived": false,
  "probability": 0.18
}
```

---

## CI/CD Pipeline

GitHub Actions automatically executes:

```text
Code Push
   ↓
Install Dependencies
   ↓
Run Tests
   ↓
Validate Data
   ↓
Train Model
   ↓
Build Docker Image
   ↓
Deploy to Render
```

---

## Monitoring

Prometheus collects:

* API request count
* Response latency
* Error rates
* Application metrics

Grafana dashboards visualize system performance and model behavior.

---

## Drift Detection

Evidently AI continuously monitors:

* Feature distribution changes
* Data drift
* Prediction drift
* Data quality metrics

---

## Deployment

The inference service is containerized using Docker and automatically deployed to Render through GitHub Actions whenever changes are pushed to the main branch.

---

## Key Learning Outcomes

* Data Versioning with DVC
* Experiment Tracking with MLflow
* Model Registry Management
* Automated Data Validation
* FastAPI Model Serving
* Docker Containerization
* CI/CD Automation with GitHub Actions
* Production Monitoring with Prometheus and Grafana
* Data Drift Detection with Evidently AI
* End-to-End MLOps Lifecycle Management

---

## Future Enhancements

* Automated retraining pipeline
* Feature store integration
* A/B model testing
* Kubernetes deployment
* Multi-model serving
* Real-time streaming inference
