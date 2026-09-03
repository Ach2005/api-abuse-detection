# Machine Learning-Based Real-Time API Abuse Detection and Automated Mitigation

A Python-based API security system that uses behavioral analysis and machine learning to detect abnormal API usage in real time and automatically apply mitigation such as rate limiting and blocking.

## Overview

APIs are commonly targeted by attackers through behaviors such as flooding, brute-force attempts, endpoint spamming, scraping, and other abnormal request patterns.

This project analyzes the behavior of API clients over multiple requests instead of relying only on a single request. It extracts behavioral features, uses machine learning to detect abuse, calculates an abuse risk score, and automatically responds to suspicious activity.

## Key Features
- Behavioral analysis of API traffic
- Machine learning-based abuse detection
- Multiple abuse scenario simulation
- Behavioral abuse score from 0–100
- Risk classification:
-- NORMAL
-- SUSPICIOUS
-- HIGH RISK
- Real-time API monitoring using FastAPI middleware
- Automatic rate limiting
- Automatic blocking after repeated violations
- Comparison of multiple machine learning models
- Model validation and evaluation
- Swagger UI for API testing

## System Workflow


**API Requests**  
↓  
**Behavioral Feature Extraction**  
↓  
**Machine Learning Detection**  
↓  
**Behavioral Abuse Scoring**  
↓  
**Risk Classification**  
↓  
**Automated Mitigation**  
↓  
**Allow / Rate Limit / Block**

## Behavioral Features

The system analyzes several characteristics of API client behavior, including:

- Session request count
- Requests per minute
- Failed authentication ratio
- API error ratio
- Unique endpoint count
- Endpoint diversity
- Mean request interval
- Request interval variation
- Night activity ratio
- Geographic movement
- Burst activity

These features help the system distinguish normal API usage from abnormal behavior.

## Machine Learning Models

The project compares the following machine learning models:

- Logistic Regression
- Decision Tree
- Random Forest
- Isolation Forest

Random Forest is used as the primary detection model because it provided the strongest validation performance among the evaluated models on the synthetic dataset.

## Risk-Based Mitigation

The system classifies API behavior into three levels:

- Risk Level	Meaning	Action
- NORMAL	Normal API behavior	ALLOW
- SUSPICIOUS	Unusual behavior	RATE LIMIT / MONITOR
- HIGH RISK	Strongly suspicious behavior	BLOCK

Rate limiting temporarily restricts clients that send requests too frequently. Repeated violations can result in the client being blocked.

## Dataset

The project uses a synthetically generated API traffic dataset because real production API security traffic was not available.

The dataset contains multiple scenarios, including:

- Normal traffic
- Brute force
- Flooding
- Endpoint spam
- Scraping
- Stealth abuse
- Geo anomaly
- Unusual timing

The latest generated dataset contains approximately 100,000 API requests, which are aggregated into behavioral sessions for analysis and machine learning.

Note: Model performance reported in this project is based on synthetic data. Real-world production traffic would be required for production-level validation.

## Project Structure



### Data

- `data/raw/` — Raw API traffic dataset
- `data/processed/` — Processed behavioral features

### Models

- `models/` — Trained Random Forest model

### Results

- `results/` — Model results, evaluation files, and graphs

### Source Code

- `src/api/` — FastAPI application, middleware, and real-time testing
- `src/data_generation/` — Synthetic API traffic generation
- `src/detection/` — Abuse detection and rate limiting
- `src/evaluation/` — Model and system evaluation
- `src/features/` — Behavioral feature extraction
- `src/models/` — Machine learning model training
- `src/scoring/` — Behavioral abuse scoring

## Technology Stack

- **Python**  
  Main programming language

- **Pandas**  
  Data processing and analysis

- **Scikit-learn**  
  Machine learning

- **FastAPI**  
  API development

- **Uvicorn**  
  ASGI server

- **Joblib**  
  Saving and loading the trained model

- **Matplotlib**  
  Evaluation visualizations

- **Swagger UI**  
  API testing and documentation

- **VS Code**  
  Development environment
  
## Installation

Clone the repository using:

git clone https://github.com/Ach2005/api-abuse-detection.git

Then move into the project directory:

cd api-abuse-detection

Install the required Python packages:

pip install pandas scikit-learn fastapi uvicorn joblib matplotlib

## Running the Real-Time API

Start the FastAPI server using:

python -m uvicorn src.api.app:app --reload

The API will be available at:

http://127.0.0.1:8000

## Swagger API Testing

FastAPI provides an interactive Swagger UI automatically.

Open:

http://127.0.0.1:8000/docs

The /detect endpoint can be used to provide behavioral features and observe:

- ML prediction
- ML abuse probability
- Abuse score
- Risk classification
- Recommended action
- Detection reasons

## Real-Time Detection Demo

With the FastAPI server running, open another terminal and execute:

python src/api/test_realtime.py

The test demonstrates:

**Normal Traffic**

↓

**ALLOW**

↓

**Suspicious Traffic**

↓

**RATE LIMIT**

↓

**Repeated Violations**

↓

**HIGH RISK**

↓

**BLOCK**


## Example Detection Result

A highly suspicious behavioral input can produce a result similar to:

- ML Prediction: 1
- ML Abuse Probability: 82.41%
- Abuse Score: 100
- Risk: HIGH RISK
- Action: BLOCK

The ML prediction indicates whether the trained model classifies the behavior as normal or abusive.

The ML abuse probability represents the model's estimated probability that the behavior is abusive.

The abuse score is a separate behavioral risk score based on multiple suspicious characteristics.

## Model Evaluation

The project evaluates the machine learning models using metrics such as:

- Precision
- Recall
- F1-score
- False Positive Rate
- ROC-AUC
- Confusion Matrix
- Cross-validation

Random Forest achieved the strongest overall validation performance on the synthetic behavioral dataset and was selected as the primary model.

## Project Objective

The objective of this project is to demonstrate how machine learning and behavioral analysis can be combined with real-time API monitoring to identify abusive usage and automatically apply appropriate mitigation.

## Future Scope

Possible future improvements include:

- Testing with real-world API traffic
- Adding database or Redis-based distributed tracking
- Integrating IP reputation and threat intelligence
- Adding authentication-specific monitoring
- Deploying the system in a cloud environment
- Adding a dedicated security monitoring dashboard
- Improving detection of low-and-slow attacks

## Disclaimer

This project is an academic/prototype implementation intended to demonstrate API abuse detection, behavioral analysis, machine learning, and automated mitigation techniques.
