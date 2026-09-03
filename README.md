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
  - NORMAL
  - SUSPICIOUS
  - HIGH RISK
- Real-time API monitoring using FastAPI middleware
- Automatic rate limiting
- Automatic blocking after repeated violations
- Comparison of multiple machine learning models
- Model validation and evaluation
- Swagger UI for API testing

## System Workflow

```text
API Requests
     ↓
Behavioral Feature Extraction
     ↓
Machine Learning Detection
     ↓
Behavioral Abuse Scoring
     ↓
Risk Classification
     ↓
Automated Mitigation
     ↓
Allow / Rate Limit / Block