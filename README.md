# Praneon Heart Predictor Backend

This repo contains a small FastAPI-based predictor service for the UCI Heart-like dataset.
It trains a RandomForest classifier (or uses synthetic data if `data/heart.csv` is not present),
serves `/predict` and `/health` endpoints, and is configured to run in Docker.

## Quick local run (recommended)
1. Create virtualenv & install:
   ```bash
   cd predict_service
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
