# main.py
# Run locally:
#  uvicorn main:app --reload --host 0.0.0.0 --port 8000

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Heart Predictor")

MODEL_FILE = "model.joblib"
SCALER_FILE = "scaler.joblib"
API_KEY = os.getenv("PREDICTOR_API_KEY")  # optional

# expected fields (UCI heart examples)
class HeartInput(BaseModel):
    age: int = Field(..., example=63)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=3)
    trestbps: float = Field(..., example=145.0)
    chol: float = Field(..., example=233.0)
    fbs: int = Field(..., example=1)
    restecg: int = Field(..., example=0)
    thalach: float = Field(..., example=150.0)
    exang: int = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=0)
    ca: int = Field(..., example=0)
    thal: int = Field(..., example=1)

# load model + scaler (joblib saved dict: {"model": clf, "feature_names": [...]})
model_data = None
scaler = None
MODEL_FEATURES = None
if os.path.exists(MODEL_FILE):
    model_data = joblib.load(MODEL_FILE)
    if isinstance(model_data, dict) and "model" in model_data:
        clf = model_data["model"]
        MODEL_FEATURES = model_data.get("feature_names")
    else:
        clf = model_data  # older style: direct estimator
    print("Loaded model from", MODEL_FILE)
else:
    raise RuntimeError(f"{MODEL_FILE} not found. Run train_model.py first.")

if os.path.exists(SCALER_FILE):
    scaler = joblib.load(SCALER_FILE)
    print("Loaded scaler from", SCALER_FILE)
else:
    print("No scaler found; predictions will use raw features.")

@app.get("/health")
def health():
    return {"status": "ok"}

def check_api_key(header_key: Optional[str]):
    if API_KEY:
        if not header_key or header_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.post("/predict")
async def predict(payload: HeartInput, x_api_key: Optional[str] = Header(None)):
    # API key check if configured
    check_api_key(x_api_key)

    # prepare feature vector in the same order as MODEL_FEATURES or default order
    if MODEL_FEATURES:
        # map features by name
        try:
            feat = np.array([getattr(payload, f) for f in MODEL_FEATURES], dtype=float).reshape(1, -1)
        except Exception:
            # fallback: try using the dataclass order
            feat = np.array([payload.age, payload.sex, payload.cp, payload.trestbps, payload.chol,
                             payload.fbs, payload.restecg, payload.thalach, payload.exang, payload.oldpeak,
                             payload.slope, payload.ca, payload.thal], dtype=float).reshape(1, -1)
    else:
        feat = np.array([payload.age, payload.sex, payload.cp, payload.trestbps, payload.chol,
                         payload.fbs, payload.restecg, payload.thalach, payload.exang, payload.oldpeak,
                         payload.slope, payload.ca, payload.thal], dtype=float).reshape(1, -1)

    if scaler is not None:
        feat = scaler.transform(feat)

    proba = clf.predict_proba(feat)[0]
    pred = int(clf.predict(feat)[0])

    return {
        "prediction": pred,
        "probability_of_positive": float(proba[1]) if len(proba) > 1 else float(proba[0]),
        "probabilities": proba.tolist(),
        "classes": clf.classes_.tolist() if hasattr(clf, "classes_") else None
    }
