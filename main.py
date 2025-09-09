from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import os

# Import our model module
from app.model import HeartDiseasePredictor

app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI backend service for predicting heart disease risk using the UCI dataset",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    age: int
    sex: int  # 1 = male, 0 = female
    cp: int  # chest pain type (0-3)
    trestbps: int  # resting blood pressure
    chol: int  # serum cholesterol in mg/dl
    fbs: int  # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    restecg: int  # resting electrocardiographic results (0-2)
    thalach: int  # maximum heart rate achieved
    exang: int  # exercise induced angina (1 = yes, 0 = no)
    oldpeak: float  # ST depression induced by exercise relative to rest
    slope: int  # slope of the peak exercise ST segment (0-2)
    ca: int  # number of major vessels colored by flourosopy (0-3)
    thal: int  # thalassemia (1-3)

class PredictionResponse(BaseModel):
    prediction: int  # 0 = no disease, 1 = disease
    probability: float  # probability of having heart disease
    risk_level: str  # "Low", "Medium", "High"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Initialize the predictor
predictor = HeartDiseasePredictor()

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    try:
        predictor.load_model()
    except Exception as e:
        # If model doesn't exist, train it
        print(f"Model not found, training new model: {e}")
        predictor.train_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_heart_disease(request: PredictionRequest):
    """Predict heart disease risk"""
    try:
        # Convert request to numpy array
        features = np.array([[
            request.age, request.sex, request.cp, request.trestbps,
            request.chol, request.fbs, request.restecg, request.thalach,
            request.exang, request.oldpeak, request.slope, request.ca,
            request.thal
        ]])
        
        # Make prediction
        prediction, probability = predictor.predict(features)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)