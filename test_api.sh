#!/bin/bash

# Test script for Heart Disease Prediction API

BASE_URL="http://localhost:8000"

echo "Testing Heart Disease Prediction API"
echo "===================================="

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# Test root endpoint
echo "2. Testing root endpoint..."
curl -s "$BASE_URL/" | python3 -m json.tool
echo ""

# Test prediction with high risk
echo "3. Testing prediction (high risk case)..."
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 70,
    "sex": 1,
    "cp": 3,
    "trestbps": 160,
    "chol": 280,
    "fbs": 1,
    "restecg": 1,
    "thalach": 130,
    "exang": 1,
    "oldpeak": 3.5,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }' | python3 -m json.tool
echo ""

# Test prediction with low risk
echo "4. Testing prediction (low risk case)..."
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "sex": 0,
    "cp": 0,
    "trestbps": 110,
    "chol": 180,
    "fbs": 0,
    "restecg": 0,
    "thalach": 180,
    "exang": 0,
    "oldpeak": 0.5,
    "slope": 1,
    "ca": 0,
    "thal": 1
  }' | python3 -m json.tool
echo ""

# Test error handling
echo "5. Testing error handling (missing field)..."
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50,
    "sex": 1
  }'
echo ""

echo "API testing completed!"