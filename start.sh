#!/bin/bash

# Heart Disease Prediction API Startup Script

echo "Heart Disease Prediction API"
echo "============================"

# Check if dependencies are available
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "FastAPI dependencies found, starting FastAPI server..."
    uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --reload
else
    echo "FastAPI dependencies not found, starting simple HTTP server..."
    python3 simple_server.py
fi