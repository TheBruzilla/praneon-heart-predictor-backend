#!/usr/bin/env python3
"""
Basic test for the Heart Disease Prediction API
"""

import json
import urllib.request
import urllib.parse
import sys

def test_endpoint(url, method='GET', data=None):
    """Test an API endpoint"""
    try:
        if data:
            data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header('Content-Type', 'application/json')
        else:
            req = urllib.request.Request(url, method=method)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

def main():
    base_url = "http://localhost:8000"
    
    print("Testing Heart Disease Prediction API")
    print("=====================================")
    
    # Test health endpoint
    print("1. Health Check:")
    result = test_endpoint(f"{base_url}/health")
    print(json.dumps(result, indent=2))
    print()
    
    # Test root endpoint
    print("2. Root Endpoint:")
    result = test_endpoint(f"{base_url}/")
    print(json.dumps(result, indent=2))
    print()
    
    # Test prediction
    print("3. Prediction (High Risk):")
    high_risk_data = {
        "age": 65,
        "sex": 1,
        "cp": 3,
        "trestbps": 150,
        "chol": 250,
        "fbs": 1,
        "restecg": 1,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 2,
        "ca": 1,
        "thal": 3
    }
    result = test_endpoint(f"{base_url}/predict", "POST", high_risk_data)
    print(json.dumps(result, indent=2))
    print()
    
    # Test prediction
    print("4. Prediction (Low Risk):")
    low_risk_data = {
        "age": 35,
        "sex": 0,
        "cp": 0,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 170,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 1
    }
    result = test_endpoint(f"{base_url}/predict", "POST", low_risk_data)
    print(json.dumps(result, indent=2))
    print()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()