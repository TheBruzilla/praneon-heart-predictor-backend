"""
Simple HTTP server for heart disease prediction API
This minimal version works without external dependencies for testing
"""

import json
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import os

class HeartDiseaseHandler(http.server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.handle_health()
        elif parsed_path.path == '/':
            self.handle_root()
        elif parsed_path.path == '/docs':
            self.handle_docs()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/predict':
            self.handle_predict()
        else:
            self.send_error(404, "Not Found")
    
    def handle_health(self):
        response = {
            "status": "healthy",
            "model_loaded": True,
            "message": "Heart Disease Prediction API is running"
        }
        self.send_json_response(response)
    
    def handle_root(self):
        response = {
            "message": "Heart Disease Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "docs": "/docs"
            },
            "description": "FastAPI backend service for predicting heart disease risk using the UCI dataset"
        }
        self.send_json_response(response)
    
    def handle_docs(self):
        docs_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Heart Disease Prediction API - Documentation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .method { color: white; padding: 5px 10px; border-radius: 3px; }
                .get { background: #61affe; }
                .post { background: #49cc90; }
                code { background: #f9f9f9; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Heart Disease Prediction API</h1>
            <p>FastAPI backend service for predicting heart disease risk using the UCI dataset.</p>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /health</h3>
                <p>Health check endpoint</p>
                <p><strong>Response:</strong></p>
                <pre><code>{
  "status": "healthy",
  "model_loaded": true
}</code></pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /predict</h3>
                <p>Predict heart disease risk</p>
                <p><strong>Request Body:</strong></p>
                <pre><code>{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}</code></pre>
                <p><strong>Response:</strong></p>
                <pre><code>{
  "prediction": 1,
  "probability": 0.87,
  "risk_level": "High"
}</code></pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /</h3>
                <p>Root endpoint with API information</p>
            </div>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(docs_html.encode())
    
    def handle_predict(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Validate required fields
            required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            for field in required_fields:
                if field not in request_data:
                    self.send_error(400, f"Missing required field: {field}")
                    return
            
            # Simple prediction logic based on risk factors
            # This is a simplified version for demonstration
            risk_score = 0
            
            # Age factor
            if request_data['age'] > 60:
                risk_score += 1
            
            # Chest pain factor
            if request_data['cp'] in [2, 3]:
                risk_score += 1
            
            # Blood pressure factor
            if request_data['trestbps'] > 140:
                risk_score += 1
            
            # Cholesterol factor
            if request_data['chol'] > 240:
                risk_score += 1
            
            # Heart rate factor
            if request_data['thalach'] < 150:
                risk_score += 1
            
            # Exercise angina factor
            if request_data['exang'] == 1:
                risk_score += 1
            
            # ST depression factor
            if request_data['oldpeak'] > 2:
                risk_score += 1
            
            # Calculate prediction and probability
            prediction = 1 if risk_score >= 4 else 0
            probability = min(0.9, max(0.1, risk_score / 7.0))
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            response = {
                "prediction": prediction,
                "probability": round(probability, 3),
                "risk_level": risk_level
            }
            
            self.send_json_response(response)
            
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server(port=8000):
    with socketserver.TCPServer(("", port), HeartDiseaseHandler) as httpd:
        print(f"Heart Disease Prediction API running on port {port}")
        print(f"Health check: http://localhost:{port}/health")
        print(f"API docs: http://localhost:{port}/docs")
        print(f"Root endpoint: http://localhost:{port}/")
        httpd.serve_forever()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run_server(port)