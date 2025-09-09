# Heart Disease Prediction API

FastAPI backend service for predicting heart disease risk using the UCI dataset. This service trains a scikit-learn RandomForest model and exposes REST endpoints for predictions and health checks.

## Features

- **Machine Learning**: RandomForest classifier trained on UCI Heart Disease dataset
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Health Monitoring**: Built-in health check endpoint
- **CORS Support**: Ready for frontend integration
- **Docker Support**: Containerized deployment
- **Cloud Ready**: Configured for Render deployment
- **Auto-training**: Automatically trains model on startup if not found

## API Endpoints

### GET `/health`
Health check endpoint that returns API status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST `/predict`
Predict heart disease risk based on patient data.

**Request Body:**
```json
{
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
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.87,
  "risk_level": "High"
}
```

### GET `/`
Root endpoint with API information and available endpoints.

### GET `/docs`
Interactive API documentation (Swagger UI).

## Feature Descriptions

| Feature | Description | Values |
|---------|-------------|---------|
| age | Age in years | 29-79 |
| sex | Gender | 1 = male, 0 = female |
| cp | Chest pain type | 0-3 (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic) |
| trestbps | Resting blood pressure (mm Hg) | 94-200 |
| chol | Serum cholesterol (mg/dl) | 126-564 |
| fbs | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false |
| restecg | Resting ECG results | 0-2 (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy) |
| thalach | Maximum heart rate achieved | 71-202 |
| exang | Exercise induced angina | 1 = yes, 0 = no |
| oldpeak | ST depression induced by exercise | 0-6.2 |
| slope | Slope of peak exercise ST segment | 0-2 (0: upsloping, 1: flat, 2: downsloping) |
| ca | Number of major vessels colored by fluoroscopy | 0-3 |
| thal | Thalassemia | 1: normal, 2: fixed defect, 3: reversible defect |

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd praneon-heart-predictor-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Training the Model

The model will automatically train on first startup. To manually train:

```bash
python -m app.model
```

## Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t heart-predictor-api .

# Run the container
docker run -p 8000:8000 heart-predictor-api
```

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
```

## Render Deployment

This application is configured for easy deployment on Render:

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Render will automatically detect the `render.yaml` configuration
4. Deploy with one click

The service will automatically:
- Install dependencies
- Train the model on startup
- Start the FastAPI server

## Frontend Integration

This API is designed to work with a Next.js frontend. Key features for integration:

- **CORS enabled** for cross-origin requests
- **JSON responses** for easy parsing
- **Detailed error messages** for debugging
- **Standardized response format**

### Example JavaScript fetch:

```javascript
const response = await fetch('https://your-api-url.onrender.com/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    age: 63,
    sex: 1,
    cp: 3,
    trestbps: 145,
    chol: 233,
    fbs: 1,
    restecg: 0,
    thalach: 150,
    exang: 0,
    oldpeak: 2.3,
    slope: 0,
    ca: 0,
    thal: 1
  })
});

const prediction = await response.json();
console.log(prediction);
```

## Model Information

- **Algorithm**: Random Forest Classifier
- **Dataset**: UCI Heart Disease Dataset
- **Features**: 13 clinical features
- **Target**: Binary classification (0 = no disease, 1 = disease)
- **Accuracy**: ~85% (varies with data)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PORT | Server port | 8000 |
| PYTHONPATH | Python path | /app |

## Development

### Project Structure

```
├── app/
│   ├── __init__.py
│   └── model.py          # ML model training and prediction
├── models/               # Trained model files (auto-generated)
├── data/                # Dataset files (auto-generated)
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── render.yaml         # Render deployment config
└── README.md           # This file
```

### Adding New Features

1. **New endpoints**: Add to `main.py`
2. **Model improvements**: Modify `app/model.py`
3. **Data preprocessing**: Update preprocessing in `app/model.py`

## License

This project is part of a full-stack machine learning demo showcasing production-ready ML deployment.