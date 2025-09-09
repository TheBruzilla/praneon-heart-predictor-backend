import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from typing import Tuple, Optional
import requests
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = "models/heart_disease_model.joblib"
        self.scaler_path = "models/scaler.joblib"
        
    def download_data(self) -> pd.DataFrame:
        """Download UCI Heart Disease dataset"""
        # UCI Heart Disease dataset URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        # Column names for the dataset
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        try:
            # Try to download the data
            df = pd.read_csv(url, names=columns, na_values='?')
            print("Downloaded data from UCI repository")
        except Exception as e:
            print(f"Could not download from UCI: {e}")
            # Create sample data if download fails
            df = self.create_sample_data()
            
        return df
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample heart disease data for testing"""
        np.random.seed(42)
        n_samples = 300
        
        data = {
            'age': np.random.randint(29, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(126, 565, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(71, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6.2, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(1, 4, n_samples),
        }
        
        # Create synthetic target variable based on some rules
        df = pd.DataFrame(data)
        target = []
        for _, row in df.iterrows():
            score = 0
            if row['age'] > 60: score += 1
            if row['cp'] in [2, 3]: score += 1
            if row['trestbps'] > 140: score += 1
            if row['chol'] > 240: score += 1
            if row['thalach'] < 150: score += 1
            if row['exang'] == 1: score += 1
            if row['oldpeak'] > 2: score += 1
            
            # Add some randomness
            if np.random.random() < 0.1:
                score = 1 - score if score in [0, 1] else score
                
            target.append(1 if score >= 3 else 0)
        
        df['target'] = target
        print("Created sample data for training")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data"""
        # Handle missing values
        df = df.dropna()
        
        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        return X.values, y.values
    
    def train_model(self) -> None:
        """Train the RandomForest model"""
        print("Starting model training...")
        
        # Download and preprocess data
        df = self.download_data()
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Save model and scaler
        self.save_model()
    
    def save_model(self) -> None:
        """Save the trained model and scaler"""
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self) -> None:
        """Load the trained model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Model loaded successfully")
        else:
            raise FileNotFoundError("Model files not found")
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Make prediction on input features"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Train or load model first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of disease
        
        return prediction, probability

if __name__ == "__main__":
    # Train model if run directly
    predictor = HeartDiseasePredictor()
    predictor.train_model()
    
    # Test prediction
    test_features = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
    prediction, probability = predictor.predict(test_features)
    print(f"Test prediction: {prediction}, probability: {probability:.3f}")