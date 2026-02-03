# 🚦 AI Model Failure Prediction System (Traffic Prediction)

An end-to-end **MLOps-focused AI system** that not only predicts traffic congestion but also **monitors the deployed model in real time and predicts model failure before it happens** using drift detection and meta-learning.

This project simulates how **production ML systems** are built, monitored, and maintained in real-world environments.

---

## 🔍 Problem Statement

Machine learning models often perform well during training but **degrade after deployment** due to:
- Data drift
- Concept drift
- Changing real-world conditions
- Model uncertainty

This project addresses that gap by building:
> **An AI system that predicts when another AI model is about to fail.**

---

## 🎯 Objectives

- Train a traffic congestion prediction model
- Deploy the model using FastAPI
- Log live predictions and performance metrics
- Monitor data drift and prediction confidence
- Predict model failure using a secondary AI (meta-model)
- Enable alerts and retraining triggers

---

## 🧠 System Architecture

Client / Sensor
↓
FastAPI Prediction API
↓
Traffic Prediction Model
↓
Prediction Logging (CSV / DB)
↓
Drift Detection & Monitoring
↓
Failure Prediction Model
↓
Alerts / Retraining


---

## 📊 Dataset

- **Source:** Kaggle – Traffic Prediction Dataset  
- **Format used in this project:**



timestamp | vehicle_count | avg_speed | weather | congestion_level


Additional features were engineered to simulate real-world conditions.

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Scikit-learn
- Pandas, NumPy

### Backend & Deployment
- FastAPI
- Uvicorn

### MLOps & Monitoring
- Prediction logging
- Data drift detection (PSI)
- Confidence monitoring
- Concept drift detection (planned)

### Tools
- Google Colab (training)
- Git & GitHub
- VS Code

---

## 📁 Project Structure



ai_pred_model/
│
├── app.py # FastAPI inference service
├── models/
│ ├── traffic_primary_model.pkl
│ └── weather_encoder.pkl
│
├── logs/
│ └── prediction_logs.csv # Auto-generated during inference
│
├── README.md
└── requirements.txt


---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Anandkumae/ai_pred_model.git
cd ai_pred_model

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run FastAPI Server
uvicorn app:app --reload

4️⃣ Test the API

Open in browser:

http://127.0.0.1:8000/docs


Example request:

{
  "vehicle_count": 45,
  "avg_speed": 22,
  "weather": "Rain"
}
