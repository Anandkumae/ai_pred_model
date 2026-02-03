from fastapi import FastAPI
import joblib
import time
import pandas as pd
from datetime import datetime
import os
import numpy as np
import json
from river.drift import ADWIN
from alert_system import alert_system

# ------------------------
# App initialization
# ------------------------
app = FastAPI()

# ------------------------
# Paths & folders
# ------------------------
MODEL_DIR = "models"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_logs.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------
# Load models
# ------------------------
model = joblib.load(os.path.join(MODEL_DIR, "traffic_primary_model.pkl"))
weather_enc = joblib.load(os.path.join(MODEL_DIR, "weather_encoder.pkl"))

# Load failure prediction model (meta-model)
FAILURE_MODEL_FILE = os.path.join(MODEL_DIR, "failure_predictor.pkl")
if os.path.exists(FAILURE_MODEL_FILE):
    failure_model = joblib.load(FAILURE_MODEL_FILE)
    print(f"✓ Loaded failure prediction model from {FAILURE_MODEL_FILE}")
else:
    failure_model = None
    print(f"⚠️ Failure prediction model not found. Run train_failure_model.py first.")

# ------------------------
# Load baseline statistics for drift detection
# ------------------------
BASELINE_FILE = "baseline_stats.json"
with open(BASELINE_FILE, "r") as f:
    baseline_stats = json.load(f)

# ------------------------
# Initialize ADWIN for concept drift detection
# ------------------------
adwin_detector = ADWIN()

# ------------------------
# PSI Calculation Function
# ------------------------
def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) to detect data drift.
    PSI > 0.2 indicates significant drift.
    """
    if len(actual) == 0:
        return 0.0
    
    expected = np.array(expected)
    actual = np.array(actual)
    
    breakpoints = np.percentile(expected, np.arange(0, 100, 100/buckets))
    breakpoints = np.append(breakpoints, expected.max())
    
    psi = 0
    for i in range(len(breakpoints)-1):
        exp_pct = ((expected >= breakpoints[i]) & 
                   (expected < breakpoints[i+1])).mean()
        act_pct = ((actual >= breakpoints[i]) & 
                   (actual < breakpoints[i+1])).mean()
        
        # Avoid division by zero
        if exp_pct > 0 and act_pct > 0:
            psi += (exp_pct - act_pct) * np.log(exp_pct / act_pct)
    
    return psi

def check_drift(feature_name, recent_values, threshold=0.2):
    """
    Check if drift detected for a specific feature.
    Returns (drift_detected, psi_score)
    """
    if feature_name not in baseline_stats:
        return False, 0.0
    
    baseline = baseline_stats[feature_name]
    psi_score = calculate_psi(baseline, recent_values)
    drift_detected = psi_score > threshold
    
    return drift_detected, psi_score

def check_confidence_risk(df_recent, window=100):
    """
    Monitor rolling confidence to detect model degradation.
    Returns (risk_level, avg_confidence, min_confidence, max_confidence)
    """
    if len(df_recent) == 0:
        return "UNKNOWN", 0.0, 0.0, 0.0
    
    # Calculate rolling average confidence
    confidences = df_recent["confidence"].values
    avg_confidence = float(np.mean(confidences))
    min_confidence = float(np.min(confidences))
    max_confidence = float(np.max(confidences))
    
    # Determine risk level based on average confidence
    if avg_confidence < 0.6:
        risk_level = "HIGH"
    elif avg_confidence < 0.75:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return risk_level, avg_confidence, min_confidence, max_confidence

def update_concept_drift(confidence):
    """
    Update ADWIN detector with prediction error proxy.
    Uses (1 - confidence) as error metric.
    Returns True if concept drift detected.
    """
    global adwin_detector
    
    # Use inverse confidence as error proxy
    # High confidence (0.9) -> low error (0.1)
    # Low confidence (0.5) -> high error (0.5)
    error = 1.0 - float(confidence)
    
    # Update ADWIN with error value
    adwin_detector.update(error)
    
    # Check if drift detected
    drift_detected = adwin_detector.drift_detected
    
    return drift_detected

# ------------------------
# Prediction endpoint
# ------------------------
@app.post("/predict")
def predict(data: dict):
    start = time.time()

    # Encode weather
    weather_encoded = weather_enc.transform([data["weather"]])[0]

    # Create DataFrame with correct feature names
    X = pd.DataFrame([{
        "vehicle_count": data["vehicle_count"],
        "avg_speed": data["avg_speed"],
        "weather": weather_encoded
    }])

    # Model inference
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    latency = time.time() - start
    
    # ------------------------
    # Concept Drift Detection (ADWIN)
    # ------------------------
    concept_drift_detected = update_concept_drift(confidence)

    # Logging (STEP 12)
    log_row = {
        "timestamp": datetime.now(),
        "vehicle_count": data["vehicle_count"],
        "avg_speed": data["avg_speed"],
        "weather": data["weather"],
        "weather_encoded": int(weather_encoded),
        "prediction": int(prediction),
        "confidence": float(confidence),
        "latency": latency,
        "concept_drift": concept_drift_detected
    }

    df_log = pd.DataFrame([log_row])

    if not os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, index=False)
    else:
        df_log.to_csv(LOG_FILE, mode="a", header=False, index=False)

    # ------------------------
    # Data Drift Detection & Confidence Monitoring
    # ------------------------
    drift_alerts = {}
    confidence_monitoring = {}
    
    # Load recent predictions (last 100) for drift analysis
    if os.path.exists(LOG_FILE):
        try:
            df_recent = pd.read_csv(LOG_FILE).tail(100)
            
            if len(df_recent) >= 30:  # Need minimum samples for reliable PSI
                # Check drift for vehicle_count
                vehicle_drift, vehicle_psi = check_drift(
                    "vehicle_count", 
                    df_recent["vehicle_count"].values
                )
                
                # Check drift for avg_speed
                speed_drift, speed_psi = check_drift(
                    "avg_speed", 
                    df_recent["avg_speed"].values
                )
                
                # Check drift for weather_encoded
                weather_drift, weather_psi = check_drift(
                    "weather_encoded", 
                    df_recent["weather_encoded"].values
                )
                
                # Convert numpy bools to Python bools for JSON serialization
                drift_alerts = {
                    "drift_detected": bool(vehicle_drift or speed_drift or weather_drift),
                    "psi_scores": {
                        "vehicle_count": round(float(vehicle_psi), 4),
                        "avg_speed": round(float(speed_psi), 4),
                        "weather_encoded": round(float(weather_psi), 4)
                    },
                    "drift_status": {
                        "vehicle_count": "DRIFT" if vehicle_drift else "OK",
                        "avg_speed": "DRIFT" if speed_drift else "OK",
                        "weather_encoded": "DRIFT" if weather_drift else "OK"
                    }
                }
                
                if drift_alerts["drift_detected"]:
                    drift_alerts["warning"] = "⚠️ Data drift detected! Model reliability may be compromised."
                
                # Confidence monitoring
                risk_level, avg_conf, min_conf, max_conf = check_confidence_risk(df_recent)
                confidence_monitoring = {
                    "risk_level": risk_level,
                    "avg_confidence": round(avg_conf, 4),
                    "min_confidence": round(min_conf, 4),
                    "max_confidence": round(max_conf, 4),
                    "samples_analyzed": int(len(df_recent))
                }
                
                if risk_level == "HIGH":
                    confidence_monitoring["warning"] = "🚨 HIGH RISK: Model confidence is very low! Model may be failing."
                elif risk_level == "MEDIUM":
                    confidence_monitoring["warning"] = "⚠️ MEDIUM RISK: Model confidence is declining. Monitor closely."
                    
        except Exception as e:
            drift_alerts = {"error": f"Drift calculation failed: {str(e)}"}
            confidence_monitoring = {"error": f"Confidence monitoring failed: {str(e)}"}

    # Concept drift status
    concept_drift_status = {
        "drift_detected": bool(concept_drift_detected),
        "method": "ADWIN",
        "description": "Monitors changes in input-output relationship using error stream"
    }
    
    if concept_drift_detected:
        concept_drift_status["warning"] = "🔴 CONCEPT DRIFT DETECTED! Model's learned patterns may no longer be valid."

    return {
        "prediction": int(prediction),
        "confidence": float(confidence),
        "latency": latency,
        "drift_detection": drift_alerts if drift_alerts else {"status": "insufficient_data"},
        "confidence_monitoring": confidence_monitoring if confidence_monitoring else {"status": "insufficient_data"},
        "concept_drift": concept_drift_status
    }

# ------------------------
# Drift Monitoring Endpoint
# ------------------------
@app.get("/drift-status")
def get_drift_status():
    """
    Get current drift status across all features.
    Analyzes the last 100 predictions.
    """
    if not os.path.exists(LOG_FILE):
        return {"error": "No prediction logs available"}
    
    try:
        df_recent = pd.read_csv(LOG_FILE).tail(100)
        
        if len(df_recent) < 30:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 30 predictions, currently have {len(df_recent)}"
            }
        
        # Check drift for all features
        vehicle_drift, vehicle_psi = check_drift("vehicle_count", df_recent["vehicle_count"].values)
        speed_drift, speed_psi = check_drift("avg_speed", df_recent["avg_speed"].values)
        weather_drift, weather_psi = check_drift("weather_encoded", df_recent["weather_encoded"].values)
        
        overall_drift = bool(vehicle_drift or speed_drift or weather_drift)
        
        # Check confidence risk
        risk_level, avg_conf, min_conf, max_conf = check_confidence_risk(df_recent)
        
        # Check concept drift occurrences in recent predictions
        concept_drift_count = 0
        if "concept_drift" in df_recent.columns:
            concept_drift_count = int(df_recent["concept_drift"].sum())
        
        return {
            "overall_drift_detected": overall_drift,
            "total_predictions_analyzed": int(len(df_recent)),
            "drift_detection": {
                "features": {
                    "vehicle_count": {
                        "psi_score": round(float(vehicle_psi), 4),
                        "status": "DRIFT" if vehicle_drift else "OK",
                        "drift_detected": bool(vehicle_drift)
                    },
                    "avg_speed": {
                        "psi_score": round(float(speed_psi), 4),
                        "status": "DRIFT" if speed_drift else "OK",
                        "drift_detected": bool(speed_drift)
                    },
                    "weather_encoded": {
                        "psi_score": round(float(weather_psi), 4),
                        "status": "DRIFT" if weather_drift else "OK",
                        "drift_detected": bool(weather_drift)
                    }
                },
                "threshold": 0.2,
                "interpretation": {
                    "psi_range": "0.0-0.1: No significant change | 0.1-0.2: Slight change | >0.2: Significant drift"
                },
                "warning": "⚠️ Data drift detected! Model reliability may be compromised." if overall_drift else None
            },
            "confidence_monitoring": {
                "risk_level": risk_level,
                "avg_confidence": round(avg_conf, 4),
                "min_confidence": round(min_conf, 4),
                "max_confidence": round(max_conf, 4),
                "thresholds": {
                    "high_risk": "< 0.6",
                    "medium_risk": "0.6 - 0.75",
                    "low_risk": ">= 0.75"
                },
                "warning": "🚨 HIGH RISK: Model confidence is very low! Model may be failing." if risk_level == "HIGH" 
                          else "⚠️ MEDIUM RISK: Model confidence is declining. Monitor closely." if risk_level == "MEDIUM"
                          else None
            },
            "concept_drift": {
                "method": "ADWIN",
                "drift_occurrences": concept_drift_count,
                "total_predictions": int(len(df_recent)),
                "drift_rate": round(concept_drift_count / len(df_recent), 4) if len(df_recent) > 0 else 0.0,
                "description": "Monitors changes in input-output relationship using adaptive windowing",
                "warning": f"🔴 Concept drift detected {concept_drift_count} times in recent predictions!" if concept_drift_count > 0 else None
            }
        }
    
    except Exception as e:
        return {"error": f"Failed to calculate drift status: {str(e)}"}

# ------------------------
# Failure Risk Prediction Endpoint (Meta-Model)
# ------------------------
@app.get("/failure-risk")
def get_failure_risk():
    """
    Predict AI failure probability using meta-model.
    Analyzes recent window metrics and returns risk assessment.
    """
    if failure_model is None:
        return {
            "error": "Failure prediction model not loaded",
            "message": "Run train_failure_model.py to train the meta-model"
        }
    
    if not os.path.exists(LOG_FILE):
        return {"error": "No prediction logs available"}
    
    try:
        # Load recent predictions
        df_recent = pd.read_csv(LOG_FILE).tail(100)
        
        if len(df_recent) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 10 predictions, currently have {len(df_recent)}"
            }
        
        # Calculate current window metrics (same as failure dataset features)
        
        # 1. PSI Score (average across all features)
        psi_scores = []
        for feature in ['vehicle_count', 'avg_speed', 'weather_encoded']:
            if feature in df_recent.columns:
                psi = calculate_psi(baseline_stats[feature], df_recent[feature].values)
                psi_scores.append(psi)
        avg_psi = float(np.mean(psi_scores)) if psi_scores else 0.0
        
        # 2. Average Confidence
        avg_confidence = float(df_recent['confidence'].mean())
        
        # 3. Average Latency
        avg_latency = float(df_recent['latency'].mean())
        
        # 4. Concept Drift Count
        concept_drift_count = 0
        if 'concept_drift' in df_recent.columns:
            concept_drift_count = int(df_recent['concept_drift'].sum())
        
        # 5. Error Trend (slope of error over time)
        confidences = df_recent['confidence'].values
        if len(confidences) >= 2:
            errors = 1.0 - confidences
            x = np.arange(len(errors))
            error_trend = float(np.polyfit(x, errors, 1)[0])
        else:
            error_trend = 0.0
        
        # Create feature vector for prediction
        features = pd.DataFrame([{
            'psi_score': avg_psi,
            'avg_confidence': avg_confidence,
            'latency': avg_latency,
            'concept_drift_count': concept_drift_count,
            'error_trend': error_trend
        }])
        
        # Predict failure probability
        failure_proba = failure_model.predict_proba(features)[0]
        failure_probability = float(failure_proba[1])  # Probability of failure (class 1)
        
        # Determine risk level
        if failure_probability > 0.7:
            risk = "HIGH"
            warning = "🚨 HIGH RISK: AI failure imminent! Immediate intervention required."
        elif failure_probability > 0.3:
            risk = "MEDIUM"
            warning = "⚠️ MEDIUM RISK: AI showing signs of degradation. Monitor closely."
        else:
            risk = "LOW"
            warning = None
        
        # Trigger automated alerts and actions
        alert_response = alert_system.handle_alert(failure_probability, {
            'psi_score': avg_psi,
            'avg_confidence': avg_confidence,
            'latency': avg_latency,
            'concept_drift_count': concept_drift_count,
            'error_trend': error_trend
        })
        
        return {
            "failure_probability": round(failure_probability, 4),
            "risk": risk,
            "warning": warning,
            "metrics": {
                "psi_score": round(avg_psi, 6),
                "avg_confidence": round(avg_confidence, 6),
                "latency": round(avg_latency, 6),
                "concept_drift_count": concept_drift_count,
                "error_trend": round(error_trend, 6)
            },
            "predictions_analyzed": int(len(df_recent)),
            "model_info": {
                "type": "RandomForestClassifier",
                "features": ["psi_score", "avg_confidence", "latency", 
                           "concept_drift_count", "error_trend"]
            },
            "alert_system": alert_response
        }
    
    except Exception as e:
        return {"error": f"Failed to predict failure risk: {str(e)}"}
