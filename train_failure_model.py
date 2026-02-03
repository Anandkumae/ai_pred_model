"""
Train Failure Prediction Meta-Model

This script trains a Random Forest classifier to predict AI failures
based on aggregated health metrics from time windows.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Configuration
LABELED_DATASET = "failure_dataset_labeled.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "failure_predictor.pkl")
MIN_SAMPLES = 50  # Minimum samples needed for training

def generate_synthetic_data(n_samples=100):
    """
    Generate synthetic failure dataset for demonstration.
    Creates realistic scenarios based on monitoring thresholds.
    """
    print(f"\n🔧 Generating {n_samples} synthetic samples...")
    
    synthetic_data = []
    
    for i in range(n_samples):
        # Randomly decide if this is a failure or healthy window
        is_failure = np.random.random() < 0.3  # 30% failure rate
        
        if is_failure:
            # Failure scenarios - at least one metric crosses threshold
            psi_score = np.random.uniform(0.15, 0.5)  # Higher drift
            avg_confidence = np.random.uniform(0.4, 0.7)  # Lower confidence
            latency = np.random.uniform(0.03, 0.1)  # Higher latency
            concept_drift_count = np.random.randint(0, 5)  # More drift
            error_trend = np.random.uniform(0.005, 0.03)  # Increasing errors
            label = 1
        else:
            # Healthy scenarios - all metrics within normal range
            psi_score = np.random.uniform(0.0, 0.2)  # Low drift
            avg_confidence = np.random.uniform(0.75, 1.0)  # High confidence
            latency = np.random.uniform(0.01, 0.04)  # Low latency
            concept_drift_count = 0  # No drift
            error_trend = np.random.uniform(-0.01, 0.01)  # Stable errors
            label = 0
        
        synthetic_data.append({
            'psi_score': psi_score,
            'avg_confidence': avg_confidence,
            'latency': latency,
            'concept_drift_count': concept_drift_count,
            'error_trend': error_trend,
            'prediction_count': np.random.randint(20, 100),
            'label': label
        })
    
    return pd.DataFrame(synthetic_data)

def train_failure_model():
    """Train Random Forest classifier for failure prediction"""
    
    print("=" * 70)
    print("FAILURE PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Load labeled dataset
    if os.path.exists(LABELED_DATASET):
        df = pd.read_csv(LABELED_DATASET)
        print(f"\n✓ Loaded {len(df)} samples from {LABELED_DATASET}")
    else:
        print(f"\n⚠️ {LABELED_DATASET} not found, using only synthetic data")
        df = pd.DataFrame()
    
    # Check if we need synthetic data
    if len(df) < MIN_SAMPLES:
        print(f"\n⚠️ Insufficient data ({len(df)} samples, need {MIN_SAMPLES})")
        synthetic_df = generate_synthetic_data(MIN_SAMPLES - len(df))
        df = pd.concat([df, synthetic_df], ignore_index=True)
        print(f"✓ Combined dataset: {len(df)} samples")
    
    # Prepare features and labels
    feature_columns = ['psi_score', 'avg_confidence', 'latency', 
                      'concept_drift_count', 'error_trend']
    
    X = df[feature_columns]
    y = df['label']
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Healthy (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"   Failure (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📚 Train/Test Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Train Random Forest
    print(f"\n🌲 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")
    
    # Evaluate
    print(f"\n📈 Model Evaluation:")
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"   Training Accuracy: {train_acc:.4f}")
    
    # Test accuracy
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, test_pred, 
                                target_names=['Healthy', 'Failure']))
    
    # Confusion matrix
    print(f"🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # Feature importance
    print(f"\n🎯 Feature Importance:")
    importances = model.feature_importances_
    for feature, importance in zip(feature_columns, importances):
        print(f"   {feature:20s}: {importance:.4f}")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\n✓ Model saved to {MODEL_FILE}")
    
    # Test prediction
    print(f"\n" + "=" * 70)
    print("TEST PREDICTIONS")
    print("=" * 70)
    
    # Test case 1: Healthy scenario
    healthy_sample = pd.DataFrame([{
        'psi_score': 0.05,
        'avg_confidence': 0.95,
        'latency': 0.02,
        'concept_drift_count': 0,
        'error_trend': 0.001
    }])
    
    healthy_prob = model.predict_proba(healthy_sample)[0]
    print(f"\n✅ Healthy Scenario:")
    print(f"   PSI: 0.05, Confidence: 0.95, Latency: 0.02")
    print(f"   Failure Probability: {healthy_prob[1]:.4f}")
    print(f"   Risk: {'HIGH' if healthy_prob[1] > 0.7 else 'MEDIUM' if healthy_prob[1] > 0.3 else 'LOW'}")
    
    # Test case 2: Failure scenario
    failure_sample = pd.DataFrame([{
        'psi_score': 0.35,
        'avg_confidence': 0.55,
        'latency': 0.08,
        'concept_drift_count': 3,
        'error_trend': 0.02
    }])
    
    failure_prob = model.predict_proba(failure_sample)[0]
    print(f"\n❌ Failure Scenario:")
    print(f"   PSI: 0.35, Confidence: 0.55, Latency: 0.08")
    print(f"   Failure Probability: {failure_prob[1]:.4f}")
    print(f"   Risk: {'HIGH' if failure_prob[1] > 0.7 else 'MEDIUM' if failure_prob[1] > 0.3 else 'LOW'}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Model is ready for deployment")
    print("2. Add /failure-risk endpoint to app.py")
    print("3. Monitor predictions and retrain with real data")
    print("=" * 70)
    
    return model

if __name__ == "__main__":
    train_failure_model()
