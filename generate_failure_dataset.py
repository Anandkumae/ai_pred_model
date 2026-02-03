"""
Generate Failure Dataset for Meta-Level Model Monitoring

This script processes prediction logs and creates a dataset that describes
model health over time windows. Each row represents a time window with
aggregated metrics that can be used to train an AI that predicts AI failure.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Configuration
LOG_FILE = "logs/prediction_logs.csv"
BASELINE_FILE = "baseline_stats.json"
OUTPUT_FILE = "failure_dataset.csv"
WINDOW_SIZE_HOURS = 1  # Time window size in hours

def calculate_psi(expected, actual, buckets=10):
    """Calculate PSI between two distributions"""
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
        
        if exp_pct > 0 and act_pct > 0:
            psi += (exp_pct - act_pct) * np.log(exp_pct / act_pct)
    
    return psi

def calculate_error_trend(confidences):
    """
    Calculate error trend (slope) using confidence as proxy.
    Error = 1 - confidence
    Returns slope of error over time.
    """
    if len(confidences) < 2:
        return 0.0
    
    errors = 1.0 - np.array(confidences)
    x = np.arange(len(errors))
    
    # Calculate slope using linear regression
    slope = np.polyfit(x, errors, 1)[0]
    return float(slope)

def generate_failure_dataset():
    """
    Generate failure dataset from prediction logs.
    Aggregates metrics into time windows.
    """
    print("=" * 70)
    print("FAILURE DATASET GENERATION")
    print("=" * 70)
    
    # Load prediction logs
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: {LOG_FILE} not found!")
        return
    
    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n✓ Loaded {len(df)} predictions from {LOG_FILE}")
    
    # Load baseline for PSI calculation
    with open(BASELINE_FILE, 'r') as f:
        baseline = json.load(f)
    
    print(f"✓ Loaded baseline statistics from {BASELINE_FILE}")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create time windows
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    
    print(f"\nTime range: {start_time} to {end_time}")
    print(f"Window size: {WINDOW_SIZE_HOURS} hour(s)")
    
    # Generate windows
    windows = []
    current_time = start_time
    
    while current_time < end_time:
        window_end = current_time + timedelta(hours=WINDOW_SIZE_HOURS)
        
        # Get predictions in this window
        window_data = df[(df['timestamp'] >= current_time) & 
                        (df['timestamp'] < window_end)]
        
        if len(window_data) > 0:
            # Calculate features
            
            # 1. PSI Score (average across all features)
            psi_scores = []
            for feature in ['vehicle_count', 'avg_speed', 'weather_encoded']:
                if feature in window_data.columns:
                    psi = calculate_psi(baseline[feature], window_data[feature].values)
                    psi_scores.append(psi)
            avg_psi = np.mean(psi_scores) if psi_scores else 0.0
            
            # 2. Average Confidence
            avg_confidence = window_data['confidence'].mean()
            
            # 3. Average Latency
            avg_latency = window_data['latency'].mean()
            
            # 4. Concept Drift Count
            concept_drift_count = 0
            if 'concept_drift' in window_data.columns:
                concept_drift_count = int(window_data['concept_drift'].sum())
            
            # 5. Error Trend (slope of error over time)
            error_trend = calculate_error_trend(window_data['confidence'].values)
            
            # Create window record
            window_record = {
                'window_start': current_time,
                'window_end': window_end,
                'psi_score': round(avg_psi, 6),
                'avg_confidence': round(avg_confidence, 6),
                'latency': round(avg_latency, 6),
                'concept_drift_count': concept_drift_count,
                'error_trend': round(error_trend, 6),
                'prediction_count': len(window_data),
                'label': -1  # -1 = unlabeled, 0 = healthy, 1 = failure
            }
            
            windows.append(window_record)
        
        current_time = window_end
    
    # Create dataset
    dataset_df = pd.DataFrame(windows)
    
    print(f"\n✓ Generated {len(dataset_df)} time windows")
    print(f"\nDataset shape: {dataset_df.shape}")
    print(f"\nFeature summary:")
    print(dataset_df.describe())
    
    # Save dataset
    dataset_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved failure dataset to {OUTPUT_FILE}")
    
    print("\n" + "=" * 70)
    print("DATASET PREVIEW")
    print("=" * 70)
    print(dataset_df.head(10).to_string())
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review the generated dataset")
    print("2. Use label_failures.py to manually label windows")
    print("3. Train a meta-model to predict failures")
    print("=" * 70)
    
    return dataset_df

if __name__ == "__main__":
    generate_failure_dataset()
