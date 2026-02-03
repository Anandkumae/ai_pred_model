"""
Simple manual test for drift detection
Run this after making some predictions
"""
import pandas as pd
import numpy as np
import json

# Load baseline
with open("baseline_stats.json", "r") as f:
    baseline = json.load(f)

# Load recent logs
df = pd.read_csv("logs/prediction_logs.csv")
print(f"Total predictions in log: {len(df)}")
print(f"\nColumns: {list(df.columns)}")

if len(df) >= 30:
    recent = df.tail(100)
    print(f"\nAnalyzing last {len(recent)} predictions...")
    
    # Calculate PSI for vehicle_count
    def calculate_psi(expected, actual, buckets=10):
        if len(actual) == 0:
            return 0.0
        expected = np.array(expected)
        actual = np.array(actual)
        breakpoints = np.percentile(expected, np.arange(0, 100, 100/buckets))
        breakpoints = np.append(breakpoints, expected.max())
        psi = 0
        for i in range(len(breakpoints)-1):
            exp_pct = ((expected >= breakpoints[i]) & (expected < breakpoints[i+1])).mean()
            act_pct = ((actual >= breakpoints[i]) & (actual < breakpoints[i+1])).mean()
            if exp_pct > 0 and act_pct > 0:
                psi += (exp_pct - act_pct) * np.log(exp_pct / act_pct)
        return psi
    
    vehicle_psi = calculate_psi(baseline["vehicle_count"], recent["vehicle_count"].values)
    speed_psi = calculate_psi(baseline["avg_speed"], recent["avg_speed"].values)
    weather_psi = calculate_psi(baseline["weather_encoded"], recent["weather_encoded"].values)
    
    print(f"\nPSI Scores:")
    print(f"  vehicle_count: {vehicle_psi:.4f} {'⚠️ DRIFT' if vehicle_psi > 0.2 else '✓ OK'}")
    print(f"  avg_speed: {speed_psi:.4f} {'⚠️ DRIFT' if speed_psi > 0.2 else '✓ OK'}")
    print(f"  weather_encoded: {weather_psi:.4f} {'⚠️ DRIFT' if weather_psi > 0.2 else '✓ OK'}")
    
    print(f"\nBaseline ranges:")
    print(f"  vehicle_count: {min(baseline['vehicle_count'])}-{max(baseline['vehicle_count'])}")
    print(f"  avg_speed: {min(baseline['avg_speed'])}-{max(baseline['avg_speed'])}")
    
    print(f"\nRecent data ranges:")
    print(f"  vehicle_count: {recent['vehicle_count'].min()}-{recent['vehicle_count'].max()}")
    print(f"  avg_speed: {recent['avg_speed'].min()}-{recent['avg_speed'].max()}")
else:
    print("\nNeed at least 30 predictions for drift analysis")
