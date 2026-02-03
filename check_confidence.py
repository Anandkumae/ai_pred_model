"""
Test confidence monitoring with varying confidence levels
"""
import pandas as pd
import numpy as np

# Load recent logs
df = pd.read_csv("logs/prediction_logs.csv")
print(f"Total predictions: {len(df)}")
print(f"\nConfidence statistics:")
print(f"  Mean: {df['confidence'].mean():.4f}")
print(f"  Min: {df['confidence'].min():.4f}")
print(f"  Max: {df['confidence'].max():.4f}")
print(f"  Std: {df['confidence'].std():.4f}")

# Check last 100 predictions
recent = df.tail(100)
avg_conf = recent['confidence'].mean()

print(f"\nLast 100 predictions:")
print(f"  Average confidence: {avg_conf:.4f}")

# Determine risk level
if avg_conf < 0.6:
    risk = "HIGH"
elif avg_conf < 0.75:
    risk = "MEDIUM"
else:
    risk = "LOW"

print(f"  Risk level: {risk}")

# Show confidence distribution
print(f"\nConfidence distribution (last 100):")
print(f"  < 0.6 (HIGH RISK): {(recent['confidence'] < 0.6).sum()} predictions")
print(f"  0.6-0.75 (MEDIUM): {((recent['confidence'] >= 0.6) & (recent['confidence'] < 0.75)).sum()} predictions")
print(f"  >= 0.75 (LOW RISK): {(recent['confidence'] >= 0.75).sum()} predictions")
