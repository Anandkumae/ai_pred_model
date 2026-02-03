"""
Manual Labeling Tool for Failure Dataset

This script helps manually label time windows as healthy (0) or failure (1)
based on the aggregated metrics.
"""

import pandas as pd
import os

DATASET_FILE = "failure_dataset.csv"
LABELED_FILE = "failure_dataset_labeled.csv"

def label_failures():
    """Interactive labeling tool for failure dataset"""
    
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Error: {DATASET_FILE} not found!")
        print("Run generate_failure_dataset.py first.")
        return
    
    # Load dataset
    df = pd.read_csv(DATASET_FILE)
    
    print("=" * 70)
    print("FAILURE DATASET LABELING TOOL")
    print("=" * 70)
    print(f"\nTotal windows: {len(df)}")
    print(f"Unlabeled: {(df['label'] == -1).sum()}")
    print(f"Healthy (0): {(df['label'] == 0).sum()}")
    print(f"Failure (1): {(df['label'] == 1).sum()}")
    
    print("\n" + "=" * 70)
    print("LABELING GUIDE")
    print("=" * 70)
    print("Consider a window as FAILURE (1) if:")
    print("  • PSI score > 0.2 (significant data drift)")
    print("  • avg_confidence < 0.6 (low model certainty)")
    print("  • concept_drift_count > 0 (relationship changed)")
    print("  • error_trend > 0.01 (errors increasing)")
    print("\nOtherwise, label as HEALTHY (0)")
    print("=" * 70)
    
    # Auto-label based on heuristics
    print("\n🤖 Auto-labeling based on heuristics...")
    
    for idx, row in df.iterrows():
        if row['label'] != -1:
            continue  # Skip already labeled
        
        # Heuristic rules
        is_failure = False
        
        if row['psi_score'] > 0.2:
            is_failure = True
        elif row['avg_confidence'] < 0.6:
            is_failure = True
        elif row['concept_drift_count'] > 0:
            is_failure = True
        elif row['error_trend'] > 0.01:
            is_failure = True
        
        df.at[idx, 'label'] = 1 if is_failure else 0
    
    print(f"✓ Auto-labeled {len(df)} windows")
    
    # Show summary
    print("\n" + "=" * 70)
    print("LABELING SUMMARY")
    print("=" * 70)
    print(f"Total windows: {len(df)}")
    print(f"Healthy (0): {(df['label'] == 0).sum()}")
    print(f"Failure (1): {(df['label'] == 1).sum()}")
    
    # Show some examples
    print("\n" + "=" * 70)
    print("HEALTHY WINDOWS (label=0)")
    print("=" * 70)
    healthy = df[df['label'] == 0].head(3)
    if len(healthy) > 0:
        print(healthy[['window_start', 'psi_score', 'avg_confidence', 
                      'concept_drift_count', 'error_trend', 'label']].to_string())
    
    print("\n" + "=" * 70)
    print("FAILURE WINDOWS (label=1)")
    print("=" * 70)
    failure = df[df['label'] == 1].head(3)
    if len(failure) > 0:
        print(failure[['window_start', 'psi_score', 'avg_confidence', 
                      'concept_drift_count', 'error_trend', 'label']].to_string())
    
    # Save labeled dataset
    df.to_csv(LABELED_FILE, index=False)
    print(f"\n✓ Saved labeled dataset to {LABELED_FILE}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review the labeled dataset")
    print("2. Train a classifier (Random Forest, XGBoost, etc.)")
    print("3. Use the model to predict future failures")
    print("=" * 70)

if __name__ == "__main__":
    label_failures()
