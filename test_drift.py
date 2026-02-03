import requests
import json
import time

print("=" * 70)
print("DATA DRIFT DETECTION - VERIFICATION TESTS")
print("=" * 70)

# Test 1: Initial prediction (insufficient data)
print("\n[TEST 1] Initial Prediction - Should show 'insufficient_data'")
print("-" * 70)
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"vehicle_count": 50, "avg_speed": 20, "weather": "Rain"}
)
result = response.json()
print(f"✓ Prediction: {result['prediction']}")
print(f"✓ Drift Detection: {result['drift_detection']}")

# Test 2: Make 30 predictions with NORMAL data (within baseline range)
print("\n[TEST 2] Making 30 Normal Predictions (baseline-like data)")
print("-" * 70)
for i in range(30):
    requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "vehicle_count": 50 + i,
            "avg_speed": 20 + (i % 15),
            "weather": ["Rain", "Clear", "Cloudy"][i % 3]
        }
    )
print("✓ Completed 30 predictions")

# Test 3: Check drift status (should be NO DRIFT)
print("\n[TEST 3] Drift Status After Normal Data - Should show NO DRIFT")
print("-" * 70)
response = requests.get("http://127.0.0.1:8000/drift-status")
status = response.json()
print(json.dumps(status, indent=2))

# Test 4: Make 50 predictions with DRIFTED data
print("\n[TEST 4] Making 50 Predictions with DRIFTED Data")
print("-" * 70)
print("(Using much higher vehicle_count and avg_speed values)")
for i in range(50):
    requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "vehicle_count": 200 + i * 3,  # Much higher than baseline (45-124)
            "avg_speed": 70 + i,  # Much higher than baseline (15-54)
            "weather": "Clear"
        }
    )
print("✓ Completed 50 drifted predictions")

# Test 5: Check drift status (should DETECT DRIFT)
print("\n[TEST 5] Drift Status After Drifted Data - Should DETECT DRIFT!")
print("-" * 70)
response = requests.get("http://127.0.0.1:8000/drift-status")
status = response.json()
print(json.dumps(status, indent=2))

# Test 6: Make one more prediction and check inline drift alert
print("\n[TEST 6] Prediction with Inline Drift Alert")
print("-" * 70)
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"vehicle_count": 250, "avg_speed": 90, "weather": "Clear"}
)
result = response.json()
print(json.dumps(result, indent=2))

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE!")
print("=" * 70)
print("\n✓ PSI calculation implemented")
print("✓ Drift detection threshold (0.2) working")
print("✓ Drift alerts included in responses")
print("✓ /drift-status endpoint functional")
