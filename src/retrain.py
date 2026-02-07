"""
RETRAIN SCRIPT
Run this once a month to refresh models with latest data.
Do NOT run more often — that causes overfitting.

Usage: python src/retrain.py
"""

import subprocess
import sys
from datetime import datetime

print("=" * 50)
print(f"MODEL RETRAIN: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 50)

# Step 1: Download fresh data
print("\n--- Step 1: Downloading latest data ---")
result = subprocess.run([sys.executable, 'src/download_data.py'],
                        capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"ERROR: {result.stderr}")
    sys.exit(1)

# Step 2: Rebuild features
print("\n--- Step 2: Building features ---")
result = subprocess.run([sys.executable, 'src/build_features.py'],
                        capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"ERROR: {result.stderr}")
    sys.exit(1)

# Step 3: Retrain models
print("\n--- Step 3: Training models ---")
result = subprocess.run([sys.executable, 'src/train_model.py'],
                        capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"ERROR: {result.stderr}")
    sys.exit(1)

print("\n" + "=" * 50)
print("RETRAIN COMPLETE")
print("Models updated with latest market data.")
print("=" * 50)
