# =====================================================
# PRODROMAL PHASE DETECTION USING ISOLATION FOREST
# Unsupervised Anomaly Detection
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

np.random.seed(42)

# ------------------------------------------
# 1. Generate Longitudinal Behavioral Data
# ------------------------------------------
days = 200

sleep = np.random.normal(7, 0.5, days)
mood = np.random.normal(7, 1, days)
steps = np.random.normal(8000, 1200, days)
fatigue = np.random.normal(3, 0.7, days)

# Simulate prodromal drift
for i in range(130, 200):
    sleep[i] -= np.random.uniform(0.5, 1.5)
    mood[i] -= np.random.uniform(0.5, 2)
    steps[i] -= np.random.uniform(1500, 3000)
    fatigue[i] += np.random.uniform(1, 2)

# ------------------------------------------
# 2. Create Dataset
# ------------------------------------------
data = pd.DataFrame({
    'sleep': sleep,
    'mood': mood,
    'steps': steps,
    'fatigue': fatigue
})

features = data[['sleep','mood','steps','fatigue']]

# ------------------------------------------
# 3. Train Isolation Forest Model
# ------------------------------------------
model = IsolationForest(contamination=0.15, random_state=42)

data['anomaly'] = model.fit_predict(features)
data['anomaly_score'] = model.decision_function(features)

# ------------------------------------------
# 4. Identify Risk Window
# ------------------------------------------
risk_days = data.index[data['anomaly'] == -1]

print("\nDetected Potential Risk Days:")
print(risk_days.tolist())

if len(risk_days) > 20:
    print("\n⚠ Sustained abnormal behavior detected — possible prodromal phase window.")
else:
    print("\nNo sustained abnormal pattern detected.")

# ------------------------------------------
# 5. Visualization
# ------------------------------------------
plt.figure(figsize=(12,5))

plt.plot(data['sleep'], label='Sleep Hours')

plt.scatter(
    data.index[data['anomaly'] == -1],
    data['sleep'][data['anomaly'] == -1],
    color='red',
    label='Detected Anomaly',
    s=60
)

plt.axvline(x=130, color='orange', linestyle='--', label='Actual Drift Start')

plt.title("Prodromal Phase Detection (Isolation Forest)")
plt.xlabel("Days")
plt.ylabel("Sleep Hours")
plt.legend()
plt.show()
