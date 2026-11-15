# =====================================================
# PRODROMAL PHASE CLASSIFICATION USING RANDOM FOREST
# Supervised Learning Model
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)

# ------------------------------------------
# 1. Generate Behavioral Data
# ------------------------------------------
days = 200

sleep = np.random.normal(7, 0.5, days)
mood = np.random.normal(7, 1, days)
steps = np.random.normal(8000, 1200, days)
fatigue = np.random.normal(3, 0.7, days)

labels = np.zeros(days)  # 0 = normal

# Simulate abnormal phase
for i in range(130, 200):
    sleep[i] -= np.random.uniform(0.5, 1.5)
    mood[i] -= np.random.uniform(0.5, 2)
    steps[i] -= np.random.uniform(1500, 3000)
    fatigue[i] += np.random.uniform(1, 2)
    labels[i] = 1  # abnormal

# ------------------------------------------
# 2. Create Dataset
# ------------------------------------------
data = pd.DataFrame({
    'sleep': sleep,
    'mood': mood,
    'steps': steps,
    'fatigue': fatigue,
    'label': labels
})

X = data[['sleep','mood','steps','fatigue']]
y = data['label']

# ------------------------------------------
# 3. Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# ------------------------------------------
# 4. Train Random Forest Model
# ------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------
# 5. Evaluate Model
# ------------------------------------------
predictions = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ------------------------------------------
# 6. Predict Full Timeline
# ------------------------------------------
data['predicted'] = model.predict(X)

# ------------------------------------------
# 7. Visualization
# ------------------------------------------
plt.figure(figsize=(12,5))

plt.plot(data['sleep'], label='Sleep Hours')

plt.scatter(
    data.index[data['predicted'] == 1],
    data['sleep'][data['predicted'] == 1],
    color='green',
    label='Predicted Abnormal',
    s=60
)

plt.axvline(x=130, color='orange', linestyle='--', label='Actual Drift Start')

plt.title("Random Forest Classification Detection")
plt.xlabel("Days")
plt.ylabel("Sleep Hours")
plt.legend()
plt.show()
