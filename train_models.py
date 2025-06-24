import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Load data
df = pd.read_csv("data/simagro_crop_simulation.csv")

# --- Health Classification ---

# Create health category labels
def classify_health(score):
    if score >= 80:
        return 'Good'
    elif score >= 50:
        return 'Moderate'
    else:
        return 'Poor'

df['Health_Label'] = df['Health Score'].apply(classify_health)

# Features for classifier
X_health = df[['Temperature(C)', 'Soil Moisture(%)', 'Pest Level', 'Fertilizer(g)']]
y_health = df['Health_Label']

# Split data
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_health, y_health, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(Xh_train, yh_train)

# Evaluate
y_pred_health = clf.predict(Xh_test)
acc = accuracy_score(yh_test, y_pred_health)
print(f"✅ Health Classifier Accuracy: {acc * 100:.2f}%")

# Save model
joblib.dump(clf, "health_classifier.pkl")


# --- Yield Regression ---

# Features for regressor
X_yield = df[['Temperature(C)', 'Soil Moisture(%)', 'Pest Level', 'Fertilizer(g)', 'Crop Height(cm)', 'Health Score']]
y_yield = df['Yield Estimate']

# Split data
Xy_train, Xy_test, yy_train, yy_test = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)

# Train regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(Xy_train, yy_train)

# Evaluate
y_pred_yield = reg.predict(Xy_test)
rmse = np.sqrt(mean_squared_error(yy_test, y_pred_yield))
print(f"✅ Yield Regressor RMSE: {rmse:.2f} kg/hectare")

# Save model
joblib.dump(reg, "yield_regressor.pkl")
