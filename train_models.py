import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# --- Load Excel Files ---
main_df = pd.read_excel("data/main_features.xlsx")
dynamic_df = pd.read_excel("data/dynamic_features.xlsx")

# --- Merge Files ---
# If both files share the same row order or a key like 'ID', use that
df = pd.concat([main_df, dynamic_df], axis=1)

# --- Drop Rows with Missing Labels ---
df = df.dropna(subset=["Health Score", "Yield_tons_per_hectare"])

# --- Final 30 Features ---
selected_features = [
    # Manual
    "Region", "Crop", "Soil_Type", "Soil_pH", "Organic_Matter", "Fertilizer_Used", "Irrigation_Used",
    "Days_to_Harvest", "Weather_Condition", "Weed_Coverage", "Temperature(C)", "Rainfall(mm)",
    "Pest Level", "Crop Height(cm)", "DayInSeason",
    
    # Sensor
    "NDVI", "SAVI", "Chlorophyll_Content", "Leaf_Area_Index", "Canopy_Coverage",
    "Soil Moisture(%)", "Wind_Speed", "Sunlight(hrs)", "Humidity", "Elevation_Data",
    "Crop_Stress_Indicator", "Crop_Growth_Stage", "Rainfall_mm", "Temperature_Celsius",
    
    # Extra for regression
    "Health Score"
]

# --- Prepare Data ---
df = df.dropna(subset=selected_features)

X = df[selected_features]
X = pd.get_dummies(X)  # Convert categorical to numeric

# --- HEALTH MODEL ---
y_health = df["Health Score"].astype(int)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_health, test_size=0.2, random_state=42)

health_model = RandomForestClassifier(n_estimators=150, random_state=42)
health_model.fit(X_train_h, y_train_h)
y_pred_h = health_model.predict(X_test_h)

print(f"✅ Health Model Accuracy: {accuracy_score(y_test_h, y_pred_h):.2f}")
joblib.dump(health_model, "health_classifier.pkl")

# --- YIELD MODEL ---
y_yield = df["Yield_tons_per_hectare"]
X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X, y_yield, test_size=0.2, random_state=42)

yield_model = RandomForestRegressor(n_estimators=150, random_state=42)
yield_model.fit(X_train_y, y_train_y)
y_pred_y = yield_model.predict(X_test_y)

rmse = mean_squared_error(y_test_y, y_pred_y, squared=False)
print(f"✅ Yield Model RMSE: {rmse:.2f} tons/hectare")
joblib.dump(yield_model, "yield_regressor.pkl")
