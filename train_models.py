import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# --- Load Data ---
main_df = pd.read_excel("data/main_features.xlsx")
dyn_df = pd.read_excel("data/dynamic_features.xlsx")

# --- Clean Column Names ---
main_df.columns = main_df.columns.str.strip()
dyn_df.columns = dyn_df.columns.str.strip()

# --- Combine Horizontally (Row-wise) ---
df = pd.concat([dyn_df.reset_index(drop=True), main_df.reset_index(drop=True)], axis=1)

# --- Drop Unnecessary or Duplicate Columns ---
df = df.loc[:, ~df.columns.duplicated()]

# --- Drop rows with missing target labels ---
df = df.dropna(subset=["Health Score", "Yield_tons_per_hectare"])

# --- Label Encode Categorical Columns ---
categorical_cols = ['Crop', 'Region', 'Soil_Type', 'Irrigation_Used', 'Weather_Condition']
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# --- Feature List (30 you shared) ---
features = [
    'NDVI', 'SAVI', 'Soil_Moisture', 'Humidity', 'Canopy_Coverage', 'Chlorophyll_Content',
    'Leaf_Area_Index', 'Pest_Hotspots', 'Temperature(C)', 'Rainfall(mm)', 'Fertilizer(g)',
    'Crop Height(cm)', 'Pest Level', 'Sunlight(hrs)', 'DayInSeason', 'Crop', 'Irrigation_Used',
    'Region', 'Soil_Type', 'Days_to_Harvest', 'Rainfall_mm', 'Temperature_Celsius',
    'Fertilizer_Used', 'Weather_Condition', 'Wind_Speed', 'Organic_Matter',
    'Crop_Stress_Indicator', 'Weed_Coverage', 'Soil_pH'
]

df = df.dropna(subset=features)

X = df[features]

# --- HEALTH CLASSIFICATION ---
def classify_health(score):
    if score < 40:
        return 0  # Poor
    elif score < 70:
        return 1  # Average
    else:
        return 2  # Good

y_health = df["Health Score"].apply(classify_health)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_health, test_size=0.2, random_state=42)

health_model = RandomForestClassifier(n_estimators=100, random_state=42)
health_model.fit(X_train_h, y_train_h)

health_acc = accuracy_score(y_test_h, health_model.predict(X_test_h))
print(f"✅ Health Model Accuracy: {health_acc:.2f}")

joblib.dump(health_model, "health_classifier.pkl")

# --- YIELD REGRESSION ---
y_yield = df["Yield_tons_per_hectare"]

X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X, y_yield, test_size=0.2, random_state=42)

yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_model.fit(X_train_y, y_train_y)

yield_rmse = mean_squared_error(y_test_y, yield_model.predict(X_test_y), squared=False)
print(f"✅ Yield Model RMSE: {yield_rmse:.2f}")

joblib.dump(yield_model, "yield_regressor.pkl")
