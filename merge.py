import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
main = pd.read_excel("main_features.xlsx")
dynamic = pd.read_excel("dynamic_features.xlsx")

print(f"Main shape: {main.shape}")
print(f"Dynamic shape: {dynamic.shape}")

# Merge datasets (basic merge for now)
merged = pd.concat([main, dynamic], axis=1)
print(f"Merged shape: {merged.shape}")

# Encode categorical variables
label_encoders = {}
for col in merged.select_dtypes(include='object').columns:
    le = LabelEncoder()
    merged[col] = merged[col].astype(str)
    merged[col] = le.fit_transform(merged[col])
    label_encoders[col] = le

# Function to predict missing values
def predict_and_fill(df, target_col):
    if df[target_col].isna().sum() == 0:
        return df  # no missing, skip

    notnull = df[df[target_col].notnull()]
    isnull = df[df[target_col].isnull()]

    if len(notnull) < 100:  # not enough data
        return df

    X = notnull.drop(columns=[target_col])
    y = notnull[target_col]
    X_missing = isnull.drop(columns=[target_col])

    # Drop cols with too many missing values in input
    X = X.dropna(axis=1, thresh=0.8 * len(X))
    X_missing = X_missing[X.columns]

    # Fill remaining missing in features
    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(X)
    X_missing = imp.transform(X_missing)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X_missing)

    df.loc[df[target_col].isna(), target_col] = y_pred
    print(f"✅ Predicted missing values in '{target_col}'")
    return df

# Important columns to predict
important_cols = [
    'Health Score',
    'Pest Level',
    'Crop Height(cm)',
    'Sunlight(hrs)',
    'Rainfall(mm)',
    'Temperature(C)'
]

# Predict and fill important targets
for col in important_cols:
    if col in merged.columns:
        merged = predict_and_fill(merged, col)

# Impute remaining missing values
num_cols = merged.select_dtypes(include=np.number).columns
merged[num_cols] = SimpleImputer(strategy='mean').fit_transform(merged[num_cols])

# Final check
print("✅ Missing values after cleaning:", merged.isna().sum().sum())

# Save final cleaned dataset
merged.to_csv("enhanced_cleaned_data.csv", index=False)
print("✅ Final cleaned data saved as 'enhanced_cleaned_data.csv'")
