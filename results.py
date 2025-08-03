import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    mean_squared_error, r2_score, precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize
import joblib
import numpy as np
import os

# ================================
# CONFIGURATION
# ================================
DYNAMIC_FILE = "dynamic_features.xlsx"
MAIN_FILE = "main_features.xlsx"
HEALTH_MODEL = "health_classifier.pkl"
YIELD_MODEL = "yield_regressor.pkl"
OUTPUT_DIR = "output_plots_final"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Load Data
# ================================
main_df = pd.read_excel(MAIN_FILE)
dyn_df = pd.read_excel(DYNAMIC_FILE)

main_df.columns = main_df.columns.str.strip()
dyn_df.columns = dyn_df.columns.str.strip()

df = pd.concat([dyn_df.reset_index(drop=True), main_df.reset_index(drop=True)], axis=1)
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(subset=["Health Score", "Yield_tons_per_hectare"])

# Encode categorical features
cat_cols = ['Crop', 'Region', 'Soil_Type', 'Irrigation_Used', 'Weather_Condition']
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Feature list
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

# ================================
# 🚨 Health Classification Model
# ================================
print("\n🔍 Evaluating Health Model...")
y_health = df["Health Score"].apply(lambda x: 0 if x < 40 else 1 if x < 70 else 2)
health_model = joblib.load(HEALTH_MODEL)
y_pred_health = health_model.predict(X)

# Unique labels
labels = sorted(y_health.unique())
label_map = {0: "Poor", 1: "Average", 2: "Good"}
target_names = [label_map.get(i, str(i)) for i in labels]

# Confusion Matrix
cm = confusion_matrix(y_health, y_pred_health, labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Health Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "health_confusion_matrix.png"))
plt.close()

# Classification Report
report = classification_report(y_health, y_pred_health, labels=labels, target_names=target_names)
with open(os.path.join(OUTPUT_DIR, "health_classification_report.txt"), 'w') as f:
    f.write(report)

# Accuracy
acc = accuracy_score(y_health, y_pred_health)
with open(os.path.join(OUTPUT_DIR, "health_accuracy.txt"), 'w') as f:
    f.write(f"Health Model Accuracy: {acc:.4f}")

# Precision-Recall Curve
if len(labels) > 1:
    y_bin = label_binarize(y_health, classes=labels)
    plt.figure(figsize=(7,5))
    for i, name in enumerate(target_names):
        if y_bin.shape[1] > i:
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_health == labels[i])
            plt.plot(recall, precision, label=name)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "health_precision_recall_curve.png"))
    plt.close()
else:
    print("⚠️ Skipping Precision-Recall Curve (only one class present).")

# ROC Curve
if len(labels) > 1:
    y_bin = label_binarize(y_health, classes=labels)
    plt.figure(figsize=(7,5))
    for i, name in enumerate(target_names):
        if y_bin.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_health == labels[i])
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.title("ROC Curve - Health Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "health_roc_curve.png"))
    plt.close()
else:
    print("⚠️ Skipping ROC Curve (only one class present).")

# Feature Importance
importances = health_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8,6))
plt.barh(np.array(features)[indices][:15], importances[indices][:15], color='skyblue')
plt.title("Top 15 Feature Importances (Health Model)")
plt.gca().invert_yaxis()
plt.savefig(os.path.join(OUTPUT_DIR, "health_feature_importance.png"))
plt.close()

print("✅ Health model evaluation complete.")

# ================================
# 🌾 Yield Regression Model
# ================================
print("\n🔍 Evaluating Yield Model...")
y_yield = df["Yield_tons_per_hectare"]
yield_model = joblib.load(YIELD_MODEL)
y_pred_yield = yield_model.predict(X)

# Metrics
r2 = r2_score(y_yield, y_pred_yield)
rmse = mean_squared_error(y_yield, y_pred_yield) ** 0.5
with open(os.path.join(OUTPUT_DIR, "yield_metrics.txt"), 'w') as f:
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"RMSE: {rmse:.4f} tons/hectare")

# Scatter plot
plt.figure(figsize=(7,5))
plt.scatter(y_yield, y_pred_yield, alpha=0.5, color='blue')
plt.plot([y_yield.min(), y_yield.max()], [y_yield.min(), y_yield.max()], 'r--')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Prediction vs Actual (Yield)")
plt.savefig(os.path.join(OUTPUT_DIR, "yield_scatter_plot.png"))
plt.close()

# Line plot
plt.figure(figsize=(8,5))
plt.plot(y_yield.values[:200], label="Actual", color="green")
plt.plot(y_pred_yield[:200], label="Predicted", color="orange")
plt.title("Yield Prediction (First 200 Samples)")
plt.xlabel("Sample")
plt.ylabel("Yield (tons/hectare)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yield_actual_vs_predicted.png"))
plt.close()

# Residual distribution
residuals = y_yield - y_pred_yield
plt.figure(figsize=(7, 4))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title("Yield Residual Distribution")
plt.xlabel("Error (tons/hectare)")
plt.savefig(os.path.join(OUTPUT_DIR, "yield_residuals.png"))
plt.close()

# Residuals vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_pred_yield, residuals, alpha=0.5, color='red')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted Yield")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.savefig(os.path.join(OUTPUT_DIR, "yield_residual_vs_predicted.png"))
plt.close()

# Feature Importance
importances = yield_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8,6))
plt.barh(np.array(features)[indices][:15], importances[indices][:15], color='orange')
plt.title("Top 15 Feature Importances (Yield Model)")
plt.gca().invert_yaxis()
plt.savefig(os.path.join(OUTPUT_DIR, "yield_feature_importance.png"))
plt.close()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[features + ["Yield_tons_per_hectare"]].corr(), cmap='coolwarm')
plt.title("Correlation Heatmap of Features")
plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation_heatmap.png"))
plt.close()

# Boxplot: Yield by Crop
if 'Crop' in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df['Crop'], y=y_yield)
    plt.title("Yield Distribution by Crop")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(OUTPUT_DIR, "yield_boxplot_by_crop.png"))
    plt.close()

print("✅ Yield model evaluation complete.")
print(f"\n📂 All results (graphs & metrics) saved in '{OUTPUT_DIR}' folder.")
