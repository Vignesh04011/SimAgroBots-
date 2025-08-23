import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import joblib

# ======================
# App Config
# ======================
st.set_page_config(page_title="🌿 SimAgro AI Dashboard", page_icon="🌾", layout="wide")
st.title("🌿 SimAgro: Smart Crop Health, Yield & Disease Predictor")
st.markdown("Use sliders or upload a leaf image to get AI-powered predictions and cobot recommendations.")

# ======================
# Load Models & Data
# ======================
df = pd.read_csv("data/simagro_crop_simulation.csv")

health_model = joblib.load("health_classifier.pkl")
yield_model = joblib.load("yield_regressor.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
disease_model = models.resnet18(weights=None)
num_ftrs = disease_model.fc.in_features
disease_model.fc = nn.Linear(num_ftrs, 38)
disease_model.load_state_dict(torch.load("best_model.pth", map_location=device))
disease_model = disease_model.to(device)
disease_model.eval()

# ======================
# Load Disease Info
# ======================
from disease_info import disease_info

# ======================
# Encoding Mappings
# ======================
crop_mapping = {0: "Wheat", 1: "Rice", 2: "Maize"}
region_mapping = {0: "North", 1: "South", 2: "East"}
soil_mapping = {0: "Sandy", 1: "Clay", 2: "Loamy"}
irrigation_mapping = {0: "No Irrigation", 1: "Irrigation Used"}
weather_mapping = {0: "Sunny", 1: "Cloudy", 2: "Rainy"}

# ======================
# Sidebar Inputs (Top 15)
# ======================
with st.sidebar:
    st.header("🧪 Primary Inputs (15)")
    NDVI = st.slider("NDVI", 0.0, 1.0, 0.6)
    SAVI = st.slider("SAVI", 0.0, 1.0, 0.5)
    Soil_Moisture = st.slider("💧 Soil Moisture (%)", 0.0, 100.0, 43.0)
    Humidity = st.slider("Humidity (%)", 0.0, 100.0, 55.0)
    Canopy_Coverage = st.slider("Canopy Coverage (%)", 0.0, 100.0, 60.0)
    Chlorophyll_Content = st.slider("Chlorophyll Content", 0.0, 100.0, 35.0)
    Leaf_Area_Index = st.slider("Leaf Area Index", 0.0, 10.0, 2.5)
    Pest_Hotspots = st.slider("Pest Hotspots", 0, 10, 1)
    Temperature_C = st.slider("🌡️ Temperature (°C)", 10.0, 45.0, 28.0)
    Rainfall_mm = st.slider("🌧️ Rainfall (mm)", 0.0, 300.0, 80.0)
    Fertilizer_g = st.slider("🌱 Fertilizer (g)", 0.0, 500.0, 120.0)
    Crop_Height_cm = st.slider("📏 Crop Height (cm)", 0.0, 200.0, 60.0)
    Pest_Level = st.slider("🌿 Pest Level (0–10)", 0.0, 10.0, 3.0)
    Sunlight_hrs = st.slider("☀️ Sunlight (hrs)", 0.0, 15.0, 8.0)
    DayInSeason = st.slider("Day in Season", 0, 150, 45)

# ======================
# Advanced Inputs
# ======================
with st.expander("🔧 Advanced Feature Settings (Optional)"):
    Crop_label = st.selectbox("🌾 Crop Type", list(crop_mapping.values()))
    Crop = [k for k, v in crop_mapping.items() if v == Crop_label][0]

    Irrigation_label = st.selectbox("🚿 Irrigation Used", list(irrigation_mapping.values()))
    Irrigation_Used = [k for k, v in irrigation_mapping.items() if v == Irrigation_label][0]

    Region_label = st.selectbox("🌍 Region", list(region_mapping.values()))
    Region = [k for k, v in region_mapping.items() if v == Region_label][0]

    Soil_label = st.selectbox("Soil Type", list(soil_mapping.values()))
    Soil_Type = [k for k, v in soil_mapping.items() if v == Soil_label][0]

    Days_to_Harvest = st.slider("📆 Days to Harvest", 10, 200, 80)
    Rainfall_mm_2 = st.slider("Rainfall Duplicate (mm)", 0.0, 300.0, Rainfall_mm)
    Temperature_C_2 = st.slider("Temperature Duplicate (°C)", 10.0, 45.0, Temperature_C)
    Fertilizer_Used = st.slider("Fertilizer Used", 0.0, 100.0, 50.0)

    Weather_label = st.selectbox("🌦️ Weather Condition", list(weather_mapping.values()))
    Weather_Condition = [k for k, v in weather_mapping.items() if v == Weather_label][0]

    Wind_Speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 5.0)
    Organic_Matter = st.slider("Organic Matter (%)", 0.0, 10.0, 3.2)
    Crop_Stress_Indicator = st.slider("Crop Stress Indicator", 0.0, 1.0, 0.2)
    Weed_Coverage = st.slider("Weed Coverage (%)", 0.0, 100.0, 10.0)
    Soil_pH = st.slider("🧪 Soil pH", 3.0, 9.0, 6.5)

# ======================
# Prepare Input Features
# ======================
input_features = np.array([[NDVI, SAVI, Soil_Moisture, Humidity, Canopy_Coverage, Chlorophyll_Content,
    Leaf_Area_Index, Pest_Hotspots, Temperature_C, Rainfall_mm, Fertilizer_g,
    Crop_Height_cm, Pest_Level, Sunlight_hrs, DayInSeason, Crop, Irrigation_Used,
    Region, Soil_Type, Days_to_Harvest, Rainfall_mm_2, Temperature_C_2,
    Fertilizer_Used, Weather_Condition, Wind_Speed, Organic_Matter,
    Crop_Stress_Indicator, Weed_Coverage, Soil_pH]])

# ======================
# Advisor Function
# ======================
def generate_cobot_recommendations(temp, moisture, pest, fertilizer, height):
    suggestions = []
    if moisture < 35: suggestions.append("💧 Moisture is low. Activate irrigation for 30–45 minutes.")
    elif moisture > 80: suggestions.append("⚠️ Soil moisture is high. Reduce watering.")
    if pest > 7: suggestions.append("🦠 High pest level detected. Apply pesticide immediately.")
    elif pest < 2: suggestions.append("✅ Pest level is low. No action needed.")
    if fertilizer < 80: suggestions.append("🌿 Fertilizer level is low. Consider applying more nutrients.")
    elif fertilizer > 300: suggestions.append("⚠️ Excess fertilizer detected. Reduce usage.")
    if height < 20: suggestions.append("📏 Crop growth is below average. Adjust nutrition and monitor.")
    elif height > 100: suggestions.append("🌾 Crop growth is optimal. Maintain current strategy.")
    if not suggestions: suggestions.append("✅ All parameters look optimal. Continue regular operations.")
    return suggestions

# ======================
# Predict Button
# ======================
if st.button("🔍 Predict"):

    # Health & Yield
    predicted_health = health_model.predict(input_features)[0]
    predicted_yield = yield_model.predict(input_features)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Predicted Health", {0: "🔴 Poor", 1: "🟡 Average", 2: "🟢 Good"}.get(predicted_health, "❓ Unknown"))
    with col2:
        st.metric("🌾 Estimated Yield", f"{predicted_yield:.2f} tons/hectare")

    # Cobot Advisor
    st.markdown("### 🤖 Virtual Cobot Advisor")
    recommendations = generate_cobot_recommendations(Temperature_C, Soil_Moisture, Pest_Level, Fertilizer_g, Crop_Height_cm)
    with st.expander("View Recommendations"):
        for rec in recommendations:
            st.write(f"- {rec}")

# ======================
# Leaf Disease Detection
# ======================
st.markdown("---")
st.subheader("🌱 Leaf Disease Detection")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = list(disease_info.keys())

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", width=250)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = disease_model(img_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]
        confidence = torch.softmax(outputs, dim=1)[0][preds.item()].item() * 100

    st.subheader("🔍 Disease Prediction")
    st.success(f"**{predicted_class}** detected with **{confidence:.2f}% confidence**.")

    st.subheader("📖 Disease Insights")
    if "healthy" in predicted_class.lower():
        st.info("✅ The plant is healthy. No immediate action required.")
    else:
        info = disease_info.get(predicted_class)
        if info:
            st.warning(f"⚠️ Disease detected: **{predicted_class}**")
            st.write(f"**Possible Cause:** {info['cause']}")
            st.write(f"**Treatment Suggestion:** {info['treatment']}")
        else:
            st.write("ℹ️ Basic info not available for this disease.")

# ======================
# Charts & Raw Data
# ======================
st.markdown("---")
st.subheader("📈 Simulation Trends")
chart_option = st.selectbox("Choose variable to visualize:", df.columns[1:])
st.line_chart(df.set_index("Day")[chart_option])

with st.expander("📊 Show Raw Data"):
    st.dataframe(df.head(10))
