import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page Setup
st.set_page_config(page_title="SimAgro AI Dashboard", page_icon="🌾", layout="centered")

st.title("🌿 SimAgro: Smart Crop Health & Yield Predictor")
st.markdown("Use the sliders below to simulate inputs and get real-time predictions based on trained AI models.")

# Load Data and Models
df = pd.read_csv("data/simagro_crop_simulation.csv")
health_model = joblib.load("health_classifier"".pkl")
yield_model = joblib.load("yield_regressor.pkl")

# --- Advisor Function ---
def generate_cobot_recommendations(temp, moisture, pest, fertilizer, height, health_score):
    suggestions = []

    if moisture < 35:
        suggestions.append("💧 Moisture is low. Activate irrigation for 30–45 minutes.")
    elif moisture > 80:
        suggestions.append("⚠️ Soil moisture is high. Reduce watering.")

    if pest > 0.7:
        suggestions.append("🦠 High pest level detected. Apply pesticide immediately.")
    elif pest < 0.2:
        suggestions.append("✅ Pest level is low. No action needed.")

    if fertilizer < 3:
        suggestions.append("🌿 Fertilizer level is low. Consider applying 4–6g/day.")
    elif fertilizer > 7:
        suggestions.append("⚠️ Excess fertilizer detected. Reduce usage.")

    if height < 20:
        suggestions.append("📏 Crop growth is below average. Adjust nutrition and monitor.")
    elif height > 80:
        suggestions.append("🌾 Crop growth is optimal. Maintain current strategy.")

    if health_score < 30:
        suggestions.append("🛑 Overall crop health is poor. Conduct field inspection.")

    if not suggestions:
        suggestions.append("✅ All parameters look optimal. Continue regular operations.")

    return suggestions

# --- Input Widgets ---
with st.sidebar:
    st.subheader("🛠️ Simulation Inputs")
    temp = st.slider("🌡️ Temperature (°C)", 10.0, 40.0, 28.0)
    moisture = st.slider("💧 Soil Moisture (%)", 0.0, 100.0, 43.0)
    pest = st.slider("🌿 Pest Level (0 to 1)", 0.0, 1.0, 0.3)
    fertilizer = st.slider("🌱 Fertilizer Used (g/day)", 0.0, 10.0, 4.0)
    height = st.slider("📏 Crop Height (cm)", 0.0, 120.0, 60.0)
    health_score = st.slider("🧬 Health Score (0–100)", 0, 100, 75)

# --- Predict Button ---
if st.button("🔍 Predict"):
    health_input = np.array([[temp, moisture, pest, fertilizer]])
    predicted_health = health_model.predict(health_input)[0]

    yield_input = np.array([[temp, moisture, pest, fertilizer, height, health_score]])
    predicted_yield = yield_model.predict(yield_input)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Predicted Health", predicted_health)
    with col2:
        st.metric("🌾 Estimated Yield", f"{predicted_yield:.2f} kg/hectare")

    st.markdown("---")

    # --- Cobot Advisor Output ---
    st.markdown("### 🤖 Virtual Cobot Advisor")
    recommendations = generate_cobot_recommendations(temp, moisture, pest, fertilizer, height, health_score)
    with st.expander("View Recommendations"):
        for rec in recommendations:
            st.write(f"- {rec}")

# --- Data Charts Section ---
st.subheader("📈 Simulation Trends from Last Run")
chart_option = st.selectbox(
    "Select variable to visualize:",
    ["Crop Height(cm)", "Health Score", "Yield Estimate", "Temperature(C)", "Soil Moisture(%)", "Pest Level"]
)

st.line_chart(df.set_index("Day")[chart_option])

# --- Optional: Show raw data
with st.expander("📊 Show Raw Data"):
    st.dataframe(df.head(10))
