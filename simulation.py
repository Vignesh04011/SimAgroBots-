import random
import pandas as pd
import numpy as np

# Constants
DAYS = 60
MAX_HEIGHT = 110  # in cm (average wheat)
FERTILIZER_WEEKLY = 30  # grams/week applied

# Helper function: Simulate environmental values
def simulate_day(day):
    stage = get_growth_stage(day)
    rainfall = random.uniform(0, 20)
    temperature = random.uniform(15, 35)
    sunlight = random.uniform(5, 10)
    pest_level = np.clip(np.random.normal(0.2, 0.1), 0, 1)
    soil_moisture = max(0, min(100, rainfall * 3 + random.uniform(-5, 5)))
    fertilizer = FERTILIZER_WEEKLY / 7  # daily
    return stage, rainfall, temperature, sunlight, pest_level, soil_moisture, fertilizer

# Growth stage logic
def get_growth_stage(day):
    if day <= 10:
        return 'Germination'
    elif day <= 30:
        return 'Vegetative'
    elif day <= 50:
        return 'Flowering'
    else:
        return 'Maturity'

# Crop height progression based on growth curve and conditions
def update_crop_height(stage, prev_height, temp, light, moisture, fert, pest):
    growth_factor = {
        'Germination': 0.2,
        'Vegetative': 0.8,
        'Flowering': 1.2,
        'Maturity': 0.3
    }[stage]
    
    # Influence factors
    env_score = 1.0
    if not (18 <= temp <= 32):
        env_score -= 0.2
    if not (60 <= moisture <= 85):
        env_score -= 0.2
    if light < 6:
        env_score -= 0.1
    if pest > 0.4:
        env_score -= 0.2

    growth = growth_factor * env_score * (1 - pest) + fert * 0.01
    new_height = min(MAX_HEIGHT, prev_height + growth)
    return round(new_height, 2)

# Health score estimation
def estimate_health(temp, moisture, pest, fert):
    score = 100
    if not (18 <= temp <= 32): score -= 15
    if not (60 <= moisture <= 85): score -= 15
    if pest > 0.4: score -= 20
    if fert < 3: score -= 10
    return max(0, round(score))

# Simulation loop
data = []
height = 0

for day in range(1, 61):
    stage, rain, temp, light, pest, moisture, fert = simulate_day(day)
    height = update_crop_height(stage, height, temp, light, moisture, fert, pest)
    health = estimate_health(temp, moisture, pest, fert)
    yield_est = (height / MAX_HEIGHT) * (health / 100) * 4000  # kg/hectare

    data.append({
        'Day': day,
        'Stage': stage,
        'Rainfall(mm)': round(rain, 1),
        'Temperature(C)': round(temp, 1),
        'Sunlight(hrs)': round(light, 1),
        'Soil Moisture(%)': round(moisture, 1),
        'Pest Level': round(pest, 2),
        'Fertilizer(g)': round(fert, 2),
        'Crop Height(cm)': height,
        'Health Score': health,
        'Yield Estimate': round(yield_est, 2)
    })

# Save as CSV
df = pd.DataFrame(data)
df.to_csv("data/simagro_crop_simulation.csv", index=False)
print("✅ Simulation complete. CSV saved at data/simagro_crop_simulation.csv")
