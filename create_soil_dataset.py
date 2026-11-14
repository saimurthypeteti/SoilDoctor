"""
SoilDoctor - Dataset B: Soil & Nutrition Dataset Generator (Phase 2)
Generates structured CSV dataset: soil_data.csv
Features:
- Crop
- pH
- Nitrogen
- Phosphorus
- Potassium
- Moisture
- Temperature
- Humidity
- Growth_Stage
- Fertilizer_Recommendation
"""

import os
import random
import pandas as pd

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "soil_data.csv")

CROPS = ["Tomato", "Rice", "Maize", "Potato", "Chilli", "Wheat"]
STAGES = ["Vegetative", "Flowering", "Fruiting", "Maturity"]
STATUS_OPTS = ["Low", "Normal", "High"]
MOIST_OPTS = ["Low", "Medium", "High"]

def determine_recommendation(crop, n, p, k, ph, moist, stage):
    if ph < 5.8:
        ph_action = "Apply agricultural lime (calcium carbonate) to buffer acidity. "
    elif ph > 7.5:
        ph_action = "Apply gypsum or elemental sulfur to reduce soil alkalinity. "
    else:
        ph_action = ""

    if n == "Low":
        nut_action = "Side-dress with soluble Urea (46% N) or organic blood meal. "
    elif p == "Low":
        nut_action = "Incorporate Di-Ammonium Phosphate (DAP 18-46-0) or rock phosphate. "
    elif k == "Low":
        nut_action = "Apply Muriate of Potash (MOP 0-0-60) or wood ash to strengthen stalks. "
    elif n == "High" and stage in ["Flowering", "Fruiting"]:
        nut_action = "Withhold nitrogen to prevent excessive vegetative drop and encourage fruit set. "
    else:
        nut_action = "Maintain balanced NPK (19-19-19) fertigation schedule. "

    if moist == "Low":
        water_action = "Increase drip irrigation duration by 25%."
    elif moist == "High":
        water_action = "Improve field drainage to avoid root hypoxia."
    else:
        water_action = "Soil moisture is optimal."

    return (ph_action + nut_action + water_action).strip()

def generate_soil_dataset(num_rows=600):
    rows = []
    for _ in range(num_rows):
        crop = random.choice(CROPS)
        stage = random.choice(STAGES)
        
        # pH distribution based on crop preference
        if crop == "Potato":
            ph = round(random.uniform(5.0, 6.4), 2)
        elif crop == "Rice":
            ph = round(random.uniform(5.5, 6.8), 2)
        else:
            ph = round(random.uniform(5.8, 7.8), 2)

        n = random.choices(STATUS_OPTS, weights=[0.40, 0.45, 0.15])[0]
        p = random.choices(STATUS_OPTS, weights=[0.25, 0.60, 0.15])[0]
        k = random.choices(STATUS_OPTS, weights=[0.25, 0.60, 0.15])[0]
        moisture = random.choices(MOIST_OPTS, weights=[0.30, 0.55, 0.15])[0]
        temp = round(random.uniform(18.0, 36.0), 1)
        humidity = round(random.uniform(40.0, 92.0), 1)

        recommendation = determine_recommendation(crop, n, p, k, ph, moisture, stage)

        rows.append({
            "Crop": crop,
            "pH": ph,
            "Nitrogen": n,
            "Phosphorus": p,
            "Potassium": k,
            "Moisture": moisture,
            "Temperature": temp,
            "Humidity": humidity,
            "Growth_Stage": stage,
            "Fertilizer_Recommendation": recommendation
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated Soil Dataset with {len(df)} records at {OUTPUT_FILE}")
    print("\nSample rows:")
    print(df.head(3))

if __name__ == "__main__":
    generate_soil_dataset()
