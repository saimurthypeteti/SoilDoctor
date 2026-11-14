import os
import cv2
from ai_engine import engine

sample_dir = os.path.join(os.path.dirname(__file__), "static", "samples")
plant_path = os.path.join(sample_dir, "tomato_early_blight.jpg")
soil_path = os.path.join(sample_dir, "loamy_soil.jpg")

print(f"Testing plant processing on {plant_path}...")
img_plant = cv2.imread(plant_path)
res_plant = engine.process_plant_image(img_plant, "Tomato")
print(f"Plant diagnosis: {res_plant['disease_name']} ({res_plant['confidence']}%) - Risk: {res_plant['risk_level']}")
print(f"Features: Color {res_plant['features']['color']['dominant_name']}, Area {res_plant['features']['shape']['area_px']}px, Affected {res_plant['features']['texture']['affected_area_pct']}%")
print(f"Base64 image keys generated: {list(res_plant['images'].keys())}")

print(f"\nTesting soil processing on {soil_path}...")
img_soil = cv2.imread(soil_path)
res_soil = engine.process_soil_image(img_soil)
print(f"Soil analysis: Texture {res_soil['visual_analysis']['texture']}, Color {res_soil['visual_analysis']['color']}")
print("AI Engine test PASSED successfully!")
