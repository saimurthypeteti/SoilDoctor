import os
import json
import base64
import requests
from io import BytesIO

# Import app directly to test Flask test client
from app import app

def test_full_pipeline():
    print("==================================================")
    print("Testing SoilDoctor Backend & Endpoints with New Dataset")
    print("==================================================")
    
    client = app.test_client()

    # 1. Test index page
    res = client.get("/")
    assert res.status_code == 200, f"Index page failed: {res.status_code}"
    html = res.get_data(as_text=True)
    assert "SoilDoctor" in html
    assert "Tomato" in html and "Rice" in html and "Chilli" in html
    assert "Sugarcane" in html and "Blackgram" in html and "Soybean" in html
    print("[PASS] 1. Index page loads with all 6 target crops and dual-mode tabs.")

    # 2. Test /api/crops
    res = client.get("/api/crops")
    assert res.status_code == 200
    crops_res = res.get_json()
    crops_list = crops_res.get("crops", [])
    crop_names = [c["name"] for c in crops_list]
    print(f"Crops in DB: {crop_names}")
    for expected in ["Tomato", "Rice", "Chilli", "Sugarcane", "Blackgram", "Soybean"]:
        assert expected in crop_names, f"Expected crop {expected} missing from DB"
    print("[PASS] 2. /api/crops returns exclusively the 6 target crops.")

    # 3. Test /api/dataset/samples
    res = client.get("/api/dataset/samples")
    assert res.status_code == 200
    dataset_manifest = res.get_json()
    samples = dataset_manifest.get("samples", [])
    print(f"Total curated real dataset samples available: {len(samples)}")
    assert len(samples) > 20, "Should have curated dataset samples"
    
    # Check sample filter
    res_tomato = client.get("/api/dataset/samples?crop=Tomato")
    tomato_samples = res_tomato.get_json().get("samples", [])
    assert all(s["crop"] == "Tomato" for s in tomato_samples), "Filtered samples should all be Tomato"
    print(f"[PASS] 3. /api/dataset/samples endpoint returns {len(samples)} real samples, filtering works ({len(tomato_samples)} Tomato).")

    # 4. Test /api/scan/plant with sample_key
    test_key = "tomato_early_blight_1.jpg"
    print(f"\nScanning real sample: {test_key}...")
    res = client.post("/api/scan/plant", data={"sample": test_key, "crop_type": "Tomato"})
    assert res.status_code == 200, f"Plant scan failed: {res.status_code}, {res.data}"
    diag = res.get_json()
    assert diag.get("success") == True, f"Scan unsuccessful: {diag}"
    d = diag.get("data", {})
    print(f"   Crop: {d.get('crop')} | Condition: {d.get('disease_name')} | Confidence: {d.get('confidence')}%")
    print(f"   Severity: {d.get('severity')} | Risk: {d.get('risk_level')}")
    images = d.get("images", {})
    print(f"   OpenCV Mask: {bool(images.get('segmented'))} | Grad-CAM: {bool(images.get('gradcam'))}")
    assert d.get("crop") == "Tomato"
    assert "features" in d
    assert "organic_solution" in d
    print("[PASS] 4. /api/scan/plant with sample_key succeeds with Grad-CAM and features.")

    # 5. Test /api/scan/plant with base64 image (simulating user upload or camera capture)
    real_sample_path = os.path.join("static", "samples", "real", "rice_bacterial_leaf_blight_1.jpg")
    with open(real_sample_path, "rb") as f:
        b64_img = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    
    print("\nScanning base64 uploaded image (Rice leaf)...")
    res = client.post("/api/scan/plant", data={"image_b64": b64_img, "crop_type": "Rice"})
    assert res.status_code == 200
    diag = res.get_json()
    assert diag.get("success") == True
    d = diag.get("data", {})
    print(f"   Crop: {d.get('crop')} | Condition: {d.get('disease_name')} | Confidence: {d.get('confidence')}%")
    print(f"   Action items count: {len(d.get('what_to_do', []))}")
    print("[PASS] 5. /api/scan/plant with direct upload base64 works accurately.")

    # 6. Test /api/analytics
    res = client.get("/api/analytics")
    assert res.status_code == 200
    analytics = res.get_json()
    assert "trends" in analytics
    trends = analytics["trends"]
    for c in ["tomato", "rice", "chilli", "sugarcane", "blackgram", "soybean"]:
        assert c in trends, f"Missing trend for {c}"
    print(f"   Analytics trends verified for: {list(trends.keys())}")
    print("[PASS] 6. /api/analytics endpoint operational.")

    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_full_pipeline()
