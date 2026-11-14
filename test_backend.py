"""
Automated test suite for SoilDoctor Backend APIs
Verifies:
- Flask routes and status codes
- Computer Vision and Grad-CAM responses
- Soil Analysis responses
- Database reads and writes
"""

import json
from app import app

def run_tests():
    client = app.test_client()
    passed = 0
    total = 0

    def assert_test(name, condition, extra=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}: {extra}")

    print("\n==================================================")
    print("Testing SoilDoctor Full Backend Pipeline...")
    print("==================================================")

    # 1. Health check
    res = client.get("/api/health")
    assert_test("GET /api/health", res.status_code == 200 and res.json.get("status") == "healthy")

    # 2. Main SPA index HTML
    res = client.get("/")
    assert_test("GET / (Index HTML)", res.status_code == 200 and b"SoilDoctor" in res.data)

    # 3. Plant scan endpoint with sample
    res = client.post("/api/scan/plant", json={"sample": "tomato_early_blight", "crop_type": "Tomato"})
    data = res.json.get("data", {})
    assert_test(
        "POST /api/scan/plant (Early Blight)",
        res.status_code == 200 and
        data.get("disease_name") == "Tomato Early Blight" and
        data.get("confidence") >= 85 and
        "gradcam" in data.get("images", {}) and
        "segmented" in data.get("images", {})
    )

    # 4. Plant scan endpoint with healthy sample
    res = client.post("/api/scan/plant", json={"sample": "rice_healthy", "crop_type": "Rice"})
    data = res.json.get("data", {})
    assert_test(
        "POST /api/scan/plant (Healthy Rice)",
        res.status_code == 200 and
        "Healthy" in data.get("disease_name", "")
    )

    # 5. Non-Plant Object Rejection Test
    import cv2, numpy as np, base64
    # Create synthetic coffee mug / blue circle on white background
    non_plant = np.full((300, 300, 3), (230, 230, 230), dtype=np.uint8)
    cv2.circle(non_plant, (150, 150), 70, (220, 50, 50), -1) # Blue circle
    _, buf = cv2.imencode(".jpg", non_plant)
    non_plant_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    res = client.post("/api/scan/plant", json={"image_b64": non_plant_b64, "crop_type": "Tomato"})
    assert_test(
        "POST /api/scan/plant (Non-Plant Object Rejection)",
        res.status_code == 400 and
        res.json.get("specimen_rejected") is True and
        "warning_message" in res.json.get("verification", {})
    )

    # 6. Real-time Specimen Verification Endpoint
    res = client.post("/api/verify/specimen", json={"image_b64": non_plant_b64})
    assert_test(
        "POST /api/verify/specimen",
        res.status_code == 200 and
        res.json.get("verification", {}).get("is_valid") is False
    )

    # 7. Soil scan endpoint
    res = client.post("/api/scan/soil", json={"sample": "loamy_soil", "lab_data": {"ph_value": 6.5, "n_status": "Low"}})
    soil_data = res.json.get("data", {})
    assert_test(
        "POST /api/scan/soil (Loamy Soil)",
        res.status_code == 200 and
        "visual_analysis" in soil_data and
        soil_data.get("nutrient_status", {}).get("ph_value") == 6.5
    )

    # 6. Scans history
    res = client.get("/api/scans")
    scans = res.json.get("scans", [])
    assert_test("GET /api/scans", res.status_code == 200 and len(scans) >= 5)

    # 7. Farms endpoint
    res = client.get("/api/farms")
    fields = res.json.get("fields", [])
    assert_test("GET /api/farms", res.status_code == 200 and len(fields) >= 3)

    # 8. Analytics endpoint
    res = client.get("/api/analytics")
    an = res.json
    assert_test("GET /api/analytics", res.status_code == 200 and "trends" in an and "metrics" in an)

    # 9. Crops encyclopedia
    res = client.get("/api/crops")
    crops = res.json.get("crops", [])
    assert_test("GET /api/crops", res.status_code == 200 and len(crops) >= 6)

    # 10. Profile endpoint
    res = client.get("/api/profile")
    prof = res.json.get("profile", {})
    assert_test("GET /api/profile", res.status_code == 200 and prof.get("full_name") == "Ramesh Kumar")

    print("\n==================================================")
    print(f"Results: {passed}/{total} tests passed.")
    print("==================================================")
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
