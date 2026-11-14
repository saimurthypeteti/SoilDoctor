"""
SoilDoctor - Intelligent Specimen Verifier & Domain Gate
Accurately detects whether an uploaded or camera-captured image is:
1. Plant / Crop Leaf (Vegetative foliage)
2. Agricultural Soil (Earth / Loam)
3. Human Being (Face / Person / Skin)
4. Non-Plant Object (Phone, Laptop, Mug, Vehicle, Furniture, Electronics, Animals, etc.)

Rejects invalid inputs before running diagnostic pipeline so non-plant objects
are never mistakenly diagnosed as plant diseases.
"""

import cv2
import numpy as np
import threading
import torch
import torchvision.models as models
from PIL import Image

class SpecimenVerifier:
    def __init__(self):
        print("Initializing SpecimenVerifier (OpenCV + MobileNetV3)...")
        # 1. OpenCV Haar Cascades for Face & Human Verification
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

        # 2. OpenCV HOG People Detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # 3. PyTorch Pretrained MobileNetV3 for 1000-Class Object Recognition
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.categories = self.weights.meta['categories']
        self.model = models.mobilenet_v3_small(weights=self.weights).eval()
        self.preprocess = self.weights.transforms()

        # Keyword sets for ImageNet category grouping
        self.plant_keywords = {
            "plant", "leaf", "tree", "grass", "corn", "ear", "flower", "daisy", "rose", "dahlia",
            "sunflower", "tulip", "acorn", "pot", "flowerpot", "greenhouse", "mushroom", "hay",
            "cabbage", "broccoli", "cauliflower", "zucchini", "cucumber", "artichoke", "bell pepper",
            "cardoon", "head cabbage", "strawberry", "orange", "lemon", "fig", "pineapple", "banana",
            "apple", "pomegranate", "grapes", "vine", "timber", "forest"
        }

        self.human_keywords = {
            "suit", "trench coat", "wig", "groom", "scuba diver", "jersey", "cardigan",
            "sweatshirt", "cloak", "jean", "pajama", "swimming trunks", "stethoscope",
            "mask", "sunglasses", "hat", "sombrero", "cowboy hat", "bonnet", "military uniform"
        }

        self.animal_keywords = {
            "dog", "cat", "puppy", "kitten", "bird", "parrot", "canary", "duck", "goose", "hen",
            "rooster", "horse", "cow", "ox", "sheep", "pig", "rabbit", "fish", "frog", "snake",
            "lizard", "spider", "bear", "tiger", "lion", "monkey", "elephant"
        }

    def verify(self, img_bgr):
        if img_bgr is None:
            return {
                "is_valid": False,
                "specimen_type": "unknown",
                "label": "Invalid Image",
                "status": "REJECTED_INVALID",
                "confidence": 0,
                "warning_message": "Could not decode image file. Please try another image."
            }

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # 1. HUMAN CHECK: Face detection + Skin tone analysis
        # -------------------------------------------------------------
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(int(min(h, w) * 0.12), int(min(h, w) * 0.12))
        )
        profiles = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(int(min(h, w) * 0.12), int(min(h, w) * 0.12))
        )

        # YCrCb Skin tone filter
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y_chan = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        # Real human skin has moderate-to-high luminance (Y > 60) and specific Cr/Cb range
        skin_mask = (y_chan > 65) & (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
        skin_pct = (np.count_nonzero(skin_mask) / (h * w)) * 100

        # HOG Person detector (downscale if large image for speed)
        small_gray = cv2.resize(gray, (240, int(240 * h / w))) if w > 240 else gray
        people, _ = self.hog.detectMultiScale(small_gray, winStride=(8, 8), padding=(4, 4), scale=1.08)

        is_human = (len(faces) > 0 and skin_pct > 5.0) or \
                   (len(profiles) > 0 and skin_pct > 5.0) or \
                   (len(people) > 0 and skin_pct > 10.0) or \
                   (len(faces) > 0 and len(profiles) > 0) or \
                   (skin_pct > 55.0 and len(people) > 0)

        if is_human:
            return {
                "is_valid": False,
                "specimen_type": "human",
                "label": "Human Person / Face",
                "status": "REJECTED_HUMAN",
                "confidence": 96,
                "warning_message": "⚠️ Human Detected! SoilDoctor is an agricultural scanner designed for crops and soil. Please point your camera at a crop leaf or farm soil.",
                "details": {
                    "faces_detected": int(len(faces) + len(profiles)),
                    "skin_area_pct": round(skin_pct, 1)
                }
            }

        # -------------------------------------------------------------
        # 2. VEGETATION OPTICS CHECK: Excess Green Index & Foliar HSV
        # -------------------------------------------------------------
        b, g, r = cv2.split(img_bgr.astype(np.float32))
        # ExG = 2*G - R - B (Vegetation Index in Agronomy)
        exg = (2.0 * g - r - b)
        exg_positive_ratio = np.count_nonzero(exg > 15) / (h * w)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Broad foliage hue range (green: 28-88, yellow-chlorotic: 15-28, brown lesions: 5-20)
        plant_mask = cv2.inRange(hsv, np.array([25, 45, 25]), np.array([95, 255, 255]))
        plant_ratio = np.count_nonzero(plant_mask) / (h * w)

        # Soil color check (earthy dark brown or reddish clay tones)
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)
        is_soil_color = (mean_r > mean_b * 1.1) and (mean_r < 150) and (mean_g < 125) and (plant_ratio < 0.20) and (exg_positive_ratio < 0.15)

        # -------------------------------------------------------------
        # 3. DEEP LEARNING MOBILENET OBJECT CLASSIFIER
        # -------------------------------------------------------------
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor_in = self.preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor_in).squeeze(0)

        probs = torch.nn.functional.softmax(logits, dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)

        top_predictions = []
        for i in range(5):
            cat_name = self.categories[top5_catid[i]]
            prob_val = round(top5_prob[i].item() * 100, 1)
            top_predictions.append({"category": cat_name, "probability": prob_val})

        top1_name = top_predictions[0]["category"].lower()
        top1_prob = top_predictions[0]["probability"]

        # -------------------------------------------------------------
        # 4. DOMAIN REASONING
        # -------------------------------------------------------------
        # Check if top 1 is human apparel
        for kw in self.human_keywords:
            if kw in top1_name:
                return {
                    "is_valid": False,
                    "specimen_type": "human",
                    "label": f"Human Clothing / Person ({top_predictions[0]['category']})",
                    "status": "REJECTED_HUMAN",
                    "confidence": int(top1_prob),
                    "warning_message": "⚠️ Person / Apparel Detected! Please scan a plant leaf or farm soil.",
                    "top_detections": top_predictions
                }

        # Check if top 1 is an animal
        for kw in self.animal_keywords:
            if kw in top1_name:
                return {
                    "is_valid": False,
                    "specimen_type": "animal",
                    "label": f"Animal / Pet ({top_predictions[0]['category']})",
                    "status": "REJECTED_OBJECT",
                    "confidence": int(top1_prob),
                    "warning_message": f"⚠️ Animal Detected ({top_predictions[0]['category']})! SoilDoctor is designed for agricultural crops. Please scan a plant leaf.",
                    "top_detections": top_predictions
                }

        # Check if clearly a plant or agricultural item
        is_plant_label = any(kw in top1_name for kw in self.plant_keywords)

        # Give the soil-color check precedence over broad ImageNet/green-pixel
        # matches. Brown soil often contains enough warm pixels to trigger the
        # generic vegetation mask, even when the whole frame is a soil sample.
        if (is_plant_label or plant_ratio > 0.18 or exg_positive_ratio > 0.15) and not is_soil_color:
            # Verified Plant
            return {
                "is_valid": True,
                "specimen_type": "plant",
                "label": "Verified Plant Foliage",
                "status": "VALID_PLANT",
                "confidence": min(99, max(88, int(85 + plant_ratio * 15))),
                "details": {
                    "foliage_ratio": round(plant_ratio, 2),
                    "exg_index": round(float(np.mean(exg)), 1),
                    "top_detection": top_predictions[0]['category']
                },
                "top_detections": top_predictions
            }

        elif is_soil_color or "soil" in top1_name or "mud" in top1_name or "sand" in top1_name:
            # Verified Soil
            return {
                "is_valid": True,
                "specimen_type": "soil",
                "label": "Verified Soil Specimen",
                "status": "VALID_SOIL",
                "confidence": 92,
                "details": {
                    "soil_tone": "Earthy Loam",
                    "top_detection": top_predictions[0]['category']
                },
                "top_detections": top_predictions
            }

        else:
            # It's an everyday manufactured object (laptop, mug, phone, vehicle, shoe, etc.)
            clean_obj_name = top_predictions[0]['category'].replace("_", " ").title()
            return {
                "is_valid": False,
                "specimen_type": "object",
                "label": f"Non-Plant Object: {clean_obj_name}",
                "status": "REJECTED_OBJECT",
                "confidence": int(top1_prob),
                "warning_message": f"⚠️ Non-Plant Object Detected ({clean_obj_name})! SoilDoctor only analyzes crops, leaves, and soil. Please upload a real plant or leaf specimen.",
                "top_detections": top_predictions
            }

class _LazySpecimenVerifier:
    """Load the pretrained ImageNet verifier only when a specimen is scanned."""

    def __init__(self):
        self._instance = None
        self._lock = threading.Lock()

    def _get_instance(self):
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = SpecimenVerifier()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)


verifier = _LazySpecimenVerifier()

if __name__ == "__main__":
    test_leaf = cv2.imread("static/samples/tomato_early_blight.jpg")
    res = verifier.verify(test_leaf)
    print("Test Tomato Leaf verification:", res["status"], res["label"], f"({res['confidence']}%)")

    test_soil = cv2.imread("static/samples/loamy_soil.jpg")
    res_s = verifier.verify(test_soil)
    print("Test Soil verification:", res_s["status"], res_s["label"], f"({res_s['confidence']}%)")

    # Generate a dummy synthetic phone/mug image to test object rejection
    dummy_mug = np.full((400, 400, 3), (200, 200, 200), dtype=np.uint8)
    cv2.circle(dummy_mug, (200, 200), 100, (40, 40, 180), -1) # Red cup
    res_obj = verifier.verify(dummy_mug)
    print("Test Non-Plant Object verification:", res_obj["status"], res_obj["label"])
