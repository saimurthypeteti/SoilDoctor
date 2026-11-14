"""
SoilDoctor - Dataset Generator (Dataset A: Plant Disease Images)
Populates dataset/ according to the Phase 1 schema:
- dataset/train/ (70%)
- dataset/validation/ (15%)
- dataset/test/ (15%)
Classes:
- Healthy
- Early_Blight
- Late_Blight
- Leaf_Spot
"""

import os
import random
import cv2
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "dataset")
CLASSES = ["Healthy", "Early_Blight", "Late_Blight", "Leaf_Spot"]
SPLITS = {
    "train": 70,       # 70 images per class
    "validation": 15,  # 15 images per class
    "test": 15         # 15 images per class
}

def create_base_leaf(w=256, h=256, green_hue=50):
    img = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    # Leaf ellipse
    angle = random.randint(-15, 15)
    axes = (random.randint(70, 95), random.randint(100, 125))
    b_val = random.randint(30, 50)
    g_val = random.randint(130, 180)
    r_val = random.randint(35, 60)
    cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (b_val, g_val, r_val), -1)
    
    # Veins
    cv2.line(img, (cx, cy + 90), (cx, cy - 90), (b_val - 10, g_val + 25, r_val - 10), 3, cv2.LINE_AA)
    for y_off in [-50, -20, 10, 40, 70]:
        cv2.line(img, (cx, cy + y_off), (cx - 50, cy + y_off - 20), (b_val - 10, g_val + 20, r_val - 10), 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy + y_off), (cx + 50, cy + y_off - 15), (b_val - 10, g_val + 20, r_val - 10), 2, cv2.LINE_AA)
    return img

def render_sample(cls_name):
    img = create_base_leaf()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    if cls_name == "Healthy":
        # Keep clean, slight natural chlorophyll variation
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif cls_name == "Early_Blight":
        # Concentric brown spots with yellow chlorotic halo
        num_spots = random.randint(3, 7)
        for _ in range(num_spots):
            sx = cx + random.randint(-45, 45)
            sy = cy + random.randint(-60, 60)
            r = random.randint(12, 24)
            # Halo (yellow)
            cv2.circle(img, (sx, sy), int(r * 1.4), (40, 190, 215), -1, cv2.LINE_AA)
            # Outer brown
            cv2.circle(img, (sx, sy), r, (22, 54, 98), -1, cv2.LINE_AA)
            # Concentric rings
            cv2.circle(img, (sx, sy), int(r * 0.65), (35, 75, 130), 2, cv2.LINE_AA)
            cv2.circle(img, (sx, sy), int(r * 0.35), (18, 40, 75), -1, cv2.LINE_AA)

    elif cls_name == "Late_Blight":
        # Irregular large water-soaked dark necrotic blotches
        num_blotches = random.randint(2, 4)
        for _ in range(num_blotches):
            bx = cx + random.randint(-40, 40)
            by = cy + random.randint(-50, 50)
            axes = (random.randint(25, 40), random.randint(18, 30))
            cv2.ellipse(img, (bx, by), axes, random.randint(0, 180), 0, 360, (25, 35, 45), -1)
            # Pale grey mold margin
            cv2.ellipse(img, (bx, by), (axes[0] + 5, axes[1] + 5), random.randint(0, 180), 0, 360, (180, 190, 190), 2)

    elif cls_name == "Leaf_Spot":
        # Small numerous circular necrotic spots (bacterial/cercospora)
        num_spots = random.randint(15, 30)
        for _ in range(num_spots):
            sx = cx + random.randint(-55, 55)
            sy = cy + random.randint(-75, 75)
            r = random.randint(3, 7)
            cv2.circle(img, (sx, sy), r + 2, (30, 180, 200), -1) # yellow halo
            cv2.circle(img, (sx, sy), r, (15, 30, 70), -1)       # dark spot center

    # Apply realistic blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def main():
    print("Generating Plant Disease Dataset (Phase 1)...")
    total_generated = 0

    for split_name, count in SPLITS.items():
        for cls_name in CLASSES:
            folder = os.path.join(BASE_DIR, split_name, cls_name)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                sample_img = render_sample(cls_name)
                fname = f"{cls_name.lower()}_{i+1:03d}.jpg"
                cv2.imwrite(os.path.join(folder, fname), sample_img)
                total_generated += 1

    print(f"Successfully generated {total_generated} leaf images across train/val/test splits!")

if __name__ == "__main__":
    main()
