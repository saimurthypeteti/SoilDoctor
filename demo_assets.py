"""
Generate realistic sample images for SoilDoctor demo:
- Tomato leaf with Early Blight (brown concentric lesions with yellow chlorotic halos)
- Healthy Rice leaf (vibrant green, linear veins)
- Maize leaf with Nitrogen Deficiency (interveinal V-shaped chlorosis)
- Rich Loamy Soil (granular dark brown texture)
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_early_blight_leaf():
    w, h = 420, 420
    img = np.full((h, w, 3), (242, 245, 243), dtype=np.uint8)
    center = (210, 210)
    
    # Base leaf shape (oval/pointed tomato leaflet)
    cv2.ellipse(img, center, (140, 185), 15, 0, 360, (38, 142, 45), -1)
    
    # Main central vein
    cv2.line(img, (180, 380), (240, 40), (70, 175, 75), 4, cv2.LINE_AA)
    # Secondary veins
    for y, x in [(100, 225), (150, 215), (200, 205), (250, 195), (300, 185)]:
        cv2.line(img, (x, y), (x - 80, y - 35), (65, 165, 70), 2, cv2.LINE_AA)
        cv2.line(img, (x, y), (x + 80, y - 25), (65, 165, 70), 2, cv2.LINE_AA)
        
    # Add realistic Early Blight disease spots (brown with concentric rings and yellow halo)
    lesions = [
        ((170, 160), 32), # Major spot
        ((250, 220), 26), # Secondary spot
        ((190, 270), 22), # Lower spot
        ((270, 140), 16), # Small spot
        ((140, 220), 14)  # Small spot
    ]
    
    for (cx, cy), r in lesions:
        # 1. Yellow chlorotic halo
        cv2.circle(img, (cx, cy), int(r * 1.5), (40, 190, 215), -1, cv2.LINE_AA)
        # 2. Outer brown margin
        cv2.circle(img, (cx, cy), r, (22, 54, 98), -1, cv2.LINE_AA)
        # 3. Concentric ring 1
        cv2.circle(img, (cx, cy), int(r * 0.7), (35, 75, 130), 2, cv2.LINE_AA)
        # 4. Concentric ring 2
        cv2.circle(img, (cx, cy), int(r * 0.4), (20, 45, 80), -1, cv2.LINE_AA)
        
    # Smooth slightly to blend natural organic look
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Save
    path = os.path.join(OUTPUT_DIR, "tomato_early_blight.jpg")
    cv2.imwrite(path, img)
    print(f"Generated: {path}")

def generate_healthy_leaf():
    w, h = 420, 420
    img = np.full((h, w, 3), (242, 245, 243), dtype=np.uint8)
    center = (210, 210)
    
    # Vibrant healthy green leaf
    cv2.ellipse(img, center, (135, 185), -10, 0, 360, (45, 165, 52), -1)
    
    # Veins
    cv2.line(img, (230, 380), (190, 40), (80, 200, 90), 4, cv2.LINE_AA)
    for y, x in [(110, 195), (160, 205), (210, 215), (260, 225), (310, 235)]:
        cv2.line(img, (x, y), (x - 75, y - 30), (75, 185, 80), 2, cv2.LINE_AA)
        cv2.line(img, (x, y), (x + 75, y - 20), (75, 185, 80), 2, cv2.LINE_AA)
        
    img = cv2.GaussianBlur(img, (3, 3), 0)
    path = os.path.join(OUTPUT_DIR, "rice_healthy.jpg")
    cv2.imwrite(path, img)
    print(f"Generated: {path}")

def generate_chlorosis_leaf():
    w, h = 420, 420
    img = np.full((h, w, 3), (242, 245, 243), dtype=np.uint8)
    center = (210, 210)
    
    # Base green leaf
    cv2.ellipse(img, center, (140, 185), 0, 0, 360, (50, 150, 60), -1)
    
    # Nitrogen deficiency V-shaped yellowing along midrib and tip
    pts = np.array([[210, 40], [170, 230], [250, 230]], np.int32)
    cv2.fillPoly(img, [pts], (45, 200, 230))
    cv2.ellipse(img, (210, 120), (45, 75), 0, 0, 360, (40, 190, 220), -1)
    
    # Veins
    cv2.line(img, (210, 380), (210, 40), (60, 180, 80), 3, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    
    path = os.path.join(OUTPUT_DIR, "maize_deficiency.jpg")
    cv2.imwrite(path, img)
    print(f"Generated: {path}")

def generate_loamy_soil():
    w, h = 420, 420
    np.random.seed(42)
    base_color = np.array([32, 58, 92], dtype=np.float32) # BGR for dark rich brown
    noise = np.random.normal(0, 22, (h, w, 3)).astype(np.float32)
    img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    
    # Add soil aggregate granules & texture
    for _ in range(120):
        cx = np.random.randint(20, w - 20)
        cy = np.random.randint(20, h - 20)
        radius = np.random.randint(3, 16)
        c_shade = int(np.random.choice([15, 25, 45, 65, 85]))
        cv2.circle(img, (cx, cy), radius, (c_shade, c_shade + 18, c_shade + 38), -1)
        
    img = cv2.GaussianBlur(img, (3, 3), 0)
    path = os.path.join(OUTPUT_DIR, "loamy_soil.jpg")
    cv2.imwrite(path, img)
    print(f"Generated: {path}")

if __name__ == "__main__":
    generate_early_blight_leaf()
    generate_healthy_leaf()
    generate_chlorosis_leaf()
    generate_loamy_soil()
    print("All demo samples generated successfully!")
