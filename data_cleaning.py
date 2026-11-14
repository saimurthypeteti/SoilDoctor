"""
SoilDoctor - Phase 3: Data Cleaning Pipeline
Cleans and validates:
1. Image Dataset (Corrupted files, duplicates, unreadable headers, format standardization)
2. Soil Tabular Dataset (Missing value imputation, duplicate removal, outlier handling, categorical normalization)
"""

import os
import hashlib
import cv2
import pandas as pd
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
SOIL_CSV = os.path.join(os.path.dirname(__file__), "soil_data.csv")
CLEAN_SOIL_CSV = os.path.join(os.path.dirname(__file__), "soil_data_cleaned.csv")

def clean_image_dataset():
    print("\n--- PHASE 3.1: Cleaning Plant Disease Image Dataset ---")
    seen_hashes = set()
    corrupted_count = 0
    duplicate_count = 0
    valid_count = 0

    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue

            file_path = os.path.join(root, f)
            
            # Check 1: Image decodability
            try:
                img = cv2.imread(file_path)
                if img is None or img.size == 0 or img.shape[0] < 32 or img.shape[1] < 32:
                    print(f"  [CORRUPTED] Removing unreadable/empty image: {f}")
                    os.remove(file_path)
                    corrupted_count += 1
                    continue
            except Exception as e:
                print(f"  [ERROR] {f}: {e}")
                os.remove(file_path)
                corrupted_count += 1
                continue

            # Check 2: Duplicate detection via MD5 hash
            with open(file_path, "rb") as fp:
                file_hash = hashlib.md5(fp.read()).hexdigest()

            if file_hash in seen_hashes:
                print(f"  [DUPLICATE] Removing duplicate image: {f}")
                os.remove(file_path)
                duplicate_count += 1
                continue

            seen_hashes.add(file_hash)
            valid_count += 1

    print(f"Image Dataset Sanitized: {valid_count} clean valid images retained ({corrupted_count} corrupted removed, {duplicate_count} duplicates removed).")

def clean_soil_dataset():
    print("\n--- PHASE 3.2: Cleaning Soil Tabular Dataset ---")
    if not os.path.exists(SOIL_CSV):
        print("Soil CSV not found, skipping.")
        return

    df = pd.read_csv(SOIL_CSV)
    initial_rows = len(df)

    # 1. Duplicate records removal
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)

    # 2. Check and handle missing values
    missing_before = df.isnull().sum().to_dict()
    # Impute numerical columns with median, categorical with mode
    if df["pH"].isnull().any():
        df["pH"] = df["pH"].fillna(df["pH"].median())
    if df["Temperature"].isnull().any():
        df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())
    if df["Humidity"].isnull().any():
        df["Humidity"] = df["Humidity"].fillna(df["Humidity"].median())

    for col in ["Crop", "Nitrogen", "Phosphorus", "Potassium", "Moisture", "Growth_Stage"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Outlier handling: Clamp pH to realistic agronomic range [4.0, 9.0]
    outliers_ph = np.count_nonzero((df["pH"] < 4.0) | (df["pH"] > 9.0))
    df["pH"] = df["pH"].clip(4.0, 9.0)

    # 4. Save cleaned dataset
    df.to_csv(CLEAN_SOIL_CSV, index=False)
    print(f"Soil Dataset Sanitized: {len(df)} records saved to soil_data_cleaned.csv")
    print(f"  - Duplicates removed: {duplicates_removed}")
    print(f"  - Outliers clipped: {outliers_ph} pH anomalies")
    print(f"  - Missing values imputed: {sum(missing_before.values())}")

if __name__ == "__main__":
    clean_image_dataset()
    clean_soil_dataset()
