"""
SoilDoctor - Real Image Extractor & Curator
Extracts crisp real sample images from 'Dataset new' for all 6 crops:
- Tomato
- Rice
- Chilli
- Sugarcane
- Blackgram
- Soybean
Saves curated images to static/samples/real/ and builds samples_manifest.json
"""

import os
import shutil
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset new")
DEST_DIR = os.path.join(BASE_DIR, "static", "samples", "real")
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")

os.makedirs(DEST_DIR, exist_ok=True)

# Mapping of crop and disease class to directory path relative to 'Dataset new'
CLASS_CONFIG = [
    # 1. Tomato (Tomata Dataset/train or test)
    {
        "crop": "Tomato",
        "condition_slug": "early_blight",
        "disease_name": "Tomato Early Blight",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("Tomata Dataset", "train", "Early_blight"),
        "key_symptoms": ["Dark brown circular lesions with concentric rings", "Chlorotic yellow halo surrounding lesions"]
    },
    {
        "crop": "Tomato",
        "condition_slug": "late_blight",
        "disease_name": "Tomato Late Blight",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("Tomata Dataset", "train", "Late_blight"),
        "key_symptoms": ["Water-soaked dark lesions", "Rapid tissue collapse and browning"]
    },
    {
        "crop": "Tomato",
        "condition_slug": "leaf_miner",
        "disease_name": "Tomato Leaf Miner",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("Tomata Dataset", "train", "Leaf Miner"),
        "key_symptoms": ["Serpentine white/translucent trails winding across leaf", "Damaged photosynthetic tissue"]
    },
    {
        "crop": "Tomato",
        "condition_slug": "magnesium_deficiency",
        "disease_name": "Tomato Magnesium Deficiency",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("Tomata Dataset", "train", "Magnesium Deficiency"),
        "key_symptoms": ["Interveinal chlorosis (yellowing between green veins)", "Older lower leaves affected first"]
    },
    {
        "crop": "Tomato",
        "condition_slug": "nitrogen_deficiency",
        "disease_name": "Tomato Nitrogen Deficiency",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("Tomata Dataset", "train", "Nitrogen Deficiency"),
        "key_symptoms": ["Generalized pale green to yellow discoloration", "Thinning leaf canopy and stunted vigor"]
    },
    {
        "crop": "Tomato",
        "condition_slug": "potassium_deficiency",
        "disease_name": "Tomato Potassium Deficiency",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("Tomata Dataset", "train", "Pottassium Deficiency"),
        "key_symptoms": ["Marginal leaf necrosis and scorching", "Curling upward of leaf edges"]
    },
    {
        "crop": "Tomato",
        "condition_slug": "spotted_wilt_virus",
        "disease_name": "Tomato Spotted Wilt Virus",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("Tomata Dataset", "train", "Spotted Wilt Virus"),
        "key_symptoms": ["Bronze or dark brown specks and rings", "Distorted leaf cupping and terminal shoot dieback"]
    },
    {
        "crop": "Tomato",
        "condition_slug": "healthy",
        "disease_name": "Tomato Healthy Leaf",
        "risk_level": "LOW",
        "rel_dir": os.path.join("Tomata Dataset", "train", "Healthy"),
        "key_symptoms": ["Vibrant uniform green foliage", "Intact leaf margins and structure"]
    },

    # 2. Rice (Rice leaf disease/Training data)
    {
        "crop": "Rice",
        "condition_slug": "bacterial_leaf_blight",
        "disease_name": "Rice Bacterial Leaf Blight",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("Rice leaf disease", "Training data", "Bacterial Leaf Blight"),
        "key_symptoms": ["Water-soaked to yellowish-white wavy marginal stripes", "Bacterial ooze droplets along leaf veins"]
    },
    {
        "crop": "Rice",
        "condition_slug": "brown_spot",
        "disease_name": "Rice Brown Spot",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("Rice leaf disease", "Training data", "Brown Spot"),
        "key_symptoms": ["Small cylindrical to oval dark brown spots", "Greyish or whitish center with yellow halo"]
    },
    {
        "crop": "Rice",
        "condition_slug": "leaf_blast",
        "disease_name": "Rice Leaf Blast",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("Rice leaf disease", "Training data", "Leaf Blast"),
        "key_symptoms": ["Diamond/spindle-shaped lesions with pointed ends", "Greyish-white center surrounded by dark brown margin"]
    },
    {
        "crop": "Rice",
        "condition_slug": "leaf_scald",
        "disease_name": "Rice Leaf Scald",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("Rice leaf disease", "Training data", "Leaf scald"),
        "key_symptoms": ["Zonate chevron-like olive-green to bleach-colored banding", "Lesions develop from leaf tip downward"]
    },
    {
        "crop": "Rice",
        "condition_slug": "narrow_brown_leaf_spot",
        "disease_name": "Rice Narrow Brown Leaf Spot",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("Rice leaf disease", "Training data", "Narrow Brown Leaf Spot"),
        "key_symptoms": ["Narrow linear short brown streaks parallel to leaf veins", "Premature foliage drying"]
    },
    {
        "crop": "Rice",
        "condition_slug": "rice_hispa",
        "disease_name": "Rice Hispa Pest Damage",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("Rice leaf disease", "Training data", "Rice Hispa"),
        "key_symptoms": ["Parallel white streaks scraped along epidermis", "Blister-like blotches where larvae feed"]
    },
    {
        "crop": "Rice",
        "condition_slug": "sheath_blight",
        "disease_name": "Rice Sheath Blight",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("Rice leaf disease", "Training data", "Sheath Blight"),
        "key_symptoms": ["Oval to irregular greenish-grey water-soaked spots", "Lesions coalesce with purplish-brown margins"]
    },
    {
        "crop": "Rice",
        "condition_slug": "healthy",
        "disease_name": "Rice Healthy Leaf",
        "risk_level": "LOW",
        "rel_dir": os.path.join("Rice leaf disease", "Training data", "Healthy Rice Leaf"),
        "key_symptoms": ["Vigorous upright green blade", "Even coloration without spots or lesions"]
    },

    # 3. Chilli (BCSS Dataset/dataset)
    {
        "crop": "Chilli",
        "condition_slug": "cercospora",
        "disease_name": "Chilli Cercospora Leaf Spot",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "chili_cercospora"),
        "key_symptoms": ["Frogeye circular spots with light grey center and dark red/brown border", "Severe premature defoliation"]
    },
    {
        "crop": "Chilli",
        "condition_slug": "mites_and_trips",
        "disease_name": "Chilli Mites & Thrips Infestation",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "chili_mites_and_trips"),
        "key_symptoms": ["Upward leaf curling (thrips) or downward boat-shaped curling (mites)", "Crinkling, thickening and bronzing"]
    },
    {
        "crop": "Chilli",
        "condition_slug": "nutritional",
        "disease_name": "Chilli Nutritional Deficiency",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "chili_nutritional"),
        "key_symptoms": ["Pale yellow chlorotic leaves with stunted growth", "Interveinal yellowing with weak branching"]
    },
    {
        "crop": "Chilli",
        "condition_slug": "powdery_mildew",
        "disease_name": "Chilli Powdery Mildew",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "chili_powdery mildew"),
        "key_symptoms": ["White powdery fungal patches on lower leaf surface", "Chlorotic blotches on upper leaf surface"]
    },
    {
        "crop": "Chilli",
        "condition_slug": "healthy",
        "disease_name": "Chilli Healthy Leaf",
        "risk_level": "LOW",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "chili_healthy"),
        "key_symptoms": ["Deep glossy green foliage", "Smooth intact lamina without curling or spots"]
    },

    # 4. Sugarcane (BCSS Dataset/dataset)
    {
        "crop": "Sugarcane",
        "condition_slug": "mosaic",
        "disease_name": "Sugarcane Mosaic Virus",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "sugercane_Mosaic"),
        "key_symptoms": ["Contrasting dark and pale green mottled patches", "Linear patchy chlorotic stripes along blade"]
    },
    {
        "crop": "Sugarcane",
        "condition_slug": "red_rot",
        "disease_name": "Sugarcane Red Rot",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "sugercane_RedRot"),
        "key_symptoms": ["Reddish lesions on leaf midrib with white cross bands", "Withering of crown foliage and internal red tissue"]
    },
    {
        "crop": "Sugarcane",
        "condition_slug": "rust",
        "disease_name": "Sugarcane Rust",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "sugercane_Rust"),
        "key_symptoms": ["Elongated small yellowish spots turning reddish-brown pustules", "Powdery brown spore ruptures on leaf underside"]
    },
    {
        "crop": "Sugarcane",
        "condition_slug": "yellow_leaf",
        "disease_name": "Sugarcane Yellow Leaf Disease",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "sugercane_Yellow"),
        "key_symptoms": ["Intense yellowing of midrib on 3rd to 6th leaves", "Yellow discoloration spreads to leaf lamina"]
    },
    {
        "crop": "Sugarcane",
        "condition_slug": "healthy",
        "disease_name": "Sugarcane Healthy Leaf",
        "risk_level": "LOW",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "sugercane_Healthy"),
        "key_symptoms": ["Long, robust, dark green arching leaf blades", "Clean white-green central midrib"]
    },

    # 5. Blackgram (BCSS Dataset/dataset)
    {
        "crop": "Blackgram",
        "condition_slug": "anthracnose",
        "disease_name": "Blackgram Anthracnose",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "blackgram_Anthracnose"),
        "key_symptoms": ["Circular to angular sunken brown lesions on leaves and pods", "Dark purplish-brown borders"]
    },
    {
        "crop": "Blackgram",
        "condition_slug": "leaf_crinkle",
        "disease_name": "Blackgram Leaf Crinkle Virus",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "blackgram_LeafCrinckle"),
        "key_symptoms": ["Severe crinkling, puckering and curling of leaves", "Enlarged, rugose leathery leaflets with twisted margins"]
    },
    {
        "crop": "Blackgram",
        "condition_slug": "powdery_mildew",
        "disease_name": "Blackgram Powdery Mildew",
        "risk_level": "MEDIUM",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "blackgram_PowderyMildew"),
        "key_symptoms": ["White powdery flour-like coating spreading on upper leaf surface", "Yellowing and premature shedding of infected foliage"]
    },
    {
        "crop": "Blackgram",
        "condition_slug": "yellow_mosaic",
        "disease_name": "Blackgram Yellow Mosaic Virus",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "blackgram_YellowMosaic"),
        "key_symptoms": ["Bright golden-yellow irregular patches interwoven with green tissue", "Complete chlorosis and severe pod reduction"]
    },
    {
        "crop": "Blackgram",
        "condition_slug": "healthy",
        "disease_name": "Blackgram Healthy Leaf",
        "risk_level": "LOW",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "blackgram_Healthy"),
        "key_symptoms": ["Smooth trifoliate emerald green leaves", "Intact ovate leaflet margins"]
    },

    # 6. Soybean (BCSS Dataset/dataset)
    {
        "crop": "Soybean",
        "condition_slug": "disease_soybean",
        "disease_name": "Soybean Leaf Disease",
        "risk_level": "HIGH",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "Disease_soybean"),
        "key_symptoms": ["Necrotic brown speckling and pustules across leaf surface", "Yellowing and premature foliage senescence"]
    },
    {
        "crop": "Soybean",
        "condition_slug": "healthy",
        "disease_name": "Soybean Healthy Leaf",
        "risk_level": "LOW",
        "rel_dir": os.path.join("BCSS Dataset", "dataset", "Healthy_soybean"),
        "key_symptoms": ["Lush trifoliate green leaves with uniform surface", "Healthy venation and vigorous vegetative vigor"]
    }
]

def extract():
    print("Starting extraction of real sample images...")
    manifest = []
    
    primary_samples = {
        "Tomato": None,
        "Rice": None,
        "Chilli": None,
        "Sugarcane": None,
        "Blackgram": None,
        "Soybean": None
    }

    for idx, cfg in enumerate(CLASS_CONFIG):
        src_path = os.path.join(DATASET_DIR, cfg["rel_dir"])
        if not os.path.exists(src_path):
            print(f"Warning: path not found: {src_path}")
            continue

        all_files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not all_files:
            print(f"Warning: no images found in {src_path}")
            continue

        random.seed(42 + idx)
        sample_filenames = random.sample(all_files, min(3, len(all_files)))

        condition_samples = []
        for s_idx, fname in enumerate(sample_filenames):
            ext = os.path.splitext(fname)[1].lower()
            dest_filename = f"{cfg['crop'].lower()}_{cfg['condition_slug']}_{s_idx+1}{ext}"
            dest_file_path = os.path.join(DEST_DIR, dest_filename)
            shutil.copyfile(os.path.join(src_path, fname), dest_file_path)

            web_path = f"/static/samples/real/{dest_filename}"
            condition_samples.append({
                "filename": dest_filename,
                "url": web_path,
                "dataset_source": os.path.join(cfg["rel_dir"], fname)
            })

            if primary_samples[cfg["crop"]] is None and ("blight" in cfg["condition_slug"] or "mosaic" in cfg["condition_slug"] or "cercospora" in cfg["condition_slug"] or "disease" in cfg["condition_slug"]):
                primary_samples[cfg["crop"]] = dest_file_path

        manifest.append({
            "id": f"{cfg['crop'].lower()}_{cfg['condition_slug']}",
            "crop": cfg["crop"],
            "condition_slug": cfg["condition_slug"],
            "disease_name": cfg["disease_name"],
            "risk_level": cfg["risk_level"],
            "key_symptoms": cfg["key_symptoms"],
            "samples": condition_samples,
            "primary_sample": condition_samples[0] if condition_samples else None
        })

    for crop, ppath in primary_samples.items():
        if ppath and os.path.exists(ppath):
            shutil.copyfile(ppath, os.path.join(SAMPLES_DIR, f"{crop.lower()}_demo.jpg"))

    # Also update tomato_early_blight.jpg and rice_healthy.jpg with REAL dataset images!
    tomato_eb = os.path.join(DEST_DIR, "tomato_early_blight_1.jpg")
    if os.path.exists(tomato_eb):
        shutil.copyfile(tomato_eb, os.path.join(SAMPLES_DIR, "tomato_early_blight.jpg"))
        
    rice_h = os.path.join(DEST_DIR, "rice_healthy_1.jpg")
    if os.path.exists(rice_h):
        shutil.copyfile(rice_h, os.path.join(SAMPLES_DIR, "rice_healthy.jpg"))

    manifest_path = os.path.join(DEST_DIR, "samples_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully curated {len(manifest)} condition classes with real dataset images!")
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    extract()
