"""
SoilDoctor - Nutrient Deficiency Model Trainer
Trained on Pepper Nutrient Deficiency (VOC XML) and CoLeaf datasets.
Saves weights to models/nutrient_model.pth and mapping to models/nutrient_classes.json.
"""

import os
import json
import random
import time
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEPPER_DIR = os.path.join(BASE_DIR, "pepper")
COLEAF_DIR = os.path.join(BASE_DIR, "CoLeaf DATASET")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

NUTRIENT_CLASSES = [
    {
        "id": 0,
        "name": "Nitrogen Deficiency",
        "symbol": "N",
        "slug": "nitrogen_deficiency",
        "description": "General uniform yellowing/chlorosis beginning on older foliage, stunted vegetative vigor.",
        "fertilizer": "Urea (46-0-0) 2% foliar spray or Calcium Ammonium Nitrate (CAN) 40 kg/acre.",
        "organic": "Enriched vermicompost (2 tons/acre) and liquid jeevamrutha foliar drench."
    },
    {
        "id": 1,
        "name": "Phosphorus Deficiency",
        "symbol": "P",
        "slug": "phosphorus_deficiency",
        "description": "Foliage turns abnormally dark green with bronze/purple discoloration, delayed flowering.",
        "fertilizer": "Single Super Phosphate (SSP 16%) 50 kg/acre or DAP (18-46-0) at root zone.",
        "organic": "Rock phosphate combined with phosphate solubilizing bacteria (PSB 2 kg/acre)."
    },
    {
        "id": 2,
        "name": "Potassium Deficiency",
        "symbol": "K",
        "slug": "potassium_deficiency",
        "description": "Marginal chlorosis leading to scorch and necrosis along leaf margins, upward curling.",
        "fertilizer": "Sulphate of Potash (SOP 0-0-50) 10 g/L foliar spray or Muriate of Potash (MOP 0-0-60).",
        "organic": "Wood ash (100 kg/acre) or composted banana waste and potassium mobilizer (KMB)."
    },
    {
        "id": 3,
        "name": "Magnesium Deficiency",
        "symbol": "Mg",
        "slug": "magnesium_deficiency",
        "description": "Distinct interveinal chlorosis on lower mature leaves, veins remain strikingly green.",
        "fertilizer": "Magnesium Sulphate (Epsom salt, MgSO4) 5 g/L foliar spray applied twice at 10-day intervals.",
        "organic": "Dolomite lime application (150 kg/acre) during field preparation."
    },
    {
        "id": 4,
        "name": "Iron Deficiency",
        "symbol": "Fe",
        "slug": "iron_deficiency",
        "description": "Interveinal chlorosis on youngest terminal leaves; leaf tissue bleaches to ivory/yellow.",
        "fertilizer": "Chelated Iron (Fe-EDTA 12%) 1.0-1.5 g/L foliar spray early morning.",
        "organic": "Fermented compost tea enriched with organic iron humate."
    },
    {
        "id": 5,
        "name": "Zinc Deficiency",
        "symbol": "Zn",
        "slug": "zinc_deficiency",
        "description": "Little leaf / rosette symptom, shortening of internodes, mottled interveinal chlorosis.",
        "fertilizer": "Zinc Sulphate Heptahydrate (ZnSO4 21%) 5 g/L neutralized with 2.5 g/L lime.",
        "organic": "Zinc solubilizing biofertilizer (ZSB) seed treatment and zinc-enriched farmyard manure."
    },
    {
        "id": 6,
        "name": "Calcium Deficiency",
        "symbol": "Ca",
        "slug": "calcium_deficiency",
        "description": "Deformed young leaves, hooked leaf tips, blossom end necrosis on fruiting vegetables.",
        "fertilizer": "Calcium Nitrate (Ca(NO3)2) 5 g/L foliar spray during vegetative and flowering stage.",
        "organic": "Crushed agricultural gypsum or bone meal powder."
    },
    {
        "id": 7,
        "name": "Boron Deficiency",
        "symbol": "B",
        "slug": "boron_deficiency",
        "description": "Thick, brittle foliage, curling downward, death of apical growing points.",
        "fertilizer": "Solubor / Borax (20% B) 1.0 g/L foliar spray.",
        "organic": "Composted poultry manure with organic boron amendments."
    },
    {
        "id": 8,
        "name": "Healthy Foliage (Balanced Nutrients)",
        "symbol": "Healthy",
        "slug": "healthy_nutrients",
        "description": "Uniform green pigmentation, balanced leaf expansion, optimal cell wall integrity.",
        "fertilizer": "Standard maintenance balanced NPK (19-19-19) 3 g/L.",
        "organic": "Maintain regular vermicompost and microbial consortia maintenance."
    }
]

PEPPER_MAP = {
    "n_leaf": 0, "n_bottomleaf": 0, "n_backleaf": 0,
    "p_leaf": 1, "p_bottomleaf": 1,
    "k_leaf": 2, "k_bottomleaf": 2, "k_backleaf": 2,
    "mg_leaf": 3, "mg_bottomleaf": 3, "mg_backleaf": 3,
    "fe_leaf": 4, "fe_bottomleaf": 4, "fe_backleaf": 4,
    "zn_leaf": 5, "zn_bottomleaf": 5, "zn_backleaf": 5,
    "healthy_leaf": 8, "healthy_bottomleaf": 8, "healthy_backleaf": 8
}

COLEAF_ZIP_MAP = {
    "nitrogen-N.zip": 0,
    "phosphorus-P.zip": 1,
    "potasium-K.zip": 2,
    "magnesium-Mg.zip": 3,
    "iron-Fe.zip": 4,
    "calcium-Ca.zip": 6,
    "boron-B.zip": 7,
    "healthy.zip": 8
}

class NutrientDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if item["type"] == "pepper_crop":
            # Load full image and crop bounding box
            img = Image.open(item["img_path"]).convert("RGB")
            bbox = item["bbox"]
            w, h = img.size
            # Clamp bbox
            xmin = max(0, min(bbox[0], w - 10))
            ymin = max(0, min(bbox[1], h - 10))
            xmax = max(xmin + 10, min(bbox[2], w))
            ymax = max(ymin + 10, min(bbox[3], h))
            crop = img.crop((xmin, ymin, xmax, ymax))
            if self.transform:
                crop = self.transform(crop)
            return crop, item["label"]
        elif item["type"] == "coleaf_zip":
            # Extract from zip
            with zipfile.ZipFile(item["zip_path"], 'r') as zf:
                data = zf.read(item["entry_name"])
                img = Image.open(BytesIO(data)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, item["label"]
        elif item["type"] == "file":
            img = Image.open(item["path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, item["label"]

def collect_nutrient_samples():
    samples = []
    
    # 1. Collect from Pepper dataset (using Pascal VOC XML bounding boxes)
    labels_dir = os.path.join(PEPPER_DIR, "Labels")
    photos_dir = os.path.join(PEPPER_DIR, "pepper_photo")

    if os.path.exists(labels_dir) and os.path.exists(photos_dir):
        xml_files = [f for f in os.listdir(labels_dir) if f.endswith(".xml")]
        random.seed(42)
        random.shuffle(xml_files)
        
        class_counts = {i: 0 for i in range(len(NUTRIENT_CLASSES))}
        MAX_PER_CLASS_PEPPER = 80

        for xf in xml_files:
            xml_path = os.path.join(labels_dir, xf)
            img_name = xf.replace(".xml", ".JPG")
            img_path = os.path.join(photos_dir, img_name)
            if not os.path.exists(img_path):
                img_name = xf.replace(".xml", ".jpg")
                img_path = os.path.join(photos_dir, img_name)
            if not os.path.exists(img_path):
                continue

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    tag = obj.find("name").text
                    if tag in PEPPER_MAP:
                        cls_id = PEPPER_MAP[tag]
                        if class_counts[cls_id] < MAX_PER_CLASS_PEPPER:
                            bnd = obj.find("bndbox")
                            bbox = (
                                int(float(bnd.find("xmin").text)),
                                int(float(bnd.find("ymin").text)),
                                int(float(bnd.find("xmax").text)),
                                int(float(bnd.find("ymax").text))
                            )
                            # Only valid boxes
                            if (bbox[2] - bbox[0] > 50) and (bbox[3] - bbox[1] > 50):
                                samples.append({
                                    "type": "pepper_crop",
                                    "img_path": img_path,
                                    "bbox": bbox,
                                    "label": cls_id
                                })
                                class_counts[cls_id] += 1
            except Exception:
                continue

    # 2. Collect from CoLeaf dataset zip packages
    if os.path.exists(COLEAF_DIR):
        for zname, cls_id in COLEAF_ZIP_MAP.items():
            zpath = os.path.join(COLEAF_DIR, zname)
            if os.path.exists(zpath):
                try:
                    with zipfile.ZipFile(zpath, 'r') as zf:
                        img_names = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        random.seed(42)
                        random.shuffle(img_names)
                        for iname in img_names[:60]:
                            samples.append({
                                "type": "coleaf_zip",
                                "zip_path": zpath,
                                "entry_name": iname,
                                "label": cls_id
                            })
                except Exception as ex:
                    print(f"Error reading {zname}: {ex}")

    return samples

def train_nutrient_model():
    print("=" * 60)
    print("TRAINING NUTRIENT DEFICIENCY MODEL (Pepper + CoLeaf)")
    print("=" * 60)

    samples = collect_nutrient_samples()
    print(f"Collected total {len(samples)} nutrient deficiency samples.")

    # Split train / val
    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = NutrientDataset(train_samples, transform=train_transform)
    val_ds = NutrientDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=0)

    num_classes = len(NUTRIENT_CLASSES)
    model = models.mobilenet_v3_small(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)

    epochs = 6
    print(f"\nTraining on CPU for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total * 100

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100 if val_total > 0 else 0
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    total_time = round(time.time() - start_time, 1)
    print(f"\nNutrient model training completed in {total_time}s!")

    model_path = os.path.join(MODELS_DIR, "nutrient_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved nutrient model weights to: {model_path}")

    classes_path = os.path.join(MODELS_DIR, "nutrient_classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump({"total_classes": num_classes, "classes": NUTRIENT_CLASSES}, f, indent=2)
    print(f"Saved nutrient classes metadata to: {classes_path}")

if __name__ == "__main__":
    train_nutrient_model()
