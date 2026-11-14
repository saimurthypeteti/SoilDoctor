"""
SoilDoctor - Primary Disease Detection Model Trainer
Trains MobileNetV3 on PlantVillage (color) and PlantDoc datasets.
Saves weights to models/disease_model.pth and mapping to models/disease_classes.json.
"""

import os
import json
import random
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PV_DIR = os.path.join(BASE_DIR, "PlantVillage-Dataset-master", "color")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class PlantVillageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_class_info(folder_name):
    """
    Parse PlantVillage folder name into clean crop and disease labels.
    Example: Pepper,_bell___Bacterial_spot -> crop: Pepper, disease: Bacterial Spot
    """
    if "___" in folder_name:
        crop_part, disease_part = folder_name.split("___", 1)
    else:
        crop_part, disease_part = folder_name, "General"

    crop = crop_part.replace("_", " ").replace(",", "").strip()
    if "Pepper" in crop:
        crop = "Pepper (Chilli)"
    elif "Corn" in crop:
        crop = "Corn"
    elif "Cherry" in crop:
        crop = "Cherry"

    disease = disease_part.replace("_", " ").strip()
    if disease.lower() == "healthy":
        disease_name = f"{crop} Healthy"
    else:
        disease_name = f"{crop} {disease}"

    slug = folder_name.lower().replace(",", "").replace("(", "").replace(")", "").replace("___", "_").replace(" ", "_").replace("-", "_")

    return {
        "folder": folder_name,
        "crop": crop,
        "disease": disease,
        "label": disease_name,
        "slug": slug
    }

def train_disease_model():
    print("=" * 60)
    print("TRAINING PRIMARY PLANT DISEASE DETECTION MODEL (PlantVillage)")
    print("=" * 60)

    if not os.path.exists(PV_DIR):
        raise FileNotFoundError(f"PlantVillage directory not found at {PV_DIR}")

    folders = sorted([d for d in os.listdir(PV_DIR) if os.path.isdir(os.path.join(PV_DIR, d))])
    print(f"Found {len(folders)} classes in PlantVillage color dataset.")

    classes_meta = []
    class_to_idx = {}
    train_samples = []
    val_samples = []

    # Sample images per class for fast CPU convergence
    MAX_TRAIN_PER_CLASS = 40
    MAX_VAL_PER_CLASS = 10

    for idx, folder in enumerate(folders):
        info = get_class_info(folder)
        info["class_id"] = idx
        classes_meta.append(info)
        class_to_idx[folder] = idx

        folder_path = os.path.join(PV_DIR, folder)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.seed(42 + idx)
        random.shuffle(files)

        train_files = files[:MAX_TRAIN_PER_CLASS]
        val_files = files[MAX_TRAIN_PER_CLASS:MAX_TRAIN_PER_CLASS + MAX_VAL_PER_CLASS]

        for f in train_files:
            train_samples.append((os.path.join(folder_path, f), idx))
        for f in val_files:
            val_samples.append((os.path.join(folder_path, f), idx))

    print(f"Total training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(val_samples)}")

    # Image Transforms
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

    train_ds = PlantVillageDataset(train_samples, transform=train_transform)
    val_ds = PlantVillageDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Model architecture: MobileNetV3-Small
    num_classes = len(classes_meta)
    model = models.mobilenet_v3_small(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)

    epochs = 6
    print(f"\nStarting training on CPU for {epochs} epochs...")
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

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total * 100

        # Evaluate validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    total_time = round(time.time() - start_time, 1)
    print(f"\nTraining completed in {total_time}s!")

    # Save model weights and classes
    model_path = os.path.join(MODELS_DIR, "disease_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved disease model weights to: {model_path}")

    classes_path = os.path.join(MODELS_DIR, "disease_classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump({"total_classes": num_classes, "classes": classes_meta}, f, indent=2)
    print(f"Saved disease classes metadata to: {classes_path}")

if __name__ == "__main__":
    train_disease_model()
