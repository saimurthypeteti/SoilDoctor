"""
SoilDoctor - CNN Disease Model Training with Transfer Learning
Trains MobileNetV3 on the NEW dataset (Tomato, Rice, Chilli, Sugarcane, Blackgram, Soybean)
Produces:
- trained_model.pth (PyTorch weights)
- class_mapping.json (Index -> Class info)
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset new")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_model.pth")
MAPPING_SAVE_PATH = os.path.join(BASE_DIR, "class_mapping.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All classes and their source folders in Dataset new
CLASS_DIRS = [
    # Tomato (8 classes)
    {"crop": "Tomato", "label": "Tomato Early Blight", "slug": "tomato_early_blight", "path": os.path.join("Tomata Dataset", "train", "Early_blight")},
    {"crop": "Tomato", "label": "Tomato Healthy", "slug": "tomato_healthy", "path": os.path.join("Tomata Dataset", "train", "Healthy")},
    {"crop": "Tomato", "label": "Tomato Late Blight", "slug": "tomato_late_blight", "path": os.path.join("Tomata Dataset", "train", "Late_blight")},
    {"crop": "Tomato", "label": "Tomato Leaf Miner", "slug": "tomato_leaf_miner", "path": os.path.join("Tomata Dataset", "train", "Leaf Miner")},
    {"crop": "Tomato", "label": "Tomato Magnesium Deficiency", "slug": "tomato_magnesium_deficiency", "path": os.path.join("Tomata Dataset", "train", "Magnesium Deficiency")},
    {"crop": "Tomato", "label": "Tomato Nitrogen Deficiency", "slug": "tomato_nitrogen_deficiency", "path": os.path.join("Tomata Dataset", "train", "Nitrogen Deficiency")},
    {"crop": "Tomato", "label": "Tomato Potassium Deficiency", "slug": "tomato_potassium_deficiency", "path": os.path.join("Tomata Dataset", "train", "Pottassium Deficiency")},
    {"crop": "Tomato", "label": "Tomato Spotted Wilt Virus", "slug": "tomato_spotted_wilt_virus", "path": os.path.join("Tomata Dataset", "train", "Spotted Wilt Virus")},

    # Rice (8 classes)
    {"crop": "Rice", "label": "Rice Bacterial Leaf Blight", "slug": "rice_bacterial_leaf_blight", "path": os.path.join("Rice leaf disease", "Training data", "Bacterial Leaf Blight")},
    {"crop": "Rice", "label": "Rice Brown Spot", "slug": "rice_brown_spot", "path": os.path.join("Rice leaf disease", "Training data", "Brown Spot")},
    {"crop": "Rice", "label": "Rice Healthy Leaf", "slug": "rice_healthy", "path": os.path.join("Rice leaf disease", "Training data", "Healthy Rice Leaf")},
    {"crop": "Rice", "label": "Rice Leaf Blast", "slug": "rice_leaf_blast", "path": os.path.join("Rice leaf disease", "Training data", "Leaf Blast")},
    {"crop": "Rice", "label": "Rice Leaf Scald", "slug": "rice_leaf_scald", "path": os.path.join("Rice leaf disease", "Training data", "Leaf scald")},
    {"crop": "Rice", "label": "Rice Narrow Brown Leaf Spot", "slug": "rice_narrow_brown_leaf_spot", "path": os.path.join("Rice leaf disease", "Training data", "Narrow Brown Leaf Spot")},
    {"crop": "Rice", "label": "Rice Hispa Damage", "slug": "rice_rice_hispa", "path": os.path.join("Rice leaf disease", "Training data", "Rice Hispa")},
    {"crop": "Rice", "label": "Rice Sheath Blight", "slug": "rice_sheath_blight", "path": os.path.join("Rice leaf disease", "Training data", "Sheath Blight")},

    # Chilli (5 classes)
    {"crop": "Chilli", "label": "Chilli Cercospora Leaf Spot", "slug": "chilli_cercospora", "path": os.path.join("BCSS Dataset", "dataset", "chili_cercospora")},
    {"crop": "Chilli", "label": "Chilli Healthy Leaf", "slug": "chilli_healthy", "path": os.path.join("BCSS Dataset", "dataset", "chili_healthy")},
    {"crop": "Chilli", "label": "Chilli Mites & Thrips", "slug": "chilli_mites_and_trips", "path": os.path.join("BCSS Dataset", "dataset", "chili_mites_and_trips")},
    {"crop": "Chilli", "label": "Chilli Nutritional Deficiency", "slug": "chilli_nutritional", "path": os.path.join("BCSS Dataset", "dataset", "chili_nutritional")},
    {"crop": "Chilli", "label": "Chilli Powdery Mildew", "slug": "chilli_powdery_mildew", "path": os.path.join("BCSS Dataset", "dataset", "chili_powdery mildew")},

    # Sugarcane (5 classes)
    {"crop": "Sugarcane", "label": "Sugarcane Healthy Leaf", "slug": "sugarcane_healthy", "path": os.path.join("BCSS Dataset", "dataset", "sugercane_Healthy")},
    {"crop": "Sugarcane", "label": "Sugarcane Mosaic Virus", "slug": "sugarcane_mosaic", "path": os.path.join("BCSS Dataset", "dataset", "sugercane_Mosaic")},
    {"crop": "Sugarcane", "label": "Sugarcane Red Rot", "slug": "sugarcane_red_rot", "path": os.path.join("BCSS Dataset", "dataset", "sugercane_RedRot")},
    {"crop": "Sugarcane", "label": "Sugarcane Rust", "slug": "sugarcane_rust", "path": os.path.join("BCSS Dataset", "dataset", "sugercane_Rust")},
    {"crop": "Sugarcane", "label": "Sugarcane Yellow Leaf", "slug": "sugarcane_yellow_leaf", "path": os.path.join("BCSS Dataset", "dataset", "sugercane_Yellow")},

    # Blackgram (5 classes)
    {"crop": "Blackgram", "label": "Blackgram Anthracnose", "slug": "blackgram_anthracnose", "path": os.path.join("BCSS Dataset", "dataset", "blackgram_Anthracnose")},
    {"crop": "Blackgram", "label": "Blackgram Healthy Leaf", "slug": "blackgram_healthy", "path": os.path.join("BCSS Dataset", "dataset", "blackgram_Healthy")},
    {"crop": "Blackgram", "label": "Blackgram Leaf Crinkle", "slug": "blackgram_leaf_crinkle", "path": os.path.join("BCSS Dataset", "dataset", "blackgram_LeafCrinckle")},
    {"crop": "Blackgram", "label": "Blackgram Powdery Mildew", "slug": "blackgram_powdery_mildew", "path": os.path.join("BCSS Dataset", "dataset", "blackgram_PowderyMildew")},
    {"crop": "Blackgram", "label": "Blackgram Yellow Mosaic", "slug": "blackgram_yellow_mosaic", "path": os.path.join("BCSS Dataset", "dataset", "blackgram_YellowMosaic")},

    # Soybean (2 classes)
    {"crop": "Soybean", "label": "Soybean Leaf Disease", "slug": "soybean_disease_soybean", "path": os.path.join("BCSS Dataset", "dataset", "Disease_soybean")},
    {"crop": "Soybean", "label": "Soybean Healthy Leaf", "slug": "soybean_healthy", "path": os.path.join("BCSS Dataset", "dataset", "Healthy_soybean")}
]

class BalancedCropDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback black image if corrupt
            image = Image.new("RGB", (224, 224), (0, 100, 0))

        if self.transform:
            image = self.transform(image)
        return image, label_idx

def build_data_splits(max_train_per_class=25, max_val_per_class=6):
    train_samples = []
    val_samples = []
    
    random.seed(42)

    for idx, cinfo in enumerate(CLASS_DIRS):
        folder_path = os.path.join(DATASET_DIR, cinfo["path"])
        if not os.path.exists(folder_path):
            print(f"Warning: path missing: {folder_path}")
            continue

        images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(images)

        n_train = min(len(images), max_train_per_class)
        n_val = min(max(0, len(images) - n_train), max_val_per_class)

        for p in images[:n_train]:
            train_samples.append((p, idx))
        for p in images[n_train:n_train + n_val]:
            val_samples.append((p, idx))

    return train_samples, val_samples

def train_model():
    print(f"Using compute device: {DEVICE}")
    train_samples, val_samples = build_data_splits(max_train_per_class=200, max_val_per_class=40)
    print(f"Total training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = BalancedCropDataset(train_samples, transform=train_transforms)
    val_ds = BalancedCropDataset(val_samples, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Transfer Learning with MobileNetV3 Small
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Unfreeze last 4 feature blocks for better crop-level discrimination
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    num_ftrs = model.classifier[3].in_features
    num_classes = len(CLASS_DIRS)
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    epochs = 12
    print(f"\n--- Training MobileNetV3 on New Dataset ({num_classes} Classes, {epochs} Epochs) ---")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_ds)
        train_acc = running_corrects.double() / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_ds) if len(val_ds) > 0 else 0
        val_acc = (val_corrects.double() / len(val_ds)) if len(val_ds) > 0 else 0

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
        scheduler.step()

    # Save model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained model saved to: {MODEL_SAVE_PATH}")

    # Save class mapping
    mapping = {
        "classes": CLASS_DIRS,
        "num_classes": num_classes
    }
    with open(MAPPING_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Class mapping saved to: {MAPPING_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
