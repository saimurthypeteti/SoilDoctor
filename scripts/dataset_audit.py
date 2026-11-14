"""
SoilDoctor Comprehensive Dataset and Architecture Audit
Inspects:
- Old datasets & model files to remove/deprecate
- PlantVillage (color)
- PlantDoc (train, test)
- Pepper nutrient dataset (images & XML labels)
- CoLeaf dataset (zip archives)
- Old dataset references across Python and JS codebase
"""

import os
import json
import zipfile
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def audit_plantvillage(pv_dir):
    print("=" * 60)
    print("1. PlantVillage Dataset Audit")
    print("=" * 60)
    color_dir = os.path.join(pv_dir, "color")
    if not os.path.exists(color_dir):
        print(f"PlantVillage color dir not found at: {color_dir}")
        return {}
    
    classes = [d for d in os.listdir(color_dir) if os.path.isdir(os.path.join(color_dir, d))]
    print(f"Total classes in PlantVillage (color): {len(classes)}")
    
    total_imgs = 0
    class_counts = {}
    corrupt_count = 0
    
    for c in sorted(classes):
        cp = os.path.join(color_dir, c)
        imgs = [f for f in os.listdir(cp) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[c] = len(imgs)
        total_imgs += len(imgs)
        
    print(f"Total images in PlantVillage: {total_imgs}")
    print("\nClasses and counts (sample 10):")
    for c, cnt in list(class_counts.items())[:10]:
        print(f"  {c}: {cnt}")
    print(f"  ... and {len(classes)-10} more classes")
    
    return {"total_classes": len(classes), "total_images": total_imgs, "class_counts": class_counts}

def audit_plantdoc(train_dir, test_dir):
    print("\n" + "=" * 60)
    print("2. PlantDoc Dataset Audit (Real-World Field Dataset)")
    print("=" * 60)
    
    train_counts = {}
    test_counts = {}
    
    if os.path.exists(train_dir):
        t_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        for c in sorted(t_classes):
            imgs = [f for f in os.listdir(os.path.join(train_dir, c)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            train_counts[c] = len(imgs)
            
    if os.path.exists(test_dir):
        te_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        for c in sorted(te_classes):
            imgs = [f for f in os.listdir(os.path.join(test_dir, c)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            test_counts[c] = len(imgs)
            
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    print(f"PlantDoc Train: {len(train_counts)} classes, {total_train} images")
    print(f"PlantDoc Test:  {len(test_counts)} classes, {total_test} images")
    
    print("\nTop 8 Train Classes:")
    for c, cnt in list(train_counts.items())[:8]:
        print(f"  {c}: {cnt} train, {test_counts.get(c, 0)} test")
        
    return {"train_classes": len(train_counts), "train_images": total_train, "test_classes": len(test_counts), "test_images": total_test}

def audit_pepper(pepper_dir):
    print("\n" + "=" * 60)
    print("3. Pepper Nutrient Deficiency Dataset Audit")
    print("=" * 60)
    
    photo_dir = os.path.join(pepper_dir, "pepper_photo")
    labels_dir = os.path.join(pepper_dir, "Labels")
    
    photo_count = len(os.listdir(photo_dir)) if os.path.exists(photo_dir) else 0
    xml_count = len(os.listdir(labels_dir)) if os.path.exists(labels_dir) else 0
    
    print(f"Total Pepper Photos: {photo_count}")
    print(f"Total Label XMLs:    {xml_count}")
    
    counts = Counter()
    tag_re = re.compile(r'<name>(.*?)</name>')
    
    if os.path.exists(labels_dir):
        xml_files = [f for f in os.listdir(labels_dir) if f.endswith('.xml')]
        for f in xml_files:
            with open(os.path.join(labels_dir, f), 'r', encoding='utf-8', errors='ignore') as fp:
                for n in tag_re.findall(fp.read()):
                    counts[n.strip()] += 1
                    
    print("\nNutrient Deficiencies & Classes found in Pepper Labels:")
    for name, cnt in counts.most_common():
        print(f"  {name}: {cnt} bounding boxes")
        
    return {"photo_count": photo_count, "xml_count": xml_count, "label_counts": dict(counts)}

def audit_coleaf(coleaf_dir):
    print("\n" + "=" * 60)
    print("4. CoLeaf Dataset Audit (Nutrient Deficiencies)")
    print("=" * 60)
    
    if not os.path.exists(coleaf_dir):
        print(f"CoLeaf directory not found at: {coleaf_dir}")
        return {}
        
    zips = [f for f in os.listdir(coleaf_dir) if f.endswith('.zip')]
    print(f"Found {len(zips)} zip packages in CoLeaf dataset:")
    
    total_imgs = 0
    coleaf_counts = {}
    for z in sorted(zips):
        zp = os.path.join(coleaf_dir, z)
        try:
            with zipfile.ZipFile(zp, 'r') as zf:
                imgs = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_imgs += len(imgs)
                name = z.replace('.zip', '')
                coleaf_counts[name] = len(imgs)
                print(f"  {name}: {len(imgs)} images")
        except Exception as e:
            print(f"  Error reading {z}: {e}")
            
    print(f"Total CoLeaf Images: {total_imgs}")
    return {"zip_count": len(zips), "total_images": total_imgs, "counts": coleaf_counts}

def audit_old_artifacts():
    print("\n" + "=" * 60)
    print("5. Old Datasets & Models Audit (To Deprecate/Remove)")
    print("=" * 60)
    
    old_items = [
        ("Dataset new", "Legacy 6-crop dataset (BCSS, Rice, Tomata)"),
        ("dataset", "Old split dataset directory"),
        ("trained_model.pth", "Old 6-crop PyTorch MobileNetV3 model trained on 25 imgs/class"),
        ("class_mapping.json", "Old 33-class mapping for legacy 6-crop dataset"),
        ("train_disease_model.py", "Old training script referencing 'Dataset new'"),
        ("soil_data.csv", "Synthetic soil CSV"),
        ("soil_data_cleaned.csv", "Synthetic soil cleaned CSV"),
        ("create_soil_dataset.py", "Old synthetic soil generator"),
        ("generate_dataset.py", "Old synthetic data generator")
    ]
    
    for item, desc in old_items:
        p = os.path.join(BASE_DIR, item)
        exists = os.path.exists(p)
        print(f"  {'[FOUND]' if exists else '[MISSING]'}: {item} ({desc})")

if __name__ == "__main__":
    print("STARTING SOILDOCTOR COMPREHENSIVE DATASET AUDIT\n")
    audit_old_artifacts()
    pv_res = audit_plantvillage(os.path.join(BASE_DIR, "PlantVillage-Dataset-master"))
    pd_res = audit_plantdoc(os.path.join(BASE_DIR, "train"), os.path.join(BASE_DIR, "test"))
    pep_res = audit_pepper(os.path.join(BASE_DIR, "pepper"))
    col_res = audit_coleaf(os.path.join(BASE_DIR, "CoLeaf DATASET"))
    print("\nAUDIT FINISHED SUCCESSFULLY.")
