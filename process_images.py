import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = 'archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'  # <- UPDATE IF NEEDED
CLASSES = {'benign': 0, 'malignant': 1}
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]
N_PER_CLASS = 200  #Changing this for fewer/more images

def collect_image_paths(label_str):
    root = os.path.join(DATA_ROOT, label_str)
    return glob.glob(os.path.join(root, '**', '*.png'), recursive=True)

def extract_magnification(path):
    return os.path.basename(os.path.dirname(path))

def extract_patient_id(path):
    # e.g. .../SOB_B_A_14-22549AB/40X/image.png
    parts = path.split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"

def process_for_magnification(mag):
    rows = []
    for label_str, label in CLASSES.items():
        all_paths = collect_image_paths(label_str)
        mag_paths = [p for p in all_paths if os.sep + mag + os.sep in p]
        if not mag_paths:
            print(f"WARNING: No images found for {mag} {label_str}!")
        mag_paths = mag_paths[:N_PER_CLASS]  #Taking only first N_PER_CLASS
        for path in mag_paths:
            patient = extract_patient_id(path)
            rows.append((path, label, patient))
    if not rows:
        print(f"WARNING: No images collected for {mag}")
        return
    df = pd.DataFrame(rows, columns=['filepath', 'label', 'patient'])
    patients = df['patient'].unique()
    if len(patients) < 2:
        print(f"Not enough patients to split for {mag}!")
        return
    train_pat, temp_pat = train_test_split(patients, test_size=0.3, random_state=42)
    val_pat, test_pat = train_test_split(temp_pat, test_size=0.5, random_state=42)
    df['split'] = df['patient'].map(lambda x: 'train' if x in train_pat else 'val' if x in val_pat else 'test')
    for split in ['train', 'val', 'test']:
        df_split = df[df['split'] == split]
        df_split[['filepath', 'label']].to_csv(f"{mag}_{split}_mini.csv", index=False)
        print(f"{mag} {split}: {len(df_split)} images")
    print(f"{mag}: Done.")

if __name__ == "__main__":
    for mag in MAGNIFICATIONS:
        process_for_magnification(mag)
