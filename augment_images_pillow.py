# PIL Augmentation for All Pokémon Cards (Realistic Scan Simulation)

import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import random
from tqdm import tqdm
import uuid

# --- Config ---
IMG_SIZE = (224, 224)
BASE_DIR = "pokemoncards/pokemon_card_images"
AUG_DIR = os.path.join(BASE_DIR, "augmented")
CSV_PATH = "pokemoncards/pokemon-cards.csv"
AUG_METADATA = "pokemoncards/augmented_metadata.csv"

# --- Setup ---
os.makedirs(AUG_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

def safe_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

df['filename'] = df.apply(lambda row: f"{row['id']}_{safe_filename(row['name'])}.jpg", axis=1)
df['path'] = df['filename'].apply(lambda x: os.path.join(BASE_DIR, x))
df = df[df['path'].apply(os.path.exists)].copy()
df['label'] = df['id']  # Use unique ID as label

# --- Augmentation Functions ---

def add_camera_effects(img):
    # Slight rotation
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    # Random mirror
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # Resize again to crop center if size changed
    img = img.resize(IMG_SIZE)

    # Colour effects
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))

    return img

# --- Augment ---
print(f"Found {len(df)} images. Augmenting each image 3x (realistic scan style)...")
augmented_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    original_path = row['path']
    try:
        img = Image.open(original_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {original_path}: {e}")
        continue

    for i in range(5):
        aug_img = img.copy()
        aug_img = add_camera_effects(aug_img)
        aug_filename = f"aug_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(AUG_DIR, aug_filename)
        aug_img.save(save_path)

        augmented_rows.append({
            "path": save_path,
            "label": row['label']
        })

# --- Save Metadata ---
aug_df = pd.DataFrame(augmented_rows)
aug_df.to_csv(AUG_METADATA, index=False)

print(f" Done. Augmented {len(aug_df)} images.")
print(f" Saved metadata to {AUG_METADATA}")