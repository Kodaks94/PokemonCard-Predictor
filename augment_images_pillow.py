# PIL Augmentation for Singleton Pokémon Cards

import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import random
from tqdm import tqdm
import uuid

IMG_SIZE = (224, 224)
BASE_DIR = "pokemoncards/pokemon_card_images"
AUG_DIR = os.path.join(BASE_DIR, "augmented")
os.makedirs(AUG_DIR, exist_ok=True)

# Load original CSV (not the augmented one)
csv_path = "pokemoncards/pokemon-cards.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError(f"Required file not found: {csv_path}")

# Generate filename and path column to match images

def safe_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

df['filename'] = df.apply(lambda row: f"{row['id']}_{safe_filename(row['name'])}.jpg", axis=1)
df['path'] = df['filename'].apply(lambda x: os.path.join(BASE_DIR, x))

# Only keep rows where file exists
existing_df = df[df['path'].apply(os.path.exists)].copy()
existing_df['label_index'] = pd.factorize(existing_df['name'])[0]

# Singleton detection
label_counts = existing_df['label_index'].value_counts()
singleton_labels = label_counts[label_counts == 1].index
singleton_df = existing_df[existing_df['label_index'].isin(singleton_labels)]

print(f"Singleton classes: {len(singleton_df)}")
augmented_rows = []

def safe_augment_pil(img):
    img = img.resize(IMG_SIZE)

    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))

    return img

print("Augmenting singleton images using PIL...")
for _, row in tqdm(singleton_df.iterrows(), total=len(singleton_df)):
    original_path = row['path']

    try:
        img = Image.open(original_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {original_path}: {e}")
        continue

    for i in range(3):
        aug_img = safe_augment_pil(img.copy())
        aug_filename = f"aug_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(AUG_DIR, aug_filename)
        aug_img.save(save_path)

        augmented_rows.append({
            "path": save_path,
            "label": row['name'],
            "label_index": row['label_index']
        })

print(f"Done. Augmented {len(augmented_rows)} images.")

# Save augmented metadata for later merging
aug_df = pd.DataFrame(augmented_rows)
aug_df.to_csv("pokemoncards/augmented_metadata.csv", index=False)
print("Saved augmented metadata to pokemoncards/augmented_metadata.csv")