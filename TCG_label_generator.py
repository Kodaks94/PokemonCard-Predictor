# Label Generator for All Pokémon Card Images (No Augmentation)

import os
import pandas as pd
import glob

# --- Config ---
BASE_DIR = "pokemoncards/pokemon_card_images"
AUG_DIR = os.path.join(BASE_DIR, "augmented")
CSV_PATH = "pokemoncards/pokemon-cards.csv"
LABEL_CSV = "pokemoncards/TCG_labels_aug.csv"

# --- Load Metadata ---
df = pd.read_csv(CSV_PATH)

def safe_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

# Original images
df['filename'] = df.apply(lambda row: f"{row['id']}_{safe_filename(row['name'])}.jpg", axis=1)
df['path'] = df['filename'].apply(lambda x: os.path.join(BASE_DIR, x))
df = df[df['path'].apply(os.path.exists)].copy()
df['label'] = df['id']  # Use unique card ID as class label

original_records = df[['path', 'label']].copy()

# Augmented images (assume .jpgs inside augmented/)
aug_files = glob.glob(os.path.join(AUG_DIR, "*.jpg"))
augmented_rows = []

for file in aug_files:
    # Assume filename is random UUID: aug_xxxxxxxx.jpg, and we map it back via folder structure
    # We'll match it to the correct label using original filename pattern
    # You must already have the augmented files generated with known label assignment
    # If not, you must store label info during augmentation phase
    # Below assumes you have an 'augmented_metadata.csv' from the augment script
    pass

# --- Load augmentation metadata (if exists) ---
AUG_METADATA = "pokemoncards/augmented_metadata.csv"
if os.path.exists(AUG_METADATA):
    aug_df = pd.read_csv(AUG_METADATA)
    augmented_records = aug_df[['path', 'label']]
else:
    print("⚠️ No augmented_metadata.csv found. Only using original cards.")
    augmented_records = pd.DataFrame(columns=['path', 'label'])

# --- Combine ---
combined = pd.concat([original_records, augmented_records], ignore_index=True)
combined.to_csv(LABEL_CSV, index=False)

print(f"✅ Saved {len(combined)} records to {LABEL_CSV}")
