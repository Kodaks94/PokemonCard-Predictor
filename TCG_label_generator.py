import os
import glob
import pandas as pd

# Directories
base_dir = "pokemoncards/pokemon_card_images"
aug_dir = os.path.join(base_dir, "augmented")

# Get all image paths
base_images = glob.glob(os.path.join(base_dir, "*.jpg"))
aug_images = glob.glob(os.path.join(aug_dir, "*.jpg"))
all_images = base_images + aug_images

print(f"Found {len(base_images)} base images and {len(aug_images)} augmented images.")

# Helper to extract label from filename
def extract_label(path):
    filename = os.path.basename(path)
    if filename.startswith("aug_"):
        return filename.split("_", 1)[1].replace(".jpg", "")  # keep the part after 'aug_'
    else:
        parts = filename.split("_", 1)
        return parts[1].replace(".jpg", "") if len(parts) > 1 else filename.replace(".jpg", "")

# Build dataframe
records = []
for path in all_images:
    label = extract_label(path)
    records.append({"path": path, "label": label})

df = pd.DataFrame(records)
df.to_csv("pokemoncards/TCG_labels_aug.csv", index=False)
print(f"Saved {len(df)} labels to pokemoncards/TCG_labels_aug.csv")
