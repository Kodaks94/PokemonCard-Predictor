import glob
import os
import pandas as pd

# Base and augmented directories
base_dir = "pokemoncards/pokemon_card_images"
aug_dir = os.path.join(base_dir, "augmented")

# Get all image paths including augmented
image_paths = glob.glob(os.path.join(base_dir, "*.jpg")) + glob.glob(os.path.join(aug_dir, "*.jpg"))

labels = []

for path in image_paths:
    filename = os.path.basename(path)

    # Handle original and augmented formats
    if filename.startswith("aug_"):
        # Example: aug_abc123_0.jpg → label = abc123_aug0
        parts = filename.split("_", 2)
        card_id = parts[1]
        aug_num = parts[2].replace(".jpg", "")
        label = f"{card_id}_aug{aug_num}"
    else:
        # Example: abc123_Pikachu.jpg → label = Pikachu
        _, name = filename.split("_", 1)
        label = name.replace(".jpg", "")

    labels.append({
        "path": path,
        "label": label
    })

# Create and save dataframe
df_labels = pd.DataFrame(labels)
df_labels.to_csv("pokemoncards/TCG_labels", index=False)

print(f"Labels generated for {len(df_labels)} images.")
