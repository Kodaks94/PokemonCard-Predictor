import pandas as pd
import os
import requests
from tqdm import tqdm
import tensorflow as tf
import numpy as np

print(os.getcwd())

# File paths
csv_path = "pokemoncards/pokemon-cards.csv"
output_csv = "pokemoncards/pokemon-cards-augmented.csv"

# Directories
save_dir = "pokemoncards/pokemon_card_images"
aug_dir = os.path.join(save_dir, "augmented")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(aug_dir, exist_ok=True)

# Smart skip if already done
existing_jpgs = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
existing_aug_jpgs = len([f for f in os.listdir(aug_dir) if f.endswith(".jpg")])
existing_total = existing_jpgs + existing_aug_jpgs

if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv)
    if existing_total == len(df_existing):
        print("All images and augmentations already exist. Skipping regeneration.")
        exit()

# Load metadata
df = pd.read_csv(csv_path)

def augment_image(img_tensor):
    img_tensor = tf.image.resize(img_tensor, [224, 224])
    img_tensor = tf.image.random_flip_left_right(img_tensor)
    img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.3)
    img_tensor = tf.image.random_contrast(img_tensor, lower=0.6, upper=1.4)
    img_tensor = tf.image.random_saturation(img_tensor, lower=0.6, upper=1.4)
    img_tensor = tf.clip_by_value(img_tensor, 0.0, 255.0)
    return tf.cast(img_tensor, tf.uint8)

# Track new rows
new_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row['image_url']
    name = row['name'].replace('/', '-')
    card_id = row['id']
    filename = f"{card_id}_{name}.jpg"
    path = os.path.join(save_dir, filename)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)

            # Load image and create 3 augmentations
            image_bytes = tf.io.read_file(path)
            img = tf.image.decode_jpeg(image_bytes, channels=3)

            for i in range(3):
                aug_img = augment_image(tf.identity(img))
                aug_filename = f"aug_{card_id}_{i}.jpg"
                aug_path = os.path.join(aug_dir, aug_filename)

                tf.io.write_file(aug_path, tf.image.encode_jpeg(aug_img))

                new_row = row.copy()
                new_row['id'] = f"aug_{card_id}_{i}"
                new_row['name'] = f"{name}_aug{i}"
                new_row['image_url'] = "N/A"
                new_row['augmented_path'] = aug_path
                new_rows.append(new_row)

    except Exception as e:
        print(f" Failed to download {url}: {e}")

# Save final CSV
aug_df = pd.DataFrame(new_rows)
full_df = pd.concat([df, aug_df], ignore_index=True)
full_df.to_csv(output_csv, index=False)
print(f" Augmented data saved to: {output_csv}")
