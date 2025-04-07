import os
import pandas as pd
import requests
from tqdm import tqdm

# Paths
CSV_PATH = "pokemoncards/pokemon-cards.csv"
SAVE_DIR = "pokemoncards/pokemon_card_images"
FAILED_LOG = "pokemoncards/failed_downloads.txt"

# Setup
os.makedirs(SAVE_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# Track failures
failed = []

print(f"Starting download of {len(df)} cards...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row['image_url']
    name = row['name'].replace('/', '-')
    card_id = row['id']
    filename = f"{card_id}_{name}.jpg"
    path = os.path.join(SAVE_DIR, filename)

    if os.path.exists(path):
        continue  # Skip if already downloaded

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
        else:
            failed.append((idx, url, f"Status {response.status_code}"))
    except Exception as e:
        failed.append((idx, url, str(e)))

# Save failures
if failed:
    with open(FAILED_LOG, 'w') as f:
        for idx, url, reason in failed:
            f.write(f"{idx},{url},{reason}\n")
    print(f"{len(failed)} images failed to download. Details saved to '{FAILED_LOG}'.")
else:
    print("All images downloaded successfully.")
