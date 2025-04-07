import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "pokemoncards/TCG_labels_aug.csv"
df = pd.read_csv(CSV_PATH)

# Factorise for consistency
df['label_index'] = pd.factorize(df['label'])[0]

# Count distribution
label_counts = df['label_index'].value_counts().sort_values(ascending=False)

# --- Summary Stats ---
print("📊 Dataset Summary")
print(f"Total samples: {len(df)}")
print(f"Unique classes: {len(label_counts)}")
print(f"Max class count: {label_counts.max()}")
print(f"Min class count: {label_counts.min()}")
print(f"Median count per class: {label_counts.median()}")
print(f"Mean count per class: {label_counts.mean():.2f}")
print(f"Classes with only 1 sample: {(label_counts == 1).sum()}")
print(f"Classes with less than 5 samples: {(label_counts < 5).sum()}")

# --- Plot ---
plt.figure(figsize=(14, 6))
sns.histplot(label_counts, bins=40, kde=False, color="skyblue")
plt.title("Class Distribution of Pokémon Cards")
plt.xlabel("Samples per Class")
plt.ylabel("Number of Classes")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("pokemoncards/class_distribution.png")

