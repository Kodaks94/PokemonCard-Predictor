# Baseline CNN Classifier for Pokémon Cards (Card ID–Based)

import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# --- Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
IMG_SIZE = (224, 224)
CSV_PATH = "pokemoncards/TCG_labels_aug.csv"
BATCH_SIZE = 32
EPOCHS = 10
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(CSV_PATH)

# Use card ID directly as label (was already saved this way in the new pipeline)
df['label_index'] = pd.factorize(df['label'])[0]
num_classes = df['label_index'].nunique()

from sklearn.utils import shuffle

# Group by class
from sklearn.utils import shuffle

val_samples = []
train_samples = []

for label, group in df.groupby('label_index'):
    group = shuffle(group, random_state=42)
    val_samples.append(group.iloc[0].copy())      # validation sample
    train_samples.extend(group.iloc[1:].copy().values.tolist())   # rest for training

val_df = pd.DataFrame(val_samples)
train_df = pd.DataFrame(train_samples, columns=df.columns)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")


# --- TF Dataset ---
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image / 255.0, label

train_ds = tf.data.Dataset.from_tensor_slices((train_df['path'].values, train_df['label_index'].values))
val_ds = tf.data.Dataset.from_tensor_slices((val_df['path'].values, val_df['label_index'].values))

train_ds = train_ds.map(load_image).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Model ---
model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Train ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# --- Evaluation ---
y_true, y_pred = [], []
for x_batch, y_batch in val_ds:
    preds = model.predict(x_batch)
    y_true.extend(y_batch.numpy())
    y_pred.extend(tf.argmax(preds, axis=1).numpy())

report = classification_report(y_true, y_pred, zero_division=0)
with open(os.path.join(MODEL_DIR, "baseline_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# --- Save Model ---
model.save(os.path.join(MODEL_DIR, "baseline_model.keras"))

# --- Plot ---
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Baseline Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, "baseline_accuracy.png"))
plt.close()
