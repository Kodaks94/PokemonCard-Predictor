# Pokemon Card Classifier - Clean Reboot for 400+ Classes

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- Config ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- Constants ---
CSV_PATH = "pokemoncards/TCG_labels_aug.csv"
MODEL_DIR = "models"
VERSION = "v_clean_reboot"
MODEL_TYPE = "EfficientNetB0"
BATCH_SIZE = 64
EPOCHS = 25
IMG_SIZE = (224, 224)
TOP_N_CLASSES = 400
Shorten_data = False
# --- Prepare Dataset ---
df = pd.read_csv(CSV_PATH)
if Shorten_data:
    # Select top classes by frequency
    top_classes = df['label'].value_counts().head(TOP_N_CLASSES).index
    df = df[df['label'].isin(top_classes)].copy()

df['label_index'] = pd.factorize(df['label'])[0]

# Train/Val Split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label_index'], random_state=42)

# TF Datasets
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image / 255.0, label

train_ds = tf.data.Dataset.from_tensor_slices((train_df['path'].values, train_df['label_index'].values))
val_ds = tf.data.Dataset.from_tensor_slices((val_df['path'].values, val_df['label_index'].values))

train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Model ---
base_model = EfficientNetB0(include_top=False, input_shape=IMG_SIZE + (3,), weights='imagenet')
base_model.trainable = False  # Start frozen

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(df['label_index'].unique()), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# --- Callbacks ---
os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_DIR, f"best_model_{VERSION}.keras")
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
]

# --- Train ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- Save ---
model.save(os.path.join(MODEL_DIR, f"classifier_model_{VERSION}.keras"))

# --- Plot Accuracy ---
plt.plot(history.history['sparse_categorical_accuracy'], label='Train Acc')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Acc')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, f"accuracy_plot_{VERSION}.png"))
plt.close()
