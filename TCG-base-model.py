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
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
# --- Constants ---
CSV_PATH = "pokemoncards/TCG_labels_aug.csv"
MODEL_DIR = "models"
VERSION = "v_clean_reboot"
MODEL_TYPE = "EfficientNetB0"
BATCH_SIZE = 32
EPOCHS = 25
IMG_SIZE = (224, 224)
TOP_N_CLASSES = 1000
Shorten_data = True
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
# --- TF Dataset Preparation ---
AUTOTUNE = tf.data.AUTOTUNE

def get_dataset(df):
    paths = df['path'].values
    labels = df['label_index'].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    return ds

train_ds = get_dataset(train_df)
val_ds = get_dataset(val_df)

train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# --- Model Definition ---

base_model = tf.keras.applications.MobileNetV2(
    include_top=False, input_shape=IMG_SIZE + (3,), weights='imagenet', pooling='avg'
)
base_model.trainable = False  # Freeze for transfer learning
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(df['label_index'].unique()), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Training ---
checkpoint_path = os.path.join(MODEL_DIR, f"{MODEL_TYPE}_{VERSION}.h5")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    steps_per_epoch= 100,
    validation_steps= 10
)

# --- Evaluation ---
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc:.4f}")

# --- Save Final Model ---
final_model_path = os.path.join(MODEL_DIR, f"{MODEL_TYPE}_{VERSION}_final.h5")
model.save(final_model_path)
print(f"Model saved to {final_model_path}")