# Pokemon Card Classifier - Full Training Pipeline (Refactored with Skip Augment if Already Done)

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
import uuid

mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
CSV_PATH = "pokemoncards/TCG_labels"
MODEL_DIR = "models"
VERSION = "v3.0"
MODEL_TYPE = "MobileNetV2_BiLSTM"
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = (224, 224)
AUG_DIR = "pokemoncards/augmented"
os.makedirs(AUG_DIR, exist_ok=True)

# --- Data Loading ---
df = pd.read_csv(CSV_PATH)
labels_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label_index'] = df['label'].map(labels_to_index)
num_classes = len(labels_to_index)

# --- Augment Singleton Classes (only once) ---
singleton_labels = df['label_index'].value_counts()[df['label_index'].value_counts() == 1].index
already_augmented = [f for f in os.listdir(AUG_DIR) if f.endswith(".jpg")]

if len(already_augmented) < len(singleton_labels) * 3:
    print("Augmenting singleton samples...")
    augmentor = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
    ])

    augmented_rows = []
    for _, row in df[df['label_index'].isin(singleton_labels)].iterrows():
        original_path = row['path']
        label = row['label']
        label_index = row['label_index']

        image_raw = tf.io.read_file(original_path)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)

        for i in range(3):
            augmented = augmentor(image, training=True)[0].numpy()
            augmented = tf.image.convert_image_dtype(augmented, tf.uint8)
            new_filename = f"aug_{uuid.uuid4().hex[:8]}.jpg"
            save_path = os.path.join(AUG_DIR, new_filename)
            tf.io.write_file(save_path, tf.image.encode_jpeg(augmented))

            augmented_rows.append({
                "path": save_path,
                "label": label,
                "label_index": label_index
            })

    df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
else:
    print("Augmented images already present. Skipping augmentation.")

# --- Train/Test Split ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_index'])

# --- Dataset ---
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_df['path'].values, train_df['label_index'].values))
val_ds = tf.data.Dataset.from_tensor_slices((val_df['path'].values, val_df['label_index'].values))

train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Model ---
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax', dtype='float32')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True)
]

# --- Train ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- Plots ---
def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "accuracy_plot.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))
    plt.close()

plot_metrics(history)

# --- Evaluation ---
y_true, y_pred = [], []
for x_batch, y_batch in val_ds:
    preds = model.predict(x_batch)
    y_true.extend(y_batch.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

actual_labels = sorted(list(set(y_true) | set(y_pred)))
actual_label_names = [label for label, idx in labels_to_index.items() if idx in actual_labels]

report = classification_report(
    y_true, y_pred,
    labels=actual_labels,
    target_names=actual_label_names,
    zero_division=0
)

with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# --- Save Model ---
val_acc = history.history['val_accuracy'][-1]
val_acc_percent = int(val_acc * 1000) / 10.0
model_name = f"pokemon_{MODEL_TYPE}_{VERSION}_{val_acc_percent:.1f}acc.keras"
model.save(os.path.join(MODEL_DIR, model_name))