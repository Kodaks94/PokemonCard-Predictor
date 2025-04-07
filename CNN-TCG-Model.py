# Pokemon Card Classifier - Final Refined Pipeline (Updated for Unique ID Labels)

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.utils import shuffle

# --- Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
mixed_precision.set_global_policy('float32')

# --- Configuration ---
CSV_PATH = "pokemoncards/TCG_labels_aug.csv"
MODEL_DIR = "models"
VERSION = "v4.0"
MODEL_TYPE = "EfficientNetV2B1_FocalLoss"
BATCH_SIZE = 64
EPOCHS = 40
IMG_SIZE = (224, 224)

# --- Load and Clean Dataset ---
df = pd.read_csv(CSV_PATH)
df['label_index'] = pd.factorize(df['label'])[0]
labels_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}
num_classes = len(labels_to_index)

# --- Manual 1-per-class validation split ---
val_samples = []
train_samples = []

for label, group in df.groupby('label_index'):
    group = shuffle(group, random_state=42)
    val_samples.append(group.iloc[0].copy())
    train_samples.extend(group.iloc[1:].copy().values.tolist())

val_df = pd.DataFrame(val_samples)
train_df = pd.DataFrame(train_samples, columns=df.columns)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# --- Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1)
])

# --- Dataset Loader ---
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_df['path'].values, train_df['label_index'].values))
val_ds = tf.data.Dataset.from_tensor_slices((val_df['path'].values, val_df['label_index'].values))

train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Focal Loss Function ---
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return loss_fn

# --- Model Architecture ---
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-5
)

base_model = EfficientNetV2B1(include_top=False, input_shape=IMG_SIZE + (3,), weights='imagenet')
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['sparse_categorical_accuracy']
)

# --- Callbacks ---
os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True),
    lr_scheduler
]

# --- Training ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- Plotting ---
def plot_metrics(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
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

with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# --- Save Model ---
val_acc = history.history['val_sparse_categorical_accuracy'][-1]
val_acc_percent = int(val_acc * 1000) / 10.0
model_name = f"pokemon_{MODEL_TYPE}_{VERSION}_{val_acc_percent:.1f}acc.keras"
model.save(os.path.join(MODEL_DIR, model_name))
