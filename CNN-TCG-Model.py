# Pokemon Card Similarity Model - Triplet Loss with Pretrained Features

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# --- Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
CSV_PATH = "pokemoncards/TCG_labels_aug.csv"
MODEL_DIR = "models"
VERSION = "v_triplet_1"
MODEL_TYPE = "EfficientNetB0_Triplet"
BATCH_SIZE = 64
EPOCHS = 20
IMG_SIZE = (224, 224)
EMBED_DIM = 128
TOP_N_CLASSES = 200

# --- Load and Reduce Dataset ---
df = pd.read_csv(CSV_PATH)
top_classes = df['label'].value_counts().head(TOP_N_CLASSES).index
df = df[df['label'].isin(top_classes)].copy()
df['label_index'] = pd.factorize(df['label'])[0]
label_to_index = dict(zip(df['label'], df['label_index']))

# --- Group by Label ---
grouped = df.groupby('label')

# --- Generate Triplets (anchor, positive, negative) ---
def make_triplets(df, num_triplets=10000):
    anchors, positives, negatives = [], [], []
    labels = df['label'].unique()
    for _ in range(num_triplets):
        pos_label = np.random.choice(labels)
        neg_label = np.random.choice(labels)
        while neg_label == pos_label:
            neg_label = np.random.choice(labels)
        pos_samples = grouped.get_group(pos_label)
        neg_samples = grouped.get_group(neg_label)
        a, p = pos_samples.sample(2).path.values
        n = neg_samples.sample(1).path.values[0]
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
    return pd.DataFrame({"anchor": anchors, "positive": positives, "negative": negatives})

triplet_df = make_triplets(df)

# --- Image Loader ---
def load_triplet(a_path, p_path, n_path):
    def load_img(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        return image / 255.0
    return load_img(a_path), load_img(p_path), load_img(n_path)

def load_triplet_map(a, p, n):
    a_img, p_img, n_img = load_triplet(a, p, n)
    return a_img, p_img, n_img

triplet_ds = tf.data.Dataset.from_tensor_slices((
    triplet_df['anchor'].values,
    triplet_df['positive'].values,
    triplet_df['negative'].values
))



def process_triplet(a, p, n):
    a_img, p_img, n_img = tf.py_function(
        func=load_triplet_map,
        inp=[a, p, n],
        Tout=[tf.float32, tf.float32, tf.float32]
    )

    # Set known shapes for each image
    a_img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    p_img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    n_img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))

    return {"anchor": a_img, "positive": p_img, "negative": n_img}, tf.zeros(())


triplet_ds = triplet_ds.map(process_triplet, num_parallel_calls=tf.data.AUTOTUNE)
triplet_ds = triplet_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --- Triplet Loss ---
def triplet_loss(margin=0.3):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return loss
    return loss

# --- Embedding Model ---
def build_embedding_model():
    base_model = EfficientNetB0(include_top=False, input_shape=IMG_SIZE + (3,), weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(EMBED_DIM, activation='relu'),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])
    return model

embedding_net = build_embedding_model()

# --- Full Triplet Model Wrapper ---
anchor_input = layers.Input(shape=IMG_SIZE + (3,), name="anchor")
positive_input = layers.Input(shape=IMG_SIZE + (3,), name="positive")
negative_input = layers.Input(shape=IMG_SIZE + (3,), name="negative")

encoded_a = embedding_net(anchor_input)
encoded_p = embedding_net(positive_input)
encoded_n = embedding_net(negative_input)

merged_output = layers.Concatenate(axis=1)([encoded_a[:, tf.newaxis, :], encoded_p[:, tf.newaxis, :], encoded_n[:, tf.newaxis, :]])
model = tf.keras.Model(inputs={"anchor": anchor_input, "positive": positive_input, "negative": negative_input}, outputs=merged_output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=triplet_loss())

# --- Callbacks ---
os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_DIR, "triplet_model.keras")
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
]

# --- Training ---
history = model.fit(triplet_ds, epochs=EPOCHS, callbacks=callbacks)

# --- Save Embedding Model ---
embedding_net.save(os.path.join(MODEL_DIR, f"embedding_net_{VERSION}.keras"))