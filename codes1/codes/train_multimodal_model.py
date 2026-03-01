import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, Concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# =========================
# GPU CONFIGURATION
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🟢 GPU detected:", gpus)
    except RuntimeError as e:
        print("⚠ GPU setup error:", e)
else:
    print("⚠ No GPU detected. Running on CPU.")

# Enable Mixed Precision (RTX 3050 Tensor Cores)
mixed_precision.set_global_policy('mixed_float16')
print("⚡ Mixed precision enabled")

# =========================
# PATHS
# =========================
BASE_DIR = r"E:\Final Year\files1\files"

IMAGE_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "image_embeddings.npy")
TEXT_SEQUENCES_PATH = os.path.join(BASE_DIR, "input_sequences.npy")
TARGET_WORDS_PATH = os.path.join(BASE_DIR, "target_words.npy")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")
MAX_SEQ_PATH = os.path.join(BASE_DIR, "max_sequence_length.txt")

# =========================
# LOAD DATA
# =========================
print("🔹 Loading embeddings...")

image_embeddings = np.load(IMAGE_EMBEDDINGS_PATH)
text_sequences = np.load(TEXT_SEQUENCES_PATH)
target_words = np.load(TARGET_WORDS_PATH)

print("Image embeddings shape:", image_embeddings.shape)
print("Text sequences shape:", text_sequences.shape)
print("Target words shape:", target_words.shape)

# =========================
# LOAD TOKENIZER
# =========================
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

word_index = tokenizer_data["config"]["word_index"]
vocab_size = len(word_index) + 1

print("Vocabulary size:", vocab_size)

# =========================
# LOAD MAX SEQUENCE LENGTH
# =========================
with open(MAX_SEQ_PATH, "r") as f:
    max_seq_len = int(f.read().strip())

print("Max sequence length:", max_seq_len)

# =========================
# MODEL ARCHITECTURE
# =========================

# Image branch
image_input = Input(shape=(2048,), name="image_input")
image_dense = Dense(256, activation="relu")(image_input)
image_dense = Dropout(0.3)(image_dense)

# Text branch
text_input = Input(shape=(max_seq_len,), name="text_input")
text_embedding = Embedding(
    input_dim=vocab_size,
    output_dim=256,
    mask_zero=True
)(text_input)

text_lstm = LSTM(256)(text_embedding)

# Fusion
merged = Concatenate()([image_dense, text_lstm])
dense_1 = Dense(256, activation="relu")(merged)

# IMPORTANT: dtype float32 for numerical stability in mixed precision
output = Dense(
    vocab_size,
    activation="softmax",
    dtype="float32"
)(dense_1)

# Build model
model = Model(inputs=[image_input, text_input], outputs=output)

# =========================
# COMPILE (Backprop Enabled Here)
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
print("🚀 Training started...")

with tf.device('/GPU:0'):
    history = model.fit(
        [image_embeddings, text_sequences],
        target_words,
        epochs=20,
        batch_size=8,  # Safe for RTX 3050 (4GB)
        validation_split=0.1,
        shuffle=True
    )

# =========================
# SAVE MODEL
# =========================
print("💾 Saving models...")

KERAS_PATH = os.path.join(BASE_DIR, "multimodal_report_generator.keras")
model.save(KERAS_PATH)

H5_PATH = os.path.join(BASE_DIR, "multimodal_report_generator.h5")
model.save(H5_PATH)

print("✅ Model saved in Keras format at:", KERAS_PATH)
print("✅ Model saved in HDF5 format at:", H5_PATH)

print("🎉 Training + Saving completed successfully!")
