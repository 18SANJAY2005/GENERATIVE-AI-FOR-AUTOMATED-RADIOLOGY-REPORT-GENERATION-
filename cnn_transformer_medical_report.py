import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LayerNormalization,
    Dropout, MultiHeadAttention
)
from tensorflow.keras.applications.resnet50 import (
    ResNet50, preprocess_input
)
from tensorflow.keras.preprocessing.image import (
    load_img, img_to_array
)

# =====================================================
# PATHS (EDIT IF NEEDED)
# =====================================================
TOKENIZER_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\tokenizer.json"
MAX_SEQ_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\max_sequence_length.txt"
MODEL_SAVE_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\transformer_medical_report.h5"

# =====================================================
# LOAD TOKENIZER & MAX SEQ LENGTH
# =====================================================
print("🔹 Loading tokenizer...")
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

VOCAB_SIZE = len(tokenizer.word_index) + 1

with open(MAX_SEQ_PATH, "r") as f:
    MAX_SEQ_LEN = int(f.read().strip())

print(f"✅ Vocabulary size: {VOCAB_SIZE}")
print(f"✅ Max sequence length: {MAX_SEQ_LEN}")

# =====================================================
# RESNET50 IMAGE FEATURE EXTRACTOR
# =====================================================
print("🔹 Loading ResNet50...")
cnn = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

def extract_image_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = cnn.predict(img)
    return features

# =====================================================
# POSITIONAL ENCODING
# =====================================================
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(angle_rads[np.newaxis, ...], tf.float32)

# =====================================================
# TRANSFORMER DECODER BLOCK
# =====================================================
class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(num_heads, d_model)
        self.cross_attn = MultiHeadAttention(num_heads, d_model)

        self.ffn = tf.keras.Sequential([
            Dense(dff, activation="relu"),
            Dense(d_model)
        ])

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()

        self.dropout = Dropout(rate)

    def call(self, x, enc_output, training=False):
        attn1 = self.self_attn(x, x, x)
        x = self.norm1(x + attn1)

        attn2 = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + attn2)

        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x

# =====================================================
# BUILD CNN + TRANSFORMER MODEL
# =====================================================
def build_transformer_model(
    vocab_size,
    max_len,
    d_model=256,
    num_heads=8,
    dff=512,
    num_layers=4
):

    # IMAGE FEATURES INPUT
    image_features = Input(shape=(2048,), name="image_features")
    img_proj = Dense(d_model)(image_features)
    img_proj = tf.expand_dims(img_proj, axis=1)

    # TEXT INPUT
    text_input = Input(shape=(max_len,), name="text_input")
    x = Embedding(vocab_size, d_model)(text_input)

    pos_enc = positional_encoding(max_len, d_model)
    x = x + pos_enc[:, :max_len, :]

    # TRANSFORMER DECODER STACK
    for _ in range(num_layers):
        x = TransformerDecoderBlock(d_model, num_heads, dff)(x, img_proj)

    # OUTPUT
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(
        inputs=[image_features, text_input],
        outputs=outputs
    )
    return model

# =====================================================
# BUILD & COMPILE MODEL
# =====================================================
print("🔹 Building Transformer model...")

model = build_transformer_model(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_SEQ_LEN
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# TRAINING TEMPLATE (UNCOMMENT WHEN READY)
# =====================================================
"""
image_features     -> shape (N, 2048)
input_sequences    -> shape (N, MAX_SEQ_LEN)
target_sequences   -> shape (N, MAX_SEQ_LEN)
"""

# model.fit(
#     [image_features, input_sequences],
#     target_sequences,
#     epochs=30,
#     batch_size=16
# )

# model.save(MODEL_SAVE_PATH)

print("\n✅ CNN + Transformer architecture ready!")
