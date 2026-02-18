import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image

# =========================
# PATHS
# =========================
BASE_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset"

MODEL_PATH = BASE_PATH + r"\multimodal_report_generator.h5"
TOKENIZER_PATH = BASE_PATH + r"\tokenizer.json"
MAX_SEQ_PATH = BASE_PATH + r"\max_sequence_length.txt"

# 👇 PUT YOUR X-RAY IMAGE PATH HERE
IMAGE_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\test_img.png"

# =========================
# LOAD MULTIMODAL MODEL
# =========================
print("🔹 Loading trained multimodal model...")
model = load_model(MODEL_PATH)

# =========================
# LOAD CNN (FEATURE EXTRACTOR)
# =========================
print("🔹 Loading ResNet50 feature extractor...")
cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# =========================
# LOAD TOKENIZER
# =========================
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_json = json.load(f)

if isinstance(tokenizer_json, list):
    tokenizer_json = tokenizer_json[0]

word_index = tokenizer_json["config"]["word_index"]

if isinstance(word_index, str):
    word_index = json.loads(word_index)

index_word = {int(v): k for k, v in word_index.items()}

# =========================
# LOAD MAX SEQUENCE LENGTH
# =========================
with open(MAX_SEQ_PATH, "r") as f:
    max_seq_len = int(f.read().strip())

# =========================
# IMAGE FEATURE EXTRACTION
# =========================
def extract_image_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    embedding = cnn_model.predict(img, verbose=0)
    return embedding

# =========================
# REPORT GENERATION
# =========================
def generate_report(image_embedding):
    sequence = [word_index.get("<start>", 1)]

    for _ in range(max_seq_len):
        padded_seq = pad_sequences([sequence], maxlen=max_seq_len, padding="post")
        preds = model.predict([image_embedding, padded_seq], verbose=0)
        next_word_id = np.argmax(preds[0])

        if next_word_id == 0:
            break

        word = index_word.get(next_word_id, "")
        if word == "<end>":
            break

        sequence.append(next_word_id)

    words = [index_word.get(i, "") for i in sequence]
    report = " ".join(words).replace("<start>", "").strip()
    return report

# =========================
# RUN PIPELINE
# =========================
print("\n🩻 Loading X-ray and extracting features...")
image_embedding = extract_image_embedding(IMAGE_PATH)

print("📝 Generating report...\n")
final_report = generate_report(image_embedding)

print("✅ GENERATED MEDICAL REPORT")
print("--------------------------------------------------")
print(final_report)
print("--------------------------------------------------")
