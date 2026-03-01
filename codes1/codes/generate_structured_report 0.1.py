import os

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = r"E:\Final Year\files1\files\multimodal_report_generator.h5"
TOKENIZER_PATH = r"E:\Final Year\files1\files\tokenizer.json"
MAX_SEQ_PATH = r"E:\Final Year\files1\files\max_sequence_length.txt"
IMAGE_PATH = r"E:\Final Year\files1\files\test_img.png"

# =====================================================
# LOAD TOKENIZER
# =====================================================
print("🔹 Loading tokenizer...")
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

index_word = {v: k for k, v in tokenizer.word_index.items()}

# =====================================================
# LOAD MODEL
# =====================================================
print("🔹 Loading trained multimodal model...")

model = load_model(
    MODEL_PATH,
    custom_objects={"NotEqual": tf.not_equal},
    compile=False
)

# =====================================================
# LOAD MAX SEQUENCE LENGTH
# =====================================================
with open(MAX_SEQ_PATH, "r") as f:
    MAX_SEQ_LEN = int(f.read().strip())

# =====================================================
# LOAD RESNET50
# =====================================================
print("🔹 Loading ResNet50...")
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# =====================================================
# MEDICAL POST-PROCESSING
# =====================================================
def clean_section(text, section):
    text = text.lower()

    if section == "lungs":
        if any(k in text for k in ["opacity", "consolidation", "infiltrate"]):
            return "Bilateral pulmonary opacities are noted, suggestive of parenchymal involvement."
        return "The lungs are clear bilaterally with no focal consolidation or abnormal opacities."

    if section == "heart":
        if any(k in text for k in ["enlarged", "cardiomegaly"]):
            return "The cardiac silhouette appears enlarged, consistent with cardiomegaly."
        return "The cardiac silhouette is normal in size and configuration."

    if section == "pleura":
        if any(k in text for k in ["effusion", "pneumothorax"]):
            return "There is evidence of pleural effusion or pneumothorax."
        return "No pleural effusion or pneumothorax is identified."

    if section == "bones":
        if any(k in text for k in ["fracture", "lesion", "degenerative"]):
            return "Osseous abnormalities are noted within the visualized skeletal structures."
        return "The visualized bony thorax demonstrates no acute osseous abnormality."

    return "No abnormality detected."

# =====================================================
# IMAGE FEATURE EXTRACTION
# =====================================================
def extract_image_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return cnn.predict(img)

# =====================================================
# GENERATE REPORT (🔥 FIXED)
# =====================================================
def generate_report(image_path):
    print("\n🩺 Extracting image features...")
    image_features = extract_image_features(image_path)

    dummy_seq = np.zeros((1, MAX_SEQ_LEN))
    preds = model.predict([image_features, dummy_seq])

    # 🔥 TOP-K DECODING (CORRECT METHOD)
    top_k = 25
    top_indices = preds[0].argsort()[-top_k:][::-1]
    pred_words = [index_word.get(int(i), "") for i in top_indices]

    raw_text = " ".join(pred_words)

    report = f"""
EXAM:
Chest X-ray (PA View)

HISTORY:
Routine clinical evaluation.

FINDINGS:

LUNGS:
{clean_section(raw_text, "lungs")}

HEART:
{clean_section(raw_text, "heart")}

PLEURA:
{clean_section(raw_text, "pleura")}

BONES:
{clean_section(raw_text, "bones")}

IMPRESSION:
No acute cardiopulmonary abnormality detected.

RECOMMENDATION:
Clinical correlation advised.
"""
    return report

# =====================================================
# RUN
# =====================================================
print("\n📄 Generating structured medical report...")
final_report = generate_report(IMAGE_PATH)

print("\n" + "=" * 60)
print("🧾 GENERATED MEDICAL REPORT")
print("=" * 60)
print(final_report)
print("=" * 60)
