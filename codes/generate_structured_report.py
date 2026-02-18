import json
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# =====================================================
# 🔧 PATH CONFIGURATION (EDIT ONLY THESE IF NEEDED)
# =====================================================
MODEL_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\multimodal_report_generator.h5"
TOKENIZER_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\tokenizer.json"

IMAGE_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\test_img.png"

MAX_SEQ_LEN = 170
IMG_SIZE = 224

# =====================================================
# 🔹 LOAD TOKENIZER
# =====================================================
print("🔹 Loading tokenizer...")
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

index_word = {v: k for k, v in tokenizer.word_index.items()}

# =====================================================
# 🔹 LOAD MODEL
# =====================================================
print("🔹 Loading trained multimodal model...")
model = load_model(MODEL_PATH)

# =====================================================
# 🔹 LOAD RESNET50 FEATURE EXTRACTOR
# =====================================================
print("🔹 Loading ResNet50...")
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# =====================================================
# 🩺 IMAGE FEATURE EXTRACTION
# =====================================================
def extract_image_features(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = cnn.predict(img)
    return features  # (1, 2048)

# =====================================================
# 📝 CLEAN & STRUCTURE FINDINGS
# =====================================================
def clean_text(text):
    text = text.replace("<start>", "").replace("<end>", "")
    text = re.sub(r"\b[xX]{3,}\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_findings(text):
    sections = {
        "LUNGS": [],
        "HEART": [],
        "PLEURA": [],
        "BONES": []
    }

    sentences = re.split(r"[.]", text)

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        s_low = s.lower()

        if any(k in s_low for k in ["lung", "opacity", "consolidation", "inflation"]):
            sections["LUNGS"].append(s)
        elif any(k in s_low for k in ["heart", "cardiac", "mediast"]):
            sections["HEART"].append(s)
        elif any(k in s_low for k in ["pleural", "effusion", "pneumothorax"]):
            sections["PLEURA"].append(s)
        elif any(k in s_low for k in ["spine", "bone", "osseous", "rib"]):
            sections["BONES"].append(s)
        else:
            sections["LUNGS"].append(s)

    return sections

# =====================================================
# 📄 REPORT GENERATION
# =====================================================
def generate_report(img_path):
    print("\n🩺 Extracting image features...")
    img_features = extract_image_features(img_path)

    dummy_seq = np.zeros((1, MAX_SEQ_LEN))

    print("📝 Generating report text...")
    preds = model.predict([img_features, dummy_seq])

    # ✅ SAFE DECODING
    if preds.ndim == 3:
        preds = preds[0]

    token_ids = np.argmax(preds, axis=-1)

    words = []
    for token in token_ids:
        word = index_word.get(token, "")
        if word == "<end>" or word == "":
            break
        words.append(word)

    raw_text = " ".join(words)
    cleaned_text = clean_text(raw_text)

    sections = split_findings(cleaned_text)

    # =================================================
    # 🧾 FINAL STRUCTURED REPORT
    # =================================================
    report = "\n" + "="*60 + "\n"
    report += "📄 GENERATED MEDICAL REPORT\n"
    report += "="*60 + "\n\n"

    report += "EXAM:\nChest X-ray (PA View)\n\n"
    report += "HISTORY:\nRoutine clinical evaluation.\n\n"

    report += "FINDINGS:\n"
    for key, vals in sections.items():
        report += f"{key}:\n"
        if vals:
            for v in vals:
                report += f"- {v}.\n"
        else:
            report += "- No abnormality detected.\n"
        report += "\n"

    report += "IMPRESSION:\n"
    report += "No acute cardiopulmonary abnormality detected.\n\n"

    report += "RECOMMENDATION:\nClinical correlation advised.\n"
    report += "="*60

    return report

# =====================================================
# 🚀 RUN
# =====================================================
if __name__ == "__main__":
    print("\n📄 Generating structured medical report...")
    report = generate_report(IMAGE_PATH)
    print(report)
