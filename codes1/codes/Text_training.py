import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------- CONFIG ----------------
INPUT_CSV = r"E:\Final Year\csv1\csv\multimodal_dataset.csv"
OUTPUT_NPY = r"E:\Final Year\files1\files\text_embeddings.npy"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_CSV)

# Ensure correct column name
assert 'findings' in df.columns, "❌ Column 'findings' not found in CSV"

texts = df['findings'].astype(str).tolist()
print(f"📄 Total text samples: {len(texts)}")

# ---------------- LOAD MODEL ----------------
print("🔄 Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # CPU friendly

# ---------------- GENERATE EMBEDDINGS ----------------
print("🧠 Generating text embeddings...")
text_embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# ---------------- SAVE ----------------
np.save(OUTPUT_NPY, text_embeddings)

print("✅ Text embeddings generated successfully!")
print(f"📁 Saved at: {OUTPUT_NPY}")
print(f"📐 Embedding shape: {text_embeddings.shape}")
