import pandas as pd
import numpy as np
import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===============================
# PATHS
# ===============================
CSV_PATH = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\csv\multimodal_dataset.csv"
SAVE_DIR = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset"

os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(CSV_PATH)

texts = df['findings'].astype(str).tolist()

print(f"📝 Total reports: {len(texts)}")

# ===============================
# TOKENIZATION
# ===============================
tokenizer = Tokenizer(
    num_words=30000,   # medical vocab can be large
    oov_token="<OOV>",
    lower=True
)

tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

print(f"📚 Vocabulary size: {vocab_size}")

# ===============================
# CREATE SEQUENCES
# ===============================
input_sequences = []
target_words = []

for text in texts:
    seq = tokenizer.texts_to_sequences([text])[0]

    for i in range(1, len(seq)):
        input_sequences.append(seq[:i])
        target_words.append(seq[i])

# ===============================
# PADDING
# ===============================
max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_seq_len,
    padding='pre'
)

target_words = np.array(target_words)

# ===============================
# SAVE FILES
# ===============================
np.save(os.path.join(SAVE_DIR, "input_sequences.npy"), input_sequences)
np.save(os.path.join(SAVE_DIR, "target_words.npy"), target_words)

with open(os.path.join(SAVE_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

with open(os.path.join(SAVE_DIR, "max_sequence_length.txt"), "w") as f:
    f.write(str(max_seq_len))

print("✅ Text tokenization & sequence preparation completed!")
print(f"📐 Max sequence length: {max_seq_len}")
print("📁 Files saved in:", SAVE_DIR)
