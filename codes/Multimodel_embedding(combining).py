import numpy as np

# Load embeddings
image_embeddings = np.load(
    r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\image_embeddings.npy"
)
text_embeddings = np.load(
    r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\text_embeddings.npy"
)

# Safety check
assert image_embeddings.shape[0] == text_embeddings.shape[0], "Sample count mismatch!"

# Combine (Late Fusion)
multimodal_embeddings = np.concatenate(
    (image_embeddings, text_embeddings),
    axis=1
)

print("✅ Multimodal fusion completed")
print("Final shape:", multimodal_embeddings.shape)

# Save
np.save(
    r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\multimodal_embeddings.npy",
    multimodal_embeddings
)

print("📁 Saved at: multimodal_embeddings.npy")
