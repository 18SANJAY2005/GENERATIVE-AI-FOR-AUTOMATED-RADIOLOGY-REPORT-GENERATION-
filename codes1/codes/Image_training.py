import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ---------- CONFIG ----------
csv_path = r"E:\Final Year\csv1\csv\multimodal_dataset.csv"
output_features = r"E:\Final Year\files1\files\image_embeddings.npy"

device = torch.device("cpu")  # Intel Iris → CPU is fine

# ---------- LOAD DATA ----------
df = pd.read_csv(csv_path)

# ---------- IMAGE TRANSFORMS ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- LOAD PRETRAINED CNN ----------
model = models.resnet50(pretrained=True)

# Remove final classification layer
model = nn.Sequential(*list(model.children())[:-1])

# Freeze weights
for param in model.parameters():
    param.requires_grad = False

model.to(device)
model.eval()

# ---------- FEATURE EXTRACTION ----------
features = []

print("🔄 Extracting image features...")

for img_path in tqdm(df['image_path']):
    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(image)
            feature = feature.view(-1).cpu().numpy()

        features.append(feature)

    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        features.append(np.zeros((2048,)))  # fallback

# ---------- SAVE FEATURES ----------
features = np.array(features)
np.save(output_features, features)

print("✅ Image feature extraction completed!")
print(f"📦 Feature shape: {features.shape}")
print(f"💾 Saved at: {output_features}")
