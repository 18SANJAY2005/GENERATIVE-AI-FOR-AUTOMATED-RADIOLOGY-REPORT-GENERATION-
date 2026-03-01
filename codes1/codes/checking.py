import pandas as pd
import os
import cv2
import pydicom
import matplotlib.pyplot as plt

# -------- Paths --------
csv_path = r"E:\Final Year\csv1\csv\iu_xray_merged.csv"
image_folder = r"E:\Final Year\dataset\dataset\images_normalized"

df = pd.read_csv(csv_path)

row = df.iloc[0]

filename = row["filename"]
findings = row["findings"]
impression = row["impression"]

image_path = os.path.join(image_folder, filename)

print("Findings:", findings)
print("Impression:", impression)
print("\nTrying image path:", image_path)

if not os.path.exists(image_path):
    print("❌ Image file not found!")
    exit()

print("✅ Image file found!")

# -------- LOAD IMAGE CORRECTLY --------
if image_path.lower().endswith(".dcm"):
    dicom = pydicom.dcmread(image_path)
    image = dicom.pixel_array
    plt.imshow(image, cmap="gray")

else:  # PNG / JPG
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap="gray")

plt.title("Chest X-ray")
plt.axis("off")
plt.show()
