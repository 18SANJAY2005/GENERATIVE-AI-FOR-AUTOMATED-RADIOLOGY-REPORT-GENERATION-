import pandas as pd

# ---------- CONFIG ----------
input_csv = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\iu_xray_merged.csv"
output_csv = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\multimodal_dataset.csv"

# ---------- LOAD DATA (FIXED ENCODING) ----------
df = pd.read_csv(input_csv, encoding='latin1')

# ---------- SELECT REQUIRED COLUMNS ----------
df = df[['image_path', 'findings']]

# ---------- BASIC CLEANING ----------
df.dropna(subset=['image_path', 'findings'], inplace=True)
df = df[df['findings'].str.strip() != ""]

# ---------- RESET INDEX ----------
df.reset_index(drop=True, inplace=True)

# ---------- SAVE CLEAN DATASET ----------
df.to_csv(output_csv, index=False)

print("✅ Multimodal dataset created successfully!")
print(f"📄 Saved at: {output_csv}")
print(f"📊 Total samples: {len(df)}")
