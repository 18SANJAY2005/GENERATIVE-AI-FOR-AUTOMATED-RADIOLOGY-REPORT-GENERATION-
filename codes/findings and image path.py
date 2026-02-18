import pandas as pd

# ---------- CONFIG ----------
excel_file = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\iu_xray_merged.csv"
output_excel = r"C:\Users\K.M.Dhoni\OneDrive\Desktop\dataset\findings_and_images.xlsx"

# ---------- READ CSV ----------
df = pd.read_csv(excel_file)

# ---------- SELECT COLUMNS ----------
# Use 'image_path' for image paths and 'findings' for findings
new_df = df[['image_path', 'findings']]

# ---------- SAVE NEW EXCEL ----------
new_df.to_excel(output_excel, index=False)

print(f"New Excel with image paths and findings saved as '{output_excel}'!")
