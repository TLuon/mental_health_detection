import pandas as pd
import os
import re
import unicodedata

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_PATH, "cleaned_data.csv")

def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)  # chuẩn hóa unicode
    s = s.lower().strip()
    return s

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("Loading raw data...")

    # load file CSV đầu tiên
    csv_file = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, csv_file))

    if "Suicide" not in df.columns:
        raise ValueError("Cột 'Suicide' không tồn tại trong dataset.")

    print("Normalizing Suicide labels...")
    df["Suicide"] = df["Suicide"].apply(normalize_text)

    # Mapping mềm
    label_mapping = {
        "not suicide post": 0,
        "potential suicide post": 1,
        "suicide post": 2
    }

    df["label"] = df["Suicide"].map(label_mapping)

    # loại NaN label (chỉ khi != 3 loại hợp lệ)
    before = len(df)
    df = df.dropna(subset=["label"])
    after = len(df)
    print(f"Dropped {before - after} rows due to invalid labels.")

    # Clean text
    print("Cleaning text...")
    df["clean_text"] = df["Tweet"].apply(clean_text)

    # Remove empty text
    df = df[df["clean_text"].str.strip() != ""]

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
