
import pandas as pd
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = "data/processed/cleaned_data.csv"
MODEL_DIR = "models/"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Processed data not found at {DATA_PATH}. Run preprocessing first.")
        sys.exit(1)

    print("Loading processed data...")
    df = pd.read_csv(DATA_PATH)

    if "clean_text" not in df.columns:
        print("[ERROR] 'clean_text' column not found in processed CSV.")
        sys.exit(1)
    if "label" not in df.columns:
        print("[ERROR] 'label' column not found in processed CSV.")
        sys.exit(1)

    # Fill NaN in clean_text and ensure str type
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    # Strip whitespace-only rows
    df["clean_text"] = df["clean_text"].str.strip()
    # Remove empty texts
    before = len(df)
    df = df[df["clean_text"] != ""].copy()
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"Removed {removed} rows with empty clean_text.")

    if len(df) == 0:
        print("[ERROR] No non-empty cleaned texts available after preprocessing.")
        sys.exit(1)

    # Ensure label is numeric
    # If labels are strings, map to ints (preserve mapping)
    if not pd.api.types.is_integer_dtype(df["label"]):
        unique = sorted(df["label"].unique())
        label_map = {v: i for i, v in enumerate(unique)}
        df["label"] = df["label"].map(label_map)
        inv_label_map = {i: v for v, i in label_map.items()}
    else:
        unique = sorted(df["label"].unique())
        label_map = {v: int(v) for v in unique}
        inv_label_map = {v: v for v in unique}

    # Check at least 2 classes
    classes = sorted(df["label"].unique())
    if len(classes) < 2:
        print(f"[ERROR] Need at least 2 label classes to train. Found classes: {classes}")
        sys.exit(1)
    else:
        print(f"Found label classes: {classes}")

    X = df["clean_text"].values
    y = df["label"].values

    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    print("Splitting and training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    print("Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Saving vectorizer...")
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Saving classifier...")
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Saving label map (inverse mapping)...")
    # save inverse mapping for predict
    with open(LABEL_MAP_PATH, "wb") as f:
        pickle.dump(inv_label_map, f)

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
