import os
import pickle
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from .preprocessing import clean_text

MODEL_DIR = "models"

# 1. LOAD MÔ HÌNH
def load_models():
    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "label_map.pkl"), "rb") as f:
        label_map = pickle.load(f)

    return vectorizer, model, label_map


# 2. HÀM DỊCH 

def translate_if_vietnamese(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "vi":
            translated = GoogleTranslator(source='vi', target='en').translate(text)
            return translated
        else:
            return text
    except Exception:
        # Nếu detect lỗi → giữ nguyên
        return text



# 3. HÀM DỰ ĐOÁN
def predict_text(text: str) -> dict:
    vectorizer, model, label_map = load_models()

    translated = translate_if_vietnamese(text)
    clean = clean_text(translated)

    X = vectorizer.transform([clean])

    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[0]
        classes = list(model.classes_)
        if 1 in classes:
            suicide_prob = float(probas[classes.index(1)])
        else:
            suicide_prob = float(max(probas))
    else:
        decision = model.decision_function(X)[0]
        suicide_prob = float(1 / (1 + np.exp(-decision)))

    return {
        "label_id": pred,
        "label": label_map[pred],
        "probability": suicide_prob,
        "original_text": text,
        "translated_text": translated
    }
