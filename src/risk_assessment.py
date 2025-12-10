import random
import os

MESSAGE_DIR = "messages"

def load_messages(level):
    filename = {
        "low": "advice_low.txt",
        "medium": "advice_medium.txt",
        "high": "advice_high.txt"
    }[level]

    path = os.path.join(MESSAGE_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines

def get_random_message(level):
    msgs = load_messages(level)
    return random.choice(msgs)

def assess_risk(prob):
    if prob >= 0.75:
        return "high"
    elif prob >= 0.40:
        return "medium"
    else:
        return "low"
