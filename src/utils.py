# src/utils.py
"""
Utility functions: save/load models, simple logging.
"""

import joblib
import os
from typing import Any

def save_model(obj: Any, path: str) -> None:
    """Save a python object (model/vectorizer) to path using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path: str) -> Any:
    """Load a python object saved by joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
