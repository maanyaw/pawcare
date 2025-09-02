# pawcare/api/utils/preprocess.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Map activity categories to numbers
ACTIVITY_MAP = {"low": 0, "medium": 1, "high": 2}

def breed_size(breed: str) -> int:
    """Very small heuristic to turn breed name into size bucket."""
    b = (breed or "").lower()
    if any(x in b for x in ["chihuahua", "pomeranian", "toy", "teacup", "shih tzu", "dachshund", "pug"]):
        return 0  # small
    if any(x in b for x in ["beagle", "poodle", "cocker", "border collie", "bulldog"]):
        return 1  # medium
    return 2      # large/default

# Allergens we flag as binary features
ALLERGENS = [
    "chicken","fish","lamb","egg","dairy","wheat","soy",
    "turkey","beef","oats","rice","millet","quinoa"
]

def allergen_flags(allergies: str) -> np.ndarray:
    s = set(a.strip().lower() for a in (allergies or "").split(",") if a.strip())
    return np.array([1 if a in s else 0 for a in ALLERGENS], dtype=np.float32)

def make_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the numeric feature matrix from the training dataframe."""
    out = pd.DataFrame()
    out["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").fillna(0)
    out["age_yr"]    = pd.to_numeric(df["age_yr"], errors="coerce").fillna(0)
    out["neutered"]  = pd.to_numeric(df["neutered"], errors="coerce").fillna(0)
    out["act_num"]   = df["activity"].map(lambda x: ACTIVITY_MAP.get(str(x).lower(), 1)).astype(float)
    out["breed_size"] = df["breed"].map(breed_size).astype(float)

    flags = np.stack(df["allergies"].map(allergen_flags).values)
    for i, name in enumerate(ALLERGENS):
        out[f"allerg_{name}"] = flags[:, i]

    return out

def make_features_one(sample: dict) -> np.ndarray:
    """Builds a 1xN feature array for a user profile dict."""
    df = pd.DataFrame([{
        "breed": sample.get("breed",""),
        "weight_kg": sample.get("weightKg", 0.0),
        "age_yr": sample.get("age", 0.0),
        "activity": sample.get("activity", "medium"),
        "neutered": 1 if sample.get("neutered", True) else 0,
        "allergies": ",".join(sample.get("allergies", [])),
    }])
    f = make_features_df(df)
    return f.values.astype(np.float32)
