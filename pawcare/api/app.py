from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, torch, torch.nn as nn, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

# --- Flask setup ---
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)

# --- Model ---
class PetModel(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Paths ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "pet_model.pt")
META_PATH = os.path.join(BASE_DIR, "models", "meta.json")
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "pet_meals.csv")

# --- Load metadata + model ---
with open(META_PATH, "r") as f:
    meta = json.load(f)

input_dim = meta.get("input_dim", 387)
encoder = SentenceTransformer(meta["encoder"])

model = PetModel(input_dim=input_dim, output_dim=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Meals DB
MEAL_DB = pd.read_csv(CSV_PATH)

# Scaler params (for numeric normalization)
scaler_mean = np.array(meta.get("scaler_mean", [0, 0, 0]))
scaler_scale = np.array(meta.get("scaler_scale", [1, 1, 1]))


# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return "PawCare API is running with ML + SHAP ✅"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # --- Inputs ---
        weight = float(data.get("weightKg", 0))
        age = float(data.get("age", 0))
        neutered = 1 if str(data.get("neutered", "no")).lower() in ["1", "yes", "true"] else 0

        text = " ".join([
            str(data.get("symptoms", "")),
            str(data.get("conditions", "")),
            " ".join(data.get("allergies", []))
        ])
        text_emb = encoder.encode([text], convert_to_tensor=True).cpu().numpy()

        # --- Normalize numeric features ---
        num_feats = np.array([weight, age, neutered], dtype=float)
        num_feats = (num_feats - scaler_mean) / scaler_scale

        # --- Final feature vector ---
        X = np.concatenate([num_feats, text_emb[0]])
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

        # --- Prediction ---
        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy()[0]

        # --- Suggested meals ---
        meals = []
        if not MEAL_DB.empty:
            for _, row in MEAL_DB.sample(min(2, len(MEAL_DB))).iterrows():
                meals.append({
                    "title": row["meal_name"],
                    "items": [i.strip() for i in str(row["ingredients"]).split(",") if i.strip()]
                })

        # --- Tips ---
        tips = [{"source": "General", "text": "Ensure access to clean water at all times."}]
        if "vomiting" in text.lower():
            tips.append({"source": "Vet", "text": "Feed bland diet, avoid fatty foods."})
        if "diabetes" in text.lower():
            tips.append({"source": "Vet", "text": "Monitor blood sugar, avoid sugary treats."})

        # --- SHAP explanations ---
        why = meta.get("shap_feature_importance", [])
        if not why:
            why = [
                {"name": "weight_kg", "value": 0.0},
                {"name": "age_yr", "value": 0.0},
                {"name": "neutered", "value": 0.0},
                {"name": "symptoms+conditions+allergies", "value": 0.0}
            ]

        # sort by importance & keep top 2
        sorted_why = sorted(why, key=lambda x: x["value"], reverse=True)[:2]
        summary = ", ".join([f"{f['name']} ({f['value']:.4f})" for f in sorted_why])


        # --- Response ---
        response = {
            "calories": round(float(y_pred[0]), 2),
            "macros": {
                "protein": round(float(y_pred[1]), 2),
                "fat": round(float(y_pred[2]), 2),
                "carb": round(float(y_pred[3]), 2),
            },
            "meals": meals,
            "tips": tips,
            "explain": {
                "top_features": sorted_why,
                "summary": f"Main drivers: {summary}"
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


