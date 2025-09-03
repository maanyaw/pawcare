from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3, os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "pawcare.db")

# --- Model definition ---
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

# --- Load model + metadata ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "pet_model.pt")
META_PATH = os.path.join(BASE_DIR, "models", "meta.json")
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "pet_meals.csv")

# ✅ FIXED: path to SQLite DB
DB_PATH = os.path.join(BASE_DIR, "pawcare.db")

with open(META_PATH, "r") as f:
    meta = json.load(f)

input_dim = meta.get("input_dim", 387)   # fallback for embeddings+numerics
encoder = SentenceTransformer(meta["encoder"])

model = PetModel(input_dim=input_dim, output_dim=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# --- Load meal DB from CSV ---
MEAL_DB = pd.read_csv(CSV_PATH)

@app.route("/", methods=["GET"])
def home():
    return "PawCare API is running with ML + SHAP ✅"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # --- numeric features
        weight = float(data.get("weightKg", 0))
        age = float(data.get("age", 0))
        neutered = 1 if str(data.get("neutered", "no")).lower() in ["1", "yes", "true"] else 0

        # --- text fields → embeddings
        text = " ".join([
            str(data.get("symptoms", "")),
            str(data.get("conditions", "")),
            " ".join(data.get("allergies", []))
        ])
        text_emb = encoder.encode([text], convert_to_tensor=True).cpu().numpy()

        # --- combine numeric + embeddings
        X = np.concatenate([[weight, age, neutered], text_emb[0]])
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

        # --- predict
        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy()[0]

        # --- Meals (randomly sample 2 from pet_meals.csv)
        meals = []
        if not MEAL_DB.empty:
            for _, row in MEAL_DB.sample(min(2, len(MEAL_DB))).iterrows():
                meals.append({
                    "title": row["meal_name"],
                    "items": [i.strip() for i in str(row["ingredients"]).split(",") if i.strip()]
                })

        # --- Tips
        tips = [{"source": "General", "text": "Ensure access to clean water at all times."}]
        if "vomiting" in text.lower():
            tips.append({"source": "Vet", "text": "Feed bland diet, avoid fatty foods."})
        if "diabetes" in text.lower():
            tips.append({"source": "Vet", "text": "Monitor blood sugar, avoid sugary treats."})

        # --- Why this plan?
        why = []
        for feat in meta.get("shap_feature_importance", []):
            why.append({
                "name": feat.get("feature", "unknown"),
                "weight": feat.get("importance", 0)
            })

        # --- Final response
        response = {
            "calories": round(float(y_pred[0]), 2),
            "macros": {
                "protein_g": round(float(y_pred[1]), 2),
                "fat_g": round(float(y_pred[2]), 2),
                "carb_g": round(float(y_pred[3]), 2),
            },
            "meals": meals,
            "tips": tips,
            "explain": {"top_features": why}
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ===================== DB Init =====================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Jobs
    c.execute("""
    CREATE TABLE IF NOT EXISTS job_applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT, phone TEXT, city TEXT,
        role TEXT, availability TEXT, exp TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # Partners
    c.execute("""
    CREATE TABLE IF NOT EXISTS partner_enquiries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT, phone TEXT, city TEXT,
        about TEXT, time_commitment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # Bookings
    c.execute("""
    CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service TEXT, location TEXT, date TEXT, time TEXT,
        pet TEXT, notes TEXT,
        golden INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    conn.commit()
    conn.close()

init_db()

# ===================== Jobs =====================
@app.route("/apply", methods=["POST"])
def apply_job():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO job_applications
        (name,email,phone,city,role,availability,exp)
        VALUES (?,?,?,?,?,?,?)""",
        (data.get("name"), data.get("email"), data.get("phone"),
         data.get("city"), data.get("role"),
         data.get("availability"), data.get("exp",""))
    )
    conn.commit(); conn.close()
    return jsonify({"status":"ok","msg":"Job application saved ✅"})

@app.route("/applications", methods=["GET"])
def view_applications():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT id,name,email,phone,city,role,availability,exp,created_at FROM job_applications ORDER BY created_at DESC")
    rows = c.fetchall(); conn.close()
    return jsonify([
        {"id":r[0],"name":r[1],"email":r[2],"phone":r[3],
         "city":r[4],"role":r[5],"availability":r[6],
         "exp":r[7],"created_at":r[8]} for r in rows
    ])

# ===================== Partners =====================
@app.route("/partner-apply", methods=["POST"])
def partner_apply():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""INSERT INTO partner_enquiries
        (name,email,phone,city,about,time_commitment)
        VALUES (?,?,?,?,?,?)""",
        (data.get("name"), data.get("email"), data.get("phone"),
         data.get("city"), data.get("about"), data.get("time"))
    )
    conn.commit(); conn.close()
    return jsonify({"status":"ok","msg":"Partner enquiry saved ✅"})

@app.route("/partner-enquiries", methods=["GET"])
def view_partners():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT id,name,email,phone,city,about,time_commitment,created_at FROM partner_enquiries ORDER BY created_at DESC")
    rows = c.fetchall(); conn.close()
    return jsonify([
        {"id":r[0],"name":r[1],"email":r[2],"phone":r[3],
         "city":r[4],"about":r[5],"time":r[6],"created_at":r[7]}
         for r in rows
    ])

# ===================== Bookings =====================
@app.route("/booking-apply", methods=["POST"])
def booking_apply():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""INSERT INTO bookings
        (service,location,date,time,pet,notes,golden)
        VALUES (?,?,?,?,?,?,?)""",
        (data.get("service"), data.get("location"), data.get("date"),
         data.get("time"), data.get("pet"), data.get("notes",""),
         1 if data.get("golden") else 0)
    )
    conn.commit(); conn.close()
    return jsonify({"status":"ok","msg":"Booking saved ✅"})

@app.route("/bookings", methods=["GET"])
def view_bookings():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT id,service,location,date,time,pet,notes,golden,created_at FROM bookings ORDER BY created_at DESC")
    rows = c.fetchall(); conn.close()
    return jsonify([
        {"id":r[0],"service":r[1],"location":r[2],
         "date":r[3],"time":r[4],"pet":r[5],"notes":r[6],
         "golden":r[7],"created_at":r[8]} for r in rows
    ])

# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

