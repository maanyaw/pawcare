# quick_train.py
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

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

# --- Load CSV ---
def load_data(csv_path="pawcare/data/pet_meals.csv"):
    return pd.read_csv(csv_path)

# --- Convert to features ---
def make_features(df, encoder):
    # Numeric cleanup
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").fillna(0)
    df["age_yr"]    = pd.to_numeric(df["age_yr"], errors="coerce").fillna(0)
    df["neutered"]  = df["neutered"].apply(lambda x: 1 if str(x).strip().lower() in ["1","yes","true"] else 0)

    X_num = df[["weight_kg", "age_yr", "neutered"]].values.astype(float)

    # Text embeddings
    texts = (
        df["symptoms"].fillna("").astype(str) + " " +
        df["conditions"].fillna("").astype(str) + " " +
        df["allergies"].fillna("").astype(str)
    ).tolist()
    X_text = encoder.encode(texts, convert_to_tensor=True).cpu().numpy()

    # Combine
    X = np.hstack([X_num, X_text])
    X = torch.tensor(X, dtype=torch.float32)

    # Targets (exclude meal_name, ingredients!)
    target_cols = ["kcal","pct_protein","pct_fat","pct_carb"]
    y = df[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = torch.tensor(y, dtype=torch.float32)

    return X, y

# --- Training ---
def train(csv_path):
    df = load_data(csv_path)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    X, y = make_features(df, encoder)

    # Scale numeric
    scaler = StandardScaler()
    X_np = X.numpy()
    X_np[:, :3] = scaler.fit_transform(X_np[:, :3])
    X = torch.tensor(X_np, dtype=torch.float32)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = PetModel(input_dim=X.shape[1], output_dim=4)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        optim.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optim.step()

        if epoch % 10 == 0:
            val_loss = loss_fn(model(X_val), y_val).item()
            print(f"Epoch {epoch:02d} | train loss {loss.item():.4f} | val loss {val_loss:.4f}")

    # Save model + metadata dir
    models_dir = os.path.join(os.path.dirname(__file__), "pawcare", "api", "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "pet_model.pt"))

    # --- SHAP (lightweight) ---
    model.eval()

    # Wrap model to accept numpy
    def predict_fn(data_np):
        data_t = torch.tensor(data_np, dtype=torch.float32)
        with torch.no_grad():
            return model(data_t).numpy()

    background = X_train[:min(20, len(X_train))].numpy()
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_val[:5].numpy(), nsamples=50)

    # Aggregate embeddings → 1 feature
    mean_abs = np.abs(shap_values[0]).mean(axis=0)
    f_importance = {
        "weight_kg": float(mean_abs[0]),
        "age_yr": float(mean_abs[1]),
        "neutered": float(mean_abs[2]),
        "symptoms+conditions+allergies": float(mean_abs[3:].mean())
    }

    meta = {
        "encoder": "all-MiniLM-L6-v2",
        "feature_names": list(f_importance.keys()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_importance": f_importance
    }

    with open(os.path.join(models_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Model + metadata (with SHAP simplified) saved in", models_dir)


if __name__ == "__main__":
    train("pawcare/data/pet_meals.csv")