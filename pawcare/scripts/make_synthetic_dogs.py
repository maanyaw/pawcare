# scripts/make_synthetic_dogs.py
import os, random, numpy as np, pandas as pd

random.seed(11); np.random.seed(11)

# --- knobs ---
N_ROWS = 2500  # change if you want more/less

breeds = [
    "Labrador Retriever","German Shepherd","Beagle","Indie","Golden Retriever",
    "Pug","Shih Tzu","Dachshund","Boxer","Siberian Husky","Rottweiler","Pomeranian"
]
activities = ["low","medium","high"]
symptom_pool = ["itchy paws","diarrhea","vomiting","lethargy","gas","dry coat"]
cond_pool = ["obesity","pancreatitis","kidney","dental","arthritis",""]
allergen_pool = ["chicken","wheat","soy","dairy","egg","fish","lamb","oats"]

# base recipes + the allergens that would block them
# (if an allergy overlaps, we skip that recipe)
BASE_RECIPES = [
    ("turkey,rice,carrot,peas", {"chicken","wheat"}),     # turkey is OK if chicken allergy
    ("fish,pumpkin,rice", {"fish","wheat"}),
    ("lamb,oats,spinach", {"lamb","oats"}),
    ("egg,sweet potato,spinach", {"egg"}),
    ("tofu,brown rice,broccoli", {"soy","wheat"}),
    ("paneer,rice,beans", {"dairy","wheat"}),
    ("mackerel,quinoa,zucchini", {"fish"}),
    ("chicken,millet,pumpkin", {"chicken"}),
]

def pick_some(pool, p=0.25):
    """Return comma-separated random subset (Bernoulli p)"""
    vals = sorted({x for x in pool if random.random() < p})
    return ",".join(vals)

def choose_ingredients(allergies_set):
    cands = [r for r, blk in BASE_RECIPES if not (allergies_set & blk)]
    # fallbacks if everything is blocked (rare)
    if not cands:
        cands = ["turkey,rice,carrot,peas","tofu,brown rice,broccoli","egg,sweet potato,spinach"]
    return random.choice(cands)

def sample_row():
    species = "dog"
    breed = random.choice(breeds)

    # weight (kg): broad dog distribution, clipped to [2, 60]
    weight = float(np.clip(np.random.normal(20, 8), 2, 60).round(1))
    age = int(np.clip(np.random.normal(5, 3), 0, 18))
    activity = random.choices(activities, weights=[0.25, 0.55, 0.20], k=1)[0]
    neutered = (random.random() < 0.75)

    allergies = pick_some(allergen_pool, 0.20)
    symptoms = pick_some(symptom_pool, 0.25)
    conditions = pick_some(cond_pool, 0.20)

    # ---- Targets ----
    # kcal via RER × activity × neuter adjustment (dogs only)
    rer = 70 * (weight ** 0.75)
    factor = {"low":1.2, "medium":1.6, "high":2.0}[activity]
    if neutered:
        factor *= 0.9
    kcal = int(round(rer * factor + np.random.normal(0, 30)))

    # Macro defaults
    pct_pro, pct_fat, pct_carb = 30, 30, 40
    # simple condition tweaks
    if "pancreatitis" in conditions:
        pct_pro, pct_fat, pct_carb = 30, 20, 50     # low fat
    elif "obesity" in conditions:
        pct_pro, pct_fat, pct_carb = 32, 25, 43     # trim fat a bit
    elif activity == "high":
        pct_pro, pct_fat, pct_carb = 32, 32, 36     # more energy density

    # Ingredients: avoid allergens
    alg_set = set(a.strip() for a in allergies.split(",") if a.strip())
    ingredients = choose_ingredients(alg_set)

    return dict(
        species=species, breed=breed, weightKg=weight, age=age, activity=activity,
        neutered=neutered, allergies=allergies, symptoms=symptoms, conditions=conditions,
        kcal=kcal, pct_pro=pct_pro, pct_fat=pct_fat, pct_carb=pct_carb,
        ingredients=ingredients
    )

def main():
    rows = [sample_row() for _ in range(N_ROWS)]
    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    out = "data/pet_meals.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  shape={df.shape}")
    # quick peek
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
