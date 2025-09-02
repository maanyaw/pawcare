# scripts/embed_kb.py
from pathlib import Path
import json, numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent  # .../api
KB = BASE / "kb"
SRC = KB / "guidelines.jsonl"
EMB = KB / "embeddings.npy"
META = KB / "meta.json"

assert SRC.exists(), f"Missing {SRC}. Create it first."

# load texts
texts, meta = [], []
with SRC.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        j = json.loads(line)
        texts.append(j["text"])
        meta.append({"id": j.get("id"), "tag": j.get("tag"), "text": j["text"]})

print(f"Loaded {len(texts)} guideline rows from {SRC}")

# embed with SBERT
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
E = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
E = E.astype("float32")

# save
np.save(EMB, E)
META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Wrote {EMB} and {META}")
