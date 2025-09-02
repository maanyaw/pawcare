// ---- CONFIG ----
const API_BASE = "http://127.0.0.1:8000";

/* ---------- bootstrap ---------- */
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("healthForm");
  const genBtn = document.getElementById("genBtn");
  const resultsCard = document.getElementById("resultsCard");

  const run = async () => {
    const profile = normalize(Object.fromEntries(new FormData(form).entries()));
    try {
      const api = await generatePlan(profile);
      const view = mapApiToView(api);
      render(view);
      resultsCard.hidden = false;
      scrollInto(resultsCard);
    } catch (err) {
      console.error("API call failed:", err);
      alert("❌ API not reachable. Is Flask running on " + API_BASE + " ?");
    }
  };

  // prevent form auto-reset after submission
  form.addEventListener("submit", (e) => { e.preventDefault(); run(); });
  if (genBtn) genBtn.addEventListener("click", (e) => { e.preventDefault(); run(); });
});

/* ---------- API call ---------- */
async function generatePlan(profile) {
  const r = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

/* ---------- Adapter ---------- */
function mapApiToView(api) {
  const kcal   = api.calories ?? 0;
  const gramsP = api.macros?.protein_g ?? 0;
  const gramsF = api.macros?.fat_g ?? 0;
  const gramsC = api.macros?.carb_g ?? 0;

  const kcalP = gramsP * 4, kcalF = gramsF * 9, kcalC = gramsC * 4;
  const total = (kcalP + kcalF + kcalC) || 1;
  const pctP  = kcalP / total;
  const pctF  = kcalF / total;
  const pctC  = kcalC / total;

  const meals = [];
  (api.meals || []).forEach(section => {
    if (section?.title) meals.push(section.title);
    (section?.items || []).forEach(x => {
      if (typeof x === "string") meals.push(x);
      else if (x?.name) meals.push(x.name);
    });
  });

  const tips = (api.tips || []).map(t => t.text ?? String(t));
  const why  = [];
  if (api.explain?.confidence != null) {
    why.push(`Model confidence: ${api.explain.confidence}`);
  }
  if (Array.isArray(api.explain?.top_features)) {
    const top = api.explain.top_features
      .map(f => `${f.name ?? f.key} (${f.weight ?? f.value})`).join(", ");
    if (top) why.push(`Top features: ${top}`);
  }
  if (Array.isArray(api.warnings) && api.warnings.length) {
    for (const w of api.warnings) why.push(`Warning: ${w}`);
  }

  return { kcal, gramsP, gramsF, gramsC, pctP, pctF, pctC, meals, tips, why };
}

/* ---------- normalize inputs ---------- */
function normalize(d) {
  const allergies = (d.allergies || "")
    .split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
  return {
    species: (d.species || "dog").toLowerCase(),   // ✅ both dog & cat supported
    breed: (d.breed || "").trim(),
    weightKg: Number(d.weightKg || d.weight || 0),
    age: Number(d.age || d.age_yr || 0),
    activity: d.activity || "medium",
    neutered: (d.neutered === true || d.neutered === "yes"),
    allergies,
    symptoms: (d.symptoms || "").toLowerCase(),
    conditions: (d.conditions || "").toLowerCase(),
  };
}

/* ---------- render ---------- */
function render(view) {
  byId("outKcal").textContent = `${view.kcal} kcal`;
  byId("outPro").textContent  = `${view.gramsP} g (${Math.round(view.pctP*100)}%)`;
  byId("outFat").textContent  = `${view.gramsF} g (${Math.round(view.pctF*100)}%)`;
  byId("outCarb").textContent = `${view.gramsC} g (${Math.round(view.pctC*100)}%)`;

  fillList("mealList", view.meals, "—");
  fillList("tipsList", view.tips, "—");
  fillList("whyList",  view.why,  "—");
}

function fillList(id, items, emptyText) {
  const ul = byId(id); ul.innerHTML = "";
  if (!items?.length) {
    const li = document.createElement("li"); li.textContent = emptyText; ul.appendChild(li);
    return;
  }
  for (const t of items) {
    const li = document.createElement("li"); li.textContent = t; ul.appendChild(li);
  }
}

/* ---------- helpers ---------- */
function byId(id){ return document.getElementById(id); }
function scrollInto(el){
  if(!el) return;
  const y = el.getBoundingClientRect().top + window.scrollY - 80;
  window.scrollTo({top:y,behavior:"smooth"});
}
