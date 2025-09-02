# 🐾 PawCare

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey?logo=flask)
![SQLite](https://img.shields.io/badge/SQLite-DB-blue?logo=sqlite)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange?logo=pytorch)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**PawCare** is an **AI-driven Pet Health and Services Platform** that combines  
🐶 **AI-powered nutrition recommendations**,  
🛍️ **sustainable shopping**,  
📅 **trusted bookings**, and  
⭐ **membership benefits** — all in one place.  

The project demonstrates a **full-stack solution**:
- 🌐 **Frontend** → Responsive website (HTML, CSS, JavaScript)  
- ⚙️ **Backend** → Flask APIs with SQLite  
- 🤖 **AI/ML** → PyTorch model + Sentence-BERT + SHAP explanations  

---

## ✨ Features

### 🌐 Frontend
- Responsive **HTML/CSS/JS** website  
- Pages for **Health, Shop, Bookings, Careers, Golden Membership**  
- Booking forms integrated with Flask backend  
- Clean, accessible design with **mobile-first layout**

### ⚙️ Backend (Flask + SQLite)
- Job Applications → `POST /apply`, `GET /applications`  
- Partner Enquiries → `POST /partner-apply`, `GET /partner-enquiries`  
- Bookings → `POST /booking-apply`, `GET /bookings` (Golden flag supported)  
- Auto-initialized SQLite DB (`pawcare.db`)  

### 🤖 AI/ML Pipeline
- Data: [`data/pet_meals.csv`](data/pet_meals.csv) with pet attributes + nutrition targets  
- Model: Custom **PyTorch MLP (`PetModel`)**  
- Text features: **Sentence-BERT** embeddings for symptoms/conditions  
- Explainability: **SHAP** (feature importance stored in `meta.json`)  
- Script: [`quick_train.py`](quick_train.py) → trains model, saves `pet_model.pt`  


<img width="1916" height="1020" alt="image" src="https://github.com/user-attachments/assets/f13c49d7-7ab3-4809-bf51-29ca10464d8b" />
<img width="1912" height="966" alt="image" src="https://github.com/user-attachments/assets/47d34a66-e898-4fd8-95e0-7d8a10fafff4" />
<img width="1919" height="571" alt="image" src="https://github.com/user-attachments/assets/bef6e8ca-1d26-4579-95fc-70a5b5b81c25" />
<img width="1917" height="744" alt="image" src="https://github.com/user-attachments/assets/22c5ca7e-bcb0-45e9-9ef2-42535c97ae1e" />
<img width="904" height="477" alt="image" src="https://github.com/user-attachments/assets/62f41b7a-f827-48b6-b76d-3d3168ec5ed7" />






