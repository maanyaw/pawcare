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

<img width="1876" height="915" alt="image" src="https://github.com/user-attachments/assets/1f1e1d5f-ca57-47bf-b47f-372140a82082" />
<img width="1326" height="674" alt="image" src="https://github.com/user-attachments/assets/b44d036c-a0d2-4b16-af89-7d86fbaede69" />
<img width="778" height="796" alt="image" src="https://github.com/user-attachments/assets/8c7a425d-c731-419e-a5e7-1a5ba79b0a8d" />
<img width="724" height="79" alt="image" src="https://github.com/user-attachments/assets/c4c02e0d-b0dc-45d5-903b-632f7a122ff5" />

<img width="1919" height="736" alt="image" src="https://github.com/user-attachments/assets/653e9fbb-b138-4ab2-ba37-fc9a1436ef64" />
<img width="1919" height="944" alt="image" src="https://github.com/user-attachments/assets/4cdc8230-7296-45ff-9931-11b717e3ea6f" />
<img width="1637" height="815" alt="image" src="https://github.com/user-attachments/assets/6e6a7f28-3831-4922-83a1-688f2ff98f88" />
<img width="365" height="323" alt="image" src="https://github.com/user-attachments/assets/c9d43b94-7467-4be6-b692-5aea1bea52a5" />
<img width="361" height="279" alt="image" src="https://github.com/user-attachments/assets/a5239632-3c5a-486c-b65b-729747269c82" />
<img width="1912" height="966" alt="image" src="https://github.com/user-attachments/assets/47d34a66-e898-4fd8-95e0-7d8a10fafff4" />
<img width="1919" height="571" alt="image" src="https://github.com/user-attachments/assets/bef6e8ca-1d26-4579-95fc-70a5b5b81c25" />
<img width="1917" height="744" alt="image" src="https://github.com/user-attachments/assets/22c5ca7e-bcb0-45e9-9ef2-42535c97ae1e" />
<img width="904" height="477" alt="image" src="https://github.com/user-attachments/assets/62f41b7a-f827-48b6-b76d-3d3168ec5ed7" />
<img width="1909" height="902" alt="image" src="https://github.com/user-attachments/assets/25c2765e-92c7-4f9e-b220-98e48fce1e96" />
<img width="1897" height="942" alt="image" src="https://github.com/user-attachments/assets/4530e467-5d65-4a7d-985b-b98e2d13280e" />
<img width="1897" height="930" alt="image" src="https://github.com/user-attachments/assets/aabe6d6d-8f2d-47ae-83b1-d43cb676c41e" />
<img width="1149" height="933" alt="image" src="https://github.com/user-attachments/assets/be6530a2-6e43-42b5-9e36-de632fbd152e" />
<img width="1916" height="944" alt="image" src="https://github.com/user-attachments/assets/0011bd31-d508-4366-8e46-a37843be534a" />
<img width="1913" height="748" alt="image" src="https://github.com/user-attachments/assets/90f0835c-c3a8-499b-afbe-5939427a788f" />
<img width="1879" height="261" alt="image" src="https://github.com/user-attachments/assets/0284d068-c414-43cc-a84e-79af3d04b9ae" />
<img width="1880" height="937" alt="image" src="https://github.com/user-attachments/assets/3042bd3c-6317-4188-842b-4549e6feb3e2" />
<img width="1893" height="775" alt="image" src="https://github.com/user-attachments/assets/e43b6861-1ac3-4191-a102-b59f25a25218" />
<img width="1898" height="783" alt="image" src="https://github.com/user-attachments/assets/41e0f90e-8552-492d-8304-0441e717f26b" />
<img width="1893" height="777" alt="image" src="https://github.com/user-attachments/assets/5ebcde88-91ce-48a8-bf02-6ea90261815b" />


## 🛠️ Tech Stack

Frontend → HTML, CSS, JavaScript

Backend → Flask, SQLite

AI/ML → PyTorch, Sentence-BERT, SHAP, Pandas, NumPy

Other → Flask-CORS, scikit-learn

## 👩‍💻 Author

**Maanya Walia**  















