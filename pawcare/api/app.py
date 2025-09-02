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
