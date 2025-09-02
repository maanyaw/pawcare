import sqlite3

# Connect (this will create pawcare.db file if it doesn’t exist)
conn = sqlite3.connect("pawcare.db")
cur = conn.cursor()

# Create table
cur.execute("""
CREATE TABLE IF NOT EXISTS job_applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL,
    city TEXT NOT NULL,
    role TEXT NOT NULL,
    availability TEXT NOT NULL,
    exp TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
print("✅ job_applications table ready in pawcare.db")
