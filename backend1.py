from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sqlite3

app = Flask(__name__)
CORS(app)

# ---------------- INIT DB FIRST ----------------
def init_db():
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        sleep_hours REAL,
        tremor REAL,
        fatigue_level REAL,
        mood_score REAL,
        prediction TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()   # ✅ MUST RUN BEFORE APP STARTS

# ---------------- LOAD MODEL ----------------
model = joblib.load("isolation_forest_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

# ---------------- TEST ROUTE ----------------
@app.route("/")
def home():
    return "Backend is running"

# ---------------- USERS ----------------
@app.route("/users", methods=["GET"])
def get_users():
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    c.execute("SELECT id, username FROM users")
    rows = c.fetchall()
    conn.close()

    users = []
    for r in rows:
        users.append({
            "u_id": r[0],
            "name": r[1],
            "email": "-",
            "age": "-",
            "created_at": "2024"
        })

    return jsonify({"data": users})


@app.route("/users", methods=["POST"])
def create_user():
    data = request.json
    print("DATA RECEIVED:", data)   # 👈 ADD THIS

    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (data["name"], "123"))

        conn.commit()
        print("USER INSERTED")   # 👈 ADD THIS

        return jsonify({"message": "User created"})
    except Exception as e:
        print("ERROR:", e)   # 👈 ADD THIS
        return jsonify({"error": str(e)})
    finally:
        conn.close()
        
# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_data = [[
        float(data["sleep_hours"]),
        float(data["tremor"]),
        float(data["fatigue_level"]),
        float(data["mood_score"])
    ]]

    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    prediction = "Anomaly" if result[0] == -1 else "Normal"

    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO records (user_id, sleep_hours, tremor, fatigue_level, mood_score, prediction)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data["user_id"],
        data["sleep_hours"],
        data["tremor"],
        data["fatigue_level"],
        data["mood_score"],
        prediction
    ))

    conn.commit()
    conn.close()

    return jsonify({"prediction": prediction})

# ---------------- HISTORY ----------------
@app.route("/history/<int:user_id>", methods=["GET"])
def history(user_id):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    c.execute("""
        SELECT sleep_hours, tremor, fatigue_level, mood_score, prediction, timestamp
        FROM records
        WHERE user_id=?
        ORDER BY timestamp ASC
    """, (user_id,))

    rows = c.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            "sleep_hours": r[0],
            "tremor": r[1],
            "fatigue_level": r[2],
            "mood_score": r[3],
            "prediction": r[4],
            "timestamp": r[5]
        })

    return jsonify(data)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)