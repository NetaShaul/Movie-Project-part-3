from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import pickle
import re
import lzma
import os
from assets_data_prep import prepare_data
import numpy as np

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)
basedir = os.path.dirname(os.path.realpath(__file__))

# -------------------------
# Load Model
# -------------------------
model = joblib.load(os.path.join(basedir, "trained_model.pkl"))

# -------------------------
# Load mappings
# -------------------------
mappings_path = os.path.join(basedir, "mappings.pkl.xz")
if os.path.exists(mappings_path):
    with lzma.open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
else:
    with open(os.path.join(basedir, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)

# -------------------------
# Load lists
# -------------------------
with open(os.path.join(basedir, 'movie_names_map.pkl'), 'rb') as f:
    movie_list = pickle.load(f)

with open(os.path.join(basedir, 'director_names_map.pkl'), 'rb') as f:
    directors_map = pickle.load(f)

# -------------------------
# Text cleaning function
# -------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()

# Preprocess movie names
for movie in movie_list:
    movie["clean"] = clean_text(movie["primaryTitle"])

# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# Get genres
# -------------------------
@app.route("/get_genres")
def get_genres():
    genres = mappings.get("unique_genres", [])
    cleaned = ["Unknown" if g in ["\\N", "\\\\N"] else g for g in genres]
    return jsonify(sorted(set(cleaned)))

# -------------------------
# Get directors
# -------------------------
@app.route("/get_directors")
def get_directors():
    try:
        directors_list = []
        for k, v in directors_map.items():
            clean_name = "Unknown" if (isinstance(v, float) and np.isnan(v)) else v
            directors_list.append({"id": k, "name": clean_name})

        return jsonify(directors_list)

    except Exception as e:
        return jsonify({"error": f"Failed to load directors: {str(e)}"}), 500

# -------------------------
# Check movie name
# -------------------------
@app.route("/check_movie", methods=["POST"])
def check_movie():
    try:
        data = request.get_json()

        if not data or "movie_name" not in data:
            return jsonify({"error": "Missing movie_name field"}), 400

        raw_name = data.get("movie_name", "")
        cleaned = clean_text(raw_name)

        # Simple matching example (you can expand)
        matches = [
            {"name": m["primaryTitle"], "year": m.get("startYear"), "tconst": m.get("tconst")}
            for m in movie_list if cleaned in m["clean"]
        ][:5]

        if matches:
            return jsonify({"status": "need_confirmation", "options": matches})
        else:
            return jsonify({"status": "no_match"})

    except Exception as e:
        return jsonify({"error": f"Movie check failed: {str(e)}"}), 500

# -------------------------
# Predict endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # -------- Validate input existence --------
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # -------- Required fields --------
        required_fields = ["tconst", "startYear", "runtimeMinutes", "genres", "directors"]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # -------- Type validation --------
        try:
            start_year = int(data["startYear"])
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value for startYear"}), 400

        try:
            runtime = float(data["runtimeMinutes"])
            if runtime <= 0:
                raise ValueError
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid runtimeMinutes"}), 400

        # -------- Create DataFrame --------
        df = pd.DataFrame([data])

        # -------- Apply preprocessing --------
        processed = prepare_data(df)

        # -------- Model prediction --------
        prediction = model.predict(processed)

        return jsonify({
            "predicted_rating": float(prediction[0])
        })

    except Exception as e:
        # -------- Generic server error --------
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=True)