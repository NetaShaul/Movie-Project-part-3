from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import pickle
import re
import lzma
import os
from assets_data_prep import prepare_data
from difflib import get_close_matches
import numpy as np

app = Flask(__name__)
basedir = os.path.dirname(os.path.realpath(__file__))

# -------------------------
# Load Model
model = joblib.load(os.path.join(basedir, "trained_model.pkl"))

# -------------------------
# Load mappings
mappings_path = os.path.join(basedir, "mappings.pkl.xz")
if os.path.exists(mappings_path):
    with lzma.open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
else:
    with open(os.path.join(basedir, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)

# -------------------------
# Load lists
with open(os.path.join(basedir, 'movie_names_map.pkl'), 'rb') as f:
    movie_list = pickle.load(f)

with open(os.path.join(basedir, 'director_names_map.pkl'), 'rb') as f:
    directors_map = pickle.load(f)

# -------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()

# Clean movie names
for movie in movie_list:
    movie["clean"] = clean_text(movie["primaryTitle"])

all_clean_names = [m["clean"] for m in movie_list]

# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
@app.route("/get_genres")
def get_genres():
    genres = mappings.get("unique_genres", [])
    cleaned = ["Unknown" if g in ["\\N", "\\\\N"] else g for g in genres]
    return jsonify(sorted(set(cleaned)))

# -------------------------
@app.route("/get_directors")
def get_directors():
    directors_list = []
    for k, v in directors_map.items():
        clean_name = "Unknown" if (isinstance(v, float) and np.isnan(v)) else v
        directors_list.append({"id": k, "name": clean_name})
    
    return jsonify(directors_list)

# -------------------------
@app.route("/check_movie", methods=["POST"])
def check_movie():
    raw_name = request.get_json(force=True).get("movie_name", "")
    cleaned = clean_text(raw_name)

    exact_matches = [m for m in movie_list if m["clean"] == cleaned]

    if exact_matches:
        matches = exact_matches
    else:
        close = get_close_matches(cleaned, all_clean_names, n=5, cutoff=0.6)
        matches = [m for m in movie_list if m["clean"] in close]

    if matches:
        return jsonify({
            "status": "need_confirmation",
            "options": [
                {
                    "name": m["primaryTitle"],
                    "year": m["startYear"],
                    "tconst": m["tconst"]
                }
                for m in matches
            ]
        })

    return jsonify({"status": "not_found"})

# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Protection if no JSON is provided
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Ensure genres always exists
        data['genres'] = data.get('genres', '')

        # Validation: Required fields
        required_fields = ['tconst', 'startYear', 'runtimeMinutes']
        for field in required_fields:
            if str(data.get(field, "")).strip() == "":
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        # Validation: Numeric fields
        try:
            data['startYear'] = int(float(data['startYear']))
            data['runtimeMinutes'] = float(data['runtimeMinutes'])
        except (ValueError, TypeError):
            return jsonify({"error": "startYear and runtimeMinutes must be numeric"}), 400

        # -------------------------
        # Create DataFrame
        df_input = pd.DataFrame([data])

        # directors (optional, but ensure it exists)
        df_input['directors'] = data.get('directors', '')

        # tconst fallback
        df_input['tconst'] = data.get('tconst', 'tt_dummy')

        # -------------------------
        # Preparation + Prediction
        df_prepared = prepare_data(df_input)
        prediction = model.predict(df_prepared)

        # -------------------------
        return jsonify({
            "predicted_rating": round(float(prediction[0]), 3)
        }), 200

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

# -------------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=True)