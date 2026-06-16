# Movie Rating Prediction System - Part 3

**Name:** Neta Or Shaul  
**Repository:** https://github.com/NetaShaul/Movie-Project-part-3

---

## 1. Project Overview

This project is a web-based Machine Learning application designed to predict movie ratings.  
Users can input movie features such as title, release year, runtime, genres, and directors, and receive a real-time predicted rating.

The system is based on a model trained in Part 2 and is deployed as a web service using Flask.

---

## 2. Technical Architecture

The system consists of three main layers:

### Frontend
- Implemented in index.html
- Provides a user-friendly interface
- Collects input data
- Sends requests using Fetch API
- Displays predictions without page reload

### Backend
- Implemented in api.py using Flask
- Handles API requests (/ and /predict)
- Validates input data
- Calls preprocessing and prediction logic

### Model & Pipeline
- Pre-trained model: trained_model.pkl
- Preprocessing: assets_data_prep.py
- Includes feature engineering and mappings

---

## 3. Installation & Setup

Clone the repository:

git clone https://github.com/NetaShaul/Movie-Project-part-3  
cd Movie-Project-part-3

Create virtual environment:

python -m venv venv

Activate environment:

Mac/Linux:  
source venv/bin/activate  

Windows:  
venv\Scripts\activate  

Install dependencies:

pip install -r requirements.txt

---

## 4. How to Run the Application

Run the server:

python api.py

Then open your browser:

http://localhost:5000

---

## 5. Input Fields

- Movie Name – string (English letters, required)
- Release Year – integer (1890–2026, required)
- Runtime – float (>0, required)
- Genres – comma-separated string (optional, default: Unknown)
- Directors – comma-separated IDs (optional)
---

## 6. Prediction Flow

1. User fills the form in the frontend  
2. Data is sent as JSON to /predict  
3. Flask receives the request  
4. Data is converted into a Pandas DataFrame  
5. prepare_data() processes the input  
6. The trained model runs predict()  
7. Result is returned as JSON  

Example response:

{
  "predicted_rating": 7.4
}

---

## 7. Project Files Description

- **api.py** – Flask backend for handling requests and predictions  
- **index.html** – Frontend UI for user input and results display  
- **assets_data_prep.py** – Preprocesses input data for the model  
- **trained_model.pkl** – Pre-trained model used for rating prediction  
- **mappings.pkl.xz** – Stores preprocessing mappings and statistics  
- **movie_names_map.pkl** – Supports movie name search and suggestions  
- **director_names_map.pkl** – Maps director IDs to names  
- **requirements.txt** – Lists required Python dependencies
---

## 8. Error Handling

The application includes validation and error handling on the backend.

- 400 Bad Request - Returned when input data is missing or invalid . for example, missing fields, incorrect data types.
- 500 Internal Server Error - Returned for unexpected errors during processing or prediction

All errors are returned as JSON responses and are displayed to the user in the frontend.

---

## 9. API Endpoints

- GET / → Loads index.html  
- POST /predict → Returns predicted rating  
- GET /get_genres → Returns available genres  
- GET /get_directors → Returns available directors  
- POST /check_movie → Suggests matching movies  

## Final Notes

Before submission, verify:

1. The server runs without errors  
2. The UI loads at http://localhost:5000  
3. Predictions work correctly  

