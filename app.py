from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load model dan label encoder
try:
    model = joblib.load("model/best_model.pkl")
    le = joblib.load("model/label_encoder.pkl")
except FileNotFoundError:
    # Handle error if files are not found
    print("Error: Model or label encoder file not found.")
    exit()


# Load data severity
try:
    df_severity_loaded = pd.read_csv('dataset/Symptom-severity.csv')
    severity_map = dict(zip(df_severity_loaded['Symptom'].str.lower().str.strip(), df_severity_loaded['weight']))
except FileNotFoundError:
    print("Error: Symptom-severity.csv not found.")
    exit()

# Tentukan fitur yang digunakan saat pelatihan
try:
    feature_cols_in_model = model.feature_names_in_
except AttributeError:
    # Fallback if feature_names_in_ is not available (e.g., older scikit-learn)
    # You might need to load the original training data or store the column names
    # For now, let's assume a fixed number of symptoms based on your notebook
    feature_cols_in_model = [f'Symptom_{i}' for i in range(1, 18)]


app = FastAPI()

class SymptomInput(BaseModel):
    symptoms: list[str]

@app.post("/predict")
def predict_disease(data: SymptomInput):
    """
    Endpoint untuk memprediksi penyakit berdasarkan daftar gejala.
    """
    input_symptoms_list = data.symptoms
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols_in_model)
    normalized_input = [sym.strip().lower().replace('_', ' ') for sym in input_symptoms_list]

    for i, symptom in enumerate(normalized_input):
        if i < len(feature_cols_in_model):
            severity_score = severity_map.get(symptom, 0)
            input_df.iloc[0, i] = severity_score


    try:
        prediction = model.predict(input_df)[0]
        predicted_label = le.inverse_transform([prediction])[0]
        return {"predicted_disease": predicted_label}
    except Exception as e:
        return {"error": str(e)}

@app.get("/symptoms")
def get_all_symptoms():
    symptoms = set()
    for col in column_symptom_map.values():
        symptoms.update(col)
    return sorted(list(symptoms))
