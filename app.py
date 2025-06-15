from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deep_translator import GoogleTranslator
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*", 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dan label encoder
try:
    model = joblib.load("model/best_model.pkl")
    le = joblib.load("model/label_encoder.pkl")
except FileNotFoundError:
    print("Model atau label encoder tidak ditemukan.")
    exit()

# Load severity map
try:
    df_severity_loaded = pd.read_csv("dataset/Symptom-severity.csv")
    severity_map = dict(zip(df_severity_loaded['Symptom'].str.lower().str.strip(), df_severity_loaded['weight']))
except FileNotFoundError:
    print("Severity file tidak ditemukan.")
    exit()

# Load deskripsi penyakit
try:
    df_description = pd.read_csv("dataset/symptom_Description.csv")
    desc_map = dict(zip(df_description['Disease'].str.lower().str.strip(), df_description['Description']))
except FileNotFoundError:
    print("Deskripsi penyakit tidak ditemukan.")
    desc_map = {}

# Load precaution penyakit
try:
    df_precaution = pd.read_csv("dataset/symptom_precaution.csv")
    precaution_map = {}
    for _, row in df_precaution.iterrows():
        disease = row['Disease'].strip().lower()
        precaution_map[disease] = [
            row[col] for col in df_precaution.columns 
            if 'Precaution' in col and pd.notnull(row[col])
        ]
except FileNotFoundError:
    print("File precaution tidak ditemukan.")
    precaution_map = {}

# Fitur model
try:
    feature_cols_in_model = model.feature_names_in_
except AttributeError:
    feature_cols_in_model = [f'Symptom_{i}' for i in range(1, 18)]

# Schema
class SymptomInput(BaseModel):
    symptoms: list[str]

# Inisialisasi translator sekali saja untuk efisiensi
translator = GoogleTranslator(source='en', target='id')

@app.post("/predict")
def predict_disease(data: SymptomInput):
    input_symptoms = [s.strip().lower().replace('_', ' ') for s in data.symptoms]
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols_in_model)

    for i, symptom in enumerate(input_symptoms):
        if i < len(feature_cols_in_model):
            input_df.iloc[0, i] = severity_map.get(symptom, 0)

    try:
        prediction = model.predict(input_df)[0]
        predicted_label = le.inverse_transform([prediction])[0].strip()
        disease_key = predicted_label.lower()

        # Ambil deskripsi & precaution
        description = desc_map.get(disease_key, "Deskripsi tidak tersedia.")
        precautions = precaution_map.get(disease_key, ["Informasi pencegahan tidak tersedia."])

        # Translasi label, deskripsi, dan precaution
        translated_label = translator.translate(predicted_label)
        translated_description = translator.translate(description)
        translated_precautions = [translator.translate(p) for p in precautions]

        return {
            "predicted_disease": predicted_label,
            "predicted_disease_translated": translated_label,
            "description": translated_description,
            "precaution": translated_precautions
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/symptoms")
def get_all_symptoms():
    symptoms_list = sorted(list(severity_map.keys()))
    translated_symptoms = []

    for symptom in symptoms_list:
        try:
            translated = translator.translate(symptom)
            translated_symptoms.append({"original": symptom, "translated": translated})
        except:
            translated_symptoms.append({"original": symptom, "translated": symptom})

    return translated_symptoms
