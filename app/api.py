from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# Kreiranje FastAPI aplikacije
app = FastAPI(
    title="Breast Cancer Diagnostic API",
    description="API za predikciju da li je tumor benigni ili maligni",
    version="1.0"
)


# Putanja do glavnog foldera projekta i sačuvanog modela
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "breast_cancer_svm.joblib"


# Učitavanje sačuvanog pipeline-a (scaler + SVM klasifikator)
model = joblib.load(MODEL_PATH)


# Opis ulaznih podataka koje API očekuje
# Pydantic automatski proverava da su sva polja prisutna i da su brojevi (float)
class PatientInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


# Opis klasa
class_names = {
    0: "benigni tumor",
    1: "maligni tumor"
}


# Početna ruta (za proveru da li je API pokrenut)
@app.get("/")
def home():
    return {
        "message": "Breast Cancer Diagnostic API is running."
    }


# Ruta za predikciju
# Pydantic polja se prebacuju nazad u imena kolona sa razmacima,
# onakva kakva su korišćena tokom treniranja
@app.post("/predict")
def predict(patient: PatientInput):
    input_data = pd.DataFrame([
        {
            "mean radius": patient.mean_radius,
            "mean texture": patient.mean_texture,
            "mean perimeter": patient.mean_perimeter,
            "mean area": patient.mean_area,
            "mean smoothness": patient.mean_smoothness,
            "mean compactness": patient.mean_compactness,
            "mean concavity": patient.mean_concavity,
            "mean concave points": patient.mean_concave_points,
            "mean symmetry": patient.mean_symmetry,
            "mean fractal dimension": patient.mean_fractal_dimension,
            "radius se": patient.radius_se,
            "texture se": patient.texture_se,
            "perimeter se": patient.perimeter_se,
            "area se": patient.area_se,
            "smoothness se": patient.smoothness_se,
            "compactness se": patient.compactness_se,
            "concavity se": patient.concavity_se,
            "concave points se": patient.concave_points_se,
            "symmetry se": patient.symmetry_se,
            "fractal dimension se": patient.fractal_dimension_se,
            "worst radius": patient.worst_radius,
            "worst texture": patient.worst_texture,
            "worst perimeter": patient.worst_perimeter,
            "worst area": patient.worst_area,
            "worst smoothness": patient.worst_smoothness,
            "worst compactness": patient.worst_compactness,
            "worst concavity": patient.worst_concavity,
            "worst concave points": patient.worst_concave_points,
            "worst symmetry": patient.worst_symmetry,
            "worst fractal dimension": patient.worst_fractal_dimension
        }
    ])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    response = {
        "prediction": int(prediction),
        "description": class_names.get(prediction, "nepoznata klasa"),
        "probabilities": {
            "benigni": round(float(probabilities[0]), 3),
            "maligni": round(float(probabilities[1]), 3)
        }
    }

    return response


# Pokretanje API-ja iz glavnog foldera projekta:
# uv run uvicorn app.api:app --reload

# Otvoriti http://127.0.0.1:8000/docs