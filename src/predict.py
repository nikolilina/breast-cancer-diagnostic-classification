from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "breast_cancer_svm.joblib"


# Učitavanje sačuvanog modela
# Model se ovde ne trenira ponovo, već se koristi prethodno sačuvan Pipeline
# (scaler + SVM klasifikator zajedno)
model = joblib.load(MODEL_PATH)


# Novi primer za predikciju
# Nazivi kolona moraju biti isti kao u trening skupu (bez id i diagnosis)
new_patient = pd.DataFrame([
    {
        "mean radius": 15.5, "mean texture": 20.0, "mean perimeter": 100.0, "mean area": 750.0,
        "mean smoothness": 0.1, "mean compactness": 0.12, "mean concavity": 0.1,
        "mean concave points": 0.07, "mean symmetry": 0.18, "mean fractal dimension": 0.06,
        "radius se": 0.4, "texture se": 1.2, "perimeter se": 3.0, "area se": 40.0,
        "smoothness se": 0.007, "compactness se": 0.025, "concavity se": 0.03,
        "concave points se": 0.012, "symmetry se": 0.02, "fractal dimension se": 0.003,
        "worst radius": 17.5, "worst texture": 26.0, "worst perimeter": 115.0, "worst area": 900.0,
        "worst smoothness": 0.14, "worst compactness": 0.3, "worst concavity": 0.3,
        "worst concave points": 0.14, "worst symmetry": 0.29, "worst fractal dimension": 0.08
    }
])


# Predikcija klase
prediction = model.predict(new_patient)
predicted_class = prediction[0]


# Opis klasa
class_names = {
    0: "benigni tumor",
    1: "maligni tumor"
}


print("Ulazni podaci")
print(new_patient)

print("\nPredikcija modela")
print(predicted_class, "-", class_names.get(predicted_class, "nepoznata klasa"))


# Verovatnoće po klasama
probabilities = model.predict_proba(new_patient)[0]

print("\nVerovatnoće po klasama")
print("Benigni (0):", round(probabilities[0], 3))
print("Maligni (1):", round(probabilities[1], 3))