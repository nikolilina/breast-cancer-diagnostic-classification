from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Osnovna podešavanja Streamlit aplikacije
st.set_page_config(
    page_title="Breast Cancer Diagnostic App",
    page_icon="🩺",
    layout="centered"
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "breast_cancer_svm.joblib"
DATA_PATH = BASE_DIR / "data" / "data.csv"


# Imena kolona (dataset nema header red)
column_names = [
    "id", "diagnosis",
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius se", "texture se", "perimeter se", "area se", "smoothness se",
    "compactness se", "concavity se", "concave points se", "symmetry se", "fractal dimension se",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Atributi su grupisani u tri sekcije (mean, se, worst) radi preglednosti forme
mean_features = column_names[2:12]
se_features = column_names[12:22]
worst_features = column_names[22:32]


# Učitavanje sačuvanog modela (Pipeline: scaler + SVM klasifikator)
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Učitavanje par primera pacijenata iz dataset-a, za brzo popunjavanje forme
@st.cache_data
def load_sample_patients():
    df = pd.read_csv(DATA_PATH, header=None, names=column_names)
    # Uzima se par benignih i par malignih primera radi raznovrsnosti izbora
    benign_samples = df[df["diagnosis"] == "B"].head(3)
    malignant_samples = df[df["diagnosis"] == "M"].head(3)
    return pd.concat([benign_samples, malignant_samples]).reset_index(drop=True)


model = load_model()
sample_patients = load_sample_patients()


# Opis klasa
class_names = {
    0: "Benigni tumor",
    1: "Maligni tumor"
}


# Naslov i opis aplikacije
st.title("Breast Cancer Diagnostic App")

st.write(
    "Aplikacija predviđa da li je tumor benigni ili maligni na osnovu "
    "30 numeričkih karakteristika ćelijskog jezgra. "
    "Izaberite primer pacijenta radi brzog popunjavanja forme, ili ručno unesite vrednosti."
)


# Izbor primera pacijenta za automatsko popunjavanje forme
st.subheader("Brzo popunjavanje primerom")

sample_labels = [
    f"Pacijent {i + 1} (stvarna dijagnoza: {row['diagnosis']})"
    for i, row in sample_patients.iterrows()
]
sample_labels.insert(0, "-- ručni unos (bez popunjavanja) --")

selected_label = st.selectbox("Izaberite primer pacijenta", sample_labels)

if selected_label != "-- ručni unos (bez popunjavanja) --":
    selected_index = sample_labels.index(selected_label) - 1
    selected_patient = sample_patients.iloc[selected_index]
else:
    selected_patient = None


# Vraća početnu vrednost polja - iz izabranog primera ako postoji, inače 0.0
def default_value(feature_name):
    if selected_patient is not None:
        return float(selected_patient[feature_name])
    return 0.0


# Forma za unos atributa, grupisana u tri proširive sekcije
st.subheader("Karakteristike ćelijskog jezgra")

input_values = {}

with st.expander("Mean vrednosti (prosečne vrednosti)", expanded=True):
    for feature in mean_features:
        input_values[feature] = st.number_input(
            feature, value=default_value(feature), format="%.5f", key=feature
        )

with st.expander("SE vrednosti (standardna greška)"):
    for feature in se_features:
        input_values[feature] = st.number_input(
            feature, value=default_value(feature), format="%.5f", key=feature
        )

with st.expander("Worst vrednosti (najekstremnije izmerene vrednosti)"):
    for feature in worst_features:
        input_values[feature] = st.number_input(
            feature, value=default_value(feature), format="%.5f", key=feature
        )


# Predikcija
if st.button("Predvidi dijagnozu"):
    input_data = pd.DataFrame([input_values])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Rezultat")

    if prediction == 1:
        st.error(f"Predikcija: {class_names[prediction]}")
    else:
        st.success(f"Predikcija: {class_names[prediction]}")

    st.write("Verovatnoće po klasama")
    probability_data = pd.DataFrame({
        "Klasa": ["Benigni (0)", "Maligni (1)"],
        "Verovatnoća": [round(probabilities[0], 3), round(probabilities[1], 3)]
    })
    st.dataframe(probability_data, hide_index=True)

    if selected_patient is not None:
        st.write("Stvarna dijagnoza izabranog primera pacijenta:", selected_patient["diagnosis"])


# Pokretanje UI aplikacije iz glavnog foldera projekta:
# uv run streamlit run app/ui.py