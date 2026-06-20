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

# Tipičan opseg vrednosti za svaki atribut (iz dataset-a), prikazuje se kao
# pomoć korisniku pri ručnom unosu da zna u kom redu veličine broj treba da bude
feature_help = {
    "mean radius": "Tipičan opseg: 6.98–28.11",
    "mean texture": "Tipičan opseg: 9.71–39.28",
    "mean perimeter": "Tipičan opseg: 43.8–188.5",
    "mean area": "Tipičan opseg: 143.5–2501.0",
    "mean smoothness": "Tipičan opseg: 0.0526–0.1634 (decimalni broj, ne procenat)",
    "mean compactness": "Tipičan opseg: 0.0194–0.3454",
    "mean concavity": "Tipičan opseg: 0.0000–0.4268",
    "mean concave points": "Tipičan opseg: 0.0000–0.2012",
    "mean symmetry": "Tipičan opseg: 0.1060–0.3040",
    "mean fractal dimension": "Tipičan opseg: 0.0500–0.0974",
    "radius se": "Tipičan opseg: 0.11–2.87",
    "texture se": "Tipičan opseg: 0.36–4.89",
    "perimeter se": "Tipičan opseg: 0.76–21.98",
    "area se": "Tipičan opseg: 6.8–542.2",
    "smoothness se": "Tipičan opseg: 0.0017–0.0311",
    "compactness se": "Tipičan opseg: 0.0023–0.1354",
    "concavity se": "Tipičan opseg: 0.0000–0.3960",
    "concave points se": "Tipičan opseg: 0.0000–0.0528",
    "symmetry se": "Tipičan opseg: 0.0079–0.0790",
    "fractal dimension se": "Tipičan opseg: 0.0009–0.0298",
    "worst radius": "Tipičan opseg: 7.93–36.04",
    "worst texture": "Tipičan opseg: 12.02–49.54",
    "worst perimeter": "Tipičan opseg: 50.4–251.2",
    "worst area": "Tipičan opseg: 185.2–4254.0",
    "worst smoothness": "Tipičan opseg: 0.0712–0.2226",
    "worst compactness": "Tipičan opseg: 0.0273–1.0580",
    "worst concavity": "Tipičan opseg: 0.0000–1.2520",
    "worst concave points": "Tipičan opseg: 0.0000–0.2910",
    "worst symmetry": "Tipičan opseg: 0.1565–0.6638",
    "worst fractal dimension": "Tipičan opseg: 0.0550–0.2075"
}


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


# Stilizovanje aplikacije - paleta inspirisana roze trakom, simbolom
# meseca borbe protiv raka dojke (Breast Cancer Awareness Month)
st.markdown("""
<style>
    :root {
        --ribbon-pink: #E91E63;
        --ribbon-pink-dark: #AD1457;
        --ribbon-pink-light: #FCE4EC;
        --text-dark: #2D2D2D;
    }

    h1, h2, h3 {
        color: var(--ribbon-pink-dark) !important;
    }

    .stButton > button {
        background-color: var(--ribbon-pink);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
    }

    .stButton > button:hover {
        background-color: var(--ribbon-pink-dark);
        color: white;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--ribbon-pink-light);
        border-radius: 8px;
    }

    [data-testid="stSidebar"] {
        background-color: var(--ribbon-pink-light);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-dark) !important;
    }

    [data-testid="stSidebar"] h3 {
        color: var(--ribbon-pink-dark) !important;
    }

    .awareness-banner {
        background-color: var(--ribbon-pink-light);
        border-left: 4px solid var(--ribbon-pink);
        padding: 0.7rem 1rem;
        border-radius: 6px;
        margin: 0.6rem 0 1.4rem 0;
        font-size: 0.9rem;
        color: var(--text-dark);
        line-height: 1.5;
    }

    .ribbon-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.6rem;
    }

    /* Radio dugmad za izbor načina unosa - stilizovano kao roze "pilule" */
    div[role="radiogroup"] {
        gap: 0.6rem;
    }

    div[role="radiogroup"] label {
        background-color: var(--ribbon-pink-light);
        border: 1.5px solid var(--ribbon-pink);
        border-radius: 20px;
        padding: 0.4rem 1rem;
    }

    div[role="radiogroup"] label:has(input:checked) {
        background-color: var(--ribbon-pink);
    }

    div[role="radiogroup"] label:has(input:checked) p {
        color: white !important;
    }

    /* st.pills dugmad za izbor konkretnog pacijenta */
    [data-testid="stPills"] button {
        border-color: var(--ribbon-pink) !important;
    }

    [data-testid="stPills"] button[kind="pillsActive"] {
        background-color: var(--ribbon-pink) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# Naslov sa emoji simbolom roze trake 
st.markdown(
    '<div class="ribbon-header"><span style="font-size:2.2rem;">🎗️</span>'
    '<h1 style="margin:0;">Breast Cancer Diagnostic App</h1></div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="awareness-banner">
    <strong>Rano otkrivanje</strong> značajno poboljšava ishod lečenja raka dojke.
    Ova aplikacija je edukativni projekat: NE ZAMENJUJE LEKARSKI PREGLED NITI DIJAGNOZU!
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Aplikacija predviđa da li je tumor benigni ili maligni na osnovu "
    "30 numeričkih karakteristika ćelijskog jezgra. Izaberite primer pacijenta "
    "radi brzog popunjavanja forme, ili ručno unesite vrednosti."
)


# Edukativni sadržaj u bočnoj traci
with st.sidebar:
    st.markdown("### O ranom otkrivanju")
    st.markdown(
        """
        Redovni samopregled i mamografski skrining su ključni alati za rano
        otkrivanje raka dojke, jer tumor otkriven u ranijoj fazi obično ima
        više opcija lečenja.

        **Mogući znaci na koje treba obratiti pažnju:**
        - novo zadebljanje ili kvržica u dojci ili pazuhu
        - promena veličine ili oblika dojke
        - promene na koži (uvlačenje, crvenilo, izgled "kore pomorandže")
        - iscedak iz bradavice koji nije mleko

        Ovi znaci ne znače sami po sebi da je u pitanju maligni tumor, mnoge
        promene su benigne - ali svaku novu ili neuobičajenu promenu treba
        proveriti kod lekara.
        """
    )
    st.divider()
    st.markdown("### O ovom modelu")
    st.markdown(
        """
        Model (SVM) je treniran na Wisconsin Breast Cancer dataset-u i
        klasifikuje tumor na osnovu 30 karakteristika ćelijskog jezgra
        dobijenih iz biopsije, ne iz mamografije ili samopregleda.

        Detaljna dokumentacija metodologije nalazi se u
        `documentation/project_documentation.pdf`.
        """
    )


# Izbor primera pacijenta ili ručnog unosa
st.subheader("Izaberite jednu od ponuđenih opcija:")

unos_mode = st.radio(
    "Način unosa podataka",
    ["Unapred definisan pacijent", "Definiši sopstvene parametre"],
    horizontal=True,
    label_visibility="collapsed"
)

selected_patient = None
selected_patient_label = None

if unos_mode == "Unapred definisan pacijent":
    sample_labels = [
        f"Pacijent {i + 1} ({row['diagnosis']})"
        for i, row in sample_patients.iterrows()
    ]
    selected_patient_label = st.pills("Izaberite pacijenta", sample_labels, default=sample_labels[0])
    if selected_patient_label is not None:
        selected_index = sample_labels.index(selected_patient_label)
        selected_patient = sample_patients.iloc[selected_index]


# Vraća početnu vrednost polja - iz izabranog primera ako postoji, inače 0.0
def default_value(feature_name):
    if selected_patient is not None:
        return float(selected_patient[feature_name])
    return 0.0


input_values = {}

if unos_mode == "Definiši sopstvene parametre":
    # Forma za ručni unos atributa, grupisana u tri proširive sekcije
    # Prikazuje se SAMO kod ručnog unosa - kod gotovog pacijenta brojevi
    # se ionako ne menjaju, pa forma nije potrebna korisniku da vidi
    st.subheader("Karakteristike ćelijskog jezgra")

    with st.expander("Mean vrednosti (prosečne vrednosti)", expanded=True):
        for feature in mean_features:
            input_values[feature] = st.number_input(
                feature, value=0.0, format="%.5f", key=f"manual_{feature}",
                help=feature_help[feature]
            )

    with st.expander("SE vrednosti (standardna greška)"):
        for feature in se_features:
            input_values[feature] = st.number_input(
                feature, value=0.0, format="%.5f", key=f"manual_{feature}",
                help=feature_help[feature]
            )

    with st.expander("Worst vrednosti (najekstremnije izmerene vrednosti)"):
        for feature in worst_features:
            input_values[feature] = st.number_input(
                feature, value=0.0, format="%.5f", key=f"manual_{feature}",
                help=feature_help[feature]
            )
else:
    # Gotov pacijent - vrednosti se uzimaju direktno iz dataset-a,
    # bez prolaska kroz number_input polja (i bez rizika od "zaglavljenog" key-a)
    if selected_patient is not None:
        for feature in column_names[2:]:
            input_values[feature] = float(selected_patient[feature])


# Predikcija
if st.button("Predvidi dijagnozu"):
    if not input_values:
        st.warning("Izaberite pacijenta ili unesite parametre pre predikcije.")
    else:
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
            st.write("Stvarna dijagnoza izabranog pacijenta:", selected_patient["diagnosis"])


# Pokretanje UI aplikacije iz glavnog foldera projekta:
# uv run streamlit run app/ui.py