from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "data.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# Imena kolona
# data.csv nema header red, pa imena kolona dodajemo ručno na osnovu opisa dataset-a
column_names = [
    "id",
    "diagnosis",
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius se", "texture se", "perimeter se", "area se", "smoothness se",
    "compactness se", "concavity se", "concave points se", "symmetry se", "fractal dimension se",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]


# Učitavanje podataka
dataset = pd.read_csv(DATA_PATH, header=None, names=column_names)

print("Prvih nekoliko redova")
print(dataset.head())

print("\nDimenzije skupa")
print(dataset.shape)


# Provera nedostajućih vrednosti i duplikata
print("\nUkupan broj nedostajućih vrednosti")
print(dataset.isna().sum().sum())

print("\nBroj duplikata")
print(dataset.duplicated().sum())


# Uklanjanje id kolone
# id je samo identifikator pacijenta i ne nosi korisnu informaciju za predikciju dijagnoze
dataset = dataset.drop(columns=["id"])


# Enkodiranje ciljne promenljive
# Malignant (M) se enkodira kao 1, benign (B) kao 0
# Malignant je pozitivna klasa jer je u medicinskom kontekstu opasnije
# propustiti maligni slučaj (False Negative) nego pogrešno označiti benigni kao sumnjiv (False Positive)
dataset["diagnosis"] = dataset["diagnosis"].map({"M": 1, "B": 0})

print("\nRaspodela ciljnih klasa")
print(dataset["diagnosis"].value_counts())


# EDA - osnovne statistike
print("\nOsnovne statistike numeričkih kolona")
print(dataset.describe())


# EDA - raspodela ciljnih klasa (grafik)
class_counts = dataset["diagnosis"].value_counts()

plt.figure(figsize=(6, 5))
class_counts.plot(kind="bar")
plt.title("Raspodela klasa (0 = benigni, 1 = maligni)")
plt.xlabel("Diagnosis")
plt.ylabel("Broj uzoraka")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "class_distribution.png")
plt.show()


# EDA - korelaciona matrica svih atributa
correlation_matrix = dataset.corr()

plt.figure(figsize=(20, 16))
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Korelacija")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Korelaciona matrica svih atributa")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_matrix.png")
plt.show()

print("\nKorelacija svakog atributa sa diagnosis (sortirano)")
print(correlation_matrix["diagnosis"].sort_values(ascending=False))


# Ulazni atributi i ciljna promenljiva
X = dataset.drop(columns=["diagnosis"])
y = dataset["diagnosis"]


# Podela na trening i test skup
# stratify čuva sličnu raspodelu klasa (0/1) i u trening i u test skupu
# random_state=42 osigurava da podela bude ista svaki put kad se kod pokrene
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDimenzije trening skupa")
print(X_train.shape)

print("\nDimenzije test skupa")
print(X_test.shape)

print("\nRaspodela klasa u trening skupu")
print(y_train.value_counts(normalize=True).round(3))

print("\nRaspodela klasa u test skupu")
print(y_test.value_counts(normalize=True).round(3))


# Skaliranje atributa
# StandardScaler transformiše svaki atribut tako da ima srednju vrednost 0 i standardnu devijaciju 1
# fit_transform se koristi SAMO na trening skupu (scaler uči prosek i std. devijaciju isključivo iz njega)
# transform (bez fit) se primenjuje na test skup, koristeći već naučene vrednosti iz trening skupa
# Ovim se sprečava data leakage - test skup ne utiče na parametre skaliranja
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Modeli za poređenje
# random_state=42 koristi se svuda gde postoji nasumičnost, radi reproduktivnosti rezultata
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

trained_models = {}
results = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Treniranje i evaluacija modela kroz cross-validaciju
for model_name, classifier in models.items():
    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    # Cross-validation na trening skupu (skaliranom)
    # recall se koristi kao glavna metrika jer je u medicinskom kontekstu
    # najvažnije ne propustiti maligni slučaj (minimizovati False Negative)
    cv_recall = cross_val_score(classifier, X_train_scaled, y_train, cv=cv, scoring="recall")
    cv_accuracy = cross_val_score(classifier, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(classifier, X_train_scaled, y_train, cv=cv, scoring="f1")
    cv_roc_auc = cross_val_score(classifier, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

    # Treniranje finalnog modela na celom (skaliranom) trening skupu
    classifier.fit(X_train_scaled, y_train)
    trained_models[model_name] = classifier

    y_pred = classifier.predict(X_test_scaled)
    y_prob = classifier.predict_proba(X_test_scaled)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_prob)

    print("CV Recall (mean):", cv_recall.mean().round(3))
    print("CV Accuracy (mean):", cv_accuracy.mean().round(3))
    print("CV F1 (mean):", cv_f1.mean().round(3))
    print("CV ROC-AUC (mean):", cv_roc_auc.mean().round(3))

    print("\nTest Accuracy:", round(test_accuracy, 3))
    print("Test Recall:", round(test_recall, 3))
    print("Test F1:", round(test_f1, 3))
    print("Test ROC-AUC:", round(test_roc_auc, 3))

    print("\nClassification report")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion matrix")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        "model": model_name,
        "cv_recall": cv_recall.mean(),
        "cv_accuracy": cv_accuracy.mean(),
        "cv_f1": cv_f1.mean(),
        "cv_roc_auc": cv_roc_auc.mean(),
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc
    })


# Poređenje modela
results_data = pd.DataFrame(results)

print("\n" + "=" * 50)
print("Poređenje modela (sortirano po CV recall)")
print("=" * 50)
print(results_data.sort_values(by="cv_recall", ascending=False))

results_data.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)


# Fino podešavanje hiperparametara (Grid Search)
# Podešavamo hiperparametre za 3 najbolja modela po cv_recall, da damo fer šansu i ostalima
# pre nego što proglasimo finalnog pobednika
top_models = results_data.sort_values(by="cv_recall", ascending=False)["model"].head(3).tolist()
print("\nModeli za fino podešavanje hiperparametara:", top_models)


# Prostor pretrage hiperparametara za svaki model
# C kod Logističke i SVM kontroliše jačinu regularizacije (manje C = jača regularizacija)
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100]
    },
    "SVM": {
        "estimator__C": [0.1, 1, 10, 100],
        "estimator__kernel": ["linear", "rbf"]
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11]
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_leaf": [1, 5, 10]
    }
}

best_models = {}

for model_name in top_models:
    print("\n" + "=" * 50)
    print("Grid Search:", model_name)
    print("=" * 50)

    base_model = models[model_name]
    grid = GridSearchCV(
        base_model,
        param_grid=param_grids[model_name],
        cv=cv,
        scoring="recall",
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)

    print("Najbolji hiperparametri:", grid.best_params_)
    print("Najbolji CV recall:", round(grid.best_score_, 3))

    best_models[model_name] = grid.best_estimator_

    y_pred = grid.best_estimator_.predict(X_test_scaled)
    print("\nTest recall sa najboljim hiperparametrima:", round(recall_score(y_test, y_pred), 3))
    print("Test accuracy sa najboljim hiperparametrima:", round(accuracy_score(y_test, y_pred), 3))


# Izbor finalnog modela
# SVM sa optimizovanim hiperparametrima (C, kernel) ima najviši CV recall nakon Grid Search-a
# Finalni model se bira na osnovu rezultata evaluacije i diskusije (CV recall kao glavni kriterijum)
final_model_name = "SVM"
final_model = best_models[final_model_name]

print("\n" + "=" * 50)
print("FINALNI MODEL:", final_model_name)
print("=" * 50)

y_pred_final = final_model.predict(X_test_scaled)
y_prob_final = final_model.predict_proba(X_test_scaled)[:, 1]

print("\nFinalne metrike na test skupu")
print("Accuracy:", round(accuracy_score(y_test, y_pred_final), 3))
print("Recall:", round(recall_score(y_test, y_pred_final), 3))
print("F1:", round(f1_score(y_test, y_pred_final), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_final), 3))

print("\nClassification report (finalni model)")
print(classification_report(y_test, y_pred_final, zero_division=0))

print("\nConfusion matrix (finalni model)")
final_cm = confusion_matrix(y_test, y_pred_final)
print(final_cm)

# Vizuelni prikaz finalne matrice konfuzije
plt.figure(figsize=(6, 5))
plt.imshow(final_cm, cmap="Blues")
plt.title(f"Matrica konfuzije - {final_model_name}")
plt.xlabel("Predviđena klasa")
plt.ylabel("Stvarna klasa")
plt.xticks([0, 1], ["Benigni (0)", "Maligni (1)"])
plt.yticks([0, 1], ["Benigni (0)", "Maligni (1)"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(final_cm[i, j]), ha="center", va="center", fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "final_confusion_matrix.png")
plt.show()


# Feature importance preko Random Forest-a
# SVM (finalni model) nema ugrađen način da pokaže važnost atributa, pa se
# Random Forest koristi kao dodatni alat za uvid u to koje su osobine najuticajnije
rf_for_importance = best_models["Random Forest"]
importances = rf_for_importance.feature_importances_

feature_importance_data = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nNajvažniji atributi prema Random Forest-u (top 10)")
print(feature_importance_data.head(10))

feature_importance_data.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

# Vizuelni prikaz top 10 najvažnijih atributa
plt.figure(figsize=(10, 6))
top_10 = feature_importance_data.head(10)
plt.barh(top_10["feature"], top_10["importance"])
plt.xlabel("Važnost atributa")
plt.title("10 najvažnijih atributa (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png")
plt.show()


# Eksportovanje finalnog modela
# Scaler i klasifikator se spajaju u jedan Pipeline pre čuvanja, tako da se
# skaliranje automatski primenjuje na nove podatke pre predikcije
final_pipeline = Pipeline(steps=[
    ("scaler", scaler),
    ("classifier", final_model)
])

MODEL_PATH = MODELS_DIR / "breast_cancer_svm.joblib"
joblib.dump(final_pipeline, MODEL_PATH)

print("\n" + "=" * 50)
print("Model sačuvan u:", MODEL_PATH)
print("=" * 50)