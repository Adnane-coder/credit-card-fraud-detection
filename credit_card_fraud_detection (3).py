# === Importation des bibliothèques nécessaires ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import joblib

# === Chargement du jeu de données ===
data = pd.read_csv(r"C:\Users\MSI\Desktop\creditcard.zip")
pd.options.display.max_columns = None

print(f"Nombre de colonnes : {data.shape[1]}")
print(f"Nombre de lignes   : {data.shape[0]}")

# === Analyse initiale ===
print("\nInformations sur les données :")
data.info()
print("\nValeurs manquantes :\n", data.isnull().sum())

# === Prétraitement ===
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])
data.drop(['Time'], axis=1, inplace=True)

# === Suppression des doublons ===
if data.duplicated().any():
    data.drop_duplicates(inplace=True)

# === Visualisation : histogramme des classes ===
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data, palette='Set2')
plt.title("Répartition des transactions (0: normales, 1: fraude)")
plt.xlabel("Classe")
plt.ylabel("Nombre de transactions")
plt.show()

# === Diagramme circulaire ===
fraud_ratio = data['Class'].value_counts(normalize=True)
plt.figure(figsize=(6, 6))
plt.pie(fraud_ratio, labels=["Normale", "Fraude"], autopct="%.2f%%", colors=["skyblue", "salmon"])
plt.title("Proportion des transactions frauduleuses")
plt.show()

# === Undersampling ===
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0].sample(n=len(fraud), random_state=42)
balanced_data = pd.concat([fraud, normal], axis=0).sample(frac=1, random_state=42)

X_under = balanced_data.drop('Class', axis=1)
y_under = balanced_data['Class']

# === Oversampling (SMOTE) ===
X = data.drop('Class', axis=1)
y = data['Class']
sm = SMOTE(random_state=42)
X_over, y_over = sm.fit_resample(X, y)

# === Fonction d'entraînement et d'évaluation ===
def train_and_evaluate(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        "Régression Logistique": LogisticRegression(max_iter=1000),
        "Arbre de Décision": DecisionTreeClassifier()
    }

    for name, clf in classifiers.items():
        print(f"\n==== {name} ({title}) ====")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Précision : {precision_score(y_test, y_pred):.4f}")
        print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")

# === Évaluation avec undersampling ===
train_and_evaluate(X_under, y_under, "Undersampling")

# === Évaluation avec oversampling ===
train_and_evaluate(X_over, y_over, "Oversampling (SMOTE)")

# === Entraînement final et sauvegarde du modèle ===
final_model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)
joblib.dump(final_model, "fraud_detection_model.pkl")

# === Chargement et prédiction sur un échantillon ===
model = joblib.load("fraud_detection_model.pkl")

sample = [[-1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2396, 0.0987, 0.3637, 0.0907,
           -0.5515, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2079, 0.0257, 0.4039, 0.2514,
           -0.0183, 0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 0.378]]  # example transaction

prediction = model.predict(sample)[0]
if prediction == 1:
    print(" Transaction frauduleuse détectée")
else:
    print(" Transaction normale")

# === Matrice de confusion ===
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normale", "Fraude"])
disp.plot(cmap='Blues')
plt.title("Matrice de confusion - Modèle final")
plt.show()

# === Courbe ROC & AUC ===
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("Courbe ROC - Modèle final")
plt.xlabel("Taux de Faux Positifs")
plt.ylabel("Taux de Vrais Positifs")
plt.legend(loc="lower right")
plt.show()

# === Export des résultats en CSV ===
results_df = X_test.copy()
results_df['Actual'] = y_test
results_df['Predicted'] = y_pred
results_df.to_csv("fraud_predictions.csv", index=False)
print(" Résultats exportés dans 'fraud_predictions.csv'")

# === Fonction de prédiction interactive ===
def predict_transaction(model, features):
    pred = model.predict([features])[0]
    return "Fraude" if pred == 1 else "Normale "

# Test de la fonction
result = predict_transaction(model, sample[0])
print("Résultat de prédiction :", result)
