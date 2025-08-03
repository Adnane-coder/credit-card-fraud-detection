# 💳 Credit Card Fraud Detection

Détection de transactions frauduleuses par Machine Learning avec preprocessing, équilibrage des données et visualisations complètes.

## 📊 Description du projet

Ce projet vise à détecter les transactions frauduleuses à partir d'un dataset déséquilibré en utilisant des techniques de Machine Learning. Il comprend le nettoyage des données, la normalisation, l’undersampling, le SMOTE, l'entraînement de modèles et leur évaluation.

## 📁 Dataset utilisé
- Source : [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Contenu : 284,807 transactions (492 frauduleuses)

## ⚙️ Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib & Seaborn
- Joblib

## 🧪 Modèles testés
- Régression Logistique
- Arbre de Décision

## ✅ Résultats
- Évaluation avec undersampling et oversampling (SMOTE)
- Visualisations : matrice de confusion, courbe ROC, camembert
- Sauvegarde du modèle (.pkl) + prédiction interactive

## 📦 Fichiers générés
- `fraud_detection_model.pkl` : modèle sauvegardé
- `fraud_predictions.csv` : prédictions sur le test set

## 🚀 Lancer le projet

```bash
pip install -r requirements.txt
python fraud_detection.py
