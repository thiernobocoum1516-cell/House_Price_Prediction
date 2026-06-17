# 🏠 House Price Prediction

## 📌 Description
Ce projet est un pipeline MLOps de prédiction des prix immobiliers basé sur du Machine Learning.

Il inclut :
- Analyse exploratoire des données (EDA)
- Prétraitement des données
- Entraînement d’un modèle de régression
- Tests automatiques avec pytest
- Intégration continue avec GitHub Actions (CI/CD)

---

## 📁 Structure du projet

laplace-immo-housing/
├── .github/
├── notebooks/
├── src/
├── tests/
├── data/
├── models/
├── requirements.txt
├── ci.yml
├── README.md


## ⚙️ Installation

Clone le projet puis installe les dépendances :

git clone https://github.com/thiernobocoum1516-cell/House_Price_Prediction.git
cd House_Price_Prediction
pip install -r requirements.txt

## 🚀 Lancer les tests
pytest tests/

## 🤖 Objectif du projet

Prédire le prix des logements à partir de leurs caractéristiques (surface, nombre de pièces, localisation, etc.).


## 🔄 CI/CD (GitHub Actions)

À chaque push sur la branche main :

* installation des dépendances
* exécution des tests
* validation automatique du code


## 👤 Auteurs

* Thierno Bocoum
* Seydina Omar Dioum
* Ndeye Codou Mbodj


## 🧠 Statut du projet

Projet en cours de développement dans un cadre d’apprentissage du Machine Learning.