# 🏠 Laplace Immo Housing – MLOps Project

Projet de prédiction du prix des maisons basé sur le dataset Ames Housing, avec une architecture orientée MLOps (Feature Engineering + Training + API).

---

## 🚀 Objectif
L'objectif de ce projet est de concevoir un système complet de prédiction des prix 
immobiliers, allant de l'analyse exploratoire des données au déploiement d'une API.

Pour cela, nous avons :
1. Effectué une analyse exploratoire approfondie (EDA) pour comprendre les données
2. Réalisé du feature engineering pour préparer les données
3. Testé plusieurs modèles (linéaires et à arbres) pour identifier le plus performant
4. Sélectionné CatBoost comme modèle final (R² = 0.9274, RMSE = 20 027 $)
5. Optimisé les hyperparamètres du modèle
6. Déployé le modèle via une API FastAPI
7. Mis en place des tests unitaires avec Pytest
8. Automatisé le déploiement continu avec GitHub Actions

Le résultat est un système fiable et robuste pour estimer le prix des maisons 
avec une erreur moyenne d'environ 20 000 $.

---

## 📁 Architecture du projet
laplace-immo-housing/
│
├── data/
│ ├── train.csv
│ ├── test.csv
  ├──sample_submission.csv
│ └── processed/
│
├── notebooks/
│ ├── EDA_END_FEATURE-ENGENERING.ipynb
│ └── modeling.ipynb
│
├── src/
│ ├── data/
│ │ └── loader.py
│ │
│ ├── features/
│ │ └── builder.py
│ │
│ ├── training/
│ │ └── train.py
│ │
│ └── api/
│ └── api.py
│
├── models/
│ ├── best_model.pkl
│ └── features.pkl
│ └── best_params.pkl
│
├── requirements.txt
├──tests
├──setup.py
└── README.md 

---

## ⚙️ Processus

### 1. Data Loading
Les données sont chargées via `loader.py` :
- suppression de `Id`
- gestion des valeurs manquantes catégorielles

---

### 2. Feature Engineering
Dans `builder.py`, création des variables :

- `QualityArea`
- `TotalSF`
- `HouseAge`
- `RemodAge`
- `TotalBath`
- `TotalPorchSF`

---

### 3. Preprocessing

Le pipeline inclut :

- Imputation intelligente (quartiers, médianes, modes)
- Encodage des variables catégorielles (LabelEncoder)
- Standardisation des variables numériques (StandardScaler)
- Sélection de features (LassoCV)

---

### 4. Modèle

Modèle de régression entraîné sur les features transformées.

Sortie :
best_model.pkl

### 5. ⚙️ Interface de prédiction (Dashboard ML)

L’application propose une interface web interactive permettant de prédire le prix d’une maison en temps réel.

---

#### 🧠 Description

L’utilisateur saisit directement les caractéristiques de la maison via un formulaire web :

- variables numériques (surface, année, qualité…)
- variables catégorielles (MSZoning, HouseStyle, etc.)

---

#### 🚀 Fonctionnement

1. L’utilisateur remplit le formulaire
2. Les données sont envoyées au backend FastAPI
Le processus de modélisation comprend les étapes suivantes :
→ Feature Engineering
→ Préprocessing (encodage + standardisation)
→ Prédiction via CatBoost
4. Le résultat est affiché instantanément

---

#### 📊 Résultats affichés

L’interface affiche aussi :

- 🧠 Modèle utilisé : CatBoostRegressor
- 📈 R² Score : 0.9274
- 📉 RMSE : 20027$
- 🔢 Nombre de features : 85

---

#### 💰 Exemple de sortie

👉 Predicted Price : **$157,258.65**

---

## Lancer le projet
### 1. Installer les dépendances

pip install -r requirements.txt

### 2. Entraîner le modèle

python src/features/builder.py --mode train --model_dir models/
python src/training/train.py

### 3. Lancer l’API

uvicorn src.api.api:app --reload

### 4. Documentation API

Accès Swagger : 
http://127.0.0.1:8000/docs

## Points clés MLOps

-Pipeline reproductible
-Feature engineering centralisé
-Séparation training / inference
-API indépendante du notebook
-Modèle sérialisé (.pkl)

## Problèmes résolus pendant le projet

-Désalignement des features (MSZoning, Id)
-Erreurs CatBoost feature mismatch
-Incohérence notebook vs API
-Gestion des valeurs manquantes

## Auteur 

Ce projet est réalisé par :

Thierno BOCOUM

Ndeye Codou MBODJ

Seydina Omar DIOUM

ici le lien de la présentation: https://canva.link/ndw34pyu6jvmfv1
