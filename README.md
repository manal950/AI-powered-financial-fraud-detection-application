# 🛡️ FINANCEGUARD AI - Système Intelligent de Détection des Fraudes Financières par Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow / Keras](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-ANN-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Projet Académique & Soutenance Universitaire** : *"Comprendre les applications du Deep Learning dans le secteur financier"*

---

## 📌 Présentation du Projet

**FINANCEGUARD AI** est une application complète, réelle et opérationnelle de détection de fraudes sur transactions financières par cartes bancaires, propulsée par un **Réseau de Neurones Artificiels (ANN - Artificial Neural Network)** développé avec **TensorFlow / Keras** et dotée d'une interface web interactive moderne sous **Streamlit**.

Contrairement à un simple prototype visuel ou statique :
- Le modèle de Deep Learning est **réellement entraîné** sur les 284 807 transactions du jeu de données officiel Kaggle (ULB Machine Learning Group).
- Le pipeline de prétraitement (`RobustScaler`) est sérialisé et appliqué rigoureusement sans fuite de données (**Zero Data Leakage**).
- L'explicabilité est assurée par le framework mathématique **SHAP (SHapley Additive exPlanations)** pour auditer chaque décision.
- Les performances sont validées scientifiquement face à un modèle de référence (**Régression Logistique**).

---

## 🌐 Déploiement en Ligne sur Streamlit Community Cloud

L'application est **prête à 100% pour le cloud** (légère : **< 350 Ko** sans le fichier CSV brut).

### Étapes simples pour la mettre en ligne :

1. **Créer un dépôt GitHub :**
   - Rendez-vous sur [GitHub.com](https://github.com) et cliquez sur **New Repository**.
   - Nommez-le par exemple `financeguard-ai` et choisissez la visibilité **Public**.
2. **Téléverser les fichiers du projet :**
   - Téléversez tous les fichiers du dossier `FinanceGuard_AI/` **SAUF `data/creditcard.csv`** (qui dépasse la limite GitHub de 100 Mo et n'est pas nécessaire en production car tous les modèles et échantillons de démonstration sont pré-entraînés).
3. **Déployer sur Streamlit Cloud :**
   - Rendez-vous sur [share.streamlit.io](https://share.streamlit.io) et connectez-vous avec votre compte GitHub.
   - Cliquez sur **Create app** > **Deploy a public app from GitHub**.
   - Sélectionnez votre dépôt (`votre-nom/financeguard-ai`), branche `main`, et fichier principal `app.py`.
   - Cliquez sur **Deploy!** : Votre application sera accessible via un lien public partageable avec vos amis et votre professeur !

---

## 🏗️ Architecture du Projet

```text
FinanceGuard_AI/
│
├── .streamlit/                         # Configuration UI Streamlit
│   └── config.toml                     # Thème Dark Navy Fintech & réglages
│
├── data/                               # Données du projet
│   ├── dataset_summary.json            # Statistiques globales du dataset Kaggle
│   ├── simulation_samples.csv          # 50 transactions réelles du test set pour simulation
│   └── sample_batch_transactions.csv   # Fichier exemple pour l'analyse par lot
│
├── models/                             # Modèles et artefacts sérialisés (< 200 Ko au total)
│   ├── fraud_model.keras               # Réseau de neurones ANN entraîné (TensorFlow/Keras)
│   ├── scaler.joblib                   # Scaler ajusté uniquement sur le train set
│   ├── baseline_logreg.joblib          # Modèle de référence Régression Logistique
│   ├── evaluation_metrics.json         # Métriques réelles, matrices de confusion, courbes ROC et PR
│   ├── shap_background.npy             # Échantillon de fond pour le calcul SHAP
│   └── global_feature_importance.json  # Importance globale des variables
│
├── .gitignore                          # Exclut le gros dataset brut (150 Mo) et les caches
├── .python-version                     # Fixe la version Python 3.11 pour le cloud
├── train_model.py                      # Script d'entraînement et d'évaluation scientifique
├── app.py                              # Application web interactive Streamlit
├── requirements.txt                    # Dépendances optimisées pour Streamlit Cloud
└── README.md                           # Documentation complète et guide de soutenance
```

---

## ⚡ Installation et Lancement Local

### 1. Prérequis
- Python 3.10 à 3.14 (ou environnement Anaconda3)

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Entraînement du modèle (Déjà réalisé !)
Les artefacts de modèles sont déjà fournis dans le dossier `models/`. Si vous souhaitez relancer l'entraînement complet :
```bash
python train_model.py
```
*(Si `creditcard.csv` n'est pas dans `data/`, le script le téléchargera automatiquement depuis OpenML).*

### 4. Lancement de l'application interactive Streamlit
```bash
streamlit run app.py
```
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`.

---

## 🧠 Architecture du Modèle de Deep Learning (ANN)

Le modèle implémenté dans `train_model.py` repose sur un perceptron multicouche optimisé pour les signaux tabulaires hautement déséquilibrés :

| Couche | Type | Neurones / Paramètres | Rôle & Justification |
| :--- | :--- | :--- | :--- |
| **Entrée** | `Input` | 30 variables | Variables financières (Time, V1 à V28 de l'ACP, Amount) |
| **Cachée 1** | `Dense + ReLU` | 64 neurones | Extraction des motifs non-linéaires + régularisation L2 ($10^{-4}$) |
| **Norm 1** | `BatchNormalization` | 64 dimensions | Stabilisation de la distribution interne des activations |
| **Drop 1** | `Dropout` | Taux = 30% | Prévention du surapprentissage (overfitting) |
| **Cachée 2** | `Dense + ReLU` | 32 neurones | Raffinement des interactions combinatoires entre composantes |
| **Norm 2** | `BatchNormalization` | 32 dimensions | Accélération de la convergence |
| **Drop 2** | `Dropout` | Taux = 20% | Renforcement de la robustesse |
| **Cachée 3** | `Dense + ReLU` | 16 neurones | Compression des signatures de fraude |
| **Sortie** | `Dense + Sigmoid`| 1 neurone | Probabilité continue de fraude comprise entre `0.0` et `1.0` |

---

## 🔬 Rigueur Scientifique et Prétraitement

1. **Partition Stratifiée Explicite (70% Train / 15% Validation / 15% Test)** :
   - Ensemble d'Entraînement : **199 364 transactions** (344 fraudes).
   - Ensemble de Validation : **42 721 transactions** (74 fraudes) — utilisé pour le *EarlyStopping* et le calibrage du seuil optimal.
   - Ensemble de Test : **42 722 transactions** (74 fraudes) — **strictement intouché** jusqu'à l'évaluation finale indépendante.
2. **Prévention Absolue du Data Leakage** :
   Le `RobustScaler` est calculé (`fit`) **strictement et uniquement sur l'ensemble d'entraînement (`X_train`)**. Les ensembles de validation et de test ne sont que transformés (`transform`), sans jamais exposer leurs paramètres au modèle pendant l'apprentissage.
3. **Gestion du Déséquilibre Extrême (Class Imbalance)** :
   Avec un taux naturel de **0.173% de fraudes** (1 fraude pour 577 transactions légitimes), une fonction de perte classique ignorerait la fraude. Nous appliquons une pondération inverse `class_weight` (poids de ~289 sur la classe 1) pour forcer le réseau à pénaliser lourdement les faux négatifs.
4. **Calibrage du Seuil Optimal sur la Validation** :
   Le seuil de décision standard (0.50) est souvent inadapté aux données hautement déséquilibrées. Le seuil optimal ($\tau^* = 0.950$) est calculé sur l'ensemble de validation en maximisant le score F1, puis appliqué à l'ensemble de test pour éradiquer 98% des faux positifs tout en conservant >81% de rappel.

---

## 📈 Résultats et Évaluation Comparative (Test Set : 42 722 Transactions - 15% Intouché)

| Métrique | Deep Learning ANN (Keras) | Régression Logistique (Baseline) | Interprétation |
| :--- | :---: | :---: | :--- |
| **ROC-AUC (Capacité Discriminante)** | **0.9703** | 0.9633 | **L'ANN surpasse le modèle linéaire** |
| **PR-AUC (Precision-Recall AP)** | **0.6432** | 0.6270 | **Excellente précision moyenne sur classe rare** |
| **Rappel (Recall) - Seuil par défaut 0.50** | **87.84%** (65 / 74) | **89.19%** (66 / 74) | Détection brute agressive |
| **Rappel (Recall) - Seuil Optimal Val ($\tau^* = 0.950$)**| **81.08%** (60 / 74) | **86.49%** (64 / 74) | **60 fraudes interceptées sur 74** |
| **Précision (Precision) - Seuil Optimal Val** | **63.16%** | 38.00% | **ANN très nettement supérieur (+25%)** |
| **Score F1 - Seuil Optimal Val** | **71.01%** | 54.05% | **ANN très nettement supérieur (+17%)** |
| **Exactitude (Accuracy) - Seuil Optimal Val** | **99.89%** | 99.68% | Moins de 35 faux positifs sur 42 648 légitimes |

---

## 💡 Guide pour la Soutenance devant le Professeur

1. **Ouvrez le Tableau de Bord** : Montrez les métriques réelles (284 807 transactions, 0.173% de fraude, 91.84% de rappel).
2. **Allez sur Détection & Simulation** : Sélectionnez *'🚨 Cas 1 : Fraude Réelle Confirmée ($529.00)'*, cliquez sur *'Lancer l'Inférence'*. Montrez la jauge de probabilité (>95%), le badge rouge *'Fraude Critique'* et l'action de blocage recommandée.
3. **Passez sur une transaction légitime** : Montrez que le modèle classe immédiatement un achat courant ($14.99) en *'Faible Risque'* (<5%).
4. **Allez sur Analyse par Lot** : Cliquez sur *'Charger le lot de démonstration (100 Tx)'* puis lancez l'audit. Montrez les alertes générées et téléchargez le rapport CSV.
5. **Démontrez l'Explicabilité (SHAP)** : Expliquez au jury comment les valeurs de Shapley décomposent la boîte noire du réseau de neurones et citez l'Article 22 du RGPD.
6. **Allez sur Performance** : Présentez le benchmark face à la Régression Logistique et utilisez le **simulateur de seuil dynamique** pour expliquer le compromis opérationnel Précision / Rappel.
