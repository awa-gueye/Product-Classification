# Product Classification - Classification des biens de consommation

> Système de classification automatique de produits e-commerce utilisant l'apprentissage automatique et le deep learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

##  Table des matières

- [À propos du projet](#à-propos-du-projet)
- [Catégories de produits](#catégories-de-produits)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Structure du projet](#structure-du-projet)
- [Technologies utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modèles de Machine Learning](#modèles-de-machine-learning)
- [API REST](#api-rest)
- [Application Web](#application-web)
- [Déploiement avec Docker](#déploiement-avec-docker)
- [Résultats](#résultats)
- [Présentation](#présentation)
- [Méthodologie](#méthodologie)
- [Contribution](#contribution)
- [Licence](#licence)

## À propos du projet

Ce projet implémente un système complet de classification de produits e-commerce capable de prédire automatiquement la catégorie d'un produit à partir de :

- **📝 Texte** : Description du produit
- **🖼️ Image** : Photo du produit  

Le système utilise des techniques avancées de machine learning et deep learning pour offrir des prédictions précises et rapides, idéales pour des applications e-commerce en production.

##  Catégories de produits

Le système classifie les produits dans **7 catégories** principales :

| Icône | Catégorie | Description |
|-------|-----------|-------------|
| 👶 | **Baby Care** | Produits de soins pour bébés |
| 💄 | **Beauty and Personal Care** | Produits de beauté et soins personnels |
| 💻 | **Computers** | Ordinateurs et accessoires informatiques |
| 🎨 | **Home Decor & Festive Needs** | Décoration et articles festifs |
| 🛋️ | **Home Furnishing** | Mobilier et ameublement |
| 🍳 | **Kitchen & Dining** | Articles de cuisine et salle à manger |
| ⌚ | **Watches** | Montres et accessoires horlogers |

## Fonctionnalités

### Capacités de classification

- ✅ Classification basée sur le texte (description produit)
- ✅ Classification basée sur l'image (photo produit)
- ✅ Scores de confiance pour chaque prédiction
- ✅ Probabilités détaillées par catégorie

### 🖥️ Interface utilisateur (Streamlit)

- ✅ Interface moderne et intuitive
- ✅ Upload d'images avec prévisualisation
- ✅ Saisie de description textuelle
- ✅ Visualisation des résultats en temps réel
- ✅ Dashboard analytique avec graphiques interactifs
- ✅ Historique complet des prédictions
- ✅ Export des données en CSV
- ✅ Design responsive (desktop/mobile)

###  API REST (FastAPI)

- ✅ Endpoints de prédiction (texte, image, multimodal)
- ✅ Documentation Swagger auto-générée
- ✅ Documentation ReDoc interactive
- ✅ Validation des données avec Pydantic
- ✅ Gestion robuste des erreurs
- ✅ Support CORS configuré
- ✅ Health check endpoint

##  Architecture

```
┌─────────────────────────────────────────┐
│         Interface Utilisateur           │
│         (Streamlit Web App)             │
└─────────────────┬───────────────────────┘
                  │
            HTTP Requests
                  │
┌─────────────────▼───────────────────────┐
│           API REST (FastAPI)            │
│  - /predict/text                        │
│  - /predict/image                       │
└─────────────────┬───────────────────────┘
                  │
          Chargement des modèles
                  │
┌─────────────────▼──────────────────────┐
│         Modèles ML/DL                  │
│  ┌────────────────────────────────┐    │
│  │  Modèle Texte (SVM + TF-IDF)   │    │
│  │  Accuracy: 95,57%              │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │  Modèle Image (VGG16)          │    │
│  │  Accuracy: 78.48%              │    │
│  └────────────────────────────────┘    │
└────────────────────────────────────────┘
```

## Structure du projet

```
Product-Classification/
│
├── 📂 api/                           # API FastAPI
│   ├── main.py                       # Point d'entrée de l'API
│   ├── requirements.txt                     
│   └── Dockerfile                    # Configuration
│
├── 📂 app/                           # Application Streamlit
│   └── streamlit_app.py              # Interface web
│
├── 📂 Data/                           # Données de l'étude
│   └── Flipkart/ 
│       ├── images/                     # Dossier contenant les images           
│       └── flipkart_com-ecommerce_sample_1050.csv  # fichier des données brutes
│
├── 📂 models/                        # Modèles entraînés sauvegardés
│   ├── final_best_model.pkl          # Modèle SVM (texte)
│   ├── tfidf_vectorizer.pkl          # Vectoriseur TF-IDF
│   ├── cnn_final.keras               # VGG16 (image)
│   └── label_encoders.pkl            # Encodeurs de labels
│
├── 📂 notebooks/                              # Notebooks du projet
│   ├── n1_analyse_exploratoire.ipynb          # Analyse exploratoire des données textuelles
│   ├── n2_prepocessing_featuring.ipynb        # Preprocessing et featuring des textes
│   ├── n3_modelisation_text.ipynb             # Modélisation des données textuelles
│   ├── n4_exploration_image.ipynb             # Analyse exploratoire des images
│   └── n5_deep_mearning_supervise.ipynb       # Modélisation des images
│
├── 📄 save_transformers.py           # Sauvegarde des transformateurs
├── 📄 requirements.txt               # Dépendances Python
├── 📄 docker-compose.yml             # Configuration Docker Compose
├── 📄 .python-version                # Version Python (3.12)
├── 📄 .gitignore                     # Fichiers à ignorer
│
└── 📄 README.md                      # Ce fichier
```

## Technologies utilisées

### Machine Learning & Deep Learning Technologies

#### Deep Learning Frameworks & Models
- **TensorFlow/Keras** - Deep learning framework pour la classification d'images et modèles de vision par ordinateur
- **EfficientNetB0** - Architecture CNN avancée pour la classification d'images 
- **VGG16** - Architecture CNN éprouvée utilisée pour la classification d'images
- **MobileNetV3-Small** - Modèle léger optimisé pour les applications en temps réel
- **CNN Custom (baseline)** - Architecture CNN personnalisée développée comme modèle de base

#### Machine Learning Algorithms (NLP & Classification)
- **Scikit-learn** - Bibliothèque complète d'algorithmes de machine learning
- **Support Vector Machines (SVM)** - Algorithmes pour la classification textuelle avec noyaux linéaires et RBF
- **Logistic Regression** - Modèle de régression logistique pour la classification binaire et multiclasse
- **Random Forest** - Algorithme d'ensemble par forêts aléatoires pour la classification
- **Gradient Boosting** - Méthodes de boosting pour améliorer les performances de prédiction
- **XGBoost** - Implémentation optimisée du gradient boosting
- **TF-IDF + SVM** - Pipeline NLP pour la classification textuelle (meilleur modèle textuel)

#### Preprocessing & Feature Engineering
- **TF-IDF Vectorization** - Extraction de caractéristiques textuelles pour le NLP
- **Image Augmentation** - Techniques d'augmentation d'images pour améliorer la robustesse des modèles
- **Feature Scaling** - Normalisation et standardisation des caractéristiques
- **Dimensionality Reduction** - Techniques pour réduire la dimensionnalité des données

#### Model Evaluation & Optimization
- **Cross-Validation** - Validation croisée pour l'évaluation robuste des modèles
- **Hyperparameter Tuning** - Optimisation des hyperparamètres via Grid Search et Random Search
- **Performance Metrics** - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Model Persistence** - Sauvegarde et chargement des modèles entraînés (pickle, h5)


### Backend & API

- **FastAPI**  - Framework web moderne et performant
- **Uvicorn** - Serveur ASGI haute performance
- **Pydantic** - Validation et sérialisation des données

### Frontend & Visualisation

- **Streamlit** - Framework pour interfaces web interactives
- **Plotly** - Bibliothèque de visualisation interactive
- **Pandas** - Manipulation et analyse de données

### Preprocessing & Utilities

- **NumPy** - Calcul numérique
- **Pillow** - Traitement d'images
- **Python-multipart** - Gestion des uploads de fichiers

### Déploiement

- **Docker** - Containerisation des applications
- **Docker Compose** - Orchestration multi-conteneurs

## Installation

### Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)
- Git
- (Optionnel) Docker et Docker Compose

### Étapes d'installation

1. **Cloner le repository**

```bash
git clone https://github.com/awa-gueye/Product-Classification.git
cd Product-Classification
```

2. **Créer un environnement virtuel** (fortement recommandé)

```bash
python -m venv venv

# Sur Windows
venv\Scripts\activate

# Sur Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

## Utilisation

### Etape 1 : Lancer l'API FastAPI

```bash
# Depuis le dossier api
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**L'API sera accessible à :**
- **Serveur** : http://localhost:8000
- **Documentation Swagger** : http://localhost:8000/docs
- **Documentation ReDoc** : http://localhost:8000/redoc

### Etape 2 : Lancer l'application Streamlit

```bash
# Depuis le dossier app
cd app
streamlit run streamlit_app.py
```

**L'application web sera accessible à :** http://localhost:8501


## Modèles de Machine Learning

### 1. Modèles Texte

**Architecture et workflow** :

```
Texte brut (nom du produit, description produit)
    ↓
[Preprocessing]
  - Nettoyage
  - Tokenisation
  - Suppression stop words
  - Lemmatisation
    ↓
[TF-IDF Vectorization]
  - N-grams (1-2)
  - Max features: 3100
    ↓
[SVM Classifier (Meilleur modèle)]
  - C optimisé
  - Gamma optimisé
    ↓
Catégorie prédite + probabilités
```

**Performances** :
- **Accuracy** : 95.57%
- **F1-Score** : 95.49%
- **Precision** : 95.62%
- **Recall** : 95.56%

**Fichiers associés** :
- `models/final_best_model.pkl` - Modèle SVM entraîné
- `models/tfidf_vectorizer.pkl` - Vectoriseur TF-IDF

### 2. Modèles Image

**Architecture détaillée** :

```
Image d'entrée (224x224x3)
    ↓
[VGG16]
  - Pré-entraîné
  - Poids gelés (frozen layers)
    ↓
[GlobalAveragePooling2D]
    ↓
[Dense Layer 512]
  - BatchNormalization
  - Dropout (0.5)
  - Activation: ReLU
    ↓
[Dense Layer 256]
  - BatchNormalization
  - Dropout (0.4)
  - Activation: ReLU
    ↓
[Dense Layer 7]
  - Activation: Softmax
    ↓
Catégorie prédite + probabilités
```

**Caractéristiques d'entraînement** :
- Transfer learning avec ResNet50
- Fine-tuning des dernières couches
- Data augmentation : rotation, flip horizontal, zoom, shear
- Early stopping 
- Optimiseur : Adam
- Loss : Categorical Crossentropy

**Performances** :
- **Accuracy** : 78.48%
- **F1-Score** : 78.62%
- **Entraînement** : 15 epochs 

**Fichier associé** :
- `models/label_encoders.pkl` - Encodeurs de labels
- `models/cnn_final.keras` - Modèle CNN entraîné

## API REST

### Endpoints disponibles

| Méthode | Endpoint | Description | Corps de la requête | Réponse |
|---------|----------|-------------|---------------------|---------|
| GET | `/` | Informations API | - | JSON metadata |
| GET | `/health` | État de santé | - | Status message |
| GET | `/categories` | Liste des catégories | - | Array de catégories |
| POST | `/predict/text` | Classification texte | `{"text": "..."}` | Prédiction + probas |
| POST | `/predict/image` | Classification image | `file: image` | Prédiction + probas |

### Exemples d'utilisation

#### Python (avec requests)

```python
import requests

# Classification par texte
response = requests.post(
    "http://localhost:8000/predict/text",
    json={"text": "Montre analogique pour homme avec bracelet en cuir véritable"}
)
result = response.json()
print(f"Catégorie: {result['predicted_class']}")
print(f"Confiance: {result['confidence']:.2%}")

# Classification par image
with open("product_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/image",
        files={"file": f}
    )
result = response.json()
print(f"Catégorie: {result['predicted_class']}")

```

#### cURL

```bash
# Classification texte
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Ordinateur portable 15 pouces gaming RTX 4070"}'

# Classification image
curl -X POST "http://localhost:8000/predict/image" \
     -F "file=@product.jpg"

```

### Format de réponse

```json
{
  "predicted_class": "Watches",
  "confidence": 0.8523,
  "probabilities": {
    "Baby Care": 0.0234,
    "Beauty and Personal Care": 0.0456,
    "Computers": 0.0123,
    "Home Decor & Festive Needs": 0.0289,
    "Home Furnishing": 0.0156,
    "Kitchen & Dining": 0.0219,
    "Watches": 0.8523
  },
  "model_used": "text",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

## Application Web - Product Classifier Pro

L'application Streamlit offre une interface professionnelle de classification de produits pour e-commerce avec un design bleu marine (#1E3A8A) et or (#D4AF37) avec thème clair/sombre adaptatif.

### Navigation Principale
- **🏠 HOME** : Page d'accueil avec présentation, fonctionnalités, catégories et performances
- **📸 IMAGE CLASSIFICATION** : Classification d'images simples ou par lot avec prévisualisation
- **📝 TEXT CLASSIFICATION** : Classification de descriptions produit uniques ou multiples
- **📊 DASHBOARD** : Analytics en temps réel avec visualisations interactives
- **🌙/☀️** : Bouton de changement de thème clair/sombre

### Fonctionnalités Clés
1. **Classification Simple** : Upload d'image ou saisie de texte pour classification instantanée
2. **Traitement Batch** : 
   - Images multiples (JPG/PNG) avec aperçu
   - Textes multiples via CSV ou saisie manuelle
3. **Visualisation des résultats** :
   - Jauge de confiance interactive
   - Graphique de probabilités par catégorie
   - Affichage de la catégorie prédite
4. **Analytics Avancés** :
   - Métriques de performance des modèles
   - Historique complet des prédictions
   - Graphiques comparatifs et distributions
   - Export des données au format CSV

### Design & Expérience Utilisateur
- **Palette** : Bleu marine (#1E3A8A) et or (#D4AF37)
- **Thèmes** : Mode clair et sombre interchangeables
- **Responsive** : Adapté desktop et mobile
- **Interactions** : Hover effects, transitions fluides, feedback visuel
- **Visualisations** : Plotly pour graphiques interactifs et modernes

### Données & Catégories
- **7 catégories** : Baby Care, Beauty & Personal Care, Computers, Home Decor & Festive Needs, Home Furnishing, Kitchen & Dining, Watches
- **Modèles** : VGG16 pour images, TF-IDF+SVM pour texte
- **Performance** : 95.6% accuracy (texte), 78.5% accuracy (images)

## Outils de déploiement testés

### 🌐 Plateformes Cloud & Hébergement
- **Streamlit Cloud** - Hébergement principal de l'application frontend (déploiement continu via GitHub)
- **Render** - Déploiement du backend FastAPI (API de classification)
- **Railway** - Testé pour le déploiement du backend

### 🐳 Conteneurisation & Orchestration
- **Docker** - Conteneurisation de l'application
- **Docker Compose** - Orchestration multi-conteneurs pour développement local

### Backend & API
- **FastAPI** - Framework backend moderne avec documentation OpenAPI automatique
- **Uvicorn** - Serveur ASGI haute performance pour FastAPI

### Gestion des dépendances & Environnements
- **Pip** - Gestionnaire de paquets Python standard
- **requirements.txt** - Fichier de dépendances versionné
- **virtualenv/venv** - Environnements virtuels isolés

### CI/CD & Automatisation
- **Automatisation Git** - Déploiement continu sur push vers main

### Stockage
- **Google Drive** - Pour le stockage des fichiers .pkl et .keras

NB : le déploiement n'a pas pu être effectué du fait des erreurs rencontrés pour le téléchargement du modèle de deep learning

## Résultats

### Comparaison des performances par modalité

| Modalité | Modèle | Accuracy | F1-Score | Précision | Rappel |
|----------|--------|----------|----------|-----------|--------|
| **Texte** | SVM (TF-IDF) | **95.57%** | **0.955** | 0.956 | 0.955 |
| **Image** | VGG16 | 78.48% | 0.7862 | - | - |

### Points clés des résultats

- ✅ **Modèle texte** : Performances excellentes avec 95.57% d'accuracy
- ✅ **Modèle image** : Résultats corrects compte tenu de la complexité visuelle
- ✅ **Transfer learning** : Amélioration significative grâce à ResNet50 pré-entraîné
- ✅ **Temps d'inférence** : Rapides et adaptés à la production

### Matrice de confusion (Modèle Texte SVM)

```
                    Prédictions
              BC   BP   CO   HD   HF   KD   WA
Réel    BC   [145   2    0    1    1    0    1]  
        BP   [ 1  143   0    2    2    1    1]  
        CO   [ 0    0  148   0    1    1    0] 
        HD   [ 2    1    0  141   3    3    0] 
        HF   [ 1    2    1    2  140   4    0]  
        KD   [ 0    1    1    3    3  142   0] 
        WA   [ 0    0    0    0    0    0  150] 

Légende des catégories :
BC = Baby Care
BP = Beauty and Personal Care
CO = Computers
HD = Home Decor & Festive Needs
HF = Home Furnishing
KD = Kitchen & Dining
WA = Watches
```

**Observations** :
- Excellente performance sur la catégorie "Watches" (100%)
- Performance solide et équilibrée sur toutes les catégories
- Peu de confusions entre catégories très différentes
- Quelques confusions marginales sur des catégories proches (ex: Home Decor vs Home Furnishing)

## Présenation 
Ci-dessous le lien pour la présentaion du projet :
https://www.canva.com/design/DAHAVETLstM/8XIpfnZDtlYXgHK9Ca-mbg/edit?utm_content=DAHAVETLstM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Méthodologie

### 1. Preprocessing des données

**Données textuelles** :
1. Nettoyage du texte
   - Suppression de la ponctuation
   - Conversion en minuscules
   - Suppression des caractères spéciaux
2. Tokenisation
3. Suppression des stop words
4. Lemmatisation pour normalisation

**Données images** :
1. Redimensionnement à 224x224 pixels
2. Normalisation des valeurs de pixels (0-1)
3. Data augmentation :
   - Rotation aléatoire (±20°)
   - Flip horizontal
   - Zoom aléatoire (±20%)
   - Shear transformation

### 2. Feature Engineering

**Modalité texte** :
- Vectorisation TF-IDF (Term Frequency-Inverse Document Frequency)
- N-grams : unigrammes et bigrammes (1-2)
- Nombre maximal de features : 10000 termes
- Normalisation L2

**Modalité image** :
- Extraction de features via ResNet50 pré-entraîné (ImageNet)
- Fine-tuning des couches supérieures
- Ajout de couches denses personnalisées
- Batch normalization et dropout pour régularisation

### 3. Modélisation et expérimentation

**Modèles testés pour le texte** :
- Support Vector Machine (SVM) ← **Retenu** (meilleure performance)
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting

**Architectures testées pour les images** :
- CNN personnalisé
- VGG16 ← **Retenu** (bon compromis performance/complexité)
- EfficientNetB0
- MobilNet

**Stratégies de fusion** :
- Early fusion (concaténation de features)
- Late fusion (combinaison de prédictions)

### 4. Évaluation et validation

**Métriques utilisées** :
- Accuracy
- F1-Score
- Precision, Recall
- Matrice de confusion

**Validation** :
- Train/Val/Test split (70/15/15)
- Validation croisée pour optimisation des hyperparamètres

## Contribution

Les contributions sont les bienvenues et encouragées ! Voici comment participer :

### Comment contribuer

1. **Fork** le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une **Pull Request**

### Guidelines de contribution

- Suivre les conventions PEP 8 pour le code Python
- Ajouter des tests pour les nouvelles fonctionnalités
- Mettre à jour la documentation si nécessaire
- Utiliser des messages de commit clairs et descriptifs
- Documenter les nouvelles fonctions et classes

### Idées de contribution

- Ajouter de nouvelles catégories de produits
- Tester d'autres architectures (BERT, Vision Transformer)
- Améliorer le dashboard avec de nouvelles visualisations
- Implémenter l'authentification API
- Ajouter le support multilingue
- Créer des notebooks d'analyse supplémentaires

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

**Résumé de la licence MIT** :
- ✅ Utilisation commerciale autorisée
- ✅ Modification autorisée
- ✅ Distribution autorisée
- ✅ Utilisation privée autorisée
- ⚠️ Aucune garantie fournie
- ⚠️ Responsabilité limitée

## Auteurs

**Awa GUEYE, Mariane DAIFERLE, Gilbert OUMSAORE, Naba Amadou Seydou TOURE**

- 🌐 GitHub : [@awa-gueye](https://github.com/awa-gueye)
- 📁 Projet : [Product-Classification](https://github.com/awa-gueye/Product-Classification)


## Ressources additionnelles
- [Documentation TensorFlow](https://www.tensorflow.org/)
- [Documentation Scikit-learn](https://scikit-learn.org/)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

[⬆ Retour en haut](#-product-classification---classification-des-biens-de-consommation)