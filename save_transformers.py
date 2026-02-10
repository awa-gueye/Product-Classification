"""
================================================================================
SCRIPT DE SAUVEGARDE DES TRANSFORMATEURS
================================================================================
Ce script extrait et sauvegarde les transformateurs nécessaires pour l'application

Transformateurs à sauvegarder :
- TfidfVectorizer (Notebook 2)
- Word2Vec model (Notebook 2)
- StandardScaler images
================================================================================
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

print("="*80)
print("SAUVEGARDE DES TRANSFORMATEURS POUR L'APPLICATION")
print("="*80)

# Chargement des données préprocessées
print("\nChargement des données du Notebook 2...")
with open('../outputs/preprocessed_data.pkl', 'rb') as f:
    preprocessed_data = pickle.load(f)

print("Contenu disponible :")
for key in preprocessed_data.keys():
    print(f"  - {key}")

# IMPORTANT : Le Notebook 2 contient les features DEJA transformées
# mais pas les transformateurs eux-mêmes
# Il faut RECREER les transformateurs avec les mêmes paramètres

print("\n" + "="*80)
print("RECREATION DES TRANSFORMATEURS")
print("="*80)

# 1. TF-IDF Vectorizer
print("\n1. Création du TfidfVectorizer...")

# Paramètres utilisés dans le Notebook 2
tfidf_params = preprocessed_data.get('tfidf_features', None)

if tfidf_params:
    max_features, ngram_range = tfidf_params
    print(f"   Paramètres détectés : max_features={max_features}, ngram_range={ngram_range}")
else:
    # Valeurs par défaut du Notebook 2
    max_features = 3000
    ngram_range = (1, 2)
    print(f"   Paramètres par défaut : max_features={max_features}, ngram_range={ngram_range}")

# Création du vectorizer (SANS données)
tfidf_vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=ngram_range,
    stop_words='english',
    min_df=2,
    max_df=0.9
)

print("   TfidfVectorizer créé (nécessite fit sur nouvelles données)")

# Sauvegarde
with open('../outputs/tfidf_vectorizer_template.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("   Sauvegardé : tfidf_vectorizer_template.pkl")

# 2. Word2Vec parameters
print("\n2. Paramètres Word2Vec...")

w2v_params = preprocessed_data.get('w2v_features', None)

if w2v_params:
    vector_size, window, min_count = w2v_params
    print(f"   Paramètres : vector_size={vector_size}, window={window}, min_count={min_count}")
else:
    vector_size = 100
    window = 5
    min_count = 2
    print(f"   Paramètres par défaut : vector_size={vector_size}, window={window}, min_count={min_count}")

w2v_config = {
    'vector_size': vector_size,
    'window': window,
    'min_count': min_count
}

with open('../outputs/word2vec_config.pkl', 'wb') as f:
    pickle.dump(w2v_config, f)

print("   Sauvegardé : word2vec_config.pkl")

# 3. StandardScaler pour features texte
print("\n3. StandardScaler pour features texte...")

scaler_text = StandardScaler()

# Fit sur les features texte existantes
X_train = preprocessed_data['X_train'].values
scaler_text.fit(X_train)

with open('../outputs/scaler_text.pkl', 'wb') as f:
    pickle.dump(scaler_text, f)

print("   Sauvegardé : scaler_text.pkl")

# 4. StandardScaler pour images (à créer dans l'app)
print("\n4. Configuration pour images...")

scaler_config = {
    'note': 'Le scaler images sera créé à la volée dans l\'application',
    'input_shape': (224, 224, 3),
    'models': ['ResNet50', 'EfficientNetB0', 'MobileNetV2']
}

with open('../outputs/image_config.pkl', 'wb') as f:
    pickle.dump(scaler_config, f)

print("   Sauvegardé : image_config.pkl")

print("\n" + "="*80)
print("RESUME DES FICHIERS SAUVEGARDES")
print("="*80)

import os
output_dir = '../outputs'

files_to_check = [
    'tfidf_vectorizer_template.pkl',
    'word2vec_config.pkl',
    'scaler_text.pkl',
    'image_config.pkl',
    'final_best_model.pkl',
    'preprocessed_data.pkl'
]

print("\nFichiers disponibles pour l'application :")
for filename in files_to_check:
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / 1024  # Ko
        print(f"  ✓ {filename} ({size:.1f} Ko)")
    else:
        print(f"  ✗ {filename} (MANQUANT)")

