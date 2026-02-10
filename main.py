"""
API ULTRA-OPTIMISÉE pour Render (1GB RAM)
"""

import os
# CONFIGURATION CRITIQUE AVANT TOUT IMPORT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Pas de logs TF
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Pas de GPU
os.environ['OMP_NUM_THREADS'] = '1'  # Limiter threads
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import joblib
import logging
from datetime import datetime
import sys
import gdown
import tempfile
import gc  # Garbage collector
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
from PIL import Image
import io

# Logging minimal
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
GOOGLE_DRIVE_IDS = {
    "tfidf_vectorizer.pkl": "1De4pUoj_IDdH3ZMYYQaTVFwwAgMo_N9y",
    "final_best_model.pkl": "1p0UXPEM5bQ2CjM6BS3YtYlxcAED6o6UA", 
    "label_encoders.pkl": "1O4EFUU6Qj_mtEb_wmBL6QjlLahe3l3yH",
    "cnn_final.keras": "1RXL7knfjXtNk6Aa3HZZQjCEUDow0QUJ7",
}

CATEGORIES = [
    "Baby Care", "Beauty and Personal Care", "Computers",
    "Home Decor & Festive Needs", "Home Furnishing", 
    "Kitchen & Dining", "Watches"
]

app = FastAPI(
    title="Product Classification API",
    docs_url=None,  # Désactiver docs pour économiser mémoire
    redoc_url=None
)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# ========== GESTION MÉMOIRE ==========
class MemoryOptimizedLoader:
    def __init__(self):
        self.vectorizer = None
        self.text_model = None
        self.label_mapping = None
        self.cnn_model = None
        self.loaded = False
        
    def load_all_with_memory_control(self):
        """Charge les modèles avec contrôle mémoire"""
        try:
            logger.warning("🚀 Démarrage avec optimisation mémoire...")
            
            # 1. Charger modèles texte (légers)
            logger.warning("1. Chargement modèles texte...")
            self._load_text_models()
            gc.collect()  # Nettoyer mémoire
            
            # 2. Charger CNN avec gestion mémoire
            logger.warning("2. Chargement CNN (peut prendre 30s)...")
            self._load_cnn_with_memory_optimization()
            
            self.loaded = True
            logger.warning("✅ Tous les modèles chargés avec succès!")
            
        except MemoryError as e:
            logger.error(f"💥 Mémoire insuffisante: {e}")
            self._emergency_fallback()
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            raise
    
    def _load_text_models(self):
        """Charge les modèles texte"""
        # TF-IDF
        self.vectorizer = self._download_model("tfidf_vectorizer.pkl")
        
        # Modèle texte
        self.text_model = self._download_model("final_best_model.pkl")
        
        # Labels
        label_data = self._download_model("label_encoders.pkl")
        if isinstance(label_data, dict):
            self.label_mapping = {v: k for k, v in label_data.items()}
        else:
            self.label_mapping = {i: cat for i, cat in enumerate(CATEGORIES)}
    
    def _load_cnn_with_memory_optimization(self):
        """Charge CNN avec optimisations"""
        try:
            # Import TensorFlow seulement si nécessaire
            import tensorflow as tf
            from tensorflow import keras
            
            # Configurer TF pour économiser mémoire
            tf.config.set_visible_devices([], 'GPU')  # Pas de GPU
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            
            # Télécharger
            file_id = GOOGLE_DRIVE_IDS["cnn_final.keras"]
            url = f"https://drive.google.com/uc?id={file_id}"
            temp_path = tempfile.mktemp(suffix='.keras')
            
            logger.warning("📥 Téléchargement CNN...")
            gdown.download(url, temp_path, quiet=True)
            
            # Charger avec optimisation
            logger.warning("🔄 Chargement en mémoire...")
            self.cnn_model = keras.models.load_model(temp_path, compile=False)
            
            # Nettoyer
            os.unlink(temp_path)
            gc.collect()
            
        except ImportError:
            logger.error("❌ TensorFlow non installé")
            self.cnn_model = None
        except Exception as e:
            logger.error(f"❌ Erreur CNN: {e}")
            self.cnn_model = None
    
    def _download_model(self, filename):
        """Télécharge un modèle"""
        file_id = GOOGLE_DRIVE_IDS[filename]
        url = f"https://drive.google.com/uc?id={file_id}"
        temp_path = tempfile.mktemp(suffix='.pkl')
        
        gdown.download(url, temp_path, quiet=True)
        model = joblib.load(temp_path)
        os.unlink(temp_path)
        return model
    
    def _emergency_fallback(self):
        """Mode dégradé si mémoire insuffisante"""
        logger.warning("⚠️ Mode dégradé: CNN désactivé")
        self.cnn_model = None
        self.loaded = True  # On fonctionne quand même sans CNN

# Initialisation
models = MemoryOptimizedLoader()

# ========== SCHÉMAS ==========
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    success: bool
    category: str
    confidence: float
    prediction_type: str
    timestamp: str
    model_available: bool = True

# ========== ÉVÉNEMENTS ==========
@app.on_event("startup")
async def startup():
    """Chargement au démarrage"""
    try:
        models.load_all_with_memory_control()
    except Exception as e:
        logger.error(f"💥 Erreur critique: {e}")
        # L'API démarre quand même en mode dégradé

# ========== ENDPOINTS ==========
@app.get("/")
async def root():
    return {
        "message": "Product Classification API",
        "status": "online",
        "text_model": "loaded",
        "cnn_model": "loaded" if models.cnn_model else "disabled",
        "memory": "optimized"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if models.loaded else "degraded",
        "text_model": True,
        "cnn_model": models.cnn_model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    if not models.loaded:
        raise HTTPException(503, "Models not loaded")
    
    try:
        # Vectorisation
        text_vec = models.vectorizer.transform([request.text])
        
        # Prédiction
        prediction = models.text_model.predict(text_vec)
        pred_idx = int(prediction[0])
        category = models.label_mapping.get(pred_idx, f"Class_{pred_idx}")
        
        # Confiance
        if hasattr(models.text_model, 'predict_proba'):
            probs = models.text_model.predict_proba(text_vec)[0]
            confidence = float(probs[pred_idx])
        else:
            confidence = 0.95
        
        return PredictionResponse(
            success=True,
            category=category,
            confidence=round(confidence, 4),
            prediction_type="text",
            timestamp=datetime.now().isoformat(),
            model_available=True
        )
        
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if not models.loaded or models.cnn_model is None:
        raise HTTPException(503, "Image model not available")
    
    try:
        # Lire et prétraiter
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB').resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prédiction (avec verbosité désactivée)
        predictions = models.cnn_model.predict(img_array, verbose=0)
        
        # Résultats
        pred_idx = int(np.argmax(predictions[0]))
        category = CATEGORIES[pred_idx] if 0 <= pred_idx < len(CATEGORIES) else "Unknown"
        confidence = float(predictions[0][pred_idx])
        
        return PredictionResponse(
            success=True,
            category=category,
            confidence=round(confidence, 4),
            prediction_type="image",
            timestamp=datetime.now().isoformat(),
            model_available=True
        )
        
    except MemoryError:
        raise HTTPException(507, "Insufficient memory for image processing")
    except Exception as e:
        raise HTTPException(500, str(e))

# ========== LANCEMENT ==========
if __name__ == "__main__":
    import uvicorn
    
    # Configuration uvicorn pour économiser mémoire
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",  # Logs minimaux
        access_log=False,  # Pas de logs d'accès
        limit_concurrency=10,  # Limiter connexions
        timeout_keep_alive=30,
    )
    
    server = uvicorn.Server(config)
    server.run()