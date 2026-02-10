"""
=============================================================================
E-COMMERCE PRODUCT CLASSIFIER - APPLICATION PROFESSIONNELLE
=============================================================================
Interface de classification intelligente de produits par IA
Design: Bleu Marine (#1E3A8A) + Blanc + Or (#D4AF37)
Thème: Clair/Sombre adaptatif
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PAGE
# =============================================================================

st.set_page_config(
    page_title="Product Classifier Pro",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CONSTANTES
# =============================================================================
API_URL = "https://product-classification-1.onrender.com"  

CATEGORIES = [
    "Baby Care",
    "Beauty and Personal Care",
    "Computers",
    "Home Decor & Festive Needs",
    "Home Furnishing",
    "Kitchen & Dining",
    "Watches"
]

# Images des catégories (Unsplash)
CATEGORY_IMAGES = {
    "Baby Care": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=300&h=200&fit=crop",
    "Beauty and Personal Care": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=300&h=200&fit=crop",
    "Computers": "https://images.unsplash.com/photo-1547082299-de196ea013d6?w=300&h=200&fit=crop",
    "Home Decor & Festive Needs": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=300&h=200&fit=crop",
    "Home Furnishing": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300&h=200&fit=crop",
    "Kitchen & Dining": "https://images.unsplash.com/photo-1556911220-bff31c812dba?w=300&h=200&fit=crop",
    "Watches": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=300&h=200&fit=crop"
}

# Images pour les features (Unsplash - professionnelles)
FEATURE_IMAGES = {
    "image_classification": "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=300&h=200&fit=crop&auto=format",
    "text_classification": "https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?w=300&h=200&fit=crop&auto=format",
    "analytics": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=300&h=200&fit=crop&auto=format",
    "batch_processing": "https://images.unsplash.com/photo-1552664730-d307ca884978?w=300&h=200&fit=crop&auto=format"
}

# Performances des modèles (à ajuster selon vos résultats réels)
MODEL_PERFORMANCE = {
    "text": {
        "name": "TF-IDF + SVM",
        "accuracy": 0.9557,
        "f1_score": 0.9549
    },
    "image": {
        "name": "VGG16",
        "accuracy": 0.7848,
        "f1_score": 0.7862
    }
}

# =============================================================================
# SESSION STATE
# =============================================================================

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# =============================================================================
# STYLES CSS PROFESSIONNELS
# =============================================================================

def apply_custom_css():
    """Styles CSS professionnels adaptatifs"""
    
    theme = st.session_state.theme
    
    if theme == 'light':
        # Thème clair
        bg_primary = "#FFFFFF"
        bg_secondary = "#F8FAFC"
        text_primary = "#1E293B"
        text_secondary = "#64748B"
        accent_blue = "#1E3A8A"
        accent_gold = "#D4AF37"
        border_color = "#E2E8F0"
        card_shadow = "0 4px 6px rgba(0, 0, 0, 0.07)"
        gradient_primary = "linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%)"
        gradient_secondary = "linear-gradient(135deg, #D4AF37 0%, #F59E0B 100%)"
    else:
        # Thème sombre
        bg_primary = "#0F172A"
        bg_secondary = "#1E293B"
        text_primary = "#F1F5F9"
        text_secondary = "#94A3B8"
        accent_blue = "#3B82F6"
        accent_gold = "#FCD34D"
        border_color = "#334155"
        card_shadow = "0 4px 6px rgba(0, 0, 0, 0.3)"
        gradient_primary = "linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)"
        gradient_secondary = "linear-gradient(135deg, #FCD34D 0%, #F59E0B 100%)"
    
    css = f"""
    <style>
    /* ============================================
       GLOBAL STYLES
       ============================================ */
    
    .stApp {{
        background: {bg_secondary};
    }}
    
    /* Cacher éléments Streamlit par défaut */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* ============================================
       NAVIGATION TOP BAR
       ============================================ */
    
    .top-nav {{
        background: {gradient_primary};
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: {card_shadow};
        border-radius: 0 0 16px 16px;
        margin: -5rem -5rem 2rem -5rem;
        position: relative;
        z-index: 100;
    }}
    
    .nav-brand {{
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    
    .nav-brand h1 {{
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    
    .nav-links {{
        display: flex;
        gap: 0.5rem;
    }}
    
    .nav-link {{
        background: rgba(255, 255, 255, 0.1);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        min-width: 120px;
        justify-content: center;
    }}
    
    .nav-link:hover {{
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }}
    
    .nav-link.active {{
        background: {accent_gold};
        color: {accent_blue};
        font-weight: 600;
    }}
    
    .nav-icon {{
        font-size: 1.2rem;
    }}
    
    .theme-toggle {{
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 500;
        min-width: 120px;
        justify-content: center;
    }}
    
    .theme-toggle:hover {{
        background: rgba(255, 255, 255, 0.3);
    }}
    
    /* ============================================
       CARDS & CONTAINERS
       ============================================ */
    
    .card {{
        background: {bg_primary};
        border-radius: 16px;
        padding: 2rem;
        box-shadow: {card_shadow};
        border: 1px solid {border_color};
        margin: 1rem 0;
    }}
    
    .card-header {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border-bottom: 2px solid {accent_gold};
        padding-bottom: 0.75rem;
    }}
    
    /* ============================================
       WELCOME PAGE
       ============================================ */
    
    .hero {{
        text-align: center;
        padding: 3rem 2rem;
        background: {gradient_primary};
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .hero-bg {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=1200&h=400&fit=crop');
        background-size: cover;
        background-position: center;
        opacity: 0.15;
        z-index: 0;
    }}
    
    .hero-content {{
        position: relative;
        z-index: 1;
    }}
    
    .hero h1 {{
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        line-height: 1.2;
    }}
    
    .hero p {{
        font-size: 1.3rem;
        opacity: 0.95;
        max-width: 700px;
        margin: 0 auto;
    }}
    
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .feature-card {{
        background: {bg_primary};
        padding: 0;
        border-radius: 16px;
        border: 1px solid {border_color};
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: {card_shadow};
    }}
    
    .feature-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        border-color: {accent_blue};
    }}
    
    .feature-image {{
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-bottom: 3px solid {accent_gold};
    }}
    
    .feature-content {{
        padding: 1.5rem;
    }}
    
    .feature-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 0.75rem;
    }}
    
    .feature-desc {{
        color: {text_secondary};
        font-size: 1rem;
        line-height: 1.6;
    }}
    
    /* ============================================
       BUTTONS
       ============================================ */
    
    .stButton>button {{
        background: {gradient_primary};
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        width: 100%;
        white-space: nowrap;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4);
    }}
    
    /* Secondary button */
    .secondary-btn {{
        background: {bg_primary} !important;
        color: {accent_blue} !important;
        border: 2px solid {accent_blue} !important;
    }}
    
    .secondary-btn:hover {{
        background: {accent_blue} !important;
        color: white !important;
    }}
    
    /* ============================================
       RESULTS DISPLAY
       ============================================ */
    
    .result-box {{
        background: {gradient_primary};
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(30, 58, 138, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .result-bg {{
        position: absolute;
        top: 0;
        right: 0;
        width: 150px;
        height: 100%;
        background-image: url('https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=300&h=200&fit=crop');
        background-size: cover;
        background-position: center;
        opacity: 0.1;
        z-index: 0;
    }}
    
    .result-content {{
        position: relative;
        z-index: 1;
    }}
    
    .result-category {{
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }}
    
    .result-confidence {{
        font-size: 1.5rem;
        opacity: 0.95;
    }}
    
    /* ============================================
       CATEGORY CARDS
       ============================================ */
    
    .category-card {{
        background: {bg_primary};
        border: 1px solid {border_color};
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: all 0.3s;
        cursor: pointer;
    }}
    
    .category-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }}
    
    .category-image {{
        width: 100%;
        height: 150px;
        object-fit: cover;
    }}
    
    .category-name {{
        padding: 1rem;
        font-weight: 700;
        color: {accent_blue};
        text-align: center;
        font-size: 1.05rem;
    }}
    
    /* ============================================
       METRICS CARDS
       ============================================ */
    
    .metric-card {{
        background: {bg_primary};
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {accent_gold};
        box-shadow: {card_shadow};
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }}
    
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {accent_blue};
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 0.95rem;
        color: {text_secondary};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* ============================================
       MODEL PERFORMANCE CARDS
       ============================================ */
    
    .model-card {{
        background: {bg_primary};
        border-radius: 16px;
        overflow: hidden;
        box-shadow: {card_shadow};
        transition: all 0.3s ease;
        border: 1px solid {border_color};
    }}
    
    .model-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.15);
    }}
    
    .model-header {{
        background: {gradient_primary};
        padding: 1.5rem;
        text-align: center;
    }}
    
    .model-title {{
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }}
    
    .model-name {{
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
    }}
    
    .model-content {{
        padding: 1.5rem;
    }}
    
    .model-metrics {{
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        flex-wrap: wrap;
    }}
    
    .model-metric {{
        text-align: center;
    }}
    
    .model-metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {accent_gold};
        margin-bottom: 0.25rem;
    }}
    
    .model-metric-label {{
        font-size: 0.85rem;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* ============================================
       BATCH PROCESSING
       ============================================ */
    
    .batch-result-card {{
        background: {bg_primary};
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid {border_color};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }}
    
    .batch-result-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    .batch-success {{
        border-left: 4px solid #10B981;
    }}
    
    .batch-error {{
        border-left: 4px solid #EF4444;
    }}
    
    /* ============================================
       FILE UPLOADER
       ============================================ */
    
    .uploadedFile {{
        border: 2px dashed {accent_blue} !important;
        border-radius: 12px !important;
        background: {bg_secondary} !important;
    }}
    
    /* ============================================
       TEXT AREA
       ============================================ */
    
    .stTextArea>div>div>textarea {{
        border: 2px solid {border_color};
        border-radius: 12px;
        background: {bg_primary};
        color: {text_primary};
    }}
    
    .stTextArea>div>div>textarea:focus {{
        border-color: {accent_blue};
    }}
    
    /* ============================================
       FOOTER
       ============================================ */
    
    .custom-footer {{
        background: {bg_secondary};
        padding: 3rem 2rem;
        margin-top: 4rem;
        border-top: 1px solid {border_color};
        text-align: center;
        border-radius: 16px 16px 0 0;
        position: relative;
        overflow: hidden;
    }}
    
    .footer-bg {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: {gradient_primary};
        opacity: 0.05;
        z-index: 0;
    }}
    
    .footer-content {{
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }}
    
    .footer-logo {{
        font-size: 1.5rem;
        font-weight: 800;
        color: {accent_blue};
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }}
    
    .footer-text {{
        color: {text_secondary};
        font-size: 0.95rem;
        margin: 0.5rem 0;
        line-height: 1.6;
    }}
    
    .footer-links {{
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }}
    
    .footer-link {{
        color: {accent_blue};
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }}
    
    .footer-link:hover {{
        color: {accent_gold};
        background: rgba(30, 58, 138, 0.1);
    }}
    
    .footer-social {{
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
    }}
    
    .social-icon {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: {bg_secondary};
        display: flex;
        align-items: center;
        justify-content: center;
        color: {accent_blue};
        text-decoration: none;
        transition: all 0.3s ease;
        border: 1px solid {border_color};
    }}
    
    .social-icon:hover {{
        background: {accent_blue};
        color: white;
        transform: translateY(-3px);
    }}
    
    .footer-divider {{
        height: 1px;
        background: {border_color};
        margin: 2rem 0;
    }}
    
    .footer-copyright {{
        color: {text_secondary};
        font-size: 0.85rem;
        margin-top: 1.5rem;
    }}
    
    /* ============================================
       NAVIGATION CONTAINER
       ============================================ */
    
    .main-nav-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
        background: {gradient_primary};
        padding: 1rem 2rem;
        border-radius: 12px;
        box-shadow: {card_shadow};
    }}
    
    .nav-buttons-container {{
        display: flex;
        gap: 10px;
        flex: 1;
    }}
    
    .theme-switch-container {{
        display: flex;
        justify-content: flex-end;
    }}
    
    /* ============================================
       RESPONSIVE
       ============================================ */
    
    @media (max-width: 768px) {{
        .hero h1 {{
            font-size: 2rem;
        }}
        
        .hero p {{
            font-size: 1.1rem;
        }}
        
        .nav-links {{
            flex-direction: column;
        }}
        
        .footer-links {{
            flex-direction: column;
            gap: 1rem;
        }}
        
        .main-nav-container {{
            flex-direction: column;
            gap: 1rem;
        }}
        
        .nav-buttons-container {{
            flex-direction: column;
            width: 100%;
        }}
        
        .theme-switch-container {{
            width: 100%;
            justify-content: center;
        }}
    }}
    
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# NAVIGATION
# =============================================================================

def render_navigation():
    """Barre de navigation avec bouton de thème icône seulement"""
    
    # Créer 5 colonnes
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.2])  
    
    # Liste des pages
    pages = [
        {"key": "home", "label": "HOME", "image": "https://images.unsplash.com/photo-1560179707-f14e90ef3623?w=70&h=70&fit=crop", "col": col1},
        {"key": "image", "label": "IMAGE CLASSIFICATION", "image": "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=70&h=70&fit=crop", "col": col2},
        {"key": "text", "label": "TEXT CLASSIFICATION", "image": "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=70&h=70&fit=crop", "col": col3},
        {"key": "dashboard", "label": "DASHBOARD", "image": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=70&h=70&fit=crop", "col": col4}
    ]
    
    # Afficher les 4 boutons de navigation
    for page in pages:
        with page["col"]:
            # Image
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 8px;">
                <img src="{page['image']}" style="width: 50px; height: 50px; border-radius: 10px; border: 2px solid rgba(30, 58, 138, 0.2);">
            </div>
            """, unsafe_allow_html=True)
            
            # Bouton
            if st.button(page["label"], key=f"nav_{page['key']}", use_container_width=True):
                st.session_state.current_page = page['key']
                st.rerun()
    
    # 5ème colonne : Bouton de thème (icône seulement)
    with col5:
        theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
        
        # CSS pour un bouton icône rond et discret
        st.markdown(f"""
        <style>
        /* Bouton icône rond */
        .theme-icon-btn {{
            background: rgba(30, 58, 138, 0.08) !important;
            border: 1.5px solid rgba(30, 58, 138, 0.15) !important;
            color: #1E3A8A !important;
            padding: 0 !important;
            border-radius: 50% !important;
            font-size: 1.2rem !important;
            font-weight: 400 !important;
            width: 40px !important;
            height: 40px !important;
            min-height: 40px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 auto !important;
            transition: all 0.2s ease !important;
        }}
        
        .theme-icon-btn:hover {{
            background: rgba(30, 58, 138, 0.12) !important;
            border-color: rgba(30, 58, 138, 0.25) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 8px rgba(30, 58, 138, 0.1) !important;
        }}
        
        /* Ajuster l'espacement */
        div[data-testid="column"]:nth-child(5) {{
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding-top: 8px;
        }}
        
        /* Cacher le texte dans le bouton */
        .theme-icon-btn > div > p {{
            display: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Espace pour aligner verticalement avec les images
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Bouton de thème (icône seulement)
        if st.button(theme_icon, key="theme_btn", 
                     help="Switch between light and dark mode",
                     use_container_width=False):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.rerun()

# =============================================================================
# UTILITAIRES API
# =============================================================================

def call_image_api(image_file):
    """Appeler API pour classification image"""
    try:
        files = {"file": image_file}
        response = requests.post(f"{API_URL}/predict/image", files=files, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Ajouter le type de prédiction
        if result and 'success' in result:
            result['prediction_type'] = 'image'
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def call_text_api(text):
    """Appeler API pour classification texte"""
    try:
        data = {"text": text}
        response = requests.post(f"{API_URL}/predict/text", json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Ajouter le type de prédiction
        if result and 'success' in result:
            result['prediction_type'] = 'text'
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Classification Error: {e.response.json() if hasattr(e, 'response') else str(e)}")
        return None

def get_api_metrics():
    """Récupérer métriques API"""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None

# =============================================================================
# FONCTIONS DE TRAITEMENT PAR LOT
# =============================================================================

def process_batch_images(image_files):
    """Traiter plusieurs images en lot"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, image_file in enumerate(image_files):
        try:
            # Mettre à jour la progression
            progress_bar.progress((i + 1) / len(image_files))
            status_text.text(f"Processing image {i+1}/{len(image_files)}: {image_file.name[:30]}...")
            
            # Appeler l'API pour chaque image
            image_file.seek(0)
            result = call_image_api(image_file)
            
            if result and result.get('success'):
                # Ajouter à l'historique
                st.session_state.prediction_history.append({
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'timestamp': datetime.now(),
                    'type': 'image',
                    'filename': image_file.name
                })
                
                results.append({
                    'filename': image_file.name,
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'success': True
                })
            else:
                results.append({
                    'filename': image_file.name,
                    'error': 'Classification failed',
                    'success': False
                })
                
        except Exception as e:
            results.append({
                'filename': image_file.name,
                'error': str(e),
                'success': False
            })
    
    progress_bar.empty()
    status_text.empty()
    return results

def process_batch_texts(texts):
    """Traiter plusieurs textes en lot"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        try:
            # Mettre à jour la progression
            progress_bar.progress((i + 1) / len(texts))
            status_text.text(f"Processing text {i+1}/{len(texts)}...")
            
            # Appeler l'API pour chaque texte
            result = call_text_api(text)
            
            if result and result.get('success'):
                # Ajouter à l'historique
                st.session_state.prediction_history.append({
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'timestamp': datetime.now(),
                    'type': 'text',
                    'text_preview': text[:50] + "..." if len(text) > 50 else text
                })
                
                results.append({
                    'text_preview': text[:50] + "..." if len(text) > 50 else text,
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'success': True
                })
            else:
                results.append({
                    'text_preview': text[:50] + "..." if len(text) > 50 else text,
                    'error': 'Classification failed',
                    'success': False
                })
                
        except Exception as e:
            results.append({
                'text_preview': text[:50] + "..." if len(text) > 50 else text,
                'error': str(e),
                'success': False
            })
    
    progress_bar.empty()
    status_text.empty()
    return results

# =============================================================================
# COMPOSANTS VISUELS
# =============================================================================

def create_confidence_gauge(confidence):
    """Jauge de confiance Plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 20, 'color': '#1E3A8A'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#1E3A8A'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#1E3A8A"},
            'bar': {'color': "#1E3A8A"},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#D4AF37",
            'steps': [
                {'range': [0, 60], 'color': '#FEE2E2'},
                {'range': [60, 80], 'color': '#FEF3C7'},
                {'range': [80, 100], 'color': '#D1FAE5'}
            ],
            'threshold': {
                'line': {'color': "#D4AF37", 'width': 4},
                'thickness': 0.8,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig

def create_probability_chart(probabilities):
    """Graphique des probabilités par catégorie"""
    df = pd.DataFrame(list(probabilities.items()), columns=['Category', 'Probability'])
    df = df.sort_values('Probability', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Category'],
        orientation='h',
        marker=dict(
            color=df['Probability'],
            colorscale=[[0, '#EFF6FF'], [0.5, '#3B82F6'], [1, '#1E3A8A']],
            line=dict(color='#D4AF37', width=2)
        ),
        text=df['Probability'].apply(lambda x: f'{x*100:.1f}%'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title={'text': "Probability Distribution", 'font': {'size': 18, 'color': '#1E3A8A'}},
        xaxis_title="Probability",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif", 'size': 12}
    )
    
    fig.update_layout(
        xaxis=dict(
            showgrid=True,
            gridcolor='#E5E7EB'
        )
    )
    
    return fig

def display_single_result(result):
    """Afficher résultat de prédiction pour une seule image/texte"""
    if result and result.get('success'):
        st.markdown(f"""
            <div class="result-box">
                <div class="result-bg"></div>
                <div class="result-content">
                    <div class="result-category">{result['category']}</div>
                    <div class="result-confidence">Confidence: {result['confidence']*100:.1f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_confidence_gauge(result['confidence']), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_probability_chart(result['all_probabilities']), use_container_width=True)
        
        # Ajouter à l'historique
        prediction_data = {
            'category': result['category'],
            'confidence': result['confidence'],
            'timestamp': datetime.now(),
            'type': result.get('prediction_type', 'unknown')
        }
        
        # Ajouter des informations supplémentaires selon le type
        if result.get('prediction_type') == 'image' and hasattr(st.session_state, 'last_uploaded_file'):
            prediction_data['filename'] = st.session_state.last_uploaded_file.name
        elif result.get('prediction_type') == 'text' and hasattr(st.session_state, 'last_text_input'):
            text = st.session_state.last_text_input
            prediction_data['text_preview'] = text[:50] + "..." if len(text) > 50 else text
        
        st.session_state.prediction_history.append(prediction_data)
        st.session_state.last_prediction = result

def display_batch_results(results, result_type="image"):
    """Afficher les résultats du traitement par lot"""
    
    if not results:
        return
    
    # Statistiques
    success_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Items", total_count)
    with col2:
        st.metric("Successful", success_count)
    with col3:
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Afficher les résultats détaillés
    st.markdown("### 📋 Batch Results Summary")
    
    for result in results:
        if result.get('success', False):
            if result_type == "image":
                st.markdown(f"""
                <div class="batch-result-card batch-success">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{result['filename']}</strong>
                            <div style="margin-top: 5px;">
                                <span style="background: #10B98120; color: #065F46; padding: 2px 8px; border-radius: 12px; font-size: 0.9rem;">
                                    {result['category']}
                                </span>
                                <span style="margin-left: 10px; color: #6B7280;">
                                    Confidence: {result['confidence']*100:.1f}%
                                </span>
                            </div>
                        </div>
                        <div style="color: #10B981; font-weight: bold;">✓</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="batch-result-card batch-success">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="color: #6B7280; font-size: 0.9rem; margin-bottom: 5px;">
                                "{result['text_preview']}"
                            </div>
                            <div>
                                <span style="background: #10B98120; color: #065F46; padding: 2px 8px; border-radius: 12px; font-size: 0.9rem;">
                                    {result['category']}
                                </span>
                                <span style="margin-left: 10px; color: #6B7280;">
                                    Confidence: {result['confidence']*100:.1f}%
                                </span>
                            </div>
                        </div>
                        <div style="color: #10B981; font-weight: bold;">✓</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            if result_type == "image":
                st.markdown(f"""
                <div class="batch-result-card batch-error">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{result['filename']}</strong>
                            <div style="margin-top: 5px; color: #DC2626;">
                                Error: {result.get('error', 'Unknown error')}
                            </div>
                        </div>
                        <div style="color: #DC2626; font-weight: bold;">✗</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="batch-result-card batch-error">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="color: #6B7280; font-size: 0.9rem; margin-bottom: 5px;">
                                "{result.get('text_preview', 'N/A')}"
                            </div>
                            <div style="color: #DC2626;">
                                Error: {result.get('error', 'Unknown error')}
                            </div>
                        </div>
                        <div style="color: #DC2626; font-weight: bold;">✗</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.success(f"✅ All results have been saved to the dashboard history. Go to Dashboard page to view detailed analytics.")

# =============================================================================
# PAGES
# =============================================================================

def page_home():
    """Page d'accueil avec design professionnel"""
    
    # Hero Section
    st.markdown(f"""
        <div class="hero">
            <div class="hero-bg"></div>
            <div class="hero-content">
                <div style="text-align: center; padding: 20px;">
                    <div style="display: inline-block; background: rgba(255, 255, 255, 0.1); 
                         padding: 20px; border-radius: 20px; margin-bottom: 30px; 
                         border: 2px solid rgba(255, 255, 255, 0.2);">
                        <img src="https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=100&h=100&fit=crop&crop=center" 
                             style="width: 80px; height: 80px; border-radius: 16px; border: 3px solid white; 
                                    margin-bottom: 15px;">
                        <h1 style="color: white; margin: 0; font-size: 2.8rem;">Product Classification System</h1>
                    </div>
                    <p style="font-size: 1.4rem; max-width: 800px; margin: 0 auto; line-height: 1.6;">
                        Advanced AI-powered product categorization for e-commerce platforms
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Features
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin: 3rem 0 2rem 0;'>Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="feature-card">
                <img src="{FEATURE_IMAGES['image_classification']}" class="feature-image" alt="Image Classification">
                <div class="feature-content">
                    <div class="feature-title">Image classification</div>
                    <div class="feature-desc">Upload product images for instant AI-powered categorization with high accuracy </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="feature-card">
                <img src="{FEATURE_IMAGES['text_classification']}" class="feature-image" alt="Text Classification">
                <div class="feature-content">
                    <div class="feature-title">Text Classification</div>
                    <div class="feature-desc">Classify products using natural language descriptions with advanced NLP algorithms for precise categorization.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="feature-card">
                <img src="{FEATURE_IMAGES['analytics']}" class="feature-image" alt="Analytics">
                <div class="feature-content">
                    <div class="feature-title">Real-time Analytics</div>
                    <div class="feature-desc">Monitor performance metrics, classification history, and system statistics in real-time dashboards.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="feature-card">
                <img src="{FEATURE_IMAGES['batch_processing']}" class="feature-image" alt="Batch Processing">
                <div class="feature-content">
                    <div class="feature-title">Batch Processing</div>
                    <div class="feature-desc">For your classification tasks, process multiple images or texts simultaneously for efficient bulk classification.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Model Performance
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin: 3rem 0 2rem 0;'>Model Performance</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        text_model = MODEL_PERFORMANCE['text']
        st.markdown(f"""
            <div class="model-card">
                <div class="model-header">
                    <div class="model-title">Text Classification Model</div>
                    <div class="model-name">{text_model['name']}</div>
                </div>
                <div class="model-content">
                    <div class="model-metrics">
                        <div class="model-metric">
                            <div class="model-metric-value">{text_model['accuracy']*100:.2f}%</div>
                            <div class="model-metric-label">Accuracy</div>
                        </div>
                        <div class="model-metric">
                            <div class="model-metric-value">{text_model['f1_score']*100:.2f}%</div>
                            <div class="model-metric-label">F1-Score</div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        image_model = MODEL_PERFORMANCE['image']
        st.markdown(f"""
            <div class="model-card">
                <div class="model-header">
                    <div class="model-title">Image Classification Model</div>
                    <div class="model-name">{image_model['name']}</div>
                </div>
                <div class="model-content">
                    <div class="model-metrics">
                        <div class="model-metric">
                            <div class="model-metric-value">{image_model['accuracy']*100:.2f}%</div>
                            <div class="model-metric-label">Accuracy</div>
                        </div>
                        <div class="model-metric">
                            <div class="model-metric-value">{image_model['f1_score']*100:.2f}%</div>
                            <div class="model-metric-label">F1-Score</div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Categories
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin: 3rem 0 2rem 0;'>Product Categories</h2>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    for idx, category in enumerate(CATEGORIES):
        with cols[idx % 4]:
            img_url = CATEGORY_IMAGES.get(category, "")
            st.markdown(f"""
                <div class="category-card">
                    <img src="{img_url}" class="category-image" alt="{category}" onerror="this.style.display='none'">
                    <div class="category-name">{category}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # About Section
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin: 3rem 0 2rem 0;'>About This Application</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card">
            <p style='font-size: 1.1rem; line-height: 1.8; color: #475569; text-align: justify;'>
            This enterprise-grade application leverages state-of-the-art machine learning models to automatically 
            classify e-commerce products into seven distinct categories. The system employs two specialized 
            classification pipelines: a computer vision model (VGG16) for image-based classification, 
            and a natural language processing model (TF-IDF + SVM) for text-based classification.
            </p>
            <p style='font-size: 1.1rem; line-height: 1.8; color: #475569; text-align: justify; margin-top: 1.5rem;'>
            Built with cutting-edge technologies including TensorFlow, scikit-learn, FastAPI, and Streamlit, 
            this solution provides real-time classification with professional-grade accuracy. The modular 
            architecture ensures scalability and easy integration into existing e-commerce platforms.
            </p>
            <div style='margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); border-radius: 12px; border-left: 4px solid #1E3A8A;'>
                <h3 style='color: #1E3A8A; margin-bottom: 1rem;'>Technologies used</h3>
                <ul style='color: #475569; font-size: 1rem; line-height: 1.8;'>
                    <li><strong>Deep Learning:</strong> TensorFlow, VGG16</li>
                    <li><strong>NLP:</strong> TF-IDF, Support Vector Machines</li>
                    <li><strong>Backend:</strong> FastAPI, Python 3.12</li>
                    <li><strong>Frontend:</strong> Streamlit with custom CSS</li>
                    <li><strong>Visualization:</strong> Plotly, Pandas</li>
                    <li><strong>Batch Processing:</strong> Multi-threaded classification</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

def page_image():
    """Page classification par image avec traitement par lot"""
    
    st.markdown("<div class='card-header'>Image classification</div>", unsafe_allow_html=True)
    
    # Onglets pour single vs batch
    tab1, tab2 = st.tabs(["Single image", "Batch processing"])
    
    with tab1:
        st.markdown("### Upload Single Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            key="single_image"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=400)
            
            with col2:
                st.markdown("<br>" * 3, unsafe_allow_html=True)
                if st.button("🔍 Classify Product", key="classify_img_single", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        # Stocker le fichier pour l'historique
                        if 'last_uploaded_file' not in st.session_state:
                            st.session_state.last_uploaded_file = uploaded_file
                        else:
                            st.session_state.last_uploaded_file = uploaded_file
                            
                        uploaded_file.seek(0)
                        result = call_image_api(uploaded_file)
                        if result:
                            display_single_result(result)
    
    with tab2:
        st.markdown("### Batch image processing")
        st.markdown("Upload multiple images for bulk classification.")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Select multiple images for batch processing",
            key="batch_images"
        )
        
        if uploaded_files:
            # Statistiques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Images Selected", len(uploaded_files))
            with col2:
                total_size = sum(f.size for f in uploaded_files) / (1024*1024)
                st.metric("Total Size", f"{total_size:.2f} MB")
            with col3:
                st.metric("Status", "Ready")
            
            # Aperçu des images (limitée à 6 images)
            st.markdown("####  Image Preview")
            if len(uploaded_files) > 0:
                cols = st.columns(min(3, len(uploaded_files)))
                for idx, file in enumerate(uploaded_files[:6]):  # Limiter à 6 images
                    with cols[idx % len(cols)]:
                        img = Image.open(file)
                        st.image(img, caption=file.name[:20] + "..." if len(file.name) > 20 else file.name, width=150)
                
                if len(uploaded_files) > 6:
                    st.info(f" ... and {len(uploaded_files) - 6} more images")
            
            # Bouton de traitement
            if st.button(" Process All Images", key="process_batch_images", use_container_width=True):
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    results = process_batch_images(uploaded_files)
                    display_batch_results(results, "image")

def page_text():
    """Page classification par texte avec traitement par lot"""
    
    st.markdown("<div class='card-header'> Text Classification</div>", unsafe_allow_html=True)
    
    # Onglets pour single vs batch
    tab1, tab2 = st.tabs([" Single text", " Batch processing"])
    
    with tab1:
        st.markdown("### Single Text Classification")
        text_input = st.text_area(
            "Enter product description",
            height=200,
            placeholder="Example: Comfortable cotton baby onesie with snap closures, perfect for newborns...",
            help="Provide a detailed product description for best results",
            key="single_text"
        )
        
        if st.button(" Classify Product", key="classify_txt_single", use_container_width=True):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    # Stocker le texte pour l'historique
                    if 'last_text_input' not in st.session_state:
                        st.session_state.last_text_input = text_input
                    else:
                        st.session_state.last_text_input = text_input
                        
                    result = call_text_api(text_input)
                    if result:
                        display_single_result(result)
            else:
                st.warning("⚠️ Please enter a product description")
    
    with tab2:
        st.markdown("###  Batch Text Processing")
        st.markdown("Enter multiple product descriptions for bulk classification. All results will be saved to the dashboard.")
        
        # Option 1: Upload CSV file
        st.markdown("#### Option 1: Upload CSV File")
        csv_file = st.file_uploader(
            "Upload CSV file with descriptions",
            type=["csv"],
            help="CSV should have a column named 'description'",
            key="batch_csv"
        )
        
        if csv_file:
            try:
                df = pd.read_csv(csv_file)
                if 'description' in df.columns:
                    text_inputs = df['description'].dropna().tolist()
                    st.success(f" Loaded {len(text_inputs)} descriptions from CSV")
                    
                    # Aperçu des données
                    with st.expander(" CSV Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button(" Process CSV Descriptions", key="process_csv", use_container_width=True):
                        with st.spinner(f"Processing {len(text_inputs)} descriptions..."):
                            results = process_batch_texts(text_inputs)
                            display_batch_results(results, "text")
                else:
                    st.error("❌ CSV file must contain a column named 'description'")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        
        # Option 2: Manual entry
        st.markdown("---")
        st.markdown("#### Option 2: Manual Entry")
        
        num_texts = st.number_input("Number of descriptions to classify", 
                                    min_value=1, max_value=20, value=3, step=1,
                                    help="Enter how many product descriptions you want to classify")
        
        text_inputs = []
        for i in range(num_texts):
            text = st.text_area(
                f"Description {i+1}",
                height=100,
                placeholder=f"Enter product description {i+1}...",
                key=f"batch_text_{i}"
            )
            if text.strip():
                text_inputs.append(text.strip())
        
        if text_inputs:
            st.info(f" {len(text_inputs)} description(s) ready for processing")
            
            if st.button(" Process All Descriptions", key="process_manual", use_container_width=True):
                with st.spinner(f"Processing {len(text_inputs)} descriptions..."):
                    results = process_batch_texts(text_inputs)
                    display_batch_results(results, "text")

def page_dashboard():
    """Page dashboard avec métriques avancées et historique complet"""
    
    st.markdown("<div class='card-header'> Analytics Dashboard</div>", unsafe_allow_html=True)
    
    # Section 1: Métriques Globales
    st.markdown("<h3 style='color: #1E3A8A; margin: 2rem 0 1.5rem 0;'> Performance Overview</h3>", unsafe_allow_html=True)
    
    # Récupérer les métriques de l'API
    metrics = get_api_metrics()
    
    # Calculer les métriques de session
    if st.session_state.prediction_history:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        # Calculer la confiance moyenne par type
        image_history = df_history[df_history['type'] == 'image']
        text_history = df_history[df_history['type'] == 'text']
        
        avg_confidence_image = image_history['confidence'].mean() if len(image_history) > 0 else 0
        avg_confidence_text = text_history['confidence'].mean() if len(text_history) > 0 else 0
        
        # Distribution par catégorie
        category_counts = df_history['category'].value_counts().to_dict()
        avg_confidence_by_cat = df_history.groupby('category')['confidence'].mean().to_dict()
        
        # Statistiques par type
        image_count = len(image_history)
        text_count = len(text_history)
        total_count = len(df_history)
    else:
        avg_confidence_image = 0
        avg_confidence_text = 0
        category_counts = {}
        avg_confidence_by_cat = {}
        image_count = 0
        text_count = 0
        total_count = 0
    
    # Row 1: Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_count}</div>
                <div class="metric-label">Total Predictions</div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #64748B;">
                    Images: {image_count} | Texts: {text_count}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accuracy_text = MODEL_PERFORMANCE['text']['accuracy'] * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #10B981;">{accuracy_text:.1f}%</div>
                <div class="metric-label">Text Model Accuracy</div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748B;">
                    F1: {MODEL_PERFORMANCE['text']['f1_score']:.3f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        accuracy_image = MODEL_PERFORMANCE['image']['accuracy'] * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #10B981;">{accuracy_image:.1f}%</div>
                <div class="metric-label">Image Model Accuracy</div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748B;">
                    F1: {MODEL_PERFORMANCE['image']['f1_score']:.3f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_confidence_session = (avg_confidence_image + avg_confidence_text) / 2 if (image_count + text_count) > 0 else 0
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #D4AF37;">{avg_confidence_session*100:.1f}%</div>
                <div class="metric-label">Avg. Confidence</div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748B;">
                    Image: {avg_confidence_image*100:.1f}% | Text: {avg_confidence_text*100:.1f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Section 2: Dernière prédiction
    if st.session_state.last_prediction:
        st.markdown("<h3 style='color: #1E3A8A; margin: 3rem 0 1.5rem 0;'>🔄 Last Prediction</h3>", unsafe_allow_html=True)
        
        last_result = st.session_state.last_prediction
        if last_result.get('success'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%); 
                            padding: 1.5rem; border-radius: 12px; border-left: 4px solid #D4AF37;">
                        <div style="font-size: 1.2rem; font-weight: 600; color: #1E3A8A; margin-bottom: 0.5rem;">
                            {last_result['category']}
                        </div>
                        <div style="color: #64748B; margin-bottom: 1rem;">
                            Type: {last_result.get('prediction_type', 'Unknown').title()}
                        </div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #1E3A8A;">
                            {last_result['confidence']*100:.1f}%
                        </div>
                        <div style="color: #64748B; font-size: 0.9rem;">
                            Confidence Score
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Graphique des probabilités pour la dernière prédiction
                if 'all_probabilities' in last_result:
                    df_probs = pd.DataFrame(list(last_result['all_probabilities'].items()), 
                                           columns=['Category', 'Probability'])
                    df_probs = df_probs.sort_values('Probability', ascending=True)
                    
                    fig = px.bar(df_probs, y='Category', x='Probability', 
                                orientation='h',
                                title="Probability Distribution (Last Prediction)",
                                color='Probability',
                                color_continuous_scale='Blues')
                    
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            title='Probability',
                            gridcolor='#E2E8F0'
                        ),
                        yaxis=dict(
                            autorange="reversed",
                            gridcolor='#E2E8F0'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Section 3: Distribution des prédictions
    if st.session_state.prediction_history:
        st.markdown("<h3 style='color: #1E3A8A; margin: 3rem 0 1.5rem 0;'> Prediction Analytics</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution par type
            type_counts = df_history['type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Predictions by Type",
                color_discrete_sequence=['#1E3A8A', '#D4AF37'],
                hole=0.4
            )
            fig.update_layout(
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top catégories par nombre de prédictions
            if category_counts:
                top_categories = pd.Series(category_counts).sort_values(ascending=False).head(5)
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_categories.values,
                        y=top_categories.index,
                        orientation='h',
                        marker_color=['#1E3A8A', '#2563EB', '#3B82F6', '#60A5FA', '#93C5FD'],
                        text=top_categories.values,
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title='Top 5 Categories',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        title='Number of Predictions',
                        gridcolor='#E2E8F0'
                    ),
                    yaxis=dict(
                        autorange="reversed",
                        gridcolor='#E2E8F0'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Section 4: Historique complet des prédictions
    if st.session_state.prediction_history:
        st.markdown("<h3 style='color: #1E3A8A; margin: 3rem 0 1.5rem 0;'> Complete Prediction History</h3>", unsafe_allow_html=True)
        
        # Préparer les données pour l'affichage
        df_display = pd.DataFrame(st.session_state.prediction_history)
        df_display['timestamp'] = df_display['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x*100:.1f}%")
        df_display['type'] = df_display['type'].apply(lambda x: x.capitalize())
        
        # Ajouter des informations supplémentaires
        if 'filename' in df_display.columns:
            df_display['input'] = df_display['filename']
        elif 'text_preview' in df_display.columns:
            df_display['input'] = df_display['text_preview'].apply(lambda x: f'"{x}"')
        else:
            df_display['input'] = 'N/A'
        
        # Trier par date
        df_display = df_display.sort_values('timestamp', ascending=False)
        
        # Afficher le tableau stylisé
        st.dataframe(
            df_display[['timestamp', 'type', 'input', 'category', 'confidence']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'timestamp': st.column_config.TextColumn('Timestamp', width='medium'),
                'type': st.column_config.TextColumn('Type', width='small'),
                'input': st.column_config.TextColumn('Input', width='large'),
                'category': st.column_config.TextColumn('Category', width='medium'),
                'confidence': st.column_config.TextColumn('Confidence', width='small')
            }
        )
        
        # Options d'export
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
                st.session_state.prediction_history = []
                st.session_state.last_prediction = None
                st.rerun()
        
        with col2:
            # Export CSV
            csv = df_display.to_csv(index=False)
            st.download_button(
                label=" Export history to CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        # Afficher un message si pas d'historique
        st.markdown("""
            <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%); 
                    border-radius: 16px; border: 2px dashed #CBD5E1; margin: 2rem 0;">
                <div style="font-size: 4rem; color: #CBD5E1; margin-bottom: 1rem;">📊</div>
                <h3 style="color: #64748B; margin-bottom: 1rem;">No Prediction History Yet</h3>
                <p style="color: #94A3B8; max-width: 600px; margin: 0 auto;">
                    Start classifying products using the Image or Text classification pages to see analytics here.
                </p>
                <div style="margin-top: 2rem;">
                    <a href="#" onclick="window.parent.postMessage({{'type': 'navigation', 'page': 'image'}}, '*')" 
                       style="background: #1E3A8A; color: white; padding: 0.75rem 1.5rem; border-radius: 8px; 
                              text-decoration: none; font-weight: 600; display: inline-block; margin: 0 0.5rem;">
                        Classify Images
                    </a>
                    <a href="#" onclick="window.parent.postMessage({{'type': 'navigation', 'page': 'text'}}, '*')" 
                       style="background: #D4AF37; color: #1E3A8A; padding: 0.75rem 1.5rem; border-radius: 8px; 
                              text-decoration: none; font-weight: 600; display: inline-block; margin: 0 0.5rem;">
                        Classify Texts
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_footer():
    """Footer professionnel amélioré"""
    current_year = datetime.now().year
    st.markdown(f"""
        <div class="custom-footer">
            <div class="footer-bg"></div>
            <div class="footer-content">
                <div class="footer-logo">
                    <img src="https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=30&h=30&fit=crop&crop=center" 
                         style="width: 30px; height: 30px; border-radius: 8px; border: 2px solid #1E3A8A;">
                    Product Classifier Pro
                </div>
                <p class="footer-text">
                    Advanced AI-powered product classification system for e-commerce platforms.<br>
                    Leveraging state-of-the-art machine learning models for accurate product categorization.
                </p>
                <div class="footer-divider"></div>
                <div class="footer-copyright">
                    © {current_year} Product Classification System. All rights reserved.<br>
                    Powered by Advanced Machine Learning & Deep Learning Models
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Application principale"""
    
    # Appliquer CSS
    apply_custom_css()
    
    # Navigation
    render_navigation()
    
    # Router vers la page appropriée
    if st.session_state.current_page == 'home':
        page_home()
    elif st.session_state.current_page == 'image':
        page_image()
    elif st.session_state.current_page == 'text':
        page_text()
    elif st.session_state.current_page == 'dashboard':
        page_dashboard()
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()