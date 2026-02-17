import streamlit as st
import torch
import pickle
import os
import requests
from model import GPTConfig, GPT

# --- CONFIGURATION DE LA PAGE & THÈME ---
st.set_page_config(page_title="UiT nanoGPT Storyteller", page_icon="🟢")

# Style CSS pour le thème Light Green (ChatGPT Style)
st.markdown("""
    <style>
    .stApp { background-color: #f0fdf4; } 
    [data-testid="stSidebar"] { background-color: #dcfce7; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #bbf7d0; border-radius: 15px; }
    .stChatMessage[data-testid="stChatMessageAssistant"] { background-color: #ffffff; border-radius: 15px; border: 1px solid #e2e8f0; }
    .stButton>button { background-color: #22c55e; color: white; border-radius: 20px; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION DU MODÈLE ---
FILE_ID = '1rLJSJQwdvRhRS8KdYjffTM-jkhPM0zGr'
CKPT_PATH = 'ckpt.pt'

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(CKPT_PATH):
        with st.spinner("Récupération du modèle (218 Mo)... Google Drive demande une confirmation antivirus."):
            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            # Première requête pour obtenir le cookie de confirmation
            response = session.get(URL, params={'id': FILE_ID}, stream=True)
            
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            
            # Si un jeton est trouvé, on relance la requête avec la confirmation
            if token:
                response = session.get(URL, params={'id': FILE_ID, 'confirm': token}, stream=True)
            
            # Écriture du fichier par morceaux (chunks)
            with open(CKPT_PATH, "wb") as f:
                for chunk in response.iter_content(32768):
                    if chunk: f.write(chunk)

    # Vérification : si le fichier est corrompu ou incomplet
    if os.path.getsize(CKPT_PATH) < 100000000: # Doit faire plus de 100Mo
        st.error("❌ Le téléchargement a échoué. Google Drive bloque peut-être l'accès.")
        if os.path.exists(CKPT_PATH): os.remove(CKPT_PATH)
        return None

    try:
        checkpoint = torch.load(CKPT_PATH, map_location='cpu')
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config)
        state_dict = checkpoint['model']
        # Nettoyage automatique des préfixes 'compile'
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement des poids : {e}")
        return None

# --- CHARGEMENT DES COMPOSANTS ---
model = download_and_load_model()

if model is None:
    st.warning("⚠️ L'application ne peut pas démarrer sans le modèle. Rafraîchissez la page.")
    st.stop()

# Chargement du dictionnaire meta.pkl (doit être sur ton GitHub)
try:
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
except FileNotFoundError:
    st.error("❌ Fichier 'meta.pkl' introuvable dans le dépôt GitHub.")
    st.stop()

# --- INTERFACE UTILISATEUR ---
st.title("🟢 UiT nanoGPT Storyteller")
st.caption("Architecture 24-layers entraînée sur TinyStories par Arthur PREVEL")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar pour les réglages
with st.sidebar:
    st.header("Paramètres")
    temp = st.slider("Créativité (Température)", 0.1, 1.2, 0.8)
    max_t = st.slider("Longueur de l'histoire", 50, 500, 200)
    if st.button("Nouvelle discussion"):
        st.session_state.messages = []
        st.rerun()

# Affichage des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrée du prompt
if prompt := st.chat_input("Il était une fois..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Génération en cours..."):
            # Encodage et génération par le modèle
            context_ids = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
            # Appel de ta fonction de génération nanoGPT
            output_ids = model.generate(context_ids, max_new_tokens=max_t, temperature=temp)[0].tolist()
            response = decode(output_ids)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
