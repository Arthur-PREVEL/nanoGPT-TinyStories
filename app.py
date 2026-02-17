import streamlit as st
import torch
import pickle
import os
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT

# --- CONFIGURATION DE LA PAGE & THÈME ---
st.set_page_config(page_title="UiT nanoGPT Storyteller", page_icon="🟢")

# Style CSS pour le thème Light Green
st.markdown("""
    <style>
    .stApp { background-color: #f0fdf4; } 
    [data-testid="stSidebar"] { background-color: #dcfce7; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #bbf7d0; border-radius: 15px; }
    .stChatMessage[data-testid="stChatMessageAssistant"] { background-color: #ffffff; border-radius: 15px; border: 1px solid #e2e8f0; }
    .stButton>button { background-color: #22c55e; color: white; border-radius: 20px; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION HUGGING FACE ---
REPO_ID = "Arthur-PREVEL/nanogpt-tinystories-depth24" 
FILENAME = "ckpt.pt"

@st.cache_resource
def load_model_from_hf():
    try:
        with st.spinner("Chargement du modèle depuis Hugging Face (218 Mo)..."):
            # Téléchargement sécurisé via Hugging Face Hub
            path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            
            checkpoint = torch.load(path, map_location='cpu')
            config = GPTConfig(**checkpoint['model_args'])
            model = GPT(config)
            
            state_dict = checkpoint['model']
            # Nettoyage des préfixes 'compile'
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            return model
    except Exception as e:
        st.error(f"Erreur lors du téléchargement depuis Hugging Face : {e}")
        return None

# --- CHARGEMENT ---
model = load_model_from_hf()

if model is None:
    st.warning("⚠️ Impossible de charger le modèle. Vérifiez la connexion à Hugging Face.")
    st.stop()

# Chargement du dictionnaire meta.pkl (doit être sur ton GitHub)
try:
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
except FileNotFoundError:
    st.error("❌ Fichier 'meta.pkl' introuvable dans ton dépôt GitHub.")
    st.stop()

# --- INTERFACE UTILISATEUR ---
st.title("🟢 UiT nanoGPT Storyteller")
st.caption("Architecture 24-layers | TinyStories Dataset | Par Arthur PREVEL")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar pour les réglages
with st.sidebar:
    st.header("Paramètres")
    temp = st.slider("Créativité (Température)", 0.1, 1.2, 0.8)
    max_t = st.slider("Longueur max (Tokens)", 50, 500, 200)
    if st.button("🔄 Nouvelle discussion"):
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
        with st.spinner("Génération..."):
            context_ids = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
            # Appel de la fonction de génération de ton modèle
            output_ids = model.generate(context_ids, max_new_tokens=max_t, temperature=temp)[0].tolist()
            response = decode(output_ids)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
