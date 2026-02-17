import streamlit as st
import torch
import pickle
import os
import requests
from model import GPTConfig, GPT

# --- CONFIGURATION & THÈME ---
st.set_page_config(page_title="UiT nanoGPT Chat", page_icon="🐦")

# CSS pour le thème Light Green style ChatGPT
st.markdown("""
    <style>
    .stApp { background-color: #f7fdf9; }
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #e8f5e9; border: 1px solid #c8e6c9; }
    .stChatMessage[data-testid="stChatMessageAssistant"] { background-color: #ffffff; border: 1px solid #e0e0e0; }
    .stButton>button { background-color: #4caf50; color: white; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DU MODÈLE ---
FILE_ID = '1rLJSJQwdvRhRS8KdYjffTM-jkhPM0zGr'
CKPT_PATH = 'ckpt.pt'

@st.cache_resource
def download_and_load_model():
    # Téléchargement si le fichier n'existe pas
    if not os.path.exists(CKPT_PATH):
        with st.spinner("Downloading model (218MB)... This might take a minute."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            response = requests.get(url, stream=True)
            with open(CKPT_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    # Chargement
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    
    state_dict = checkpoint['model']
    # Nettoyage des préfixes torch.compile si nécessaire
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- CHARGEMENT DES METADATA ---
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

model = download_and_load_model()

# --- INTERFACE ---
st.title("🟢 UiT nanoGPT Storyteller")
st.caption("Modèle : Depth-24 layers | Dataset : TinyStories")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar pour les réglages
with st.sidebar:
    st.header("Settings")
    temp = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.8)
    tokens = st.slider("Story Length", 50, 500, 200)

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Write the start of a story..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération
    with st.chat_message("assistant"):
        input_ids = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
        with st.spinner("Writing..."):
            # Simulation de génération (Remplace par ta fonction model.generate)
            y = model.generate(input_ids, tokens, temperature=temp)[0].tolist()
            response = decode(y)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
