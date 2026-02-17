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
    if not os.path.exists(CKPT_PATH):
        with st.spinner("Downloading model (218MB) from Google Drive..."):
            # Nouvelle méthode pour contourner l'avertissement "fichier volumineux"
            def get_confirm_token(response):
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        return value
                return None

            def save_response_content(response, destination):
                CHUNK_SIZE = 32768
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk: 
                            f.write(chunk)

            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            response = session.get(URL, params={'id': FILE_ID}, stream=True)
            token = get_confirm_token(response)

            if token:
                params = {'id': FILE_ID, 'confirm': token}
                response = session.get(URL, params=params, stream=True)
            
            save_response_content(response, CKPT_PATH)

    # Vérification : si le fichier est tout petit, c'est qu'il y a eu un souci
    if os.path.getsize(CKPT_PATH) < 1000000: # moins de 1Mo
        os.remove(CKPT_PATH)
        st.error("Erreur de téléchargement : le fichier est corrompu ou Google bloque l'accès.")
        return None

    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    
    state_dict = checkpoint['model']
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
