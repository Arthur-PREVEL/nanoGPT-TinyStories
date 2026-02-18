import streamlit as st
import torch
import pickle
import time
import datetime
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLE
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="UiT nanoGPT Assistant", 
    page_icon="✨", 
    layout="centered"
)

# CSS "Clean Interface" inspiré de Snowflake mais avec l'identité UiT
st.markdown("""
    <style>
    /* Fond global très propre */
    .stApp { background-color: #ffffff; }
    
    /* Suppression du padding haut pour un look 'App' */
    .block-container { padding-top: 3rem; }

    /* Style des bulles de chat */
    .stChatMessage { 
        padding: 1rem; 
        border-radius: 12px; 
        margin-bottom: 1rem; 
    }
    
    /* Bulle Utilisateur : Vert UiT Professionnel */
    [data-testid="stChatMessageUser"] { 
        background-color: #059669; 
    }
    /* Texte utilisateur en blanc */
    [data-testid="stChatMessageUser"] p { color: white !important; }

    /* Bulle Assistant : Gris très léger (Style ChatGPT/Claude) */
    [data-testid="stChatMessageAssistant"] { 
        background-color: #f8fafc; 
        border: 1px solid #e2e8f0;
    }
    /* Texte assistant en noir/gris foncé */
    [data-testid="stChatMessageAssistant"] p { color: #1e293b !important; }

    /* Titres et Labels */
    h1, h2, h3 { color: #064e3b !important; font-family: 'Helvetica', sans-serif; }
    
    /* Input field stylisé */
    .stChatInput > div { border-color: #cbd5e1 !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTES & DONNÉES
# -----------------------------------------------------------------------------

REPO_ID = "Arthur-PREVEL/nanogpt-tinystories-depth24" 
FILENAME = "ckpt.pt"

# Suggestions affichées au démarrage (Pills)
SUGGESTIONS = {
    "🐶 A dog named Max": "Once upon a time, there was a dog named Max who loved to...",
    "🏰 The Magic Castle": "Lily found a secret door that led to a big, shining castle...",
    "🍪 The Lost Cookie": "Tom was sad because he lost his favorite cookie in the...",
    "🤖 The Happy Robot": "A little robot wanted to make friends, so he went to the...",
}

# -----------------------------------------------------------------------------
# 3. CHARGEMENT DU MODÈLE (BACKEND)
# -----------------------------------------------------------------------------

@st.cache_resource
def load_resources():
    try:
        # Téléchargement & Chargement Modèle
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        checkpoint = torch.load(path, map_location='cpu')
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config)
        
        state_dict = checkpoint['model']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

        # Chargement Meta (Tokenization)
        with open('meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        
        return model, meta['stoi'], meta['itos']
    except Exception as e:
        return None, None, None

model, stoi, itos = load_resources()

# Helpers pour encoder/decoder
if stoi and itos:
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    st.error("Erreur critique: Impossible de charger le modèle ou meta.pkl")
    st.stop()

# -----------------------------------------------------------------------------
# 4. FONCTIONS UTILITAIRES UI
# -----------------------------------------------------------------------------

def clear_conversation():
    st.session_state.messages = []
    st.session_state.selected_suggestion = None

@st.dialog("Project Disclaimers")
def show_disclaimer():
    st.write("### INF-3600 Project Info")
    st.caption("""
    This AI assistant uses a **24-layer Transformer** trained from scratch on the **TinyStories** dataset.
    
    * **Capabilities:** Generates simple narratives with coherent structure.
    * **Limitations:** May exhibit hallucinations or grammatical loops typical of small language models (18M params).
    * **Privacy:** No data is stored externally. Everything runs in ephemeral memory.
    
    *Author: Arthur PREVEL - UiT The Arctic University of Norway*
    """)

# Simulation de streaming pour l'effet visuel "Claude/ChatGPT"
def stream_text(full_text):
    for word in full_text.split(" "):
        yield word + " "
        time.sleep(0.02) # Vitesse d'écriture

# -----------------------------------------------------------------------------
# 5. INTERFACE PRINCIPALE
# -----------------------------------------------------------------------------

# Initialisation de l'état
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- HEADER ---
# On affiche un header différent si c'est la page d'accueil ou si le chat est actif
has_history = len(st.session_state.messages) > 0

if has_history:
    col1, col2 = st.columns([6, 1])
    col1.title("🟢 nanoGPT Assistant")
    col2.button("Restart", icon="🔄", on_click=clear_conversation, use_container_width=True)
else:
    # Grand logo pour l'état vide
    st.markdown("<div style='text-align: center; margin-bottom: 2rem; font-size: 4rem;'>✨</div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>How can I help you write today?</h1>", unsafe_allow_html=True)

# --- ZONE D'AFFICHAGE DU CHAT ---
if has_history:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- INPUT & LOGIQUE DE DÉMARRAGE ---

# Gestion des suggestions (Pills)
user_prompt = None
selected_suggestion = None

# Si pas d'historique, on affiche les suggestions au milieu
if not has_history:
    st.write("") # Spacer
    st.write("") 
    
    # Input field central
    user_prompt = st.chat_input("Start a story...", key="main_input")
    
    # Suggestions sous l'input
    selected_suggestion = st.pills(
        "Try these examples:",
        options=SUGGESTIONS.keys(),
        selection_mode="single"
    )
    
    # Bouton Disclaimer discret
    st.markdown("<br>", unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
    with col_c2:
        if st.button("ℹ️ Model Architecture & Info", type="tertiary", use_container_width=True):
            show_disclaimer()

# Si historique, l'input est géré par Streamlit automatiquement en bas, 
# mais on doit récupérer la valeur si c'est un prompt "suivi"
if has_history:
    user_prompt = st.chat_input("Continue the story...")

# --- LOGIQUE DE TRAITEMENT ---

# Priorité : Input texte > Suggestion cliquée
final_prompt = None
if user_prompt:
    final_prompt = user_prompt
elif selected_suggestion and not has_history:
    # Si on clique sur une pilule, on prend le texte associé
    final_prompt = SUGGESTIONS[selected_suggestion]

# Exécution
if final_prompt:
    # 1. Affichage User
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    if not has_history:
        st.rerun() # Pour rafraîchir l'interface et passer en mode "Chat"
    else:
        with st.chat_message("user"):
            st.markdown(final_prompt)

    # 2. Génération Bot
    with st.chat_message("assistant"):
        with st.spinner("Generating narrative structure..."):
            # Encodage
            context_ids = torch.tensor(encode(final_prompt), dtype=torch.long)[None, ...]
            
            # Génération (Temperature basse pour la démo clean)
            output_ids = model.generate(context_ids, max_new_tokens=300, temperature=0.2)[0].tolist()
            full_response = decode(output_ids)
            
            # Affichage en streaming
            response_placeholder = st.empty()
            streamed_text = ""
            for chunk in stream_text(full_response):
                streamed_text += chunk
                response_placeholder.markdown(streamed_text + "▌")
            
            response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
