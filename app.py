import streamlit as st
import torch
import pickle
import os
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT

# --- PAGE CONFIGURATION & THEME ---
st.set_page_config(page_title="UiT nanoGPT Storyteller", page_icon="🟢")

# Custom CSS for the "Full Green" theme with white text for chat messages
st.markdown("""
    <style>
    /* Light green main background */
    .stApp { 
        background-color: #f0fdf4; 
    }
    
    /* Pale green sidebar */
    [data-testid="stSidebar"] { 
        background-color: #dcfce7; 
    }

    /* Force chat message text to white */
    .stChatMessage p {
        color: white !important;
    }

    /* User Bubble: Deep Emerald */
    .stChatMessage[data-testid="stChatMessageUser"] { 
        background-color: #065f46; 
        border-radius: 15px;
        padding: 15px;
    }

    /* Assistant Bubble: Forest Green */
    .stChatMessage[data-testid="stChatMessageAssistant"] { 
        background-color: #059669; 
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #047857;
    }

    /* Keep headers and labels dark for readability on light background */
    h1, h2, h3, label {
        color: #064e3b !important;
    }
    
    /* Custom Green Button */
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HUGGING FACE CONFIGURATION ---
REPO_ID = "Arthur-PREVEL/nanogpt-tinystories-depth24" 
FILENAME = "ckpt.pt"

@st.cache_resource
def load_model_from_hf():
    try:
        with st.spinner("Loading Artificial Intelligence..."):
            # Secure download from Hugging Face Hub
            path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            checkpoint = torch.load(path, map_location='cpu')
            
            # Reconstruct config and model
            config = GPTConfig(**checkpoint['model_args'])
            model = GPT(config)
            
            # Clean state_dict keys (removes torch.compile prefixes)
            state_dict = checkpoint['model']
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            model.eval()
            return model
    except Exception as e:
        st.error(f"Hugging Face Error: {e}")
        return None

# Load model
model = load_model_from_hf()

if model is None:
    st.stop()

# Load meta.pkl for character mapping (must be in your GitHub repo)
try:
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
except FileNotFoundError:
    st.error("❌ 'meta.pkl' not found in your GitHub repository!")
    st.stop()

# --- CHAT INTERFACE ---
st.title("🟢 UiT nanoGPT Storyteller")
st.write("Exploring scaling laws on the TinyStories dataset (24-layer depth model).")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    # Default temperature at 0.1 for maximum stability
    temp = st.slider("Creativity (Temperature)", 0.1, 1.2, 0.1) 
    max_t = st.slider("Story Length (Tokens)", 50, 500, 250)
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt input
if prompt := st.chat_input("Once upon a time..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Writing the story..."):
            # Encode input and generate from model
            context_ids = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
            # Using the generate function from your model.py
            output_ids = model.generate(context_ids, max_new_tokens=max_t, temperature=temp)[0].tolist()
            response = decode(output_ids)
            st.markdown(response)
    
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
