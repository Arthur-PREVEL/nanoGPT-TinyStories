import streamlit as st
import torch
import pickle
import os
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UiT nanoGPT Storyteller", page_icon="🟢")

# --- HIGH-CONTRAST CSS THEME ---
st.markdown("""
    <style>
    /* Main page background: Clean light slate */
    .stApp { 
        background-color: #f8fafc; 
    }
    
    /* Sidebar: Pale green */
    [data-testid="stSidebar"] { 
        background-color: #f0fdf4; 
    }

    /* Target all text inside chat bubbles to be pure white and bold */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span {
        color: #ffffff !important;
        font-size: 1.05rem;
    }

    /* User Message Box: Bold Emerald Green */
    [data-testid="stChatMessageUser"] { 
        background-color: #059669 !important; 
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Bot Message Box: Deep Forest Green (Absolute Contrast) */
    [data-testid="stChatMessageAssistant"] { 
        background-color: #064e3b !important; 
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #065f46;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* External text (Title, Labels): Dark Forest Green */
    h1, h2, h3, label, .stMarkdown {
        color: #064e3b !important;
        font-weight: 600;
    }
    
    /* Sidebar Buttons */
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING LOGIC ---
REPO_ID = "Arthur-PREVEL/nanogpt-tinystories-depth24" 
FILENAME = "ckpt.pt"

@st.cache_resource
def load_model():
    try:
        with st.spinner("Initializing 24-layer Model..."):
            # Fetch weights from Hugging Face
            path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            checkpoint = torch.load(path, map_location='cpu')
            
            # Reconstruct architecture
            config = GPTConfig(**checkpoint['model_args'])
            model = GPT(config)
            
            # Remove potential torch.compile prefixes
            state_dict = checkpoint['model']
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            model.eval()
            return model
    except Exception as e:
        st.error(f"Hardware/Network Error: {e}")
        return None

# Load model and character mapping
model = load_model()
if model is None: st.stop()

try:
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
except FileNotFoundError:
    st.error("Error: meta.pkl missing from GitHub repository.")
    st.stop()

# --- CHAT ENGINE ---
st.title("🟢 UiT nanoGPT Storyteller")
st.write("Depth-24 Transformer Model | Character-Level Infrerence")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("Model Parameters")
    # Low temperature (0.1) prevents logical collapse
    temp = st.slider("Temperature (Creativity)", 0.1, 1.2, 0.1) 
    max_t = st.slider("Max Story Length (Tokens)", 50, 500, 250)
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Interaction
if prompt := st.chat_input("Start a story (e.g., 'Once upon a time...')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing narrative..."):
            # Process input and generate new tokens
            context = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
            output = model.generate(context, max_t, temperature=temp)[0].tolist()
            response = decode(output)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
