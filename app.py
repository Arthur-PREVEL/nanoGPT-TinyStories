import streamlit as st
import torch
import pickle
import os
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UiT nanoGPT Storyteller", page_icon="🟢")

# Custom CSS for high-contrast "Emerald Night" theme
st.markdown("""
    <style>
    /* Very light green background for the page */
    .stApp { 
        background-color: #f0fdf4; 
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { 
        background-color: #dcfce7; 
    }

    /* Force all chat text to white for contrast */
    .stChatMessage p, .stChatMessage span {
        color: #ffffff !important;
        font-weight: 400;
    }

    /* User Message "Box": Dark Emerald Green */
    .stChatMessage[data-testid="stChatMessageUser"] { 
        background-color: #064e3b !important; 
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }

    /* Bot Message "Box": Forest Green */
    .stChatMessage[data-testid="stChatMessageAssistant"] { 
        background-color: #059669 !important; 
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #047857;
    }

    /* Make headers and slider labels dark green for visibility */
    h1, h2, h3, label, .stMarkdown {
        color: #064e3b;
    }
    
    /* Green sidebar button */
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 10px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HUGGING FACE SETUP ---
REPO_ID = "Arthur-PREVEL/nanogpt-tinystories-depth24" 
FILENAME = "ckpt.pt"

@st.cache_resource
def load_model_from_hf():
    try:
        with st.spinner("Downloading weights from Hugging Face Hub..."):
            path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            checkpoint = torch.load(path, map_location='cpu')
            
            # Reconstruct model from checkpoint args
            config = GPTConfig(**checkpoint['model_args'])
            model = GPT(config)
            
            # Remove 'compile' prefixes if present
            state_dict = checkpoint['model']
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            model.eval()
            return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Initialization
model = load_model_from_hf()
if model is None:
    st.stop()

# Load meta.pkl for character encoding/decoding
try:
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
except FileNotFoundError:
    st.error("Missing 'meta.pkl' in the repository!")
    st.stop()

# --- CHAT INTERFACE ---
st.title("🟢 UiT nanoGPT Storyteller")
st.write("Using a 24-layer depth model trained on the TinyStories dataset.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("Parameters")
    # Low temperature (0.1) provides stable, logical text generation
    temp = st.slider("Creativity (Temperature)", 0.1, 1.2, 0.1) 
    max_t = st.slider("Story Length (Tokens)", 50, 500, 250)
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt
if prompt := st.chat_input("Enter the start of a story..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating story..."):
            # Prepare context and generate tokens
            context_ids = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
            output_ids = model.generate(context_ids, max_new_tokens=max_t, temperature=temp)[0].tolist()
            response = decode(output_ids)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
